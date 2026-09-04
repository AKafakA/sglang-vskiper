"""Selection-style FlexiDepth paper-suite evals through an SGLang server.

Use this for multiple-choice tasks where free-form generation is the wrong
metric. The current implementation covers HellaSwag, Winogrande, and MMLU with
the paper's 5-shot setting and SGLang's normalized-logprob ``select`` path,
which is the local no-install substitute for lm-evaluation-harness style choice
scoring.

Launch the server separately with the desired mode/env, then run this script
against that port. It saves per-example JSONL outputs for parity checks.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import datasets

from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import download_and_cache_file, normalize_base_url, read_jsonl

TASK = os.environ.get("FD_TASK", "hellaswag").lower()
N = int(os.environ.get("FD_N", "100"))
SHOTS = int(os.environ.get("FD_SHOTS", "5"))
OUT = os.environ.get("FD_OUT", "")
HOST = os.environ.get("FD_HOST", "127.0.0.1")
PORT = int(os.environ.get("FD_PORT", "30000"))
PARALLEL = int(os.environ.get("FD_PARALLEL", "64"))
MODE = os.environ.get("FD_MODE_LABEL", "unknown")


def _hellaswag_example(row: dict[str, Any], include_answer: bool) -> str:
    text = row["activity_label"] + ": " + row["ctx"] + " "
    if include_answer:
        text += row["endings"][int(row["label"])]
    return text


def _load_hellaswag(n: int, shots: int) -> tuple[str, list[dict[str, Any]]]:
    url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
    filename = download_and_cache_file(url)
    rows = list(read_jsonl(filename))
    few_shot = "\n\n".join(_hellaswag_example(rows[i], True) for i in range(shots))
    items = []
    for idx, row in enumerate(rows[:n]):
        items.append(
            {
                "id": str(idx),
                "question": _hellaswag_example(row, False),
                "choices": list(row["endings"]),
                "label": int(row["label"]),
                "activity_label": row["activity_label"],
            }
        )
    return few_shot, items


def _winogrande_example(row: dict[str, Any], include_answer: bool) -> str:
    text = "Fill in the blank with the correct option.\nSentence: "
    text += row["sentence"].strip() + "\nAnswer:"
    if include_answer:
        answer_idx = int(row["answer"]) - 1
        text += " " + [row["option1"], row["option2"]][answer_idx]
    return text


def _load_winogrande(n: int, shots: int) -> tuple[str, list[dict[str, Any]]]:
    train = datasets.load_dataset(
        "allenai/winogrande", "winogrande_xl", split="train"
    )
    validation = datasets.load_dataset(
        "allenai/winogrande", "winogrande_xl", split=f"validation[:{n}]"
    )
    few_shot = "\n\n".join(_winogrande_example(train[i], True) for i in range(shots))
    items = []
    for idx, row in enumerate(validation):
        choices = [row["option1"], row["option2"]]
        items.append(
            {
                "id": str(idx),
                "question": _winogrande_example(row, False),
                "choices": choices,
                "label": int(row["answer"]) - 1,
                "sentence": row["sentence"],
            }
        )
    return few_shot, items


_MMLU_LETTERS = ["A", "B", "C", "D"]


def _mmlu_prompt(row: dict[str, Any], include_answer: bool) -> str:
    subject = row["subject"].replace("_", " ")
    text = (
        "The following are multiple choice questions (with answers) about "
        f"{subject}.\n\n"
    )
    text += row["question"].strip()
    for letter, choice in zip(_MMLU_LETTERS, row["choices"]):
        text += f"\n{letter}. {choice}"
    text += "\nAnswer:"
    if include_answer:
        text += " " + _MMLU_LETTERS[int(row["answer"])]
    return text


def _round_robin_by_subject(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    subjects = []
    for row in rows:
        subject = row["subject"]
        if subject not in by_subject:
            by_subject[subject] = []
            subjects.append(subject)
        by_subject[subject].append(row)

    selected = []
    cursor = 0
    while len(selected) < n:
        made_progress = False
        for subject in subjects:
            bucket = by_subject[subject]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                made_progress = True
                if len(selected) >= n:
                    break
        if not made_progress:
            break
        cursor += 1
    return selected


def _load_mmlu(n: int, shots: int) -> tuple[str, list[dict[str, Any]]]:
    dev_rows = list(datasets.load_dataset("cais/mmlu", "all", split="dev"))
    test_rows = list(datasets.load_dataset("cais/mmlu", "all", split="test"))

    dev_by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in dev_rows:
        dev_by_subject.setdefault(row["subject"], []).append(row)

    items = []
    for idx, row in enumerate(_round_robin_by_subject(test_rows, n)):
        subject_dev = dev_by_subject.get(row["subject"], [])
        few_shot = "\n\n".join(
            _mmlu_prompt(example, True) for example in subject_dev[:shots]
        )
        items.append(
            {
                "id": str(idx),
                "question": _mmlu_prompt(row, False),
                "choices": [f" {letter}" for letter in _MMLU_LETTERS],
                "label": int(row["answer"]),
                "subject": row["subject"],
                "choice_texts": list(row["choices"]),
                "few_shot": few_shot,
            }
        )
    return "", items


def _load_task(task: str, n: int, shots: int):
    if task == "hellaswag":
        return _load_hellaswag(n, shots)
    if task == "winogrande":
        return _load_winogrande(n, shots)
    if task == "mmlu":
        return _load_mmlu(n, shots)
    raise ValueError(
        f"unsupported FD_TASK={task!r}; expected hellaswag, winogrande, or mmlu"
    )


def _row_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in ("activity_label", "sentence", "subject", "choice_texts"):
        if key in item:
            metadata[key] = item[key]
    return metadata


def main() -> None:
    import sglang as sgl

    few_shot, items = _load_task(TASK, N, SHOTS)
    set_default_backend(RuntimeEndpoint(normalize_base_url(HOST, PORT)))
    arguments = [
        {
            "few_shot": item.get("few_shot", few_shot),
            "question": item["question"],
            "choices": item["choices"],
        }
        for item in items
    ]

    @sgl.function
    def few_shot_select(s, few_shot, question, choices):
        s += few_shot + "\n\n" + question
        s += sgl.select("answer", choices=choices)

    start = time.perf_counter()
    states = few_shot_select.run_batch(
        arguments,
        temperature=0.0,
        num_threads=PARALLEL,
        progress_bar=True,
    )
    elapsed = time.perf_counter() - start

    rows = []
    correct = []
    for item, state in zip(items, states):
        prediction = state["answer"]
        pred_idx = item["choices"].index(prediction)
        is_correct = pred_idx == item["label"]
        correct.append(is_correct)
        try:
            meta = state.get_meta_info("answer")
        except Exception:
            meta = {}
        rows.append(
            {
                "task": TASK,
                "mode": MODE,
                "id": item["id"],
                "prediction": prediction,
                "pred_idx": pred_idx,
                "label": item["label"],
                "score": float(is_correct),
                "choices": item["choices"],
                "question": item["question"],
                "meta_info": meta,
                **_row_metadata(item),
            }
        )

    accuracy = float(np.mean(correct)) if correct else 0.0
    summary = {
        "task": TASK,
        "mode": MODE,
        "num_examples": len(items),
        "num_shots": SHOTS,
        "metric": "acc_norm_via_sglang_select",
        "score": accuracy,
        "elapsed_s": elapsed,
        "parallel": PARALLEL,
    }

    if OUT:
        out_path = Path(OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        with out_path.with_suffix(out_path.suffix + ".summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(
        "FD-SELECT-EVAL "
        f"[{MODE}] {TASK}: acc_norm={accuracy:.3f} "
        f"(n={len(items)}, shots={SHOTS}) elapsed={elapsed:.1f}s",
        flush=True,
    )
    print("FD-SELECT-EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
