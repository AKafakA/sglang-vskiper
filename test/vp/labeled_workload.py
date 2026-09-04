"""Labeled real-workload construction and scoring for VP serving tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import re
import string
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Optional


DECODE_DATASETS = ("gsm8k", "coqa", "humaneval")
# Broader-decode workloads (v1.3): standalone named workloads only — they are
# deliberately NOT part of DECODE_DATASETS so the decode_mix/mixed composition
# contracts stay byte-stable.
EXTENDED_DECODE_DATASETS = (
    "gsm8k_cot",
    "ifeval",
    "mmlu_pro",
    "mmlu_pro_cot",
    # Long-context writing row (sealed axis A1-A6, 2026-09-04). Phase is
    # "decode" because these generate 1024 tokens -- ~55% of cell time is
    # decode -- unlike LongBench's gov_report/multi_news in PREFILL_DATASETS,
    # whose 512-token outputs over the same documents are prefill-dominated.
    "scrolls_gov_report",
    "scrolls_summ_screen_fd",
    "scrolls_qmsum",
)
PREFILL_DATASETS = (
    "mmlu",
    "hellaswag",
    "winogrande",
    "gov_report",
    "multi_news",
)
ALL_DATASETS = DECODE_DATASETS + EXTENDED_DECODE_DATASETS + PREFILL_DATASETS
WORKLOAD_DATASETS = {
    **{name: (name,) for name in ALL_DATASETS},
    "decode_mix": DECODE_DATASETS,
    "mixed": DECODE_DATASETS + PREFILL_DATASETS[:3],
    "prefill_core": PREFILL_DATASETS[:3],
    "prefill_long": PREFILL_DATASETS[3:],
    # Sealed writing row: SCROLLS trio, window-FILTERED (never truncated).
    "longctx_writing": (
        "scrolls_gov_report",
        "scrolls_summ_screen_fd",
        "scrolls_qmsum",
    ),
    "mixed_25": DECODE_DATASETS + PREFILL_DATASETS[:3],
    "mixed_50": DECODE_DATASETS + PREFILL_DATASETS[:3],
    "mixed_75": DECODE_DATASETS + PREFILL_DATASETS[:3],
    "mixed_all8": DECODE_DATASETS + PREFILL_DATASETS,
}
WORKLOAD_PHASE = {
    **{name: "decode" for name in DECODE_DATASETS},
    **{name: "decode" for name in EXTENDED_DECODE_DATASETS},
    **{name: "prefill" for name in PREFILL_DATASETS},
    "decode_mix": "decode",
    "mixed": "mixed",
    "prefill_core": "prefill",
    "prefill_long": "prefill",
    "longctx_writing": "decode",
    "mixed_25": "mixed",
    "mixed_50": "mixed",
    "mixed_75": "mixed",
    "mixed_all8": "mixed",
}
DEFAULT_DECODE_WEIGHTS = {
    # Proportional to the MEASURED window-fit survivors (330/145/38 of
    # 972/338/272) so the three pools drain together under sample_items'
    # weighted draw; equal weights would exhaust qmsum's 38 rows first.
    "scrolls_gov_report": 0.643,
    "scrolls_summ_screen_fd": 0.283,
    "scrolls_qmsum": 0.074,
    "gsm8k": 0.45,
    "coqa": 0.45,
    "humaneval": 0.10,
}
DEFAULT_PREFILL_WEIGHTS = {
    "mmlu": 1.0 / 3.0,
    "hellaswag": 1.0 / 3.0,
    "winogrande": 1.0 / 3.0,
    "gov_report": 0.0,
    "multi_news": 0.0,
}
DEFAULT_DATASET_WEIGHTS = {**DEFAULT_DECODE_WEIGHTS, **DEFAULT_PREFILL_WEIGHTS}

PROTOCOL_SCHEMA_VERSION = 4
PROTOCOL_SUITE_ID = "flexidepth-lm-eval-0.4.9.1-serving-v1"
# Uniform generation penalty applied to every request body of a suite.
#
# The default leaves every request body and every metadata row byte-identical to
# a suite built before the penalty became a builder input, so the frozen payload
# does not move. The value itself is ALWAYS declared in the suite summary — an
# absent declaration is an error rather than an implicit 0.0 — and it always
# binds the suite identity hashes, so a rebuild that drops the policy changes the
# hashes instead of silently producing a penalty-free suite.
DEFAULT_FREQUENCY_PENALTY = 0.0
FREQUENCY_PENALTY_FIELD = "frequency_penalty"
DATASET_PROTOCOLS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "id": "lm-eval-0.4.9.1:gsm8k-v3:5shot-multiturn",
        "source": "openai/gsm8k:main",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 256,
    },
    "gsm8k_cot": {
        "id": "lm-eval-0.4.9.1:gsm8k_cot-v3:8shot-fixed-multiturn",
        "source": "openai/gsm8k:main",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 256,
    },
    "mmlu_pro": {
        "id": "lm-eval-0.4.9.1:mmlu_pro-native:5shot-letter-chat",
        "source": "TIGER-Lab/MMLU-Pro",
        "evaluation_split": "test",
        "quality_semantics": "declared_native_variant",
        "task_reference_max_output_len": 16,
    },
    "mmlu_pro_cot": {
        "id": "lm-eval-0.4.9.1:mmlu_pro-cot:5shot-cot-chat",
        "source": "TIGER-Lab/MMLU-Pro",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 1024,
    },
    "ifeval": {
        "id": "lm-eval-0.4.9.1:ifeval-v4:zero-shot-chat",
        "source": "google/IFEval",
        "evaluation_split": "train",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 1280,
    },
    "coqa": {
        "id": "lm-eval-0.4.9.1:coqa-v3:zero-shot-raw",
        "source": "EleutherAI/coqa",
        "evaluation_split": "validation",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 256,
    },
    "scrolls_gov_report": {
        "id": "lm-eval:longbench_gov_report:zero-shot-chat",
        "source": "tau/scrolls",
        "evaluation_split": "validation",
        "quality_semantics": "longbench_rouge_offline",
        "task_reference_max_output_len": 1024,
    },
    "scrolls_summ_screen_fd": {
        "id": "lm-eval:scrolls_summscreenfd:zero-shot-chat",
        "source": "tau/scrolls",
        "evaluation_split": "validation",
        "quality_semantics": "longbench_rouge_offline",
        "task_reference_max_output_len": 1024,
    },
    "scrolls_qmsum": {
        "id": "lm-eval:scrolls_qmsum:zero-shot-chat",
        "source": "tau/scrolls",
        "evaluation_split": "validation",
        "quality_semantics": "longbench_rouge_offline",
        "task_reference_max_output_len": 1024,
    },
    "humaneval": {
        "id": "lm-eval-0.4.9.1:humaneval-v1:zero-shot-raw",
        "source": "openai/openai_humaneval",
        "evaluation_split": "test",
        "quality_semantics": "paper_exact",
        "task_reference_max_output_len": 1024,
    },
    "mmlu": {
        "id": "lm-eval-0.4.9.1:mmlu-v1:5shot-multiturn-prefill-probe",
        "source": "cais/mmlu",
        "evaluation_split": "test",
        "quality_semantics": "paper_prompt_prefill_probe",
        "task_reference_max_output_len": 1,
    },
    "hellaswag": {
        "id": "lm-eval-0.4.9.1:hellaswag-v1:5shot-multiturn-prefill-probe",
        "source": "Rowan/hellaswag",
        "evaluation_split": "validation",
        "quality_semantics": "paper_prompt_prefill_probe",
        "task_reference_max_output_len": 1,
    },
    "winogrande": {
        "id": "lm-eval-0.4.9.1:winogrande-v1:5shot-multiturn-prefill-probe",
        "source": "allenai/winogrande:winogrande_xl",
        "evaluation_split": "validation",
        "quality_semantics": "paper_prompt_prefill_probe",
        "task_reference_max_output_len": 1,
    },
    "gov_report": {
        "id": "longbench-v2:gov_report:zero-shot",
        "source": "THUDM/LongBench",
        "evaluation_split": "test",
        "quality_semantics": "longbench_rouge_l",
        "task_reference_max_output_len": 512,
    },
    "multi_news": {
        "id": "longbench-v2:multi_news:zero-shot",
        "source": "THUDM/LongBench",
        "evaluation_split": "test",
        "quality_semantics": "longbench_rouge_l",
        "task_reference_max_output_len": 512,
    },
}


@dataclass
class WorkloadItem:
    dataset: str
    item_id: str
    phase: str
    prompt: str | list[dict[str, str]]
    prompt_kind: str
    reference_output_len: Optional[int]
    metric: str
    gold: Any
    protocol_id: str
    quality_semantics: str
    task_reference_max_output_len: Optional[int]
    stop: list[str] = field(default_factory=list)
    evaluator_data: dict[str, Any] = field(default_factory=dict)
    source_item_id: Optional[str] = None
    replica_index: int = 0


def _load_module(name: str, filename: str):
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _gsm8k_answer(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def _gsm8k_example(row: dict[str, Any], include_answer: bool) -> str:
    text = f"Question: {row['question'].strip()}\nAnswer:"
    if include_answer:
        text += " " + row["answer"].strip()
    return text


def _gsm8k_chat_messages(
    train: Any, row: dict[str, Any], fewshot_indices: list[int]
) -> list[dict[str, str]]:
    messages = []
    for shot_index in fewshot_indices:
        shot = train[shot_index]
        messages.extend(
            [
                {"role": "user", "content": _gsm8k_example(shot, False)},
                {"role": "assistant", "content": shot["answer"]},
            ]
        )
    messages.append({"role": "user", "content": _gsm8k_example(row, False)})
    return messages


def load_gsm8k(limit: int, include_train_pool: bool = False) -> list[WorkloadItem]:
    import datasets

    train = datasets.load_dataset("openai/gsm8k", "main", split="train")
    test = datasets.load_dataset("openai/gsm8k", "main", split="test")
    protocol = DATASET_PROTOCOLS["gsm8k"]
    fewshot_rng = random.Random(1234)
    items: list[WorkloadItem] = []

    def _emit(split_name: str, rows: Any, forbid_self: bool) -> None:
        # test is emitted first and unchanged (same order + same few-shot draws) so
        # eval-split items stay byte-identical when include_train_pool is False.
        for index, row in enumerate(rows):
            if len(items) >= limit:
                return
            fewshot_indices = fewshot_rng.sample(range(len(train)), 5)
            while forbid_self and index in fewshot_indices:
                fewshot_indices = fewshot_rng.sample(range(len(train)), 5)
            messages = _gsm8k_chat_messages(train, row, fewshot_indices)
            items.append(
                WorkloadItem(
                    dataset="gsm8k",
                    item_id=f"{split_name}:{index}",
                    phase="decode",
                    prompt=messages,
                    prompt_kind="chat_messages",
                    reference_output_len=None,
                    metric="gsm8k_strict_match",
                    gold=_gsm8k_answer(row["answer"]),
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    stop=["Question:", "</s>", "<|im_end|>"],
                    evaluator_data={
                        "source_split": split_name,
                        "fewshot_indices": fewshot_indices,
                        "fewshot_seed": 1234,
                    },
                )
            )

    _emit("test", test, forbid_self=False)
    if include_train_pool:
        # Perf/throughput lane only: train rows join the SERVING load (~8.8k total).
        # The quality lane scores the test split only, so no leak.
        _emit("train", train, forbid_self=True)
    return items


# The 8 fixed chain-of-thought exemplars from lm-eval 0.4.9.1
# lm_eval/tasks/gsm8k/gsm8k-cot.yaml (fewshot_config.samples, sampler
# first_n) — reproduced VERBATIM; any wording drift breaks protocol parity.
GSM8K_COT_FEWSHOT: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in"
        " the grove today. After they are done, there will be 21 trees. How"
        " many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some"
        " more were planted. So there must have been 21 - 15 = 6. The answer"
        " is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how"
        " many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The"
        " answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how"
        " many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total"
        " they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The"
        " answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has"
        " 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some"
        " to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his"
        " mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and"
        " dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers"
        " were installed each day, from monday to thursday. How many"
        " computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more"
        " computers were added. So 5 * 4 = 20 computers were added. 9 + 20"
        " is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On"
        " wednesday, he lost 2 more. How many golf balls did he have at the"
        " end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he"
        " had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf"
        " balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money"
        " does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 ="
        " 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The"
        " answer is 8.",
    ),
)


def _gsm8k_cot_question(question: str) -> str:
    # doc_to_text of gsm8k-cot.yaml: "Q: {{question}}\n\nA:"
    return f"Q: {question.strip()}\n\nA:"


def load_gsm8k_cot(limit: int) -> list[WorkloadItem]:
    import datasets

    test = datasets.load_dataset("openai/gsm8k", "main", split="test")
    protocol = DATASET_PROTOCOLS["gsm8k_cot"]
    shot_messages = []
    for shot_question, shot_target in GSM8K_COT_FEWSHOT:
        shot_messages.extend(
            [
                {"role": "user", "content": _gsm8k_cot_question(shot_question)},
                {"role": "assistant", "content": shot_target},
            ]
        )
    items = []
    for index, row in enumerate(test.select(range(min(limit, len(test))))):
        messages = [
            *shot_messages,
            {"role": "user", "content": _gsm8k_cot_question(row["question"])},
        ]
        items.append(
            WorkloadItem(
                dataset="gsm8k_cot",
                item_id=f"test:{index}",
                phase="decode",
                prompt=messages,
                prompt_kind="chat_messages",
                reference_output_len=None,
                metric="gsm8k_cot_strict_match",
                gold=_gsm8k_answer(row["answer"]),
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=protocol[
                    "task_reference_max_output_len"
                ],
                stop=["Q:", "</s>", "<|im_end|>"],
                evaluator_data={
                    "source_split": "test",
                    "fewshot": "gsm8k_cot_v3_first8_fixed",
                },
            )
        )
    return items


_MMLU_PRO_LETTERS = "ABCDEFGHIJ"


def _mmlu_pro_question(row: dict) -> str:
    # lm-eval mmlu_pro doc_to_text shape: Question / Options / Answer
    lines = ["Question:", row["question"].strip(), "Options:"]
    # The published dataset pads every options list to 10 with "N/A"; the
    # official MMLU-Pro loader removes those placeholders before lettering.
    options = [o for o in row["options"] if o != "N/A"]
    for letter, option in zip(_MMLU_PRO_LETTERS, options):
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def _load_mmlu_pro_shots(cot: bool) -> dict[str, list[dict]]:
    import datasets

    validation = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    by_category: dict[str, list[dict]] = {}
    for row in validation:
        by_category.setdefault(row["category"], []).append(row)
    shots: dict[str, list[dict]] = {}
    for category, rows in by_category.items():
        messages = []
        for row in rows[:5]:
            if cot:
                # validation cot_content begins "A: Let's think step by
                # step." — the user turn already carries that cue, so strip
                # the FULL fixed prefix (review P2: avoid duplicating it).
                target = row["cot_content"].strip()
                for prefix in ("A: Let's think step by step.", "A:"):
                    if target.startswith(prefix):
                        target = target[len(prefix):].strip()
                        break
                question = _mmlu_pro_question(row) + "\nAnswer: Let's think step by step."
            else:
                target = f"The answer is ({row['answer']})."
                question = _mmlu_pro_question(row) + "\nAnswer:"
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": target},
                ]
            )
        shots[category] = messages
    return shots


def load_mmlu_pro(limit: int, cot: bool) -> list[WorkloadItem]:
    import datasets

    dataset_name = "mmlu_pro_cot" if cot else "mmlu_pro"
    protocol = DATASET_PROTOCOLS[dataset_name]
    test = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    shots = _load_mmlu_pro_shots(cot)
    missing = sorted({row["category"] for row in test} - set(shots))
    if missing:
        raise ValueError(
            f"MMLU-Pro test categories missing validation shots: {missing}"
        )
    items = []
    for index, row in enumerate(test):
        if len(items) >= limit:
            break
        suffix = "\nAnswer: Let's think step by step." if cot else "\nAnswer:"
        messages = [
            *shots[row["category"]],
            {"role": "user", "content": _mmlu_pro_question(row) + suffix},
        ]
        items.append(
            WorkloadItem(
                dataset=dataset_name,
                item_id=f"test:{index}",
                phase="decode",
                prompt=messages,
                prompt_kind="chat_messages",
                reference_output_len=None,
                metric="mmlu_pro_letter_match",
                gold=str(row["answer"]),
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=protocol[
                    "task_reference_max_output_len"
                ],
                stop=["Question:", "</s>", "<|im_end|>"],
                evaluator_data={
                    "source_split": "test",
                    "category": row["category"],
                    "fewshot": "mmlu_pro_validation_first5_per_category",
                },
            )
        )
    return items


def mmlu_pro_letter_match(prediction: str, gold: str) -> float:
    # lm-eval mmlu_pro filters, patterns VERBATIM: primary
    # 'answer is \(?([ABCDEFGHIJ])\)?', fallback r'.*[aA]nswer:\s*\(?([ABCDEFGHIJ])\)?'
    match = re.search(r"answer is \(?([ABCDEFGHIJ])\)?", prediction)
    if match is None:
        match = re.search(r".*[aA]nswer:\s*\(?([ABCDEFGHIJ])\)?", prediction)
    if match is None:
        return 0.0
    return float(match.group(1) == str(gold).strip())


def load_ifeval(limit: int) -> list[WorkloadItem]:
    import datasets

    rows = datasets.load_dataset("google/IFEval", split="train")
    protocol = DATASET_PROTOCOLS["ifeval"]
    items = []
    for index, row in enumerate(rows.select(range(min(limit, len(rows))))):
        items.append(
            WorkloadItem(
                dataset="ifeval",
                item_id=f"train:{index}",
                phase="decode",
                prompt=[{"role": "user", "content": row["prompt"]}],
                prompt_kind="chat_messages",
                reference_output_len=None,
                # Scored ONLY by the third-party harness offline
                # (score_lmeval_offline.py); no in-repo reimplementation of
                # the IFEval instruction registry.
                metric="ifeval_offline",
                gold=None,
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=protocol[
                    "task_reference_max_output_len"
                ],
                stop=[],
                evaluator_data={
                    "source_split": "train",
                    "key": row["key"],
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                },
            )
        )
    return items


def _coqa_questions(row: dict[str, Any]) -> list[str]:
    questions = row["questions"]
    if isinstance(questions, dict):
        values = questions.get("input_text") or questions.get("question") or []
        return [str(value) for value in values]
    return [str(value) for value in questions]


def _coqa_gold_answers(row: dict[str, Any], turn: int) -> list[str]:
    answers = row["answers"]
    if isinstance(answers, dict):
        values = []
        for key in ("input_text", "answer"):
            candidate = answers.get(key)
            if isinstance(candidate, list) and turn < len(candidate):
                values.append(str(candidate[turn]))
            elif isinstance(candidate, str) and turn == 0:
                values.append(candidate)
        primary = [value for value in values if value]
    elif isinstance(answers, list) and turn < len(answers):
        answer = answers[turn]
        if isinstance(answer, dict):
            primary = [
                str(answer[key])
                for key in ("input_text", "answer")
                if key in answer
            ]
        else:
            primary = [str(answer)]
    else:
        primary = []

    additional = row.get("additional_answers") or {}
    for annotator in additional.values() if isinstance(additional, dict) else []:
        if not isinstance(annotator, dict):
            continue
        values = annotator.get("input_text") or []
        if isinstance(values, list) and turn < len(values):
            primary.append(str(values[turn]))
    deduplicated = []
    seen = set()
    for value in primary:
        normalized = value.lower()
        if value and normalized not in seen:
            seen.add(normalized)
            deduplicated.append(value)
    return deduplicated


def _coqa_official_prompt(
    row: dict[str, Any],
) -> tuple[str, list[str], int]:
    questions = _coqa_questions(row)
    if not questions:
        return "", [], -1
    target_turn = len(questions) - 1
    gold = _coqa_gold_answers(row, target_turn)
    history_answers = [
        (_coqa_gold_answers(row, turn) or [""])[0]
        for turn in range(target_turn)
    ]
    prompt = (row.get("story") or row.get("passage") or "") + "\n\n"
    for question, answer in zip_longest(questions, history_answers):
        if question is None:
            break
        prompt += f"Q: {question}\n\n"
        prompt += f"A: {answer}\n\n" if answer is not None else "A:"
    return prompt, gold, target_turn


def load_coqa(limit: int, include_train_pool: bool = False) -> list[WorkloadItem]:
    import datasets

    protocol = DATASET_PROTOCOLS["coqa"]

    def _load(split: str) -> Any:
        for source in ("EleutherAI/coqa", "stanfordnlp/coqa"):
            try:
                return datasets.load_dataset(source, split=split)
            except Exception:
                continue
        return datasets.load_dataset("coqa", split=split, trust_remote_code=True)

    items: list[WorkloadItem] = []

    def _emit(split_name: str, data: Any) -> None:
        # validation is emitted first and unchanged, so eval-split items stay
        # byte-identical when include_train_pool is False.
        for story_index, row in enumerate(data):
            if len(items) >= limit:
                return
            prompt, gold, target_turn = _coqa_official_prompt(row)
            if not gold:
                continue
            items.append(
                WorkloadItem(
                    dataset="coqa",
                    item_id=f"{split_name}:{story_index}:{target_turn}",
                    phase="decode",
                    prompt=prompt,
                    prompt_kind="raw_completion",
                    reference_output_len=None,
                    metric="coqa_f1",
                    gold=gold,
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    stop=["\nQ:"],
                    evaluator_data={
                        "source_split": split_name,
                        "target_turn": target_turn,
                    },
                )
            )

    _emit("validation", _load("validation"))
    if include_train_pool:
        # Perf/throughput lane only: train stories join the SERVING load (~7.7k total).
        # The quality lane scores the validation split only, so no leak.
        _emit("train", _load("train"))
    return items


def load_humaneval(limit: int) -> list[WorkloadItem]:
    import datasets

    data = datasets.load_dataset("openai/openai_humaneval", split="test")
    protocol = DATASET_PROTOCOLS["humaneval"]
    items = []
    for index, row in enumerate(data):
        task_id = str(row.get("task_id", index))
        items.append(
            WorkloadItem(
                dataset="humaneval",
                item_id=task_id,
                phase="decode",
                prompt=row["prompt"],
                prompt_kind="raw_completion",
                reference_output_len=None,
                metric="humaneval_pass_at_1",
                gold=None,
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=protocol[
                    "task_reference_max_output_len"
                ],
                stop=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
                evaluator_data={
                    "source_split": "test",
                    "task_id": task_id,
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                    "canonical_solution": row.get("canonical_solution", ""),
                },
            )
        )
        if len(items) >= limit:
            break
    return items


def load_selection_dataset(dataset: str, limit: int) -> list[WorkloadItem]:
    module = _load_module("vp_flexidepth_select_eval", "flexidepth_select_eval.py")
    _few_shot, rows = module._load_task(dataset, limit, 5)
    source_split = {
        "mmlu": "test",
        "hellaswag": "validation",
        "winogrande": "validation",
    }[dataset]
    items = []
    protocol = DATASET_PROTOCOLS[dataset]
    for row in rows:
        label = int(row["label"])
        if row["selection_mode"] == "multiple_input":
            prompt_variants = [
                (
                    f"{row['id']}:candidate:{candidate_index}",
                    [
                        *row["fewshot_messages"],
                        {"role": "user", "content": candidate},
                    ],
                    candidate_index,
                )
                for candidate_index, candidate in enumerate(row["choices"])
            ]
        else:
            prompt_variants = [
                (
                    str(row["id"]),
                    list(row["messages"]),
                    None,
                )
            ]
        for item_id, prompt, candidate_index in prompt_variants:
            items.append(
                WorkloadItem(
                    dataset=dataset,
                    item_id=item_id,
                    phase="prefill",
                    prompt=prompt,
                    prompt_kind="chat_messages",
                    reference_output_len=1,
                    metric="selection_prefill_probe",
                    gold={
                        "index": label,
                        "choice": str(row["choices"][label]),
                    },
                    protocol_id=protocol["id"],
                    quality_semantics=protocol["quality_semantics"],
                    task_reference_max_output_len=protocol[
                        "task_reference_max_output_len"
                    ],
                    stop=["\n"],
                    evaluator_data={
                        "source_split": source_split,
                        "fewshot_indices": row.get("fewshot_indices") or [],
                        "selection_metric": row["selection_metric"],
                        "selection_mode": row["selection_mode"],
                        "candidate_index": candidate_index,
                        "continuation": row.get("continuation"),
                    },
                )
            )
    return items


def load_scrolls_summary(
    dataset: str,
    limit: int,
    tokenizer: Any,
    context_length: int,
) -> list[WorkloadItem]:
    """SCROLLS long-context summarisation from locally staged jsonl.

    Rows whose prompt + reference output would exceed the model window are
    DROPPED, not truncated (owner ruling A1, 2026-09-04): truncating would score
    ROUGE against a summary of a document the model only half saw, making the
    figure incomparable to any published SCROLLS number -- which is the reason
    the axis uses SCROLLS validation splits at all. Filtering happens here
    rather than in `render_workload_rows` because the reference output length is
    a per-dataset constant, so both sides of the window test are known at load
    time; doing it later would break that function's gap-free `index` contract.

    Quality is NOT scored in-repo. The metric defers to the third-party harness
    (lm-eval `longbench_gov_report` / `scrolls_*`), matching the `ifeval_offline`
    precedent -- an in-repo ROUGE reimplementation would violate the standing
    rule that quality comes only from third-party harnesses.
    """

    scrolls = _load_module("vp_scrolls_local", "scrolls_local.py")
    protocol = DATASET_PROTOCOLS[dataset]
    reference_output_len = int(scrolls.GENLEN[dataset])
    items: list[WorkloadItem] = []
    considered = 0
    for record in scrolls.iter_records(dataset):
        if len(items) >= limit:
            break
        considered += 1
        prompt = scrolls.render_prompt(dataset, record)
        prompt_tokens = len(tokenizer(prompt)["input_ids"])
        if not scrolls.fits_window(prompt_tokens, dataset, context_length):
            continue
        items.append(
            WorkloadItem(
                dataset=dataset,
                item_id=str(record["id"]),
                phase="decode",
                prompt=[{"role": "user", "content": prompt}],
                prompt_kind="chat_messages",
                reference_output_len=reference_output_len,
                metric="rouge_offline",
                gold=list(record["answers"]),
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=protocol[
                    "task_reference_max_output_len"
                ],
                stop=[],
                evaluator_data={
                    "source_split": protocol["evaluation_split"],
                    "prompt_tokens": prompt_tokens,
                    "window_filtered": True,
                    "context_length": context_length,
                },
            )
        )
    if not items:
        raise ValueError(
            f"{dataset}: no staged row fits prompt + {reference_output_len} "
            f"<= {context_length} (considered {considered})"
        )
    return items


def load_longbench_summary(
    dataset: str, limit: int, tokenizer: Any, max_length: int = 7000
) -> list[WorkloadItem]:
    exporter = _load_module("vp_longbench_export", "export_longbench_autobench.py")
    longbench = exporter.load_longbench_module()
    records = longbench.load_records(dataset, limit)
    protocol = DATASET_PROTOCOLS[dataset]
    items = []
    for index, record in enumerate(records):
        reference_output_len = int(longbench.GENLEN[dataset])
        row = exporter.render_row(
            longbench,
            tokenizer,
            dataset,
            record,
            max_length,
            reference_output_len,
        )
        items.append(
            WorkloadItem(
                dataset=dataset,
                item_id=str(index),
                phase="prefill",
                prompt=row["messages"],
                prompt_kind="chat_messages",
                reference_output_len=reference_output_len,
                metric="rouge_l",
                gold=row["answers"],
                protocol_id=protocol["id"],
                quality_semantics=protocol["quality_semantics"],
                task_reference_max_output_len=reference_output_len,
            )
        )
    return items


def load_dataset_items(
    dataset: str,
    limit: int,
    tokenizer: Optional[Any] = None,
    include_train_pool: bool = False,
    context_length: Optional[int] = None,
) -> list[WorkloadItem]:
    if dataset in {
        "scrolls_gov_report",
        "scrolls_summ_screen_fd",
        "scrolls_qmsum",
    }:
        if tokenizer is None:
            raise ValueError(f"{dataset} requires a tokenizer for the window filter")
        if context_length is None:
            raise ValueError(
                f"{dataset} requires context_length: the window filter is the "
                "sealed alternative to truncation (ruling A1) and must not "
                "silently default"
            )
        return load_scrolls_summary(dataset, limit, tokenizer, context_length)
    if dataset == "gsm8k":
        return load_gsm8k(limit, include_train_pool)
    if dataset == "gsm8k_cot":
        return load_gsm8k_cot(limit)
    if dataset == "ifeval":
        return load_ifeval(limit)
    if dataset == "mmlu_pro":
        return load_mmlu_pro(limit, cot=False)
    if dataset == "mmlu_pro_cot":
        return load_mmlu_pro(limit, cot=True)
    if dataset == "coqa":
        return load_coqa(limit, include_train_pool)
    if dataset == "humaneval":
        return load_humaneval(limit)
    if dataset in {"mmlu", "hellaswag", "winogrande"}:
        return load_selection_dataset(dataset, limit)
    if dataset in {"gov_report", "multi_news"}:
        if tokenizer is None:
            raise ValueError(f"{dataset} requires a tokenizer")
        return load_longbench_summary(dataset, limit, tokenizer)
    raise ValueError(f"unsupported dataset: {dataset}")


def parse_weights(raw: str) -> dict[str, float]:
    if not raw:
        return {}
    result = {}
    for item in raw.split(","):
        name, separator, value = item.strip().partition("=")
        if not separator:
            raise ValueError(f"invalid dataset weight {item!r}; expected name=value")
        result[name] = float(value)
    return result


def _weighted_dataset_choice(
    rng: random.Random,
    available: list[str],
    weights: dict[str, float],
) -> str:
    values = [max(0.0, weights.get(name, 1.0)) for name in available]
    if not any(values):
        values = [1.0] * len(available)
    return rng.choices(available, weights=values, k=1)[0]


def sample_items(
    pools: dict[str, list[WorkloadItem]],
    num_requests: int,
    phase_mode: str,
    decode_ratio: float,
    seed: int,
    weights: Optional[dict[str, float]] = None,
    repeat_exhausted: bool = False,
) -> list[WorkloadItem]:
    if phase_mode not in {"decode", "prefill", "mixed"}:
        raise ValueError(f"invalid phase mode: {phase_mode}")
    if not 0.0 <= decode_ratio <= 1.0:
        raise ValueError("decode_ratio must be in [0, 1]")
    supplied_weights = dict(weights or {})
    weights = {**DEFAULT_DATASET_WEIGHTS, **supplied_weights}
    if len(pools) == 1:
        only_dataset = next(iter(pools))
        if only_dataset not in supplied_weights:
            weights[only_dataset] = 1.0
    originals = {name: list(items) for name, items in pools.items()}
    working = {name: list(items) for name, items in originals.items()}
    for name, items in working.items():
        random.Random(_stable_seed(seed, name)).shuffle(items)
    refill_epochs = Counter()
    occurrences = Counter()
    rng = random.Random(seed)
    sampled = []
    while len(sampled) < num_requests:
        source = originals if repeat_exhausted else working
        available = [name for name, items in source.items() if items]
        if not available:
            break
        if phase_mode == "mixed":
            target_phase = "decode" if rng.random() < decode_ratio else "prefill"
            phase_available = [
                name
                for name in available
                if source[name][-1].phase == target_phase
            ]
            if phase_available:
                available = phase_available
        else:
            available = [
                name for name in available if source[name][-1].phase == phase_mode
            ]
            if not available:
                break
        available = [name for name in available if weights.get(name, 1.0) > 0]
        if not available:
            break
        selected = _weighted_dataset_choice(rng, available, weights)
        if not working[selected]:
            refill_epochs[selected] += 1
            working[selected] = list(originals[selected])
            random.Random(
                _stable_seed(seed, f"{selected}:refill:{refill_epochs[selected]}")
            ).shuffle(working[selected])
        item = working[selected].pop()
        source_item_id = item.source_item_id or item.item_id
        occurrence_key = (item.dataset, source_item_id)
        replica_index = occurrences[occurrence_key]
        occurrences[occurrence_key] += 1
        if replica_index:
            item = replace(
                item,
                item_id=f"{source_item_id}:replica:{replica_index}",
                source_item_id=source_item_id,
                replica_index=replica_index,
            )
        sampled.append(item)
    if len(sampled) < num_requests:
        raise ValueError(
            f"requested {num_requests} unique rows but only sampled {len(sampled)} "
            f"for phase_mode={phase_mode}"
        )
    return sampled


def validate_frequency_penalty(value: Any) -> float:
    """Normalize the uniform generation penalty into a recordable float."""

    penalty = float(value)
    if not math.isfinite(penalty) or penalty < 0.0:
        raise ValueError(
            f"frequency_penalty must be finite and non-negative, got {value!r}"
        )
    return penalty


def _prompt_hash(prompt_ids: list[int]) -> str:
    payload = json.dumps(prompt_ids, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_list(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    shape = getattr(tokenized, "shape", None)
    if shape is not None:
        tokenized = tokenized.tolist()
    if (
        isinstance(tokenized, (list, tuple))
        and len(tokenized) == 1
        and isinstance(tokenized[0], (list, tuple))
    ):
        tokenized = tokenized[0]
    return [int(token) for token in tokenized]


def _render_prompt_ids(tokenizer: Any, item: WorkloadItem) -> list[int]:
    if item.prompt_kind == "chat_messages":
        if not isinstance(item.prompt, list):
            raise ValueError(f"{item.dataset}:{item.item_id} has invalid chat prompt")
        tokenized = tokenizer.apply_chat_template(
            item.prompt, tokenize=True, add_generation_prompt=True
        )
        return _token_list(tokenized)
    if item.prompt_kind == "raw_completion":
        if not isinstance(item.prompt, str):
            raise ValueError(f"{item.dataset}:{item.item_id} has invalid raw prompt")
        return _token_list(
            tokenizer.encode(item.prompt, add_special_tokens=False)
        )
    raise ValueError(
        f"{item.dataset}:{item.item_id} has unknown prompt kind {item.prompt_kind!r}"
    )


def render_workload_rows(
    items: list[WorkloadItem],
    phase_mode: str,
    tokenizer: Any,
    context_length: int,
    decode_output_policy: str = "remaining_model_context",
    fixed_decode_output_len: Optional[int] = None,
    sampling_overrides: Optional[dict[str, dict[str, Any]]] = None,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if tokenizer is None:
        raise ValueError("a tokenizer is required for context-derived output budgets")
    if context_length <= 1:
        raise ValueError("context_length must be greater than one")
    penalty = validate_frequency_penalty(frequency_penalty)
    requests = []
    metadata = []
    for index, item in enumerate(items):
        request_id = f"{item.dataset}:{item.item_id}"
        source_item_id = item.source_item_id or item.item_id
        source_request_id = f"{item.dataset}:{source_item_id}"
        prompt_ids = _render_prompt_ids(tokenizer, item)
        prompt_len = len(prompt_ids)
        if prompt_len >= context_length:
            raise ValueError(
                f"{item.dataset}:{item.item_id} prompt has {prompt_len} tokens, "
                f"which does not fit context length {context_length}"
            )
        is_prefill_probe = phase_mode == "prefill" or (
            phase_mode == "mixed" and item.phase == "prefill"
        )
        if is_prefill_probe:
            output_len = 1
            output_policy = "prefill_probe_one_token"
        elif decode_output_policy == "remaining_model_context":
            output_len = context_length - prompt_len
            output_policy = "remaining_model_context"
        elif decode_output_policy == "task_reference_limit":
            if not item.task_reference_max_output_len:
                raise ValueError(
                    f"{item.dataset}:{item.item_id} has no task reference limit"
                )
            output_len = min(
                context_length - prompt_len,
                int(item.task_reference_max_output_len),
            )
            output_policy = "task_reference_limit"
        elif decode_output_policy == "fixed_token_capacity":
            if fixed_decode_output_len is None or fixed_decode_output_len <= 0:
                raise ValueError(
                    "fixed_token_capacity requires a positive fixed output length"
                )
            if prompt_len + fixed_decode_output_len > context_length:
                raise ValueError(
                    f"{item.dataset}:{item.item_id} has only "
                    f"{context_length - prompt_len} output positions, fewer than "
                    f"the fixed capacity length {fixed_decode_output_len}"
                )
            output_len = fixed_decode_output_len
            output_policy = "fixed_token_capacity"
        else:
            raise ValueError(
                f"unsupported decode output policy {decode_output_policy!r}"
            )
        extra_request_body: dict[str, Any] = {
            "temperature": 0.0,
            "ignore_eos": output_policy == "fixed_token_capacity",
        }
        if item.stop and output_policy != "fixed_token_capacity":
            extra_request_body["stop"] = item.stop
        sampling_policy = dict((sampling_overrides or {}).get(item.dataset, {}))
        if sampling_policy and output_policy == "fixed_token_capacity":
            raise ValueError(
                f"{item.dataset} sampling overrides cannot change fixed-token capacity"
            )
        extra_request_body.update(sampling_policy)
        if penalty != DEFAULT_FREQUENCY_PENALTY:
            if FREQUENCY_PENALTY_FIELD in extra_request_body:
                raise ValueError(
                    f"{item.dataset}:{item.item_id} already carries a "
                    f"{FREQUENCY_PENALTY_FIELD} sampling override; the uniform "
                    "generation penalty must not be applied on top of it"
                )
            extra_request_body[FREQUENCY_PENALTY_FIELD] = penalty
        requests.append(
            {
                "request_id": request_id,
                "prompt": prompt_ids,
                "prompt_len": prompt_len,
                "output_len": output_len,
                "extra_request_body": extra_request_body,
            }
        )
        quality_eligible = (
            output_policy != "fixed_token_capacity"
            and item.metric != "selection_prefill_probe"
            and not (
                item.reference_output_len is not None
                and output_len < item.reference_output_len
                and item.metric == "rouge_l"
            )
        )
        metadata.append(
            {
                "index": index,
                "request_id": request_id,
                "source_request_id": source_request_id,
                "replica_index": item.replica_index,
                "dataset": item.dataset,
                "source_split": item.evaluator_data.get("source_split"),
                "phase": item.phase,
                "metric": item.metric,
                "gold": item.gold,
                "reference_output_len": item.reference_output_len,
                "requested_output_len": output_len,
                "prompt_len": prompt_len,
                "context_length": context_length,
                "output_policy": output_policy,
                "fixed_output_tokens": (
                    fixed_decode_output_len
                    if output_policy == "fixed_token_capacity"
                    else None
                ),
                "task_reference_max_output_len": (
                    item.task_reference_max_output_len
                ),
                "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
                "protocol_suite_id": PROTOCOL_SUITE_ID,
                "protocol_id": item.protocol_id,
                "quality_semantics": item.quality_semantics,
                "prompt_kind": item.prompt_kind,
                "quality_eligible": quality_eligible,
                "sampling_policy": sampling_policy,
                "prompt_sha256": _prompt_hash(prompt_ids),
                "evaluator_data": item.evaluator_data,
            }
        )
    return requests, metadata


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_workload(
    workload: str,
    num_requests: int,
    seed: int,
    tokenizer: Optional[Any],
    context_length: int,
    datasets_filter: Optional[list[str]] = None,
    phase_mode: Optional[str] = None,
    decode_ratio: float = 0.5,
    weights: Optional[dict[str, float]] = None,
    repeat_exhausted: bool = False,
    decode_output_policy: str = "remaining_model_context",
    fixed_decode_output_len: Optional[int] = None,
    sampling_overrides: Optional[dict[str, dict[str, Any]]] = None,
    frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
    include_train_pool: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if workload not in WORKLOAD_DATASETS:
        raise ValueError(f"unsupported workload {workload!r}")
    datasets_selected = tuple(datasets_filter or WORKLOAD_DATASETS[workload])
    unknown = sorted(set(datasets_selected) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    selected_phase = phase_mode or WORKLOAD_PHASE[workload]
    per_dataset_limit = num_requests
    pools = {
        name: load_dataset_items(
            name,
            per_dataset_limit,
            tokenizer,
            include_train_pool,
            context_length=context_length,
        )
        for name in datasets_selected
    }
    items = sample_items(
        pools,
        num_requests,
        selected_phase,
        decode_ratio,
        seed,
        weights,
        repeat_exhausted,
    )
    requests, metadata = render_workload_rows(
        items,
        selected_phase,
        tokenizer,
        context_length,
        decode_output_policy=decode_output_policy,
        fixed_decode_output_len=fixed_decode_output_len,
        sampling_overrides=sampling_overrides,
        frequency_penalty=frequency_penalty,
    )
    counts = Counter(row["dataset"] for row in metadata)
    phase_counts = Counter(row["phase"] for row in metadata)
    replica_counts = Counter(
        row["dataset"] for row in metadata if int(row["replica_index"]) > 0
    )
    source_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_ids: dict[str, set[str]] = defaultdict(set)
    prompt_hashes: dict[str, set[str]] = defaultdict(set)
    for row in metadata:
        dataset = str(row["dataset"])
        source_ids[dataset].add(str(row["source_request_id"]))
        prompt_hashes[dataset].add(str(row["prompt_sha256"]))
        if row.get("source_split"):
            source_split_counts[dataset][str(row["source_split"])] += 1
    summary = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_suite_id": PROTOCOL_SUITE_ID,
        "dataset_protocols": {
            name: DATASET_PROTOCOLS[name] for name in datasets_selected
        },
        "output_policy": {
            "decode": decode_output_policy,
            "fixed_output_tokens": fixed_decode_output_len,
            "prefill": "one_token_probe",
            "context_length": context_length,
        },
        "sampling_overrides": sampling_overrides or {},
        FREQUENCY_PENALTY_FIELD: validate_frequency_penalty(frequency_penalty),
        "workload": workload,
        "phase_mode": selected_phase,
        "decode_ratio_requested": decode_ratio,
        "seed": seed,
        "num_requests": len(requests),
        "datasets": dict(sorted(counts.items())),
        "phases": dict(sorted(phase_counts.items())),
        "quality_eligible": sum(bool(row["quality_eligible"]) for row in metadata),
        "repeat_exhausted": repeat_exhausted,
        "replicated_requests": sum(
            int(row["replica_index"] > 0) for row in metadata
        ),
        "replicated_requests_by_dataset": dict(sorted(replica_counts.items())),
        "source_splits": {
            dataset: dict(sorted(splits.items()))
            for dataset, splits in sorted(source_split_counts.items())
        },
        "unique_source_requests": len(
            {str(row["source_request_id"]) for row in metadata}
        ),
        "unique_source_requests_by_dataset": {
            dataset: len(values) for dataset, values in sorted(source_ids.items())
        },
        "unique_prompts": len({str(row["prompt_sha256"]) for row in metadata}),
        "unique_prompts_by_dataset": {
            dataset: len(values)
            for dataset, values in sorted(prompt_hashes.items())
        },
    }
    return requests, metadata, summary


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = "".join(
        character for character in text if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def numeric_exact_match(prediction: str, gold: str) -> float:
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", prediction)
    predicted = numbers[-1].replace(",", "") if numbers else ""
    expected = str(gold).replace(",", "").strip()
    try:
        return float(abs(float(predicted) - float(expected)) < 1e-4)
    except Exception:
        return float(predicted == expected)


def gsm8k_strict_match(prediction: str, gold: str) -> float:
    match = re.search(r"#### (-?[0-9.,]+)", prediction)
    if match is None:
        return 0.0
    predicted = match.group(1).replace(",", "").rstrip(".")
    expected = str(gold).replace(",", "").rstrip(".").strip()
    return float(predicted == expected)


def gsm8k_cot_strict_match(prediction: str, gold: str) -> float:
    # lm-eval gsm8k-cot strict-match filter, pattern VERBATIM (the trailing
    # unescaped dot is part of the upstream pattern), take_first. The
    # regexes_to_ignore normalization removes exactly ONE trailing period
    # (\.$) — never rstrip, which would over-normalize "5.." to "5" and
    # score answers upstream rejects.
    match = re.search(r"The answer is (\-?[0-9\.\,]+).", prediction)
    if match is None:
        return 0.0
    predicted = re.sub(
        r"\.$", "", match.group(1).replace(",", "").replace("$", "")
    )
    expected = re.sub(r"\.$", "", str(gold).replace(",", "").strip())
    return float(predicted == expected)


def coqa_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    if len(gold_answers) <= 1:
        return max(token_f1(prediction, answer) for answer in gold_answers)
    total = 0.0
    for held_out in range(len(gold_answers)):
        references = gold_answers[:held_out] + gold_answers[held_out + 1 :]
        total += max(token_f1(prediction, answer) for answer in references)
    return total / len(gold_answers)


def _lcs_length(left: list[str], right: list[str]) -> int:
    state = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, 1):
            current = state[index]
            state[index] = (
                previous + 1
                if left_token == right_token
                else max(state[index], state[index - 1])
            )
            previous = current
    return state[-1]


def rouge_l(prediction: str, reference: str) -> float:
    predicted = prediction.split()
    expected = reference.split()
    if not predicted or not expected:
        return 0.0
    common = _lcs_length(predicted, expected)
    precision = common / len(predicted)
    recall = common / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def choice_accuracy(prediction: str, gold: dict[str, Any]) -> float:
    match = re.search(r"\b([A-Z])\b", prediction.upper())
    if match:
        return float(match.group(1) == str(gold["letter"]).upper())
    return float(_normalize_text(str(gold["text"])) in _normalize_text(prediction))


def humaneval_pass_at_1(
    prediction: str, evaluator_data: dict[str, Any], allow_code_execution: bool
) -> tuple[Optional[float], str]:
    if not allow_code_execution:
        return None, "code_execution_disabled"
    try:
        from humaneval_execution import check_correctness
    except (ImportError, ModuleNotFoundError):
        return None, "humaneval_executor_unavailable"
    problem = {
        "task_id": evaluator_data["task_id"],
        "prompt": evaluator_data["prompt"],
        "test": evaluator_data["test"],
        "entry_point": evaluator_data["entry_point"],
        "canonical_solution": evaluator_data.get("canonical_solution", ""),
    }
    result = check_correctness(problem, prediction, 3.0)
    return float(bool(result.get("passed"))), "scored"


def score_prediction(
    metadata: dict[str, Any], prediction: str, allow_code_execution: bool = False
) -> tuple[Optional[float], str]:
    if not metadata.get("quality_eligible", True):
        if metadata.get("metric") == "selection_prefill_probe":
            return None, "quality_scored_by_selection_evaluator"
        if metadata.get("output_policy") in {
            "fixed_token_capacity",
            "production_max_equal_work",
        }:
            return None, "capacity_workload_not_quality_evidence"
        return None, "quality_not_defined_for_prefill_truncation"
    metric = metadata["metric"]
    gold = metadata.get("gold")
    if metric == "gsm8k_strict_match":
        return gsm8k_strict_match(prediction, str(gold)), "scored"
    if metric == "gsm8k_cot_strict_match":
        return gsm8k_cot_strict_match(prediction, str(gold)), "scored"
    if metric == "mmlu_pro_letter_match":
        return mmlu_pro_letter_match(prediction, str(gold)), "scored"
    if metric == "ifeval_offline":
        # IFEval verdicts come only from the third-party harness offline;
        # serving-side scoring is intentionally not defined.
        return None, "offline_third_party_scoring"
    if metric == "rouge_offline":
        # SCROLLS/LongBench ROUGE comes only from the third-party harness
        # (lm-eval `metrics.get_rouge_score`); no in-repo reimplementation.
        return None, "offline_third_party_scoring"
    if metric == "numeric_exact_match":
        return numeric_exact_match(prediction, str(gold)), "scored"
    if metric == "coqa_f1":
        first_line = prediction.strip().split("\n")[0]
        return coqa_f1(first_line, [str(answer) for answer in gold]), "scored"
    if metric == "token_f1":
        first_line = prediction.strip().split("\n")[0]
        return max(token_f1(first_line, str(answer)) for answer in gold), "scored"
    if metric == "choice_accuracy":
        return choice_accuracy(prediction, gold), "scored"
    if metric == "rouge_l":
        return max(rouge_l(prediction, str(reference)) for reference in gold), "scored"
    if metric == "humaneval_pass_at_1":
        return humaneval_pass_at_1(
            prediction, metadata.get("evaluator_data") or {}, allow_code_execution
        )
    raise ValueError(f"unsupported metric: {metric}")


def score_benchmark_record(
    record: dict[str, Any],
    metadata: list[dict[str, Any]],
    allow_code_execution: bool = False,
) -> dict[str, Any]:
    predictions = record.get("generated_texts") or []
    request_ids = record.get("request_ids") or []
    errors = record.get("errors") or []
    requested_output_lens = record.get("requested_output_lens") or []
    server_output_lens = record.get("server_reported_output_lens") or []
    retokenized_output_lens = record.get("retokenized_output_lens") or []
    effective_output_lens = record.get("output_lens") or []
    output_len_sources = record.get("output_len_sources") or []
    finish_reasons = record.get("finish_reasons") or []
    e2e_latencies = record.get("e2e_latencies") or []
    if request_ids:
        if len(request_ids) != len(predictions):
            raise ValueError(
                f"request_ids has {len(request_ids)} rows but predictions has "
                f"{len(predictions)}"
            )
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("benchmark request_ids are not unique")
        metadata_by_id = {row["request_id"]: row for row in metadata}
        if len(metadata_by_id) != len(metadata):
            raise ValueError("metadata request_ids are not unique")
        unknown = [
            request_id
            for request_id in request_ids
            if request_id not in metadata_by_id
        ]
        if unknown:
            raise ValueError(f"benchmark contains unknown request_ids: {unknown[:5]}")
        aligned_metadata = [metadata_by_id[request_id] for request_id in request_ids]
        alignment_source = "request_id"
    else:
        aligned_metadata = metadata
        alignment_source = "legacy_position"
    count = min(len(predictions), len(aligned_metadata))
    per_dataset: dict[str, list[float]] = defaultdict(list)
    per_source_split: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_request = []
    statuses = Counter()
    cap_hits = Counter()
    for index in range(count):
        meta = aligned_metadata[index]
        error = errors[index] if index < len(errors) else ""
        if error:
            score, status = None, "request_error"
        else:
            score, status = score_prediction(
                meta, str(predictions[index]), allow_code_execution
            )
        statuses[status] += 1
        if score is not None:
            per_dataset[meta["dataset"]].append(score)
            source_split = meta.get("source_split")
            if source_split:
                per_source_split[meta["dataset"]][str(source_split)].append(score)
        finish_reason = (
            finish_reasons[index] if index < len(finish_reasons) else None
        )
        if finish_reason == "length":
            cap_hits[meta["dataset"]] += 1
        per_request.append(
            {
                "index": index,
                "request_id": meta["request_id"],
                "source_request_id": meta.get("source_request_id"),
                "replica_index": meta.get("replica_index", 0),
                "dataset": meta["dataset"],
                "source_split": meta.get("source_split"),
                "metric": meta["metric"],
                "score": score,
                "status": status,
                "error": error,
                "prediction": predictions[index],
                "requested_output_len": (
                    requested_output_lens[index]
                    if index < len(requested_output_lens)
                    else None
                ),
                "server_output_len": (
                    server_output_lens[index]
                    if index < len(server_output_lens)
                    else None
                ),
                "retokenized_output_len": (
                    retokenized_output_lens[index]
                    if index < len(retokenized_output_lens)
                    else None
                ),
                "effective_output_len": (
                    effective_output_lens[index]
                    if index < len(effective_output_lens)
                    else None
                ),
                "output_len_source": (
                    output_len_sources[index]
                    if index < len(output_len_sources)
                    else None
                ),
                "finish_reason": finish_reason,
                "cap_hit": finish_reason == "length",
                "e2e_latency_s": (
                    e2e_latencies[index] if index < len(e2e_latencies) else None
                ),
            }
        )
    performance_fields = (
        "metric_accounting_version",
        "duration",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "total_output_tokens_retokenized",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "output_throughput_retokenized",
        "total_throughput",
        "total_throughput_retokenized",
        "mean_e2e_latency_ms",
        "p90_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "mean_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
        "concurrency",
        "max_concurrent_requests",
        "request_rate",
        "server_input_usage_reported",
        "declared_input_fallbacks",
        "server_usage_reported",
        "retokenized_fallbacks",
        "missing_finish_reasons",
        "cap_hit_count",
    )
    return {
        "tag": record.get("tag"),
        "dataset_name": record.get("dataset_name"),
        "alignment_source": alignment_source,
        "scored_requests": count,
        "status_counts": dict(sorted(statuses.items())),
        "cap_hits_by_dataset": dict(sorted(cap_hits.items())),
        "quality": {
            dataset: {
                "n": len(scores),
                "mean": sum(scores) / len(scores) if scores else math.nan,
            }
            for dataset, scores in sorted(per_dataset.items())
        },
        "quality_by_source_split": {
            dataset: {
                source_split: {
                    "n": len(scores),
                    "mean": sum(scores) / len(scores) if scores else math.nan,
                }
                for source_split, scores in sorted(splits.items())
            }
            for dataset, splits in sorted(per_source_split.items())
        },
        "performance": {
            field: record.get(field)
            for field in performance_fields
            if field in record
        },
        "per_request": per_request,
    }
