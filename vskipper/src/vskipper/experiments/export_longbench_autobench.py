"""Export selected LongBench tasks for official ``sglang.benchmark.serving``.

The benchmark's ``autobench`` loader preserves request order, supports chat
messages, and forwards only explicit ``extra_request_body`` fields. Answers stay
in a separate metadata file so they are never sent to the server.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from transformers import AutoTokenizer


LONG_BENCH = Path(__file__).resolve().with_name("longbench_eval.py")


def load_longbench_module():
    spec = importlib.util.spec_from_file_location("vp_longbench_eval", LONG_BENCH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def stable_int_from_tokens(input_ids: List[int]) -> int:
    vals = input_ids[:64]
    digest = hashlib.sha1(",".join(str(int(x)) for x in vals).encode("ascii"))
    return int(digest.hexdigest()[:16], 16)


def profile_indices(rows: List[Dict[str, Any]], policy: str, profile_count: int) -> List[int]:
    if profile_count <= 0:
        return []
    if policy == "hash":
        base = stable_int_from_tokens(rows[0]["prompt_prefix"])
        return [
            (base + i * 1315423911 + int(row["prompt_len"])) % profile_count
            for i, row in enumerate(rows)
        ]
    if policy == "length":
        return [int(row["prompt_len"]) % profile_count for row in rows]
    if policy in {"round_robin", "balanced"}:
        return [i % profile_count for i, _ in enumerate(rows)]
    return [0 for _ in rows]


def chat_token_ids(tok: Any, messages: List[Dict[str, str]]) -> List[int]:
    try:
        ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
            return_dict=False,
        )
    except TypeError:
        ids = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def render_row(lb: Any, tok: Any, task: str, rec: Dict[str, Any], maxlen: int, genlen: int) -> Dict[str, Any]:
    tmpl = lb.PROMPT[task]
    template_ids = tok(
        tmpl.format(context="", input=rec["input"]),
        add_special_tokens=False,
    )["input_ids"]
    budget = maxlen - genlen - len(template_ids) - 64
    if budget <= 0:
        raise ValueError(f"maxlen={maxlen} too small for task={task} genlen={genlen}")

    cids = tok(rec["context"], add_special_tokens=False)["input_ids"]
    if len(cids) > budget:
        half = budget // 2
        cids = cids[:half] + cids[-(budget - half) :]

    content = tmpl.format(context=tok.decode(cids), input=rec["input"])
    messages = [{"role": "user", "content": content}]
    prompt_ids = chat_token_ids(tok, messages)
    metric = lb.METRIC[task]
    extra_key = "task=qa" if metric == "f1" else "task=summary"
    return {
        "task": task,
        "messages": messages,
        "answers": list(rec["answers"]),
        "genlen": genlen,
        "metric": metric,
        "ctx_len": len(cids),
        "prompt_len": len(prompt_ids),
        "prompt_prefix": prompt_ids[:64],
        "extra_key": extra_key,
    }


def build_rows(lb: Any, tok: Any, tasks: Iterable[str], n: int, maxlen: int, genlen_cap: int) -> List[Dict[str, Any]]:
    task_list = [task for task in tasks if task]
    records = {task: lb.load_records(task, n) for task in task_list}
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        for task in task_list:
            rec = records[task][i]
            genlen = int(lb.GENLEN[task])
            if genlen_cap > 0:
                genlen = min(genlen, genlen_cap)
            row = render_row(lb, tok, task, rec, maxlen, genlen)
            row["task_index"] = i
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--tasks", default="gov_report")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--maxlen", type=int, default=7000)
    parser.add_argument("--genlen-cap", type=int, default=64)
    parser.add_argument("--profile-count", type=int, default=3)
    parser.add_argument("--profile-assignment", default="balanced")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    lb = load_longbench_module()
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = build_rows(
        lb,
        tok,
        [task.strip() for task in args.tasks.split(",")],
        args.n,
        args.maxlen,
        args.genlen_cap,
    )

    indices = profile_indices(rows, args.profile_assignment, args.profile_count)
    request_rows = []
    metadata_rows = []
    for idx, row in enumerate(rows):
        profile_idx = indices[idx] if indices else None
        extra_key = row["extra_key"]
        if profile_idx is not None:
            extra_key = f"{extra_key}|vp_profile={profile_idx}"
        request_rows.append(
            {
                "messages": row["messages"],
                "output_len": row["genlen"],
                "prompt_len": row["prompt_len"],
                "extra_request_body": {
                    "extra_key": extra_key,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            }
        )
        metadata_rows.append(
            {
                "idx": idx,
                "task": row["task"],
                "task_index": row["task_index"],
                "answers": row["answers"],
                "genlen": row["genlen"],
                "metric": row["metric"],
                "ctx_len": row["ctx_len"],
                "prompt_len": row["prompt_len"],
                "extra_key": extra_key,
                "profile_idx": profile_idx,
            }
        )

    write_jsonl(Path(args.requests), request_rows)
    write_jsonl(Path(args.metadata), metadata_rows)
    summary = {
        "model": args.model,
        "tasks": [task.strip() for task in args.tasks.split(",") if task.strip()],
        "n_per_task": args.n,
        "rows": len(rows),
        "maxlen": args.maxlen,
        "genlen_cap": args.genlen_cap,
        "profile_count": args.profile_count,
        "profile_assignment": args.profile_assignment,
        "requests": args.requests,
        "metadata": args.metadata,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
