#!/usr/bin/env python3
"""Score official SGLang output details against a labeled workload manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from labeled_workload import read_jsonl, score_benchmark_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-jsonl", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args()

    benchmark_records = read_jsonl(args.bench_jsonl)
    metadata = read_jsonl(args.metadata)
    scores = [
        score_benchmark_record(record, metadata, args.allow_code_execution)
        for record in benchmark_records
    ]
    payload = {
        "bench_jsonl": str(args.bench_jsonl),
        "metadata": str(args.metadata),
        "scores": scores,
    }
    output = args.output or args.bench_jsonl.with_suffix(".score.json")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(scores),
                "quality": [score.get("quality", {}) for score in scores],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
