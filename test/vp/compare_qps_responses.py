#!/usr/bin/env python3
"""Compare saved QPS responses by exact frozen request identity."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RESULT_FILE = re.compile(r"^.+_rep\d+\.jsonl$")


def load_responses(path: Path) -> tuple[list[str], dict[str, str]]:
    ordered_ids: list[str] = []
    responses: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            request_ids = record.get("request_ids")
            generated = record.get("generated_texts")
            if not isinstance(request_ids, list) or not isinstance(generated, list):
                raise ValueError(
                    f"{path}:{line_number} lacks request_ids/generated_texts arrays"
                )
            if len(request_ids) != len(generated):
                raise ValueError(
                    f"{path}:{line_number} request/response counts differ"
                )
            for request_id, response in zip(request_ids, generated):
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError(f"{path}:{line_number} has an invalid request ID")
                if request_id in responses:
                    raise ValueError(f"{path} repeats request ID {request_id!r}")
                ordered_ids.append(request_id)
                responses[request_id] = str(response)
    if not ordered_ids:
        raise ValueError(f"{path} contains no saved responses")
    return ordered_ids, responses


def compare_responses(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_ids, left = load_responses(left_path)
    right_ids, right = load_responses(right_path)
    if left_ids != right_ids:
        raise ValueError("saved response artifacts do not have identical request order")
    mismatches = [
        request_id
        for request_id in left_ids
        if left[request_id] != right[request_id]
    ]
    return {
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "requests": len(left_ids),
        "exact_matches": len(left_ids) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatch_request_ids": mismatches,
        "status": "passed" if not mismatches else "failed",
    }


def compare_response_directories(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    def result_files(directory: Path) -> dict[str, Path]:
        files = {
            path.name: path
            for path in directory.iterdir()
            if path.is_file() and RESULT_FILE.fullmatch(path.name)
        }
        if not files:
            raise ValueError(f"{directory} contains no benchmark result files")
        return files

    left_files = result_files(left_dir)
    right_files = result_files(right_dir)
    if left_files.keys() != right_files.keys():
        raise ValueError("result directories do not contain identical cell names")
    cells = {
        name: compare_responses(left_files[name], right_files[name])
        for name in sorted(left_files)
    }
    mismatch_count = sum(cell["mismatch_count"] for cell in cells.values())
    return {
        "left_dir": str(left_dir.resolve()),
        "right_dir": str(right_dir.resolve()),
        "cell_count": len(cells),
        "requests": sum(cell["requests"] for cell in cells.values()),
        "exact_matches": sum(cell["exact_matches"] for cell in cells.values()),
        "mismatch_count": mismatch_count,
        "cells": cells,
        "status": "passed" if mismatch_count == 0 else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--left-dir", type=Path)
    parser.add_argument("--right-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    file_mode = args.left is not None or args.right is not None
    directory_mode = args.left_dir is not None or args.right_dir is not None
    if file_mode == directory_mode:
        parser.error("select exactly one of file mode or directory mode")
    if file_mode:
        if args.left is None or args.right is None:
            parser.error("file mode requires --left and --right")
        result = compare_responses(args.left, args.right)
    else:
        if args.left_dir is None or args.right_dir is None:
            parser.error("directory mode requires --left-dir and --right-dir")
        result = compare_response_directories(args.left_dir, args.right_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
