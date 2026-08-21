#!/usr/bin/env python3
"""Compare deterministic direct/V2 FlexiDepth parity tensor files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = {path.name: path for path in args.left.glob("*.pt")}
    right = {path.name: path for path in args.right.glob("*.pt")}
    names = sorted(left.keys() & right.keys())
    if not names:
        raise ValueError("FlexiDepth parity traces have no shared tensors")

    comparisons = []
    for name in names:
        left_tensor = torch.load(left[name], map_location="cpu", weights_only=True)
        right_tensor = torch.load(right[name], map_location="cpu", weights_only=True)
        comparisons.append(
            {
                "name": name,
                "same_shape": tuple(left_tensor.shape) == tuple(right_tensor.shape),
                "exact": bool(torch.equal(left_tensor, right_tensor)),
            }
        )
    report = {
        "left_records": len(left),
        "right_records": len(right),
        "shared_records": len(names),
        "left_only": sorted(left.keys() - right.keys()),
        "right_only": sorted(right.keys() - left.keys()),
        "exact_records": sum(item["exact"] for item in comparisons),
        "comparisons": comparisons,
    }
    report["status"] = (
        "passed"
        if not report["left_only"]
        and not report["right_only"]
        and report["exact_records"] == report["shared_records"]
        else "failed"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "left_records", "right_records", "shared_records", "exact_records"
    )}, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
