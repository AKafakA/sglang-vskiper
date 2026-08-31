#!/usr/bin/env python3
"""Gate A (work==0) self-test: PASS when all 4 arms share per-request in/out;
FAIL when any arm's output length differs from the V-dec-rs reference."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import _fixtures as fx
import cross_arm_work_gate as gate

ARMS = ["P-def", "FD-eager", "V-dec", "V-dec-rs"]
RIDS = [f"gsm8k:{i}" for i in range(5)]
INS = [10, 20, 30, 40, 50]
OUTS = [100, 110, 120, 130, 140]


def _load_arms(root: Path, out_overrides: dict[str, list[int]]) -> dict:
    arms = {}
    for arm in ARMS:
        outs = out_overrides.get(arm, OUTS)
        path = fx.write_cell(root / f"{arm}.jsonl", RIDS, INS, outs)
        arms[arm] = gate.load_cell(path)
    return arms


def run() -> bool:
    case = fx.Case("Gate A work==0")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # PASS: identical work across all arms (reference = V-dec-rs).
        arms = _load_arms(root / "pass", {})
        case.expect_pass(gate.evaluate("gsm8k_qps6_rep1", arms, None, "V-dec-rs"), "identical work")

        # FAIL: FD-eager generated one extra token on request 2.
        bad = list(OUTS)
        bad[2] += 1
        arms = _load_arms(root / "fail", {"FD-eager": bad})
        verdict = gate.evaluate("gsm8k_qps6_rep1", arms, None, "V-dec-rs")
        case.expect_fail(verdict, "output-length mismatch")
        case.check(
            any(m.get("arm") == "FD-eager" and m.get("kind") == "length_mismatch"
                for m in verdict["mismatches"]),
            "mismatch attributed to FD-eager",
        )
    return case.done()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
