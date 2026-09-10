#!/usr/bin/env python3
"""Gate A (work==0) self-test: PASS when all 4 arms share per-request in/out;
FAIL when any arm's output length differs from the V-dec-rs reference.

Ported at 7f66e4688a without its `_fixtures` module, which had been removed at
390c3bf745 -- so this file could not import, and because it matches `test_*.py` it
broke `pytest test/vp/` collection for everything else. Rewritten as a plain pytest
whose only helper is grounded in what `cross_arm_work_gate.load_cell` actually reads
(the LAST record's `request_ids` / `input_lens` / `output_lens` arrays).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VP = Path(__file__).resolve().parents[1]
if str(VP) not in sys.path:
    sys.path.insert(0, str(VP))

import cross_arm_work_gate as gate  # noqa: E402

ARMS = ["P-def", "FD-eager", "V-dec", "V-dec-rs"]
RIDS = [f"gsm8k:{i}" for i in range(5)]
INS = [10, 20, 30, 40, 50]
OUTS = [100, 110, 120, 130, 140]
REFERENCE = "V-dec-rs"


def _write_cell(path: Path, rids, ins, outs) -> Path:
    """Write the minimal artifact `load_cell` consumes: one record, three arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"request_ids": list(rids), "input_lens": list(ins), "output_lens": list(outs)}
        )
        + "\n"
    )
    return path


def _load_arms(root: Path, out_overrides: dict[str, list[int]]) -> dict:
    return {
        arm: gate.load_cell(
            _write_cell(root / f"{arm}.jsonl", RIDS, INS, out_overrides.get(arm, OUTS))
        )
        for arm in ARMS
    }


def test_identical_work_passes(tmp_path: Path) -> None:
    verdict = gate.evaluate(
        "gsm8k_qps6_rep1", _load_arms(tmp_path, {}), None, REFERENCE
    )
    assert verdict["pass"] is True, verdict["mismatches"]
    assert verdict["mismatches"] == []


def test_one_extra_token_fails_and_is_attributed(tmp_path: Path) -> None:
    """The defect this gate exists for: a single extra generated token on one arm.

    Cross-arm claims on unmatched work cost ~$100 and two weeks, so the gate must fail
    on a one-token difference AND say which arm produced it.
    """
    bad = list(OUTS)
    bad[2] += 1
    verdict = gate.evaluate(
        "gsm8k_qps6_rep1",
        _load_arms(tmp_path, {"FD-eager": bad}),
        None,
        REFERENCE,
    )
    assert verdict["pass"] is False
    assert any(
        m.get("arm") == "FD-eager" and m.get("kind") == "length_mismatch"
        for m in verdict["mismatches"]
    ), verdict["mismatches"]
