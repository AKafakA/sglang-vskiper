#!/usr/bin/env python3
"""GR-1a must actually FIRE from the pairing driver, and must REFUSE unequal work.

This test exists because of two failures that are the same failure. First,
`cross_arm_work_gate.py` had no caller at all for months -- cross-arm claims on unmatched
work already cost this project ~$100 and two weeks, with a final warning attached. Then,
when it was finally wired into `run_paired_campaign.py`, the call omitted `--out`, which the
gate's CLI marks required: every invocation would have died on argparse with exit 2, the
wrapper would have logged that as a work-identity FAILURE, and a ten-hour 54-cell run would
have produced a table in which every pair read "unquotable" for a reason having nothing to do
with the data.

Both are invisible to a passing gate. So this asserts the REFUSAL, not the pass -- a gate
never observed failing is not evidence (D-256, D-614, D-627 all passed their gates).

Fixtures are built to `cross_arm_work_gate.load_cell`'s ACTUAL reader: the LAST json record
of the artifact, carrying logical_request_ids / input_lens / output_lens.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

VP = Path(__file__).resolve().parents[1]
if str(VP) not in sys.path:
    sys.path.insert(0, str(VP))

import run_paired_campaign as driver  # noqa: E402


def write_cell(path: Path, lens: list[tuple[int, int]]) -> Path:
    """An artifact shaped the way load_cell reads it: last line wins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(json.dumps({"note": "earlier record load_cell must ignore"}) + "\n")
        handle.write(json.dumps({
            "logical_request_ids": [f"r{i}" for i in range(len(lens))],
            "input_lens": [a for a, _ in lens],
            "output_lens": [b for _, b in lens],
        }) + "\n")
    return path


def build(tmp_path: Path, base: list[tuple[int, int]], treat: list[tuple[int, int]]):
    """A (rep, dataset) laid out exactly as run_arm_cells writes it."""
    spec = {
        "tree": str(VP.parent.parent),
        "arms": {"baseline": "upstream", "treatment": "integrated_it4"},
    }
    rates = {"r11p25": 11.25}
    suite = driver.suite_name("gsm8k", "r11p25")
    for arm, lens in (("upstream", base), ("integrated_it4", treat)):
        write_cell(tmp_path / "rep1" / "gsm8k" / arm / "cells" /
                   f"{suite}_qps11p25_rep1.jsonl", lens)
    return spec, rates


def test_equal_work_passes(tmp_path):
    same = [(10, 20), (11, 21), (12, 22)]
    spec, rates = build(tmp_path, same, list(same))
    failed = driver.cross_arm_work_gate(spec, "gsm8k", rates, 1, tmp_path)
    assert failed == [], f"identical work must pass, got {failed}"


def test_unequal_output_lens_is_REFUSED(tmp_path):
    """The whole point. One arm generating different tokens is not a comparable pair."""
    base = [(10, 20), (11, 21), (12, 22)]
    treat = [(10, 20), (11, 21), (12, 999)]        # one request does different work
    spec, rates = build(tmp_path, base, treat)
    failed = driver.cross_arm_work_gate(spec, "gsm8k", rates, 1, tmp_path)
    assert failed == ["r11p25"], f"unequal work must REFUSE the rate, got {failed}"


def test_the_gate_actually_ran_and_left_its_verdict(tmp_path):
    """Guards the --out defect specifically: an argparse death would leave no verdict file
    while still 'failing' the rate, which looks identical to a real refusal from the log."""
    same = [(10, 20), (11, 21)]
    spec, rates = build(tmp_path, same, list(same))
    driver.cross_arm_work_gate(spec, "gsm8k", rates, 1, tmp_path)
    verdicts = list((tmp_path / "rep1" / "gsm8k").glob("workgate_*.json"))
    assert verdicts, "the gate wrote no verdict file -- it did not really run"


def test_a_missing_arm_artifact_fails_rather_than_skips(tmp_path):
    """An unchecked pair is a FAILED pair -- audit D-624 #3 found the gate itself defaulting
    to pass when no --arm was given."""
    same = [(10, 20)]
    spec, rates = build(tmp_path, same, list(same))
    for p in (tmp_path / "rep1" / "gsm8k" / "integrated_it4" / "cells").glob("*.jsonl"):
        p.unlink()
    failed = driver.cross_arm_work_gate(spec, "gsm8k", rates, 1, tmp_path)
    assert failed == ["r11p25"], f"a missing artifact must fail the rate, got {failed}"
