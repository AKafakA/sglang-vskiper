#!/usr/bin/env python3
"""A measurement cell must submit its WHOLE suite; a fixed --duration-s is refused.

Owner, 2026-09-10: *"there exist no fixed duration allowed"*.

A wall-clock deadline makes each rate submit a different number of requests (rate x duration)
and truncates whatever is still in flight. Two cells at two rates then did different work,
which is fatal for a Q* curve -- whose entire content is a comparison ACROSS rates -- and for
a paired table, whose unit is a delta between two cells.

The defect survived because it lived in DRIVER SCRIPTS: eleven on the box carry a hardcoded
--duration-s, and each one that got cleaned left the other ten. So the rule now lives in the
runner, and this test asserts the REFUSAL rather than the pass -- a gate never observed
failing is not evidence (D-256, D-614, D-627 all passed their gates).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

VP = Path(__file__).resolve().parents[1]
if str(VP) not in sys.path:
    sys.path.insert(0, str(VP))

import run_qps_evaluation as runner  # noqa: E402

ROWS = 3600


def suite(tmp_path: Path) -> tuple[Path, Path]:
    requests = tmp_path / "gsm8k.d179.requests.jsonl"
    metadata = tmp_path / "gsm8k.d179.metadata.jsonl"
    with requests.open("w") as rq, metadata.open("w") as md:
        for i in range(ROWS):
            rq.write(json.dumps({"request_id": f"r{i}", "prompt_token_ids": [1, 2, 3]}) + "\n")
            md.write(json.dumps({"request_id": f"r{i}", "prompt_kind": "chat_messages"}) + "\n")
    return requests, metadata


def cell_args(tmp_path: Path, *, min_prompts: int, duration_s: float,
              diagnostic: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        experiment="upstream", min_prompts=min_prompts, duration_s=duration_s,
        partial_suite_diagnostic=diagnostic, output_dir=tmp_path / "cells",
        workload_records={},
    )


def run(tmp_path, **kw):
    requests, metadata = suite(tmp_path)
    return runner._run_cell(cell_args(tmp_path, **kw), "gsm8k.d179", 8.0, 1, requests, metadata)


def test_fixed_duration_is_REFUSED(tmp_path):
    """180 s at 8 qps submits 1440 of 3600 -- and at 18 qps it would submit 3240. Two rates,
    two different amounts of work, one curve. This is the case that must not run."""
    with pytest.raises(ValueError) as caught:
        run(tmp_path, min_prompts=400, duration_s=180)
    message = str(caught.value)
    assert "REFUSING" in message
    assert "1440 of 3600" in message, message
    assert "rows / qps" in message


def test_the_refusal_tells_the_caller_the_derived_duration(tmp_path):
    """A refusal that does not say what to do instead gets worked around, not fixed."""
    with pytest.raises(ValueError) as caught:
        run(tmp_path, min_prompts=400, duration_s=180)
    assert "--min-prompts 3600" in str(caught.value)
    assert "--duration-s 450" in str(caught.value)   # 3600 / 8


def test_derived_duration_passes_the_check(tmp_path):
    """rows/qps = 3600/8 = 450 s submits exactly the suite. It must get PAST this check --
    it fails later on the missing endpoint, which is a different error entirely."""
    with pytest.raises(Exception) as caught:
        run(tmp_path, min_prompts=ROWS, duration_s=450)
    assert "REFUSING" not in str(caught.value)


def test_diagnostic_escape_is_explicit(tmp_path):
    """Smokes may submit a partial suite, but only by SAYING SO -- and by contract such a
    cell is never performance evidence."""
    with pytest.raises(Exception) as caught:
        run(tmp_path, min_prompts=32, duration_s=2, diagnostic=True)
    assert "REFUSING" not in str(caught.value)
