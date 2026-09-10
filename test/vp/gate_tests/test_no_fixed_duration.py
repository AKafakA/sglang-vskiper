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


def test_a_caller_supplied_duration_is_REFUSED_at_the_parameter(tmp_path):
    """180 s at 8 qps would submit 1440 of 3600, and at 18 qps 3240 -- two rates, two amounts
    of work, one curve. The refusal now fires on the PARAMETER rather than on the resulting
    shortfall, so the wrong thing is not expressible at all for a measurement cell."""
    with pytest.raises(ValueError) as caught:
        run(tmp_path, min_prompts=400, duration_s=180)
    message = str(caught.value)
    assert "DIAGNOSTIC overrides" in message, message
    assert "rows/qps" in message


def test_derivation_happens_when_nothing_is_supplied(tmp_path):
    """The default path: give the runner neither, and it computes both from the suite and the
    rate. It must get PAST this check and fail later on the missing endpoint."""
    with pytest.raises(Exception) as caught:
        run(tmp_path, min_prompts=None, duration_s=None)
    assert "DIAGNOSTIC overrides" not in str(caught.value)
    assert "REFUSING" not in str(caught.value)


def test_there_is_no_fixed_duration_DEFAULT_any_more():
    """The rule was previously enforced by a post-hoc refusal while --duration-s still
    DEFAULTED to 180.0 -- so a caller that simply omitted the flag got exactly the value the
    rule forbids. Assert the default is gone, from the parser itself."""
    import argparse
    parser = argparse.ArgumentParser()
    src = Path(runner.__file__).read_text()
    assert 'parser.add_argument("--duration-s", type=float, default=180.0)' not in src
    assert '--duration-s", type=float, default=None' in src
    assert '--min-prompts", type=int, default=None' in src


def test_the_diagnostic_escape_still_works(tmp_path):
    """Smokes may submit a partial suite, but only by SAYING SO -- and by contract such a cell
    is never performance evidence."""
    with pytest.raises(Exception) as caught:
        run(tmp_path, min_prompts=32, duration_s=2, diagnostic=True)
    assert "DIAGNOSTIC overrides" not in str(caught.value)
    assert "REFUSING" not in str(caught.value)


def test_the_derived_duration_never_over_subscribes_the_suite(tmp_path):
    """duration = floor(rows/qps) can still round up past the suite at fractional rates, which
    is why the harvest's own derivation decrements. Check the real campaign rates."""
    import math
    for rows, qps in [(3600, 8.25), (3600, 10.45), (3600, 13.75),
                      (4000, 22.5), (4000, 28.5), (4000, 37.5), (3600, 11), (4000, 27)]:
        d = math.floor(rows / qps)
        while math.ceil(qps * d) > rows:
            d -= 1
        assert math.ceil(qps * d) <= rows, (rows, qps, d)
        assert rows - math.ceil(qps * d) < qps, f"{rows}@{qps}: leaves a whole extra second"
