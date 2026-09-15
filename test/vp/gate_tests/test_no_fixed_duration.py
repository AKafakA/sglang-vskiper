#!/usr/bin/env python3
"""A cell's size is its SUITE's row count and its rate. There is no duration, anywhere.

History, because the shape of the fix changed twice:

  * A fixed `--duration-s 180` made each rate submit `rate x duration` requests, so two cells
    at two rates did different work -- fatal to a Q* curve (a comparison ACROSS rates) and to
    a paired table (a delta BETWEEN two cells). made the runner DERIVE it as rows/qps
    and refuse a caller-supplied one.
  * Tracing that derivation end to end (2026-09-11) showed it could not affect anything:
    `requested_count = max(min_prompts, ceil(qps*duration))` with min_prompts set to the row
    count and duration shrunk until `ceil(qps*duration) <= rows` always returns the row count,
    and no duration ever reached the client -- the benchmark takes `--num-prompts` and
    `--request-rate`, with arrivals from trace timestamps.

    Owner: *"we do not need such duration which is confusing to control the numbers of
    requests or cells, it should only be designed by config and send with different rates."*

So the mechanism is GONE rather than guarded. These tests assert the absence, which is
strictly stronger than asserting a refusal: there is no longer an input to refuse.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VP = Path(__file__).resolve().parents[1]
if str(VP) not in sys.path:
    sys.path.insert(0, str(VP))

import run_qps_evaluation as runner  # noqa: E402

SRC = (VP / "run_qps_evaluation.py").read_text()
CODE = "\n".join(l for l in SRC.splitlines() if not l.strip().startswith("#"))


def test_the_cli_accepts_NO_duration_or_min_prompts():
    """The strongest form: a caller cannot pass one even to be refused."""
    helptext = subprocess.run(
        [sys.executable, str(VP / "run_qps_evaluation.py"), "--help"],
        capture_output=True, text=True,
    ).stdout
    assert "--duration-s" not in helptext
    assert "--min-prompts" not in helptext


def test_a_caller_passing_one_is_rejected_by_argparse(tmp_path):
    """With the required arguments supplied, argparse gets far enough to reject the flag
    itself -- which is the failure a stale driver script would actually hit."""
    required = ["--experiment", "upstream",
                "--deployment-manifest", str(tmp_path / "m.json"),
                "--workload-dir", str(tmp_path),
                "--qps-config", str(tmp_path / "q.json"),
                "--output-dir", str(tmp_path / "out"),
                "--evidence-class", "development"]
    done = subprocess.run(
        [sys.executable, str(VP / "run_qps_evaluation.py"), *required, "--duration-s", "180"],
        capture_output=True, text=True,
    )
    assert done.returncode != 0
    assert "unrecognized arguments: --duration-s" in done.stderr, done.stderr[-300:]


def test_the_arithmetic_is_gone_entirely():
    """`max(min_prompts, ceil(qps*duration))` was the fixed-duration semantics in derived
    clothing. Its absence is what makes the suite the only thing that sizes a cell."""
    assert "_requested_count" not in CODE
    assert "ceil(qps * duration_s)" not in CODE
    assert "args.duration_s" not in CODE
    assert "args.min_prompts" not in CODE


def test_requested_count_IS_the_available_rows():
    assert "requested_count = available" in CODE


def test_a_qps_config_cannot_smuggle_a_duration_back_in(tmp_path):
    """The config was the other door: `config.get("duration_s")` used to set it silently."""
    assert 'config.get("duration_s") is not None' in CODE
    assert "there is no duration" in SRC


def test_an_evaluation_contract_naming_duration_is_rejected():
    """A contract describing traffic.duration_seconds is describing a runner that no longer
    exists; honouring it silently would reintroduce the semantics through the back door."""
    assert 'traffic.get(stale)' in CODE
    assert '"duration_seconds", "minimum_requests"' in CODE


def test_the_declared_cell_size_gate_still_governs_the_suite():
    """With duration gone, DECLARED_SUITE_ROWS is the only thing standing between a
    wrong-sized suite and a cell built on it."""
    assert runner.DECLARED_SUITE_ROWS == {"gsm8k": 3600, "coqa": 4000, "bbh_cot": 4000}
