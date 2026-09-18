#!/usr/bin/env python3
"""The paired table refuses an ungated rep, and reports a straddle as parity.

The plan requires paired analysis bound into the run flow (D-624 #5). It was not: the driver
writes cells and computes no delta, and the one tool with the right method was written for the
v1.3 layout — it globs `ovn-*/results/`, looks for `workgate_*.log` where the driver writes
`.json`, and hardcodes the VOID knees (BBH 35, CoQA 23) against live values of 25 and 27.

These tests encode the two behaviours that matter more than the arithmetic: an ungated rep is
REFUSED loudly rather than dropped into the mean (D-593 — a missing rep and a failed rep must
never look alike), and an interval containing zero is reported as parity.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "vskipper/src/vskipper/experiments/paired_analysis.py"

BASE = {"median_ttft_ms": 100.0, "mean_ttft_ms": 110.0, "p99_ttft_ms": 200.0,
        "median_tpot_ms": 30.0, "mean_tpot_ms": 32.0, "p99_tpot_ms": 50.0,
        "median_e2e_latency_ms": 1000.0, "mean_e2e_latency_ms": 1100.0,
        "p99_e2e_latency_ms": 5000.0, "output_throughput": 2000.0}


def _cell(root: Path, arm: str, suite: str, rep: int, scale: float) -> None:
    d = root / arm / "cells"
    d.mkdir(parents=True, exist_ok=True)
    rec = {k: v * scale for k, v in BASE.items()}
    (d / f"{suite}_qps10p45_rep{rep}.jsonl").write_text(json.dumps(rec) + "\n")


def _build(tmp: Path, reps: int, *, gated: bool = True, scale=lambda r: 0.90,
           gate_pass: bool = True) -> Path:
    out = tmp / "out"
    suite = "gsm8k_eqw_r10p45"
    for rep in range(1, reps + 1):
        ds = out / f"rep{rep}" / "gsm8k"
        _cell(ds, "upstream", suite, rep, 1.0)
        _cell(ds, "integrated_it4", suite, rep, scale(rep))
        if gated:
            (ds / f"workgate_{suite}.json").write_text(json.dumps(
                {"gate": "A_work_identity", "pass": gate_pass, "mismatch_count": 0 if gate_pass else 7}))
    return out


def _run(out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), str(out), *extra],
                          capture_output=True, text=True)


def test_a_gated_pair_produces_a_delta(tmp_path):
    done = _run(_build(tmp_path, 3))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "gsm8k" in done.stdout and "gsm8k_eqw_r10p45" in done.stdout
    assert "n = 3 gated reps" in done.stdout
    # treatment latencies are 0.90x the baseline -> -10 %
    assert "-10.00 %" in done.stdout


def test_an_UNGATED_rep_is_refused_loudly_not_dropped(tmp_path):
    """The D-593 rule: a rep with no GR-1a verdict must not quietly vanish into the mean."""
    out = _build(tmp_path, 3)
    (out / "rep2" / "gsm8k" / "workgate_gsm8k_eqw_r10p45.json").unlink()
    done = _run(out)
    assert "REFUSED reps" in done.stdout
    assert "rep2" in done.stdout and "GR-1a did not run" in done.stdout
    assert "n = 2 gated reps" in done.stdout, "the surviving reps still report, but as n=2"


def test_a_FAILED_workgate_is_refused(tmp_path):
    done = _run(_build(tmp_path, 2, gate_pass=False))
    assert done.returncode == 1
    assert "GR-1a FAILED" in done.stdout
    assert "no gated (dataset, rate) pair produced a delta" in done.stdout


def test_a_straddle_is_reported_as_PARITY(tmp_path):
    """Deltas of -2 %, +2 %, -1 %, +1 % average near zero with a wide interval."""
    out = _build(tmp_path, 4, scale=lambda r: {1: 0.98, 2: 1.02, 3: 0.99, 4: 1.01}[r])
    done = _run(out)
    assert done.returncode == 0
    assert "PARITY (straddles 0)" in done.stdout


def test_a_consistent_improvement_is_not_called_parity(tmp_path):
    out = _build(tmp_path, 4, scale=lambda r: {1: 0.80, 2: 0.81, 3: 0.79, 4: 0.80}[r])
    done = _run(out)
    assert "treatment better" in done.stdout


def test_one_rep_reports_a_mean_with_NO_interval(tmp_path):
    """n=1 is a point estimate. It must not be dressed as a confidence interval."""
    done = _run(_build(tmp_path, 1))
    assert "n=1, no interval" in done.stdout


def test_throughput_direction_is_inverted_relative_to_latency(tmp_path):
    """Lower latency is better; lower TPS is worse. A single sign rule would mislabel one."""
    out = _build(tmp_path, 4, scale=lambda r: {1: 0.80, 2: 0.81, 3: 0.79, 4: 0.80}[r])
    done = _run(out)
    lines = {l.split()[0] + " " + l.split()[1]: l for l in done.stdout.splitlines()
             if l.strip().startswith(("TTFT", "TPOT", "E2E", "output"))}
    assert "treatment better" in lines["TTFT p50"], lines["TTFT p50"]
    assert "treatment worse" in lines["output TPS"], lines["output TPS"]


def test_the_json_report_is_machine_readable(tmp_path):
    out = _build(tmp_path, 3)
    target = tmp_path / "report.json"
    done = _run(out, "--json", str(target))
    assert done.returncode == 0
    report = json.loads(target.read_text())
    assert report["baseline"] == "upstream" and report["treatment"] == "integrated_it4"
    assert any(r["metric"] == "TTFT p50" and r["n"] == 3 for r in report["rows"])


def test_a_MISSING_ARM_DIRECTORY_is_reported_not_skipped(tmp_path):
    """Found by running this tool against the live headline's partial output rather than only
    against fixtures: an absent arm directory took a silent `break`, so the tool said "no gated
    pair" without naming which dataset was missing or why.

    This runs at the END of a campaign, where an absent arm means that dataset never completed.
    Silently skipping makes "never ran" look identical to "not in the spec" -- D-593's rule one
    level up."""
    out = _build(tmp_path, 2)
    import shutil
    shutil.rmtree(out / "rep2" / "gsm8k" / "integrated_it4")
    done = _run(out)
    assert "REFUSED reps" in done.stdout
    assert "rep2 gsm8k" in done.stdout
    assert "integrated_it4" in done.stdout and "never ran" in done.stdout
    assert "n = 1 gated rep" in done.stdout, "the surviving rep still reports"


def test_both_arms_missing_is_reported_once_naming_both(tmp_path):
    out = _build(tmp_path, 1)
    import shutil
    for arm in ("upstream", "integrated_it4"):
        shutil.rmtree(out / "rep1" / "gsm8k" / arm)
    done = _run(out)
    assert done.returncode == 1
    assert "upstream, integrated_it4" in done.stdout
