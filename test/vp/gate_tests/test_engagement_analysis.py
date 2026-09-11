#!/usr/bin/env python3
"""The engagement/ridge reporter, pinned on not inventing its own inputs.

`enter_rows=176` is the constant a reviewer calls tuning. The defence is evidence, and
evidence produced by a tool that supplies its own device peaks is not evidence — that is the
manufactured-justification pattern already in this project's log. So the peaks and the band
are BOTH declared on the command line, and these tests hold that line.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "test/vp/engagement_analysis.py"
SRC = TOOL.read_text()
BASE = ["--enter-rows", "176", "--exit-rows", "144",
        "--peak-tflops", "312", "--peak-bw-gbs", "1935"]


def _cell(root: Path, arm: str, name: str, occ: list[int]) -> None:
    d = root / "rep1" / "ds" / arm / "cells"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}_qps1_rep1.load.jsonl").write_text(
        "\n".join(json.dumps({"running_requests": v}) for v in occ) + "\n")


def _run(*args) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


def test_the_tool_holds_NO_copy_of_a_served_constant_or_a_device_peak():
    """A second copy of enter_rows here would drift from the served design."""
    code = "\n".join(l for l in SRC.splitlines() if not l.strip().startswith("#"))
    body = code[code.index("def occupancy"):]
    for literal in ("176", "144", "1935", "312"):
        assert literal not in body, f"{literal} is hardcoded below the docstring"
    assert 'required=True' in SRC


def test_a_cell_that_never_reaches_enter_rows_is_NAMED_not_averaged(tmp_path: Path):
    _cell(tmp_path, "t", "low", [10, 40, 99])
    _cell(tmp_path, "t", "high", [10, 200, 400])
    out = _run(tmp_path, *BASE)
    assert out.returncode == 0, out.stderr
    assert "NEVER" in out.stdout
    assert "low" in out.stdout.split("NEVER reach enter_rows")[1]
    assert "decode leg is inert" in out.stdout


def test_idle_samples_do_not_count_as_low_occupancy(tmp_path: Path):
    """Zero running requests is not a regime; averaging it in would fake a low p50."""
    _cell(tmp_path, "t", "c", [0, 0, 0, 0, 200, 200])
    out = _run(tmp_path, *BASE, "--json", tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["cells"][0]["samples"] == 2
    assert data["cells"][0]["p50"] == 200


def test_the_ridge_is_computed_from_the_DECLARED_peaks(tmp_path: Path):
    _cell(tmp_path, "t", "c", [200])
    out = _run(tmp_path, *BASE, "--json", tmp_path / "r.json")
    ridge = json.loads((tmp_path / "r.json").read_text())["ridge_rows"]
    assert abs(ridge - 312e12 / 1935e9) < 1e-6
    # a different device gives a different ridge -- nothing is baked in
    out2 = _run(tmp_path, "--enter-rows", "176", "--exit-rows", "144",
                "--peak-tflops", "989", "--peak-bw-gbs", "3350",
                "--json", tmp_path / "r2.json")
    assert out2.returncode == 0
    assert abs(json.loads((tmp_path / "r2.json").read_text())["ridge_rows"] - 295.2) < 1.0


def test_an_inverted_band_is_refused(tmp_path: Path):
    _cell(tmp_path, "t", "c", [200])
    out = _run(tmp_path, "--enter-rows", "144", "--exit-rows", "176",
               "--peak-tflops", "312", "--peak-bw-gbs", "1935")
    assert out.returncode == 2 and "hysteresis" in out.stderr


def test_an_empty_directory_is_refused_not_reported_as_clean(tmp_path: Path):
    out = _run(tmp_path, *BASE)
    assert out.returncode == 2 and "no *.load.jsonl" in out.stderr


def test_a_cell_with_only_idle_samples_is_refused(tmp_path: Path):
    """All-zero occupancy means the cell did not run; reporting p50=0 would hide that."""
    _cell(tmp_path, "t", "c", [0, 0, 0])
    out = _run(tmp_path, *BASE)
    assert out.returncode == 2 and "NO NON-IDLE SAMPLES" in out.stdout
