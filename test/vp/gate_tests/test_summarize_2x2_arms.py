#!/usr/bin/env python3
"""The 2x2 summariser: it must READ the new layout, and refuse a non-upstream arm C.

Two defects this pins, both of the same family the project keeps re-finding.

1. **Dead tool.** The summariser globbed `<root>/A|B|C|D/`, which is the hand-run layout.
   `run_quality_2x2.py` writes `<root>/<arm-name>/<workload>/`, so pointed at the corrected
   campaign it exits FATAL on arm A -- at ~04:00Z, after the GPU is spent.

2. **The gate that could not see its own condition.** The 2x2's rule is "no quality collapse
   vs UPSTREAM sglang" (D-587), and for months arm C was ARMS["stock"] -- this fork with the
   skipper off. Every downstream reader saw only the arm LABEL, so the violation was
   invisible. The manifest now records the FACT, and the summariser refuses without it.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "test/vp/summarize_2x2.py"


def _cell(root: Path, arm_dir: str, workload: str, value: float,
          upstream: bool | None = None) -> None:
    d = root / arm_dir / workload
    d.mkdir(parents=True, exist_ok=True)
    (d / "results_x.json").write_text(json.dumps({
        "results": {"gsm8k": {"exact_match,flexible-extract": value,
                              "exact_match_stderr,flexible-extract": 0.01}}}))
    manifest = {"arm": arm_dir, "workload": workload, "status": "passed"}
    if upstream is not None:
        manifest["upstream_baseline"] = upstream
    (d / "quality_manifest.json").write_text(json.dumps(manifest))


def _campaign(root: Path, c_upstream: bool | None = True) -> list[str]:
    _cell(root, "armA", "gsm8k", 0.77)
    _cell(root, "armB", "gsm8k", 0.71)
    _cell(root, "upstream", "gsm8k", 0.76, upstream=c_upstream)
    _cell(root, "integrated_alwaysskip", "gsm8k", 0.70, upstream=False)
    return ["--arm-dir", "A=armA", "--arm-dir", "B=armB",
            "--arm-dir", "C=upstream", "--arm-dir", "D=integrated_alwaysskip"]


def _run(*args) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)],
                          capture_output=True, text=True)


def test_it_reads_the_committed_drivers_layout(tmp_path: Path):
    args = _campaign(tmp_path)
    out = _run(tmp_path, "--dataset", "gsm8k", *args)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "0.7700" in out.stdout and "0.7000" in out.stdout


def test_without_the_mapping_it_fails_LOUDLY_not_silently(tmp_path: Path):
    _campaign(tmp_path)
    out = _run(tmp_path, "--dataset", "gsm8k")
    assert out.returncode != 0
    assert "FATAL" in (out.stdout + out.stderr)


def test_a_non_upstream_arm_C_is_REFUSED(tmp_path: Path):
    """The exact violation that stood for months."""
    args = _campaign(tmp_path, c_upstream=False)
    out = _run(tmp_path, "--dataset", "gsm8k", *args)
    assert out.returncode != 0
    blob = out.stdout + out.stderr
    assert "NOT served by upstream" in blob and "refusing to compute" in blob.lower()


def test_a_manifest_predating_the_field_is_UNKNOWABLE_not_assumed(tmp_path: Path):
    args = _campaign(tmp_path, c_upstream=None)
    out = _run(tmp_path, "--dataset", "gsm8k", *args)
    assert out.returncode != 0
    assert "UNKNOWABLE" in (out.stdout + out.stderr)


def test_the_historical_escape_hatch_exists_and_is_NAMED_as_dangerous(tmp_path: Path):
    args = _campaign(tmp_path, c_upstream=None)
    out = _run(tmp_path, "--dataset", "gsm8k", *args, "--skip-upstream-check")
    assert out.returncode == 0, out.stdout + out.stderr
    help_text = _run("--help").stdout
    assert "Never for a new campaign" in help_text


def test_the_driver_records_the_fact_not_just_the_label():
    driver = (ROOT / "test/vp/run_lmeval_quality.py").read_text()
    assert '"upstream_baseline": bool(args.upstream_baseline)' in driver


def test_arm_C_is_described_as_UPSTREAM_in_the_legend():
    text = TOOL.read_text()
    assert "UPSTREAM SGLang" in text
    assert '"C": "stock base model, OUR serving stack"' not in text


def test_arms_may_live_under_DIFFERENT_roots(tmp_path: Path):
    """A and B carry over from the earlier campaign; C and D are re-measured elsewhere."""
    old, new = tmp_path / "old", tmp_path / "new"
    _cell(old, "A", "gsm8k", 0.77)
    _cell(old, "B", "gsm8k", 0.71)
    _cell(new, "upstream", "gsm8k", 0.76, upstream=True)
    _cell(new, "integrated_alwaysskip", "gsm8k", 0.70, upstream=False)
    out = _run(new, "--dataset", "gsm8k",
               "--arm-dir", f"A={old}/A", "--arm-dir", f"B={old}/B",
               "--arm-dir", "C=upstream", "--arm-dir", "D=integrated_alwaysskip")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "0.7700" in out.stdout and "0.7000" in out.stdout
