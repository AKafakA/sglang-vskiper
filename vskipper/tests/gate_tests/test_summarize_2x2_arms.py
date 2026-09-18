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
TOOL = ROOT / "vskipper/src/vskipper/experiments/summarize_2x2.py"


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
    driver = (ROOT / "vskipper/src/vskipper/experiments/run_lmeval_quality.py").read_text()
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


def test_latex_emit_writes_a_row_and_macros(tmp_path: Path):
    """The paper must never carry a difference-of-differences computed anywhere but here."""
    args = _campaign(tmp_path)
    out = _run(tmp_path, "--dataset", "gsm8k", *args,
               "--latex", tmp_path / "quality_rows.tex")
    assert out.returncode == 0, out.stdout + out.stderr
    row = (tmp_path / "quality_rows.tex").read_text()
    assert row.startswith("GSM8K &") and row.rstrip().endswith("\\\\")
    assert "0.7700" in row and "0.7000" in row          # arm A and arm D, verbatim
    macros = (tmp_path / "quality_macros.tex").read_text()
    for name in ("vpQGsmBminusA", "vpQGsmDminusC", "vpQGsmDod", "vpQGsmArmD"):
        assert f"\\newcommand{{\\{name}}}" in macros, name


def test_latex_emit_uses_the_lmeval_KEY_not_a_prettier_name(tmp_path: Path):
    """D-641: the filter IS the measurement -- gsm8k strict-match flips the sign of (B-A)."""
    args = _campaign(tmp_path)
    _run(tmp_path, "--dataset", "gsm8k", *args, "--latex", tmp_path / "quality_rows.tex")
    assert "exact\\_match,flexible-extract" in (tmp_path / "quality_rows.tex").read_text()


def test_latex_emit_is_REFUSED_when_arm_C_is_not_upstream(tmp_path: Path):
    """A table row is the most quotable artifact there is; it must not outrun the gate."""
    args = _campaign(tmp_path, c_upstream=False)
    out = _run(tmp_path, "--dataset", "gsm8k", *args, "--latex", tmp_path / "q.tex")
    assert out.returncode != 0
    assert not (tmp_path / "q.tex").exists(), "it wrote a row despite refusing"


def test_a_REFUSED_cell_must_not_enter_the_2x2(tmp_path: Path):
    """Found in production 2026-09-12: coqa arm D was REFUSED on zero_empty and the
    summariser published its score anyway. A gate that fires and a consumer that ignores it
    is this project's most-repeated defect."""
    args = _campaign(tmp_path)
    d = tmp_path / "integrated_alwaysskip" / "gsm8k" / "quality_manifest.json"
    record = json.loads(d.read_text())
    record.update(status="refused", failed_gates=["zero_empty"])
    d.write_text(json.dumps(record))
    out = _run(tmp_path, "--dataset", "gsm8k", *args, "--latex", tmp_path / "q.tex")
    assert out.returncode != 0
    blob = out.stdout + out.stderr
    assert "status='refused'" in blob and "never publish a refused number" in blob
    assert not (tmp_path / "q.tex").exists(), "it emitted a row for a refused cell"


def test_a_missing_status_is_refused_not_assumed(tmp_path: Path):
    args = _campaign(tmp_path)
    d = tmp_path / "armA" / "gsm8k" / "quality_manifest.json"
    record = json.loads(d.read_text()); record.pop("status", None)
    d.write_text(json.dumps(record))
    out = _run(tmp_path, "--dataset", "gsm8k", *args)
    assert out.returncode != 0 and "status=None" in (out.stdout + out.stderr)
