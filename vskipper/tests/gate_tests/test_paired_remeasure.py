"""A refused cell is re-measured on a fresh boot, only that rate, bounded, with the rejection kept.

Owner rule (cell-measurement rules, 2026-09-10; restated 2026-09-21 after the H100 upstream
0.75x cell was refused on sporadic server-side aborts): an empty or rejected cell is
re-measured, never accepted and never abandoned. CPU-only: the boot is faked.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("rpc", ROOT / "vskipper/src/vskipper/experiments/run_paired_campaign.py")
rpc = importlib.util.module_from_spec(spec)
sys.modules["rpc"] = rpc
spec.loader.exec_module(rpc)

SPEC = {"tree": str(ROOT), "python": sys.executable}
RATES = {"r10p5": 10.5, "r13p3": 13.3}


def _fake_boot(outcomes: list[dict[str, str]], boots: list[dict[str, float]]):
    """Each call consumes one outcome: {label: 'ok' | 'reject' | 'crash'}."""
    def boot(spec, dataset, rates, arm, rep, cell_root, port, attempt):
        boots.append(dict(rates))
        cells = cell_root / ("cells" if attempt == 1 else f"cells_retry{attempt}")   # as run_arm_boot does
        cells.mkdir(exist_ok=True)
        (cells / "run_manifest.json").write_text('{"attempt": %d}\n' % attempt)
        (cell_root / "runner.log").write_text(f"attempt {attempt}\n")
        outcome = outcomes.pop(0)
        for label in rates:
            suite = rpc.suite_name(dataset, label)
            base = f"{suite}_qps{rates[label]:g}_rep{rep}"
            (cells / f"{base}.arrival.requests.jsonl").write_text("{}\n")
            if outcome[label] == "ok":
                (cells / f"{base}.jsonl").write_text("{}\n")
                (cells / f"{base}.load.jsonl").write_text("{}\n")
            elif outcome[label] == "reject":
                (cells / f"INVALID.{base}.jsonl").write_text("{}\n")
                (cells / f"{base}.load.jsonl").write_text("{}\n")
                (cells / f"{base}.command.json").write_text('{"status": "rejected"}\n')
        return 0 if all(v == "ok" for v in outcome.values()) else 1
    return boot


def test_refused_rate_is_remeasured_alone_and_rejection_kept(tmp_path, monkeypatch):
    boots: list[dict[str, float]] = []
    monkeypatch.setattr(rpc, "run_arm_boot",
                        _fake_boot([{"r10p5": "ok", "r13p3": "reject"}, {"r13p3": "ok"}], boots))
    rc = rpc.run_arm_cells(SPEC, "gsm8k", RATES, "upstream_g1024", 1, tmp_path, 30000)
    assert rc == 0
    assert boots == [RATES, {"r13p3": 13.3}]          # the second boot measures ONLY the refused rate
    cells = tmp_path / "rep1/gsm8k/upstream_g1024/cells"
    assert rpc.cell_artifact(cells.parent, rpc.suite_name("gsm8k", "r13p3")) is not None
    shelf = cells / "rejected/attempt1"
    assert (shelf / "run_manifest.json").is_file()
    assert any(p.name.startswith("INVALID.") for p in shelf.iterdir())
    assert not any(p.name.startswith("INVALID.") for p in cells.iterdir())
    assert not rpc.refused_cells(cells, "gsm8k", RATES)
    assert (cells / "run_manifest.retry2.json").is_file()            # the re-measure's manifest kept beside
    assert not any((cells.parent / "cells_retry2").glob("*_qps*"))    # nothing left behind in the retry dir


def test_bounded_attempts_then_unquotable(tmp_path, monkeypatch):
    boots: list[dict[str, float]] = []
    monkeypatch.setattr(rpc, "run_arm_boot",
                        _fake_boot([{"r10p5": "ok", "r13p3": "reject"}] + [{"r13p3": "reject"}] * 5, boots))
    rc = rpc.run_arm_cells(SPEC, "gsm8k", RATES, "vskipper", 2, tmp_path, 30000)
    assert rc != 0
    assert len(boots) == rpc.MAX_CELL_ATTEMPTS
    cells = tmp_path / "rep2/gsm8k/vskipper/cells"
    assert rpc.refused_cells(cells, "gsm8k", RATES) == {"r13p3": 13.3}
    assert sorted(p.name for p in (cells / "rejected").iterdir()) == ["attempt1", "attempt2"]


def test_crashed_run_with_no_artifact_is_remeasured(tmp_path, monkeypatch):
    boots: list[dict[str, float]] = []
    monkeypatch.setattr(rpc, "run_arm_boot",
                        _fake_boot([{"r10p5": "crash", "r13p3": "crash"}, {"r10p5": "ok", "r13p3": "ok"}], boots))
    rc = rpc.run_arm_cells(SPEC, "coqa", RATES, "vskipper", 1, tmp_path, 30000)
    assert rc == 0 and len(boots) == 2 and boots[1] == RATES


def test_runner_command_never_resumes():
    spec = {"tree": str(ROOT), "python": "py", "suites_dir": "/s", "source_revision": "x",
            "arms": {"baseline": "upstream_g1024", "treatments": ["vskipper"]}, "upstream_arms": {}}
    monkey = rpc.arm_is_upstream
    rpc.arm_is_upstream = lambda s, a: False
    try:
        base = rpc.runner_command(spec, "vskipper", Path("/m"), Path("/q"), Path("/c"), 1)
        again = rpc.runner_command(spec, "vskipper", Path("/m"), Path("/q"), Path("/c_retry2"), 1)
    finally:
        rpc.arm_is_upstream = monkey
    # a re-measure boots a FRESH runner into its own directory (Codex r2 BLOCKER: the
    # runner refuses a resume whose qps config differs), so resume is never requested
    assert "--resume-completed" not in base and "--resume-completed" not in again
