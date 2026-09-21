#!/usr/bin/env python3
"""The skipping gate must read the counter block that applies to the SERVED ARM.

Two arms, two counter blocks, and reading only one of them is a bug in each direction:

  integrated_it4          admission-gated. regime_switch.counters counts decode PASSES as
                          skip vs prod_allrun. This is where was caught: 16,702
                          passes, skip 0, an entire night of quality numbers describing
                          the production all-RUN body.
  integrated_alwaysskip   the 2x2's arm D. Its defining property is `regime_switch: False`,
                          so those counters are permanently zero and reading them says
                          "nothing ran" about an arm that routed 100% of its tokens.
                          Evidence lives in fd_c3.counters, as TOKENS.

The fix must make the second work WITHOUT weakening the first.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VP = Path(__file__).resolve().parents[1]
GATE = VP / "gates" / "verify_skipping_executed.py"


def _snapshot(path: Path, *, regime=None, c3=None) -> Path:
    vp: dict = {}
    if regime is not None:
        vp["regime_switch"] = {"enabled": True, "counters": {"decode": regime}}
    else:
        vp["regime_switch"] = {"enabled": False,
                               "counters": {"decode": {"skip": 0, "prod_allrun": 0}}}
    if c3 is not None:
        vp["fd_c3"] = {"enabled": True, "counters": c3}
    path.write_text(json.dumps({"internal_states": [{"vp_runtime": vp}]}))
    return path


def _run(before: Path, after: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(GATE), "--before", str(before), "--after", str(after), *extra],
        capture_output=True, text=True,
    )


# --- admission-gated arm: the catch must survive ---


def test_d627_case_is_still_refused(tmp_path):
    """16,702 decode passes, skip 0 -- the original defect, verbatim."""
    b = _snapshot(tmp_path / "b.json", regime={"skip": 0, "prod_allrun": 18})
    a = _snapshot(tmp_path / "a.json", regime={"skip": 0, "prod_allrun": 16720})
    r = _run(b, a)
    assert r.returncode == 1
    # message updated by: the check now spans both phases, so "never skipped" became
    # "nothing routed in either phase"
    assert "NOTHING ROUTED" in r.stdout


def test_gated_arm_that_did_skip_passes(tmp_path):
    b = _snapshot(tmp_path / "b.json", regime={"skip": 0, "prod_allrun": 0})
    a = _snapshot(tmp_path / "a.json", regime={"skip": 900, "prod_allrun": 100})
    r = _run(b, a)
    assert r.returncode == 0, r.stdout
    assert "regime_switch" in r.stdout


def test_share_below_the_floor_is_RECORDED_not_refused(tmp_path):
    """CHANGED BY (owner: "we need just record but not as the hard gates").

    This used to refuse, on the reasoning that a mostly-no-skip cell attributes the skipper's
    quality to a system that largely was not it. That reasoning holds for the QUALITY lane and
    fails for the PERF lane: below `enter_rows` the load-aware design runs prod_allrun by
    construction, so a low decode share is the design behaving correctly, not a broken
    measurement. Refusing it discarded the gsm8k 0.75 x Q* cell, which reproduces v1.3's
    published row within its CI on five of eight metrics.

    The share is still computed and printed, so a reader can judge the cell; it no longer
    decides the exit code on its own. What still refuses is NOTHING routed anywhere."""
    b = _snapshot(tmp_path / "b.json", regime={"skip": 0, "prod_allrun": 0})
    a = _snapshot(tmp_path / "a.json", regime={"skip": 100, "prod_allrun": 900})
    r = _run(b, a)
    assert r.returncode == 0, r.stdout
    assert "10.0%" in r.stdout
    assert "RECORDED, not refused" in r.stdout


# --- always-route arm: the case that exposed the bug ---


def test_always_route_arm_is_certified_from_fd_c3(tmp_path):
    """Real numbers from the 2026-09-10 P0 smoke: 10 -> 4975 tokens, 0 in the all-RUN band."""
    b = _snapshot(tmp_path / "b.json",
                  c3={"fd_tokens_skip_body": 10, "fd_tokens_prod_allrun_band": 0})
    a = _snapshot(tmp_path / "a.json",
                  c3={"fd_tokens_skip_body": 4975, "fd_tokens_prod_allrun_band": 0})
    r = _run(b, a)
    assert r.returncode == 0, r.stdout
    assert "fd_c3" in r.stdout and "100.0%" in r.stdout


def test_always_route_arm_that_did_not_skip_is_refused(tmp_path):
    """The fix must not become a rubber stamp for the arm it was added to support."""
    b = _snapshot(tmp_path / "b.json",
                  c3={"fd_tokens_skip_body": 0, "fd_tokens_prod_allrun_band": 0})
    a = _snapshot(tmp_path / "a.json",
                  c3={"fd_tokens_skip_body": 0, "fd_tokens_prod_allrun_band": 5000})
    r = _run(b, a)
    assert r.returncode == 1
    # message updated by: the check now spans both phases, so "never skipped" became
    # "nothing routed in either phase"
    assert "NOTHING ROUTED" in r.stdout


# --- neither block: refuse, never assume ---


def test_no_counter_block_at_all_is_refused(tmp_path):
    b = _snapshot(tmp_path / "b.json")
    a = _snapshot(tmp_path / "a.json")
    r = _run(b, a)
    assert r.returncode != 0
    assert "neither an enabled regime_switch nor fd_c3" in (r.stdout + r.stderr)


def test_source_changing_mid_run_is_refused(tmp_path):
    """A restart between snapshots makes the delta meaningless."""
    b = _snapshot(tmp_path / "b.json", regime={"skip": 5, "prod_allrun": 5})
    a = _snapshot(tmp_path / "a.json",
                  c3={"fd_tokens_skip_body": 500, "fd_tokens_prod_allrun_band": 0})
    r = _run(b, a)
    assert r.returncode != 0
    assert "counter source changed" in (r.stdout + r.stderr)


# --- [D-849] pinned arm below the band: nothing routed is the design, not a defect ---


def _pinned_snapshot(path: Path, *, admitted_fd: int, skip: int, allrun: int) -> Path:
    vp = {"regime_switch": {"enabled": True, "version": 2,
                            "admission": {"enabled": True},
                            "counters": {"decode": {"skip": skip, "prod_allrun": allrun},
                                         "admission": {"coverage_dense_violation_rows": 0}}},
          "admission_pins": {"enabled": True, "admitted_stock": 40, "admitted_fd": admitted_fd,
                             "cross_body_prefix_reuse": 0}}
    path.write_text(json.dumps({"internal_states": [{"vp_runtime": vp}]}))
    return path


def test_pinned_arm_with_no_fd_pins_is_recorded_not_refused(tmp_path):
    b = _pinned_snapshot(tmp_path / "b.json", admitted_fd=0, skip=0, allrun=0)
    a = _pinned_snapshot(tmp_path / "a.json", admitted_fd=0, skip=0, allrun=800)
    r = _run(b, a)
    assert r.returncode == 0, r.stdout
    assert "BY CONSTRUCTION" in r.stdout


def test_pinned_arm_with_fd_pins_but_no_skip_is_refused(tmp_path):
    b = _pinned_snapshot(tmp_path / "b.json", admitted_fd=0, skip=0, allrun=0)
    a = _pinned_snapshot(tmp_path / "a.json", admitted_fd=12, skip=0, allrun=800)
    r = _run(b, a)
    assert r.returncode == 1 and "NOTHING ROUTED" in r.stdout


def test_pin_violation_counters_are_refused(tmp_path):
    b = _pinned_snapshot(tmp_path / "b.json", admitted_fd=0, skip=0, allrun=0)
    a = _pinned_snapshot(tmp_path / "a.json", admitted_fd=12, skip=500, allrun=300)
    d = json.loads(a.read_text()); d["internal_states"][0]["vp_runtime"]["admission_pins"]["cross_body_prefix_reuse"] = 1
    a.write_text(json.dumps(d))
    r = _run(b, a)
    assert r.returncode == 1 and "other body" in r.stdout
