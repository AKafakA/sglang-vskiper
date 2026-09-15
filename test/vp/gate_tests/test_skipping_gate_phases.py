#!/usr/bin/env python3
"""The skipping gate counts BOTH phases from the per-body counters (D-736), records a low
share instead of refusing it, and REFUSES counters that do not add up or a dead escape.

[D-734] The paragraph below was written while the prefill dispatch machinery was silently OFF
(env-gated on a variable D-609 had deleted); the "{dense: 0, fd: 0} counts transitions" reading
was wrong -- the counters were zero because nothing counted. Kept as history of the mistake.

[D-697] The gate read `regime_switch.counters.decode` alone and refused the gsm8k 0.75 x Q*
cell as "never skipped" -- while that cell's treatment arm routed 4,925 prefill passes over
6,977,016 prefill tokens. The refused cell reproduces v1.3's published row within its
confidence interval on five of eight metrics.

`counters.prefill` is NOT the prefill equivalent: it reads {dense: 0, fd: 0} on that same cell
because it counts switch TRANSITIONS, not routed work. Routed prefill work lives in
`batch_composition`.

D-627 was this gate blind to a mechanism that was OFF. This was the same gate blind to a
mechanism that was ON -- worse, because it discards real results while looking like diligence.
The floor that survives is "something routed SOMEWHERE"; the share floor became a recorded note
(owner: "we need just record but not as the hard gates").
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "test/vp/gates/verify_skipping_executed.py"


def snap(tmp, name, *, decode_skip, decode_allrun, prefill_passes, prefill_routed=True,
         prefill_fd=None, prefill_dense=0, engagement_samples=None):
    """[D-736] Snapshots carry the prefill leg's real per-body counters: by default every prefill
    pass ran the routed body (fd == passes); tests override to model dense passes, dead counters
    ({0, 0} with passes > 0) and a dead engagement escape."""
    if prefill_fd is None:
        prefill_fd = prefill_passes - prefill_dense
    if engagement_samples is None:
        engagement_samples = prefill_fd  # a live escape folds a sample per routed pass
    counters = {"decode": {"skip": decode_skip, "prod_allrun": decode_allrun},
                "prefill": {"dense": prefill_dense, "fd": prefill_fd}}
    if engagement_samples is not None:
        counters["prefill_engagement"] = {"ema": 0.49, "samples": engagement_samples, "syncs": 0}
    d = {"internal_states": [{"vp_runtime": {
        "schema_version": 1,
        "served_design": {"arm": "integrated_it4",
                          "active_phases": ["decode", "prefill"] if prefill_routed else ["decode"]},
        "regime_switch": {"enabled": True, "version": 1,
                          "prefill": {"enabled": True, "min_tokens": 1536, "engagement_min": 0.35},
                          "counters": counters},
        "batch_composition": {"prefill_passes": prefill_passes, "prefill_tokens": prefill_passes * 1400,
                              "decode_passes": decode_skip + decode_allrun, "mixed_passes": 0},
    }}]}
    p = tmp / name
    p.write_text(json.dumps(d))
    return p


def run(before, after, *extra):
    return subprocess.run([sys.executable, str(GATE), "--before", str(before),
                           "--after", str(after), *extra], capture_output=True, text=True)


def test_prefill_only_routing_is_RECORDED_not_refused(tmp_path):
    """The real 0.75 x Q* shape: decode inert below enter_rows, prefill routing 7M tokens."""
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3)
    a = snap(tmp_path, "a.json", decode_skip=0, decode_allrun=28932, prefill_passes=2798)
    done = run(b, a)
    assert done.returncode == 0, done.stdout
    assert "RECORDED, not refused" in done.stdout
    assert "the skipper executed" in done.stdout


def test_the_share_is_still_reported_so_the_cell_can_be_judged(tmp_path):
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3)
    a = snap(tmp_path, "a.json", decode_skip=0, decode_allrun=28932, prefill_passes=2798)
    out = run(b, a).stdout
    assert "skip share" in out and "%" in out


def test_decode_skipping_still_passes_normally(tmp_path):
    """The 0.95 x Q* shape -- must not have been broken by the change."""
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=28932, prefill_passes=2798)
    a = snap(tmp_path, "a.json", decode_skip=25396, decode_allrun=38602, prefill_passes=4925)
    done = run(b, a)
    assert done.returncode == 0
    assert "RECORDED, not refused" not in done.stdout, "74% share must not trip the note"


def test_NOTHING_ROUTED_ANYWHERE_is_still_REFUSED(tmp_path):
    """D-627's actual failure, and the one thing the floor must still catch: the measurement
    describes the production body while claiming to describe the skipper."""
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=0)
    a = snap(tmp_path, "a.json", decode_skip=0, decode_allrun=16702, prefill_passes=0)
    done = run(b, a)
    assert done.returncode == 1
    assert "NOTHING ROUTED" in done.stdout


def test_an_arm_that_does_not_route_prefill_does_not_get_credit_for_it(tmp_path):
    """batch_composition counts prefill passes whether or not the ARM routes that phase, so the
    gate must gate on active_phases -- otherwise a decode-only arm banks prefill it never routed."""
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3, prefill_routed=False)
    a = snap(tmp_path, "a.json", decode_skip=0, decode_allrun=28932, prefill_passes=2798, prefill_routed=False)
    done = run(b, a)
    assert done.returncode == 1, "decode-only arm with no decode skipping must still refuse"
    assert "NOTHING ROUTED" in done.stdout


def test_the_source_label_is_stable_across_snapshots(tmp_path):
    """The label is compared before/after to catch a server restart. Embedding the COUNTS in it
    made every run look like a reconfiguration -- which is exactly what happened on the first
    attempt at this fix."""
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3)
    a = snap(tmp_path, "a.json", decode_skip=25396, decode_allrun=38602, prefill_passes=4925)
    out = run(b, a).stdout + run(b, a).stderr
    assert "counter source changed mid-run" not in out, out[-300:]


# --- the counters are the evidence, and they must add up -------------------------------

def _snap_pair(tmp, **kw):
    b = snap(tmp, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3, engagement_samples=0, **kw)
    a = snap(tmp, "a.json", decode_skip=100, decode_allrun=400, prefill_passes=2798,
             engagement_samples=40, **kw)
    return b, a


def test_healthy_prefill_counters_pass_and_are_printed(tmp_path):
    b, a = _snap_pair(tmp_path)
    done = run(b, a)
    assert done.returncode == 0, done.stdout
    assert "routed body (fd)           : 2795" in done.stdout and "engagement escape" in done.stdout


def test_dead_prefill_counters_with_passes_are_REFUSED(tmp_path):
    """The exact shape every campaign cell had before D-734: passes ran, counters read {0, 0}."""
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3, prefill_fd=0, engagement_samples=0)
    a = snap(tmp_path, "a.json", decode_skip=100, decode_allrun=400, prefill_passes=2798, prefill_fd=0, engagement_samples=0)
    done = run(b, a)
    assert done.returncode == 1 and "do not add up" in done.stdout, done.stdout


def test_counters_that_do_not_sum_to_the_scheduler_passes_are_REFUSED(tmp_path):
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3)
    a = snap(tmp_path, "a.json", decode_skip=100, decode_allrun=400, prefill_passes=2798, prefill_fd=2000)
    done = run(b, a)
    assert done.returncode == 1 and "do not add up" in done.stdout, done.stdout


def test_dense_passes_are_counted_not_assumed_routed(tmp_path):
    """CoQA after the fix: most passes demoted to dense by the escape; fd small but > 0."""
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3, prefill_dense=0, engagement_samples=0)
    a = snap(tmp_path, "a.json", decode_skip=0, decode_allrun=15000, prefill_passes=372, prefill_dense=360,
             engagement_samples=12)
    done = run(b, a)
    assert done.returncode == 0, done.stdout
    assert "dense body                 : 360" in done.stdout


def test_dead_engagement_escape_is_REFUSED(tmp_path):
    b = snap(tmp_path, "b.json", decode_skip=0, decode_allrun=20, prefill_passes=3, engagement_samples=0)
    a = snap(tmp_path, "a.json", decode_skip=100, decode_allrun=400, prefill_passes=2798, engagement_samples=0)
    done = run(b, a)
    assert done.returncode == 1 and "escape is not executing" in done.stdout, done.stdout
