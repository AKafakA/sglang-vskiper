"""Graph-coverage accounting.

Records which decode rows ran captured vs eager. This is the instrument that
makes the occupancy runaway visible instead of silent: rows above the captured
ceiling fall to the eager path, and without this accounting that degradation is
invisible.

The self-sized capture ladder that once lived here was removed -- it was opt-in,
unexercised by any arm, and its attestation defaulted to reporting a derivation
that never happened. See
codex/asplos-plan/2026-08-21-removed-feature-register.md. What survives is
measured from the decode runner, which is the coverage oracle by construction."""

from __future__ import annotations

from sglang.srt.vpipe.design import (  # [D-609]
    SERVED_EXECUTION_MODE, skipper_deployed,
)

import contextlib
from sglang.srt.environ import envs
from typing import Iterator, Sequence
import msgspec
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch


_fd_skip_decode_deployed_cache: Optional[bool] = None
_graph_lifecycle_depth = 0


COVERAGE_REASON_ROWS = "rows_gt_max_bs"
COVERAGE_REASON_INELIGIBLE = "not_graph_eligible"
COVERAGE_REASON_NO_RUNNER = "no_graph_runner"
COVERAGE_REASONS = (
    COVERAGE_REASON_ROWS,
    COVERAGE_REASON_INELIGIBLE,
    COVERAGE_REASON_NO_RUNNER,
)
class CoverageDenseCounters(msgspec.Struct):
    """Process-wide (c3) evidence counters, served via ``/server_info``.

    Two-site parity (F2/F18): site A (dispatch-seam decision) writes
    ``dense_overflow_*``; site B (dense-body entry at the first ROUTED layer,
    deduped per pass via ``vp_fd_coverage_counted``) writes ``dense_body_*``.
    The gate is ``decisions == passes`` and ``rows == rows`` — removing either
    site fails the cross-check.
    """

    # Site A — dispatch-seam decision witnesses.
    dense_overflow_decisions: int = 0
    dense_overflow_rows: int = 0
    overflow_reason: dict[str, int] = {}
    # Site B — dense-body positive witnesses (the Candidate-A lesson: the body
    # that ran is attested at the body, not at the decision).
    dense_body_passes: int = 0
    dense_body_rows: int = 0
    # Negative witnesses (invariant: exactly 0 in serving; >0 => cell VOID).
    eager_skip_decode_layer_calls: int = 0
    # Body-mix accounting per SS5 (token == decode row here).
    fd_tokens_skip_body: int = 0
    fd_tokens_prod_allrun_band: int = 0
    fd_tokens_dense_overflow: int = 0
    # Composition observability.
    dispatch_graph_steps: int = 0
    regime_observe_eager: int = 0
    coverage_stamp_w1_composed: int = 0
    # Graph-lifecycle mark evidence (F1): guarded sections entered, and each
    # mid-serving recapture with its forward-pass index (visible evidence, not
    # a silent counter hole).
    graph_lifecycle_sections: int = 0
    recapture_events: list[int] = []

    @classmethod
    def create(cls) -> "CoverageDenseCounters":
        return cls(
            overflow_reason={reason: 0 for reason in COVERAGE_REASONS},
            recapture_events=[],
        )
def vp_graph_lifecycle_active() -> bool:
    """True inside capture()/warmup()/recapture in the decode graph runner.

    ``torch.cuda.is_current_stream_capturing()`` excludes only the capture
    invocation itself; the backend runs every body twice EAGERLY before
    capture, and warmup/recapture also execute bodies eagerly on decode
    dummies — the sentinel must not count any of them.
    """

    return _graph_lifecycle_depth > 0
def _serving_decode_layer_call(forward_batch: Any) -> bool:
    mode = forward_batch.forward_mode
    if mode is None or not mode.is_decode():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if vp_graph_lifecycle_active():
        return False
    return True
def record_eager_skip_decode_layer_call(forward_batch: Any) -> None:
    """Negative witness at the entry of every FD skip decode body.

    Invariant: exactly 0 in serving.  Excluded: stream capture, and the
    graph-lifecycle sections (capture warmups, boot warmup, recapture).
    """

    if _serving_decode_layer_call(forward_batch):
        _c3_counters.eager_skip_decode_layer_calls += 1
_c3_counters = CoverageDenseCounters.create()
def coverage_dense_enabled() -> bool:
    """The ``SGLANG_VP_COVERAGE_DENSE`` kill switch (default ON, strict parse)."""

    return envs.SGLANG_VP_COVERAGE_DENSE.get()
def fd_skip_decode_deployed() -> bool:
    """Cached "FD weights loaded AND full_graph execution mode" predicate.

    Mirrors the env pattern the dispatch seam already uses
    (``model_runner.py`` full-graph dispatch recording); both values are
    boot-constant, so one read per process is exact.  Production (no FD
    weights) is False -> zero stamps, zero counters, byte-identical (E-C3).
    """

    global _fd_skip_decode_deployed_cache
    if _fd_skip_decode_deployed_cache is None:
        # [D-609] arm + design, not environment
        _fd_skip_decode_deployed_cache = bool(
            skipper_deployed() and SERVED_EXECUTION_MODE == "full_graph"
        )
    return _fd_skip_decode_deployed_cache
def coverage_dense_armed() -> bool:
    """Joint F14 scope: C-A stamping, C-C seam observe, and C-D self-sizing
    are all gated here — OFF is a mechanism-inert byte-parity boot."""

    return fd_skip_decode_deployed() and coverage_dense_enabled()
@contextlib.contextmanager
def vp_graph_lifecycle_mark() -> Iterator[None]:
    global _graph_lifecycle_depth
    _graph_lifecycle_depth += 1
    _c3_counters.graph_lifecycle_sections += 1
    try:
        yield
    finally:
        _graph_lifecycle_depth -= 1
def record_recapture_event(step_index: int) -> None:
    _c3_counters.recapture_events.append(int(step_index))
def record_dense_body_pass(forward_batch: Any) -> None:
    """Site-B positive witness (F2/F18): the dense body EXECUTED.

    Called from the dense fall-through of a ROUTED decoder layer only, so the
    first routed layer (16) of a coverage-stamped pass increments exactly once
    per pass (dedup via ``vp_fd_coverage_counted``).
    """

    if not forward_batch.vp_fd_decode_coverage_dense:
        return
    if forward_batch.vp_fd_coverage_counted:
        return
    forward_batch.vp_fd_coverage_counted = True
    rows = int(forward_batch.batch_size)
    _c3_counters.dense_body_passes += 1
    _c3_counters.dense_body_rows += rows
    _c3_counters.fd_tokens_dense_overflow += rows
def stamp_coverage_dense(
    forward_batch: Any,
    *,
    runner: Any,
    can_run_graph: bool,
    w1_active: bool,
) -> bool:
    """Site A: stamp one uncovered decode pass dense and account the decision.

    Caller contract (the model-runner dispatch seam): decode pass, FD
    deployed, kill switch ON, and the seam ``observe(rows)`` already issued.
    Returns True when the pass was stamped (caller resets via try/finally).
    """

    rows = int(forward_batch.batch_size)
    if can_run_graph:
        _c3_counters.dispatch_graph_steps += 1
        return False
    # Overflow OR oracle unavailable — same branch (R-E, fail-closed dense).
    forward_batch.vp_fd_decode_coverage_dense = True
    # cut8: invalidate the per-pass seam memo, which reads this stamp.
    forward_batch.vp_seam_batch_routed = None
    forward_batch.vp_seam_batch_eager = None
    # Same pairing as the W1 low band: every layer delegates to the
    # production attention backend, and the routed-backend plan is
    # suppressed for this pass (F4).
    forward_batch.fd_full_graph_force_production_attention = True
    _c3_counters.dense_overflow_decisions += 1
    _c3_counters.dense_overflow_rows += rows
    if runner is None:
        reason = COVERAGE_REASON_NO_RUNNER
    else:
        reason = runner._last_reject_reason or COVERAGE_REASON_INELIGIBLE
        _c3_counters.regime_observe_eager += 1
    _c3_counters.overflow_reason[reason] += 1
    if w1_active:
        _c3_counters.coverage_stamp_w1_composed += 1
    return True
def account_covered_dispatch(forward_batch: Any, band_body: Optional[str]) -> None:
    """Body-mix accounting for one covered (graph) decode dispatch."""

    rows = int(forward_batch.batch_size)
    if band_body == "prod_allrun":
        _c3_counters.fd_tokens_prod_allrun_band += rows
    else:
        _c3_counters.fd_tokens_skip_body += rows
def reset_coverage_stamps(forward_batch: Any) -> None:
    """try/finally reset for both stamps + the attention pairing (SS2.1)."""

    forward_batch.vp_fd_decode_coverage_dense = False
    forward_batch.vp_seam_batch_routed = None
    forward_batch.vp_seam_batch_eager = None
    forward_batch.vp_fd_coverage_counted = False
    forward_batch.fd_full_graph_force_production_attention = False


BODY_GRAPH_SKIP = "graph_skip"
BODY_GRAPH_ALLRUN = "graph_allrun"
BODY_DENSE_EAGER = "dense_eager"
BODY_EAGER_SKIP = "eager_skip"


def coverage_dense_counters() -> CoverageDenseCounters:
    return _c3_counters

def reset_coverage_dense_state() -> None:
    """Reset all module state (CPU tests only; serving never resets)."""

    global _c3_counters, _graph_lifecycle_depth
    global _fd_skip_decode_deployed_cache
    _c3_counters = CoverageDenseCounters.create()
    _graph_lifecycle_depth = 0
    _fd_skip_decode_deployed_cache = None

def coverage_parity_ok(counters: Optional[CoverageDenseCounters] = None) -> bool:
    """Two-site parity gate: decisions==passes and rows==rows (SS5)."""

    value = _c3_counters if counters is None else counters
    return (
        value.dense_overflow_decisions == value.dense_body_passes
        and value.dense_overflow_rows == value.dense_body_rows
    )

def cdopt_skip_only_bucket_legal(
    capture_bs: Sequence[int], bucket: int, enter_rows: int
) -> bool:
    """Corrected C-D-opt legality (F6): a skip-only capture of ``bucket`` is
    legal only when the bucket's MINIMUM mapped raw row count (previous bucket
    + 1) is at or above the W1 ``enter_rows`` — the band advances from RAW
    rows while the replay key uses the PADDED bucket, so any raw count mapping
    to this bucket below ``enter_rows`` would compose (bucket, prod_allrun)
    against a missing graph and crash at replay.  The halver itself stays
    deferred (SS2.4); only the predicate ships, so the corrected premise is
    executable evidence.
    """

    ordered = sorted(int(value) for value in capture_bs)
    if int(bucket) not in ordered:
        raise ValueError(f"bucket {bucket} is not in the capture ladder")
    index = ordered.index(int(bucket))
    min_mapped_raw = 1 if index == 0 else ordered[index - 1] + 1
    return min_mapped_raw >= int(enter_rows)

def select_decode_body(
    rows: int,
    max_bs: Optional[int],
    oracle_ok: bool,
    fd_enabled: bool,
    band_state: Optional[str],
    w1_enabled: bool,
) -> str:
    """Pure mirror of the total per-pass body composition, strict priority
    coverage > band:

    - not covered (rows above the realized cap, oracle unavailable, or no
      runner) -> stock dense eager (fail-closed, R-E);
    - covered, W1 low band -> captured production all-RUN body;
    - covered otherwise -> captured skip body.

    ``EAGER_SKIP`` is not a reachable output for ANY input (R-A totality).
    ``band_state`` is the W1 hysteresis state (``prod_allrun``/``skip``) and
    only selects among CAPTURED bodies — band staleness can never produce
    eager skip.
    """

    covered = oracle_ok and max_bs is not None and int(rows) <= int(max_bs)
    if not covered:
        return BODY_DENSE_EAGER
    if not fd_enabled:
        return BODY_GRAPH_ALLRUN
    if w1_enabled and band_state == "prod_allrun":
        return BODY_GRAPH_ALLRUN
    return BODY_GRAPH_SKIP
