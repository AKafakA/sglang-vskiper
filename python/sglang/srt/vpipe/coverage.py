"""Graph-coverage accounting and the self-sized capture ladder.

Records which decode rows ran captured vs eager, why capture stopped (OOM,
error, memory reserve), and the realised ladder. This is the instrument that
makes the occupancy runaway visible instead of silent."""

from __future__ import annotations

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
    # V4/V2 scheduler-owned route body (F3 tripwire; pinned 0 while the V4
    # decode scheduler is disabled — nonzero forces the C-E obligation).
    v4_route_execute_decode_calls: int = 0
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
LADDER_STOP_POOL_CEILING = "pool_ceiling"
LADDER_STOP_USER_FLAG = "user_flag"
RESERVE_PROVENANCE_DEFAULT_ZERO = "default_zero_unmeasured"
RESERVE_PROVENANCE_DECLARED = "declared_campaign_config"
LADDER_SOURCE_SELF_SIZED = "self_sized"
LADDER_SOURCE_USER_FLAG = "user_flag"
LADDER_STOP_MEMORY_RESERVE = "memory_reserve"
LADDER_STOP_CAPTURE_OOM = "capture_oom"
LADDER_STOP_CAPTURE_ERROR = "capture_error"
class CoverageLadderRecord(msgspec.Struct):
    """Per-boot C-D capture-ladder provenance, served via ``/server_info``.

    The realized ladder itself (``capture_bs`` / ``max_bs``) lives on the
    decode graph runner — the oracle IS the runner state (R-C); this record
    carries only derivation provenance and trim history.
    """

    ladder_source: str = LADDER_SOURCE_SELF_SIZED
    # The derivation target: the pool ceiling (self_sized) or the verbatim
    # user cap (user_flag).
    target_max_bs: int = 0
    req_to_token_pool_size: int = 0
    stop_reason: str = LADDER_STOP_POOL_CEILING
    reserve_bytes: int = 0
    reserve_provenance: str = RESERVE_PROVENANCE_DEFAULT_ZERO
    # Each trim: {"bucket": int, "stop_reason": str} in occurrence order.
    capture_trim_events: list[dict[str, Any]] = []
    derived: bool = False
    # The realized SELF-SIZED ladder for THIS boot: recorded once at the single
    # derivation site (post alignment/clamp filtering) and shrunk in lockstep
    # with every descent/reserve trim. Source of truth for per-boot ladder
    # idempotency: any later capture cycle (mid-serving recapture, runner
    # rebuild) reuses this list instead of re-deriving from the pool ceiling —
    # a recapture must capture only what previously succeeded, never re-run
    # the multi-minute descent (measured 2026-08-04: a mid-serving recapture
    # restarted the 4096 descent at bucket 1344 after the boot had already
    # descended to 1216, stalling the scheduler and killing every in-flight
    # stream). Always [] on a user_flag boot (stock behavior verbatim).
    realized_capture_bs: list[int] = []

    @classmethod
    def create(cls) -> "CoverageLadderRecord":
        return cls(capture_trim_events=[], realized_capture_bs=[])
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
        _fd_skip_decode_deployed_cache = bool(
            os.environ.get("SGLANG_FD_WEIGHTS", "").strip()
            and os.environ.get("SGLANG_FD_EXECUTION_MODE", "").strip().lower()
            == "full_graph"
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
def coverage_reserve_bytes() -> int:
    return int(envs.SGLANG_VP_COVERAGE_RESERVE_BYTES.get())
def coverage_reserve_provenance() -> str:
    return (
        RESERVE_PROVENANCE_DECLARED
        if envs.SGLANG_VP_COVERAGE_RESERVE_BYTES.is_set()
        else RESERVE_PROVENANCE_DEFAULT_ZERO
    )
def note_ladder_derivation(
    *, source: str, target_max_bs: int, pool_size: int
) -> None:
    """Record the C-D derivation decision (called from the single existing
    derivation site, ``get_batch_sizes_to_capture``)."""

    _c3_ladder.ladder_source = source
    _c3_ladder.target_max_bs = int(target_max_bs)
    _c3_ladder.req_to_token_pool_size = int(pool_size)
    _c3_ladder.stop_reason = (
        LADDER_STOP_POOL_CEILING
        if source == LADDER_SOURCE_SELF_SIZED
        else LADDER_STOP_USER_FLAG
    )
    _c3_ladder.reserve_bytes = coverage_reserve_bytes()
    _c3_ladder.reserve_provenance = coverage_reserve_provenance()
    _c3_ladder.derived = True
    # A fresh derivation supersedes any stored realized ladder; the site
    # records the new one via record_realized_ladder after filtering.
    _c3_ladder.realized_capture_bs = []
def record_realized_ladder(capture_bs: Sequence[int]) -> None:
    """Store THIS boot's realized self-sized ladder (post alignment/clamp
    filtering) for per-boot idempotent reuse. Called from the derivation site
    exactly once per fresh SELF-SIZED derivation; user_flag boots never store
    (stock behavior verbatim)."""

    _c3_ladder.realized_capture_bs = [int(value) for value in capture_bs]
def persisted_self_sized_ladder() -> Optional[list[int]]:
    """The already-derived (and possibly descent-trimmed) SELF-SIZED ladder of
    THIS boot, or ``None`` when no self-sized ladder was realized yet.

    This makes the self-sized derivation idempotent per boot: a later capture
    cycle — the mid-serving ``recapture_if_needed`` path, or a rebuilt decode
    graph runner — reuses exactly the buckets that previously realized, so it
    never re-derives from the pool ceiling and never re-runs the bounded
    descent while requests are streaming. ``reset_coverage_dense_state()``
    (CPU tests / fresh boot semantics) clears it, so a new boot re-derives.
    """

    if not _c3_ladder.derived:
        return None
    if _c3_ladder.ladder_source != LADDER_SOURCE_SELF_SIZED:
        return None
    if not _c3_ladder.realized_capture_bs:
        return None
    return list(_c3_ladder.realized_capture_bs)
def coverage_self_sized_ladder_active() -> bool:
    """True when THIS boot derived a self-sized ladder (descent/trim armed).

    A verbatim user ladder is never trimmed — an OOM on it fails loudly,
    exactly like the parent tag."""

    return (
        coverage_dense_armed()
        and _c3_ladder.derived
        and _c3_ladder.ladder_source == LADDER_SOURCE_SELF_SIZED
    )
def note_capture_trim(*, bucket: int, stop_reason: str, realized_max: int) -> None:
    """Record one descent/reserve trim; update the ladder stop reason when the
    trim bounds the realized maximum."""

    if stop_reason not in (
        LADDER_STOP_MEMORY_RESERVE,
        LADDER_STOP_CAPTURE_OOM,
        LADDER_STOP_CAPTURE_ERROR,
    ):
        raise ValueError(f"unknown capture trim stop_reason: {stop_reason!r}")
    _c3_ladder.capture_trim_events.append(
        {"bucket": int(bucket), "stop_reason": stop_reason}
    )
    # Keep the stored per-boot ladder in lockstep with the runner's trim so a
    # later capture cycle reuses only buckets that actually realized.
    _c3_ladder.realized_capture_bs = [
        value for value in _c3_ladder.realized_capture_bs if value != int(bucket)
    ]
    if int(bucket) > int(realized_max):
        _c3_ladder.stop_reason = stop_reason
def trim_candidates(candidates: Sequence[int], bucket: int) -> list[int]:
    """Pure bounded-descent step: drop ``bucket``; each step strictly shrinks
    the candidate set (termination), and an empty result fails loudly."""

    remaining = [value for value in candidates if value != bucket]
    if len(remaining) == len(candidates):
        raise ValueError(
            f"capture descent asked to trim {bucket}, which is not a candidate"
        )
    if not remaining:
        raise RuntimeError(
            "capture descent exhausted every decode graph bucket; refusing to "
            "serve with no captured decode coverage"
        )
    return remaining
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
    forward_batch.vp_fd_coverage_counted = False
    forward_batch.fd_full_graph_force_production_attention = False
_c3_ladder = CoverageLadderRecord.create()
