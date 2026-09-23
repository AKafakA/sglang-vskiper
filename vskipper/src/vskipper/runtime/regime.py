"""Occupancy regime switch.

A single fail-closed knob with two legs: a prefill token threshold and a decode
row hysteresis. It exists because the routed decode body is only efficient while
CUDA-graph-captured -- above the captured ceiling it runs eager and
un-amortised, which is the occupancy runaway. The switch keeps execution inside
the regime the capture ladder actually covers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
from vskipper.runtime.common import (
    ADMISSION_CRITERION_MONOTONE_DECODE,
    ADMISSION_CRITERION_PHASE_STICKY,
    ADMISSION_DECODE_AFTER_FD_PREFILL_FD,
    ADMISSION_DECODE_AFTER_STOCK_PREFILL_STOCK,
    RegimeSwitchConfig,
    ADMISSION_DECODE_DOWNGRADE_BAND_LOW,
    ADMISSION_DECODE_DOWNGRADE_BAND_LOW_ANY,
    ADMISSION_DECODE_UPGRADE_BAND_HIGH,
    ADMISSION_PREFILL_DEMOTION_ADMISSION,
    ADMISSION_PREFILL_TOKENS_UNCACHED,
)
from vskipper.runtime.common import (
    DECODE_BODY_HIGH,
    DECODE_BODY_LOW,
    REQUEST_BODY_FD,
    REQUEST_BODY_STOCK,
    _DECODE_BODIES,
    _REQUEST_BODIES,
)


PREFILL_BODY_DENSE = "dense"
PREFILL_BODY_FD = "fd"
class DecodeHysteresis:
    """Stateful enter/exit band for the decode regime decision.

    ``update(rows)`` enters the high (skip) regime at ``rows >= enter_rows``,
    exits to the low (production-all-RUN) regime at ``rows <= exit_rows``, and
    holds the current regime inside the band so live-``bs`` oscillation cannot
    flap the body.  The initial regime (below the band) is the low body.

    When the decode leg is disabled the switch is off for decode: every call
    returns :data:`DECODE_BODY_HIGH` so the unchanged M3.38 skip body governs.
    """

    def __init__(self, cfg: RegimeSwitchConfig) -> None:
        self._cfg = cfg
        self._state = DECODE_BODY_LOW
        self._exit_streak = 0

    @property
    def state(self) -> str:
        return self._state

    def update(self, rows: int, kv_tokens: Optional[int] = None) -> str:
        if not self._cfg.decode.enabled:
            return DECODE_BODY_HIGH
        decode = self._cfg.decode
        if decode.kv_criterion:
            # Lane-2 cut1: key the band on resident KV tokens (rows x context).
            if kv_tokens is None:
                raise ValueError(
                    "regime switch decode kv criterion needs the batch's "
                    "seq_lens_sum; the caller passed none"
                )
            metric, enter, exit_ = int(kv_tokens), decode.enter_kv_tokens, decode.exit_kv_tokens
        else:
            metric, enter, exit_ = int(rows), decode.enter_rows, decode.exit_rows
        if metric >= enter:
            self._state = DECODE_BODY_HIGH
            self._exit_streak = 0
        elif metric <= exit_:
            # c2: only a SUSTAINED run of sub-exit passes flips to dense; a
            # transient dip (interrupted by any pass above exit_rows) does not.
            self._exit_streak += 1
            if self._exit_streak >= self._cfg.decode.exit_dwell:
                self._state = DECODE_BODY_LOW
        else:
            self._exit_streak = 0
        return self._state
def compose_regime_variant_label(
    lora_variant: Optional[str], regime_body: Optional[str]
) -> Optional[str]:
    """Compose the decode ``ShapeKey.variant_label`` from LoRA + regime body.

    The regime body is a ``"|"``-joined suffix on the existing LoRA variant, so
    no ``ShapeKey`` schema change is needed and capture/replay key on
    ``(bucket, regime)``.  A ``None`` regime body (the decode leg off) yields
    exactly the LoRA variant — byte-identical keying.  FlexiDepth conditional
    graphs disallow LoRA, so in practice the label is just the regime body.
    """

    if regime_body is None:
        return lora_variant
    if lora_variant is None:
        return regime_body
    return f"{lora_variant}|{regime_body}"
def decode_regime_from_variant_label(label: Optional[str]) -> Optional[str]:
    """Extract the decode regime body from a composed ``variant_label``.

    Inverse of :func:`compose_regime_variant_label` for the regime component:
    returns the body when the (last ``"|"``-separated) component is a known
    decode body, else ``None`` (switch off, or a pure LoRA label).
    """

    if label is None:
        return None
    body = label.rsplit("|", 1)[-1]
    return body if body in _DECODE_BODIES else None
def regime_body_dispatches_stock_decode(regime_body: Optional[str]) -> bool:
    """Whether a decode regime body dispatches the STOCK production decode graph.

    W1 I6b (fork option (b)): the ``prod_allrun`` low band is served by the
    stock production decode path — the plain base-Llama forward with production
    (flashinfer) decode attention and NO FlexiDepth hooks, captured as its own
    stock decode CUDA graph — because production-all-RUN *through* the FD
    conditional graph is neither byte-identical (0/48) nor same-speed (+4%) vs
    stock (GPU parity, 2026-07-23).  The ``skip`` high band keeps the M3.38
    FlexiDepth conditional/whole-model body.  ``None`` (switch off) keeps the
    unchanged single-body FlexiDepth dispatch.  The decode graph backend keys on
    this exact predicate, so the low-vs-stock dispatch decision is CPU-testable.
    """

    return regime_body == DECODE_BODY_LOW
class DecodeRegimeDispatch:
    """Stateful decode-leg regime dispatcher owned by the decode graph runner.

    Encapsulates the hysteresis, the per-body pass counters, and the
    capture-body enumeration so the whole decode dispatch decision is
    CPU-testable without the torch-heavy runner.  Inactive (config ``None`` or
    the decode leg disabled) means no regime: capture bodies ``[None]``, the
    observed body is ``None``, counters are ``None`` — byte-identical dispatch.
    """

    def __init__(self, cfg: Optional[RegimeSwitchConfig]) -> None:
        self._active = cfg is not None and cfg.decode.enabled
        self._hysteresis = DecodeHysteresis(cfg) if self._active else None
        self._counts = {DECODE_BODY_LOW: 0, DECODE_BODY_HIGH: 0}
        self._current_body: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def current_body(self) -> Optional[str]:
        return self._current_body

    def capture_bodies(self) -> list[Optional[str]]:
        """Regime bodies to capture per bucket (Strategy B captures both)."""

        if not self._active:
            return [None]
        return [DECODE_BODY_LOW, DECODE_BODY_HIGH]

    def observe(
        self, rows: int, kv_tokens: Optional[int] = None
    ) -> Optional[str]:
        """Advance the hysteresis from raw pre-pad rows (and, under the cut1
        kv criterion, the batch's resident KV tokens); count and hold the body."""

        if not self._active:
            self._current_body = None
            return None
        body = self._hysteresis.update(int(rows), kv_tokens)
        self._counts[body] += 1
        self._current_body = body
        return body

    def counters(self) -> Optional[dict[str, int]]:
        """Per-body decode pass counts, or ``None`` when the decode leg is off."""

        if not self._active:
            return None
        return dict(self._counts)

    def pin(self, body: str) -> Optional[str]:
        """[D-849] Record the body a PINNED decode pass executes, without
        advancing the hysteresis (under pinning the band state lives in the
        scheduler's mirror; the runner only sees uniform sub-batches, whose
        rows/KV would mis-drive a band). Counts the pass so
        ``regime_switch.counters.decode`` keeps meaning "decode passes by
        executed body", which is what verify_skipping_executed consumes."""

        if not self._active:
            self._current_body = None
            return None
        if body not in _DECODE_BODIES:
            raise ValueError(f"unknown decode body {body!r}")
        self._counts[body] += 1
        self._current_body = body
        return body

    def observe_or_pin(
        self,
        rows: int,
        kv_tokens: Optional[int],
        pinned_decode_body: Optional[str],
    ) -> Optional[str]:
        """``pin`` when the pass carries a pin, else the version-1 ``observe``."""

        if pinned_decode_body is not None:
            return self.pin(pinned_decode_body)
        return self.observe(rows, kv_tokens)
def prefill_regime_decision(
    extend_num_tokens: int,
    batch_size: int,
    is_mixed: bool,
    running_bs: int,
    cfg: RegimeSwitchConfig,
) -> str:
    """Choose the prefill body for one scheduled pass.

    Returns :data:`PREFILL_BODY_FD` (grouped FlexiDepth) at or above the
    row-corrected token threshold, else :data:`PREFILL_BODY_DENSE` (production
    dense).  On a MIXED pass ``extend_num_tokens`` is inflated by the running
    decode rows, so the true prompt-token count subtracts ``running_bs``.

    When the prefill leg is disabled the switch is off for prefill: always
    return :data:`PREFILL_BODY_FD` so the unchanged FD phase gate governs.
    """

    if not cfg.prefill.enabled:
        return PREFILL_BODY_FD
    effective_tokens = (
        extend_num_tokens - running_bs if is_mixed else extend_num_tokens
    )
    threshold = cfg.prefill.min_tokens - cfg.prefill.row_correction_alpha * (
        batch_size
    )
    # [D-596] The prefill band's UPPER token gate is GONE. It demoted every pass
    # above 6144 tokens to the dense body on the premise (2026-08-20) that the
    # routed body loses on large packed passes. Three things falsified that:
    #   * the engagement escape (D-339) was added afterwards to make exactly this
    #     decision from measured routing share, and token brackets applied FIRST
    #     as hard bounds, pre-empting it;
    #   * the routed body was rebuilt (count-GEMM, on-device compaction, D-574
    #     layer-policy deletion) -- the "thin projector lanes" it was calibrated
    #     against are not what runs;
    #   * the per-token profile INVERTS its premise: cost falls with pass size,
    #     98 us/tok below 1536 down to 72.7 at 6-8k, against dense ~78-85.
    # Measured directly (D-579, CSD3 duo, single variable, GR-1a PASS on all four
    # cells): removing it converts +9.5 % of gsm8k overload passes from dense to
    # routed and buys TTFT -6.1 % / E2E -2.8 % there, while being PROVABLY INERT
    # on coqa -- both arms recorded byte-identical prefill counters (1413 dense /
    # 8 routed) because coqa's passes never reach the bound.
    # It was never a capture constraint either: prefill_cuda_graph_runner.py:873
    # captures BOTH bodies at every shape, so passes just above the bound had a
    # routed graph ready and were sent to dense regardless.
    if effective_tokens >= threshold:
        return PREFILL_BODY_FD
    return PREFILL_BODY_DENSE
class PrefillEngagementTracker:
    """[R2, lane-2 tax-removal track, 2026-09-07] The prefill escape gate's
    engagement EMA, fed from the binary-cohort device counters WITHOUT a
    host stall.

    Today's read (``stats.cpu().tolist()`` at the start of the pass after a
    routed pass) waits for the previous pass's kernels — under the overlap
    scheduler that is a full-pass stall on the forward thread (served coqa
    profile: 13 x ~182 ms in one 80-step window). This tracker takes the same
    readings at the same stream positions as non-blocking copies into pinned
    slots with a CUDA event each, folds them into the EMA when their events
    have completed, and only synchronises when the pending samples could
    change the ONE predicate the EMA feeds (``ema < engagement_min``): a
    sample s in [0, 1] moves the EMA monotonically (``0.8*ema + 0.2*s``), so
    with k unfolded samples the EMA lies inside [lo_k, hi_k] computed with
    the same float ops; if the whole interval sits on one side of the floor
    the decision is already known. Every decision, every served body and
    every counter is therefore identical to the synchronous reader; the
    stall survives only on passes where the reader would have needed the
    value (the descent to demotion, the boot pass, at most one per probe).

    CPU stats tensors (unit tests, non-CUDA devices) are read synchronously —
    byte-identical to the old path.
    """

    DECAY = 0.8

    def __init__(self) -> None:
        self.ema: Optional[float] = None
        self.samples = 0
        self.syncs = 0
        self._baseline: Optional[tuple] = None
        self._pending: list = []  # [(per_device_slots, event)] in stream order
        self._slots: dict = {}  # device -> [pinned int64[3], pinned int64[3]]
        self._slot_next: dict = {}

    # -- stream-side -------------------------------------------------------
    def snapshot(self, stats_by_device) -> None:
        """Record the counters as they stand at this point of the stream."""
        import torch

        # One reading = one (slot, event) pair PER DEVICE: CUDA events are
        # device-bound, so each device's copy gets its own event recorded on
        # that device's current stream (Codex review 2026-09-07, MAJOR).
        items = []
        for device, stats in list(stats_by_device.items()):
            if not stats.is_cuda:
                items.append((None, stats.detach().to("cpu").clone(), None))
                continue
            ring = self._slots.get(device)
            if ring is None:
                ring = [
                    torch.empty(3, dtype=torch.int64, pin_memory=True)
                    for _ in range(2)
                ]
                self._slots[device] = ring
                self._slot_next[device] = 0
            idx = self._slot_next[device]
            self._slot_next[device] = (idx + 1) % len(ring)
            slot = ring[idx]
            if any(slot is s for pend in self._pending for s, _, _ in pend):
                # The ring wrapped onto an unfolded reading: fold everything
                # first (never happens with one pass in flight; correctness only).
                self.sync_all()
            slot.copy_(stats.detach(), non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            items.append((slot, None, event))
        self._pending.append(items)
        if all(event is None for _, _, event in items):
            self.fold_ready()

    # -- host-side ---------------------------------------------------------
    @property
    def pending(self) -> int:
        return len(self._pending)

    @property
    def baseline_missing(self) -> bool:
        return self._baseline is None and not self._pending

    def _totals(self, items) -> tuple:
        run = 0
        project = 0
        for slot, cpu_values, _ in items:
            values = (cpu_values if slot is None else slot).tolist()
            run += int(values[1])
            project += int(values[2])
        return (run, project)

    def _fold(self, totals: tuple) -> None:
        base = self._baseline
        if base is not None:
            d_run = totals[0] - base[0]
            d_project = totals[1] - base[1]
            d_total = d_run + d_project
            if d_total > 0:
                sample = d_project / d_total
                ema = self.ema
                # Literal coefficients, exactly as the synchronous reader wrote
                # them: `(1.0 - 0.8)` is not `0.2` in float and shifted the
                # EMA by one ulp on the box (pytest 2026-09-07 20:41Z).
                self.ema = sample if ema is None else 0.8 * ema + 0.2 * sample
                self.samples += 1
        self._baseline = totals

    def fold_ready(self) -> None:
        """Fold every reading whose copy has completed, in stream order."""
        while self._pending:
            items = self._pending[0]
            if any(event is not None and not event.query() for _, _, event in items):
                return
            self._pending.pop(0)
            self._fold(self._totals(items))

    def sync_all(self) -> None:
        if not self._pending:
            return
        self.syncs += 1
        while self._pending:
            items = self._pending.pop(0)
            for _, _, event in items:
                if event is not None:
                    event.synchronize()
            self._fold(self._totals(items))

    def demote(self, engagement_min: Optional[float]) -> bool:
        """Exactly ``ema is not None and ema < engagement_min`` as the
        synchronous reader would compute it, synchronising only when the
        pending readings could change the answer."""
        if engagement_min is None:
            return False
        self.fold_ready()
        if self._pending:
            if self.ema is None:
                self.sync_all()
            else:
                lo = self.ema
                hi = self.ema
                for _ in range(len(self._pending)):
                    lo = 0.8 * lo + 0.2 * 0.0
                    hi = 0.8 * hi + 0.2 * 1.0
                if hi < engagement_min:
                    return True
                if lo >= engagement_min:
                    return False
                self.sync_all()
        return self.ema is not None and self.ema < engagement_min

    def realized(self) -> dict:
        self.sync_all()
        return {"ema": self.ema, "samples": int(self.samples), "syncs": int(self.syncs)}


def regime_switch_zero_counters() -> dict[str, dict[str, int]]:
    """Per-phase-by-body pass counters, zeroed.

    The scaffold (I1-I4) always emits zeros; the forward-path integration
    (I5/I6) replaces these with the real device-side pass counts.  Counters are
    non-deterministic runtime evidence and are stripped from the deployment
    identity SHA, so an on-state expectation must never assert them.
    """

    return {
        "prefill": {PREFILL_BODY_DENSE: 0, PREFILL_BODY_FD: 0},
        "decode": {DECODE_BODY_LOW: 0, DECODE_BODY_HIGH: 0},
        "admission": admission_split_zero_counters(),
    }


# ---------------------------------------------------------------------------
# [D-849, 2026-09-21] Per-request body pinning.
# ---------------------------------------------------------------------------


def admission_split_zero_counters() -> dict[str, int]:
    """Model-runner-side evidence of pinned decode dispatch, zeroed.

    ``split_passes``: decode steps whose running batch carried both pins and
    were partitioned; ``split_rows_stock/fd``: rows dispatched through each
    sub-pass of a partitioned step; ``coverage_dense_violation_rows``: rows of
    an fd-pinned (sub-)pass that the (c3) coverage stamp forced dense -- a pin
    violation, MUST be 0 in a served cell; ``stock_eager_passes``: stock-pinned
    (sub-)passes served by the eager dense fall-through (uncovered rows);
    ``ladder_chunked_passes``: decode steps whose pinned batch exceeded the
    captured ladder and was cut into <= ladder-top chunks (each a captured
    replay); ``ladder_chunk_replays``: the chunk replays those steps issued;
    ``forced_run_passes`` / ``forced_run_rows``: mixed (sub-)passes served by
    ONE routed replay and the stock-pinned rows forced to RUN inside them.
    """

    return {
        "split_passes": 0,
        "split_rows_stock": 0,
        "split_rows_fd": 0,
        "coverage_dense_violation_rows": 0,
        "stock_eager_passes": 0,
        "ladder_chunked_passes": 0,
        "ladder_chunk_replays": 0,
        "forced_run_passes": 0,
        "forced_run_rows": 0,
    }


def request_body_decode_body(pin: str) -> str:
    """The decode body a pinned request executes (stock -> prod_allrun, fd -> skip)."""

    if pin == REQUEST_BODY_STOCK:
        return DECODE_BODY_LOW
    if pin == REQUEST_BODY_FD:
        return DECODE_BODY_HIGH
    raise ValueError(f"unknown request body pin {pin!r}")


def request_body_prefill_body(pin: str) -> str:
    """The prefill body a pinned request executes (stock -> dense, fd -> fd)."""

    if pin == REQUEST_BODY_STOCK:
        return PREFILL_BODY_DENSE
    if pin == REQUEST_BODY_FD:
        return PREFILL_BODY_FD
    raise ValueError(f"unknown request body pin {pin!r}")


def pin_from_band_state(state: str) -> str:
    """Map the decode band state to the pin new admissions receive."""

    if state == DECODE_BODY_HIGH:
        return REQUEST_BODY_FD
    if state == DECODE_BODY_LOW:
        return REQUEST_BODY_STOCK
    raise ValueError(f"unknown decode band state {state!r}")


def upgrade_pinned_rows(pinner: "AdmissionPinner", reqs) -> int:
    """[D-849 add. 12] Apply the one-way stock->fd decode upgrade to every
    decode-pinned stock row of the running batch that has not switched yet,
    BEFORE the step's forward batch is built (ForwardBatch.init_new reads
    ``req.vp_body``). A request whose generated K/V now spans two bodies is
    never inserted into the radix cache at finish. Returns the number of rows
    upgraded this step."""

    if not pinner.upgrades_enabled or pinner.state != DECODE_BODY_HIGH:
        return 0
    upgraded = 0
    for req in reqs:
        if req.vp_decode_pinned and req.vp_body == REQUEST_BODY_STOCK and not req.vp_decode_switched:
            req.vp_body = req.vp_decode_body = pinner.upgrade_pin(REQUEST_BODY_STOCK)
            req.vp_decode_upgraded = True
            req.vp_decode_switched = True
            req.vp_skip_finish_insert = True
            upgraded += 1
    return upgraded


def downgrade_pinned_rows(pinner: "AdmissionPinner", reqs) -> int:
    """[D-849 add. 35] The one-switch rule's other direction: every decode-pinned
    fd row that has not switched yet drops to the dense body once the band is
    LOW (the dense body is the faster one below the crossover, and it reads
    routed K/V without harm -- the FD->stock leg). Never back. Returns the
    number of rows downgraded this step."""

    if not pinner.downgrades_enabled or pinner.state != DECODE_BODY_LOW:
        return 0
    downgraded = 0
    after_promotion = pinner.downgrades_after_promotion
    for req in reqs:
        if not (req.vp_decode_pinned and req.vp_body == REQUEST_BODY_FD) or req.vp_decode_demoted:
            continue
        if req.vp_decode_switched and not after_promotion:
            continue  # one-switch rule: a promoted row stays routed
        req.vp_body = req.vp_decode_body = pinner.downgrade_pin(REQUEST_BODY_FD)
        req.vp_decode_switched = True
        req.vp_decode_demoted = True
        req.vp_skip_finish_insert = True
        downgraded += 1
    return downgraded


def monotone_assign_rows(pinner: "AdmissionPinner", reqs) -> int:
    """[D-849 add. 23] Monotone lane, once per decode step BEFORE the forward batch
    is built: band HIGH -> every row runs routed and becomes promoted; band LOW ->
    promoted rows keep the routed body (never demoted) and the others are forced
    to RUN inside the same routed replay (a stock-only batch runs the stock graph,
    exactly version 1). Returns the number of rows promoted this step."""

    if not pinner.monotone:
        return 0
    high = pinner.state == DECODE_BODY_HIGH
    promoted = 0
    for req in reqs:
        if high and not req.vp_promoted:
            req.vp_promoted = True
            promoted += 1
        req.vp_body = REQUEST_BODY_FD if (high or req.vp_promoted) else REQUEST_BODY_STOCK
    if promoted:
        pinner._promoted += promoted
    return promoted


class AdmissionPinner:
    """Scheduler-owned mirror of the decode band; decides each request's pin.

    Fed ONCE per decode step from ``Scheduler.run_batch`` with the same two
    numbers the runner's ``observe`` receives (rows, resident KV tokens), so the
    band state it holds is exactly the version-1 state, but owned by the
    scheduler thread: under the overlap scheduler the runner's dispatch runs on
    the forward thread, and reading its state at admission would be racy and
    could diverge across TP ranks. Inactive (config ``None`` or pinning off)
    means ``pin_for_admission()`` returns ``None`` and nothing is pinned --
    byte-identical version-1 behaviour.

    Cold start (no decode step observed yet) is the band's initial low state,
    so the first admissions pin ``stock`` (``admission.cold_start``).
    """

    def __init__(self, cfg: Optional[RegimeSwitchConfig]) -> None:
        self._active = cfg is not None and cfg.pinning
        self._cfg = cfg
        self._hysteresis = DecodeHysteresis(cfg) if self._active else None
        self._steps = 0
        self._flips = 0
        self._admitted = {REQUEST_BODY_STOCK: 0, REQUEST_BODY_FD: 0}
        # [phase-sticky] per-request plans decided at the prefill->decode boundary
        self._plans = {"stock_stock": 0, "stock_fd": 0, "fd_fd": 0, "fd_stock": 0}
        self._upgrades = 0  # [D-849 add. 12] stock->fd decode upgrades at band HIGH
        self._downgrades = 0  # [D-849 add. 35] fd->stock decode downgrades at band LOW (one switch per request)
        self._promoted = 0  # [D-849 add. 23] monotone lane: rows promoted by their first routed step
        # [D-849 add. 13] version-1 engagement demotion at admission: the verdict comes
        # from the runner's PrefillEngagementTracker (set_engagement_demote); the dense
        # streak / probe window is the version-1 one, kept here per admission round.
        self._engagement_demote = None
        self._dense_streak = 0
        self._prefill_demoted_rounds = 0
        self._prefill_probe_rounds = 0

    @property
    def monotone(self) -> bool:
        """[D-849 add. 23] the monotone lane: no admission pins, one namespace,
        version-1 prefill and band; decode rows promoted once, never demoted."""
        return self._active and self._cfg.admission.criterion == ADMISSION_CRITERION_MONOTONE_DECODE

    @property
    def phase_sticky(self) -> bool:
        return self._active and self._cfg.admission.criterion == ADMISSION_CRITERION_PHASE_STICKY

    def set_engagement_demote(self, verdict) -> None:
        """[D-849 add. 13] ``verdict()`` -> True when the version-1 engagement
        escape would demote the next routed pass (the runner tracker's
        ``demote(engagement_min)``)."""

        self._engagement_demote = verdict

    def prefill_pin(self, round_prompt_tokens: int, round_requests: int = 1) -> Optional[str]:
        """[phase-sticky] The PREFILL body of an admission round = the version-1
        pass decision made at admission: the token bracket
        (``prefill_regime_decision`` on the round's prompt tokens, before any
        cache hit -- the key that finds them depends on this) and, under
        ``prefill_demotion: "admission"``, the engagement demotion with its
        one-probe-per-window streak. With the prefill leg off, or under the
        band-state criterion, the band state decides (the version-2 pin)."""

        if not self._active or self.monotone:
            return None  # monotone: the version-1 pass criterion decides at dispatch
        if not self.phase_sticky or not self._cfg.prefill.enabled:
            return self.current_pin()
        variant = prefill_regime_decision(
            int(round_prompt_tokens), int(round_requests), False, 0, self._cfg
        )
        if (
            variant == PREFILL_BODY_FD
            and self._cfg.admission.prefill_demotion == ADMISSION_PREFILL_DEMOTION_ADMISSION
            and self._cfg.prefill.engagement_min is not None
            and self._engagement_demote is not None
            and self._engagement_demote()
        ):
            if self._dense_streak < self._cfg.prefill.engagement_probe_every:
                variant = PREFILL_BODY_DENSE
                self._dense_streak += 1
                self._prefill_demoted_rounds += 1
            else:
                self._dense_streak = 0
                self._prefill_probe_rounds += 1
        elif variant == PREFILL_BODY_FD:
            self._dense_streak = 0
        return REQUEST_BODY_FD if variant == PREFILL_BODY_FD else REQUEST_BODY_STOCK

    def decode_pin_at_boundary(self, prefill_pin: str) -> str:
        """[phase-sticky] The DECODE body of a request leaving prefill: the band
        state now (``decode_after_fd_prefill: "band"``), or FD after an FD
        prefill (``"fd"``). A stock prefill decodes STOCK under
        ``decode_after_stock_prefill: "stock"`` (the served design, add. 28:
        routed generation over a dense-computed prompt is the checkpoint's loop
        mode) and follows the band under the legacy ``"band"``."""

        if prefill_pin not in _REQUEST_BODIES:
            raise ValueError(f"unknown request body pin {prefill_pin!r}")
        if not self.phase_sticky:
            return prefill_pin
        if (
            prefill_pin == REQUEST_BODY_STOCK
            and self._cfg.admission.decode_after_stock_prefill == ADMISSION_DECODE_AFTER_STOCK_PREFILL_STOCK
        ):
            return REQUEST_BODY_STOCK
        if (
            prefill_pin == REQUEST_BODY_FD
            and self._cfg.admission.decode_after_fd_prefill == ADMISSION_DECODE_AFTER_FD_PREFILL_FD
        ):
            return REQUEST_BODY_FD
        return pin_from_band_state(self._hysteresis.state)

    @property
    def prefill_tokens_uncached(self) -> bool:
        """[D-849 add. 17] the round's prefill criterion counts uncached tokens."""
        return (
            self._active
            and self.phase_sticky
            and self._cfg.admission.prefill_tokens == ADMISSION_PREFILL_TOKENS_UNCACHED
        )

    @property
    def prefill_min_tokens(self) -> int:
        """[D-849 add. 22] the prefill leg's token threshold (the round walk stops there)."""
        return int(self._cfg.prefill.min_tokens) if self._active else 0

    @property
    def upgrades_enabled(self) -> bool:
        """[D-849 add. 12] one-way stock->fd decode upgrade at band HIGH."""
        return (
            self._active
            and self.phase_sticky
            and self._cfg.admission.decode_upgrade == ADMISSION_DECODE_UPGRADE_BAND_HIGH
        )

    @property
    def downgrades_enabled(self) -> bool:
        """[D-849 add. 35/40] one-time fd->stock decode downgrade at band LOW."""
        return (
            self._active
            and self.phase_sticky
            and self._cfg.admission.decode_downgrade in (ADMISSION_DECODE_DOWNGRADE_BAND_LOW, ADMISSION_DECODE_DOWNGRADE_BAND_LOW_ANY)
        )

    @property
    def downgrades_after_promotion(self) -> bool:
        """[D-849 add. 40] a promoted row may be demoted once at band LOW (never re-promoted)."""
        return self.downgrades_enabled and self._cfg.admission.decode_downgrade == ADMISSION_DECODE_DOWNGRADE_BAND_LOW_ANY

    def downgrade_pin(self, decode_pin: str) -> str:
        """[D-849 add. 35] The decode body an fd-pinned request runs from the
        next step on: ``stock`` once the band is LOW (below the crossover the
        dense body is the faster one), else unchanged. Counts each downgrade."""

        if decode_pin not in _REQUEST_BODIES:
            raise ValueError(f"unknown request body pin {decode_pin!r}")
        if (
            decode_pin == REQUEST_BODY_FD
            and self.downgrades_enabled
            and self._hysteresis.state == DECODE_BODY_LOW
        ):
            self._downgrades += 1
            return REQUEST_BODY_STOCK
        return decode_pin

    def upgrade_pin(self, decode_pin: str) -> str:
        """[D-849 add. 12] The decode body a stock-pinned request runs from the
        next step on: ``fd`` once the band is HIGH (routed decode pays for the
        whole running batch, the per-step band says), else unchanged. Never
        the other way: a routed decode stays routed. Counts each upgrade."""

        if decode_pin not in _REQUEST_BODIES:
            raise ValueError(f"unknown request body pin {decode_pin!r}")
        if (
            decode_pin == REQUEST_BODY_STOCK
            and self.upgrades_enabled
            and self._hysteresis.state == DECODE_BODY_HIGH
        ):
            self._upgrades += 1
            return REQUEST_BODY_FD
        return decode_pin

    def record_plan(self, prefill_pin: str, decode_pin: str) -> None:
        if not self._active:
            return
        key = f"{prefill_pin}_{decode_pin}"
        if key not in self._plans:
            raise ValueError(f"[D-849] plan {key!r} is not a served plan")
        self._plans[key] += 1

    @property
    def active(self) -> bool:
        return self._active

    @property
    def state(self) -> Optional[str]:
        return self._hysteresis.state if self._active else None

    def observe_decode_step(self, rows: int, kv_tokens: Optional[int]) -> None:
        """Advance the mirror by one decode step (the whole running batch)."""

        if not self._active:
            return
        before = self._hysteresis.state
        after = self._hysteresis.update(int(rows), kv_tokens)
        self._steps += 1
        if after != before:
            self._flips += 1

    def observe_idle(self) -> None:
        """[D-849] The mirror is advanced by decode steps only, so after a drain
        to an empty running batch it keeps the last busy state; a request
        admitted onto an idle engine would then inherit it (2026-09-21 gate: the
        first request after a drained tiny-band cell was pinned to the routed
        body). Admission onto an empty batch observes the empty batch first:
        zero rows, zero resident K/V -- the level the band actually has."""

        if not self._active:
            return
        before = self._hysteresis.state
        after = self._hysteresis.update(0, 0)
        if after != before:
            self._flips += 1

    def current_pin(self) -> Optional[str]:
        """The pin a request admitted NOW would receive (``None`` when
        inactive); does not count anything."""

        if not self._active:
            return None
        return pin_from_band_state(self._hysteresis.state)

    def record_admission(self, pin: str) -> None:
        """Count one request pinned to ``pin``."""

        if not self._active:
            return
        if pin not in _REQUEST_BODIES:
            raise ValueError(f"unknown request body pin {pin!r}")
        self._admitted[pin] += 1

    def pin_for_admission(self) -> Optional[str]:
        """The pin for a request admitted NOW, counted (``None`` when inactive)."""

        pin = self.current_pin()
        if pin is not None:
            self.record_admission(pin)
        return pin

    def counters(self) -> Optional[dict[str, int]]:
        if not self._active:
            return None
        return {
            "admitted_stock": self._admitted[REQUEST_BODY_STOCK],
            "admitted_fd": self._admitted[REQUEST_BODY_FD],
            "steps_observed": self._steps,
            "band_flips": self._flips,
            "plan_stock_stock": self._plans["stock_stock"],
            "plan_stock_fd": self._plans["stock_fd"],
            "plan_fd_fd": self._plans["fd_fd"],
            "plan_fd_stock": self._plans["fd_stock"],
            "decode_upgrades_stock_fd": self._upgrades,
            "decode_downgrades_fd_stock": self._downgrades,
            "monotone_promoted": self._promoted,
            "prefill_demoted_rounds": self._prefill_demoted_rounds,
            "prefill_probe_rounds": self._prefill_probe_rounds,
        }


def prefill_variant_for_pass(
    pin: Optional[str],
    extend_num_tokens: int,
    batch_size: int,
    is_mixed: bool,
    running_bs: int,
    cfg: RegimeSwitchConfig,
    tracker: Optional["PrefillEngagementTracker"],
    dense_streak: int,
) -> tuple[str, int]:
    """The prefill body for one pass and the updated engagement dense streak.

    A PINNED pass (``pin`` not None) runs its pin's body unconditionally: the
    token bracket and the engagement demotion are bypassed (they would hand a
    routed-pinned request a dense prefill, i.e. a mixed-body request again);
    the streak is untouched. An unpinned pass reproduces the version-1
    decision verbatim: the bracket (``prefill_regime_decision``, or FD on a
    mixed pass with mixed switching disabled), then the engagement demotion
    with one routed probe pass per ``engagement_probe_every`` dense passes.
    """

    if pin is not None:
        return request_body_prefill_body(pin), dense_streak
    if is_mixed and not cfg.prefill.include_mixed:
        variant = PREFILL_BODY_FD
    else:
        variant = prefill_regime_decision(
            extend_num_tokens, batch_size, is_mixed, running_bs, cfg
        )
    if (
        variant == PREFILL_BODY_FD
        and cfg.prefill.engagement_min is not None
        and tracker is not None
        and tracker.demote(cfg.prefill.engagement_min)
    ):
        if dense_streak < cfg.prefill.engagement_probe_every:
            variant = PREFILL_BODY_DENSE
            dense_streak += 1
        else:
            dense_streak = 0
    elif variant == PREFILL_BODY_FD:
        dense_streak = 0
    return variant, dense_streak


def batch_pin_of(pins: list[Optional[str]]) -> Optional[str]:
    """The uniform pin of a batch: ``None`` when no row is pinned, the pin when
    every pinned row agrees, ``"mixed"`` when both pins are present."""

    seen = {p for p in pins if p is not None}
    if not seen:
        return None
    if len(seen) == 1:
        (pin,) = seen
        if pin not in _REQUEST_BODIES:
            raise ValueError(f"unknown request body pin {pin!r}")
        return pin
    return "mixed"


def pinned_decode_body_of(forward_batch: Any) -> Optional[str]:
    """The decode body a pinned (uniform) forward batch must execute, or
    ``None`` for an unpinned batch. A "mixed" batch reaching a dispatch site is
    a defect (the model runner partitions it first) and fails closed."""

    pin = forward_batch.vp_body
    if pin is None:
        return None
    if pin == "mixed":
        # [D-849 forced-RUN] a mixed step executes ONE routed replay: the
        # stock-pinned rows are forced to RUN inside it.
        return DECODE_BODY_HIGH
    return request_body_decode_body(pin)
