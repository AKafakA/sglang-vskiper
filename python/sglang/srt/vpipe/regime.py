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
from sglang.srt.vpipe.common import (
    RegimeSwitchConfig,
)
from sglang.srt.vpipe.common import (
    DECODE_BODY_HIGH,
    DECODE_BODY_LOW,
    _DECODE_BODIES,
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
    if (
        cfg.prefill.max_tokens is not None
        and effective_tokens > cfg.prefill.max_tokens
    ):
        # Large packed passes: the routed body's cohort costs exceed the
        # savings (measured regime boundary — see max_tokens field note).
        return PREFILL_BODY_DENSE
    if effective_tokens >= threshold:
        return PREFILL_BODY_FD
    return PREFILL_BODY_DENSE
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
    }
