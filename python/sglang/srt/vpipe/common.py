"""Shared leaf helpers.

Symbols needed by more than one module in the package. They live here so the
import graph stays acyclic -- every module may import `common`, and `common`
imports nothing from the package.
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    replace,
    asdict,
    field,
)
from typing import (
    Iterable,
    Sequence,
    Any,
    Mapping,
    Optional,
)
from pathlib import Path
from math import ceil
import hashlib
import json
import math
import msgspec
import signal
import json as _json
import os

from sglang.srt.vpipe.design import (  # the design is code
    SERVED_EXECUTION_MODE,
    SERVED_FUSED_PROJECT_INPUT,
    SERVED_FUSED_ROUTER_NORM,
    SERVED_GATE_MODE,
    SERVED_REGIME_SWITCH,
    active_arm,
    active_arm_name,
    arm_compact_enabled,
    arm_gate_mode,
    arm_phases,
    mechanism,
    skipper_deployed,
)
import torch
import torch.nn.functional as F
from sglang.srt.vpipe.env import (
    FD_COMPACT_ROUTED_QKV_ENV,
    FD_CONTIGUOUS_ROUTED_QKV_ENV,
    FD_ROUTED_QKV_CAPACITIES_ENV,
    FD_ROUTED_QKV_CAPACITY_MULTIPLE_ENV,
    FD_ROUTED_QKV_MIN_ROWS_ENV,
    FD_COMPACT_PHASES_ENV,
    FD_PREFILL_CUBLAS_ENV,
    FD_PREFILL_FALLBACK_ENV,
    VP_DECODE_COVERAGE_ENV,
    VP_DECODE_COVERAGE_MAX_BS_ENV,
    _VP_DECODE_COVERAGE,
    FD_GATE_MODE_ENV,
    FD_FUSED_ROUTER_NORM_ENV,
    FD_LOW_ROW_POLICY_ENV,
    _VALID_ACTIVE_PHASES,
    _NATIVE_DENSE_UNBOUNDED_ROWS,
    _VALID_GATE_MODES,
    _VALID_LOW_ROW_POLICIES,
    FD_EXECUTION_FULL_GRAPH,
    FD_EXECUTION_DIRECT_EAGER,
    FD_EXECUTION_MODE_ENV,
    _VALID_EXECUTION_MODES,
    FD_ACTIVE_PHASES_ENV,
    FULL_GRAPH_MOCK_SEED_ENV,
    FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO_ENV,
    FULL_GRAPH_MOCK_TOKEN_SKIP_RATE_ENV,
    FULL_GRAPH_SKIPPER_ENV,
    REGIME_SWITCH_ENV,
    _MOCK_CONFIG_ENVS,
    DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER,
)
from sglang.srt.vpipe.types import (
    FullGraphActionBatch,
    FULL_GRAPH_ACTION_CONTRACT,
    FullGraphSkipperAdapter,
    LogicalAction,
    RUN_PROJECT_EXECUTION,
    _LOGICAL_ACTION_CODES,
)
from sglang.srt.vpipe.skipper import (
    available_skippers,
    build_skipper,
    configured_full_graph_skipper_name,
)
from sglang.srt.vpipe.kv_commit import (
    _device_key,
    _trace_counter,
    _fdvp_state,
    _fdvp_trace_enabled,
    _FDVP_TRACE_STATE,
)


REGIME_SWITCH_CONFIG_VERSION = 2
DECODE_BODY_LOW = "prod_allrun"
DECODE_BODY_HIGH = "skip"
_DECODE_BODIES = frozenset((DECODE_BODY_LOW, DECODE_BODY_HIGH))
# [D-849, 2026-09-21] Per-request body pinning. A request's body is decided ONCE
# at admission from the switch's band state and kept for its whole lifetime
# (prefill and every decode step), so every served request is exactly one
# model's computation: "stock" = the base model's dense body (decode
# prod_allrun, prefill dense), "fd" = the routed FlexiDepth body (decode skip,
# prefill fd). Per-PASS selection (version 1) let a request receive both bodies;
# at the GSM8K knee 3,163 of 3,600 served requests matched neither model's
# token sequence and ran away at 4.7 % against the checkpoint's own 2.2 %.
# VP_BODY_MIXED marks a decode ForwardBatch that carries both pins; the model
# runner partitions it into two uniform sub-passes (never a forced-RUN row
# inside the routed graph, which is neither byte-identical nor same-speed vs
# the stock graph -- see regime_body_dispatches_stock_decode).
REQUEST_BODY_STOCK = "stock"
REQUEST_BODY_FD = "fd"
_REQUEST_BODIES = frozenset((REQUEST_BODY_STOCK, REQUEST_BODY_FD))
VP_BODY_MIXED = "mixed"
ADMISSION_CRITERION_BAND_STATE = "decode_band_state"
ADMISSION_PREFILL_DEMOTION_OBSERVE_ONLY = "observe_only"
ADMISSION_MIXED_STEP_PARTITION = "partition"
_FD_PARITY_EPOCHS = {}
def full_graph_contiguous_routed_qkv_config(
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[bool, int, int, dict[int, tuple[float, float]]]:
    """Return fixed-capacity cuBLAS settings for complementary QKV routes."""

    values = os.environ if environ is None else environ
    raw_enabled = str(
        values.get(FD_CONTIGUOUS_ROUTED_QKV_ENV, "0")
    ).strip().lower()
    if raw_enabled in {"1", "true", "yes", "on"}:
        enabled = True
    elif raw_enabled in {"0", "false", "no", "off", ""}:
        enabled = False
    else:
        raise ValueError(
            f"{FD_CONTIGUOUS_ROUTED_QKV_ENV} must be a boolean value"
        )

    try:
        min_rows = int(values.get(FD_ROUTED_QKV_MIN_ROWS_ENV, "128"))
        capacity_multiple = int(
            values.get(FD_ROUTED_QKV_CAPACITY_MULTIPLE_ENV, "16")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "invalid contiguous routed QKV configuration"
        ) from error
    if min_rows <= 0:
        raise ValueError(f"{FD_ROUTED_QKV_MIN_ROWS_ENV} must be positive")
    if capacity_multiple <= 0:
        raise ValueError(
            f"{FD_ROUTED_QKV_CAPACITY_MULTIPLE_ENV} must be positive"
        )

    raw_capacities = str(
        values.get(FD_ROUTED_QKV_CAPACITIES_ENV, "") or ""
    ).strip()
    capacities: dict[int, tuple[float, float]] = {}
    if raw_capacities:
        for entry in raw_capacities.split(","):
            fields = [field.strip() for field in entry.split(":")]
            if len(fields) != 3:
                raise ValueError(
                    f"invalid {FD_ROUTED_QKV_CAPACITIES_ENV} entry: "
                    f"{entry!r}"
                )
            try:
                layer_id = int(fields[0])
                run_fraction = float(fields[1])
                project_fraction = float(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"invalid {FD_ROUTED_QKV_CAPACITIES_ENV} entry: "
                    f"{entry!r}"
                ) from error
            if layer_id < 0 or layer_id in capacities:
                raise ValueError(
                    f"duplicate or negative layer in "
                    f"{FD_ROUTED_QKV_CAPACITIES_ENV}: {layer_id}"
                )
            if not 0.0 < run_fraction <= 1.0:
                raise ValueError(
                    f"RUN capacity in {FD_ROUTED_QKV_CAPACITIES_ENV} "
                    "must be in (0, 1]"
                )
            if not 0.0 < project_fraction <= 1.0:
                raise ValueError(
                    f"PROJECT capacity in {FD_ROUTED_QKV_CAPACITIES_ENV} "
                    "must be in (0, 1]"
                )
            capacities[layer_id] = (run_fraction, project_fraction)
    if enabled and not capacities:
        raise ValueError(
            f"{FD_CONTIGUOUS_ROUTED_QKV_ENV}=1 requires "
            f"{FD_ROUTED_QKV_CAPACITIES_ENV}"
        )
    if not enabled and capacities:
        raise ValueError(
            f"{FD_ROUTED_QKV_CAPACITIES_ENV} requires "
            f"{FD_CONTIGUOUS_ROUTED_QKV_ENV}=1"
        )
    return enabled, min_rows, capacity_multiple, capacities
def _fixed_capacity_mapped_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    row_map: torch.Tensor,
    count: torch.Tensor,
    output: torch.Tensor,
    *,
    capacity: int,
) -> None:
    """Run a packed cuBLAS common lane and an exact mapped overflow lane."""

    rows = int(input_tensor.shape[0])
    if capacity <= 0 or capacity > rows:
        raise ValueError("routed projection capacity is outside its row map")
    from sglang.srt.vpipe.cohort import (
        scatter_cohort,
    )
    from sglang.srt.vpipe.kernel import (
        pack_rows,
    )
    from sglang.srt.vpipe.cohort import (
        mapped_linear,
    )

    packed_input = pack_rows(
        input_tensor,
        row_map,
        count,
        capacity=capacity,
    )
    packed_output = F.linear(packed_input, weight)
    scatter_cohort(output, packed_output, row_map, count)
    mapped_linear(
        input_tensor,
        weight,
        row_map,
        count,
        output,
        row_offset=capacity,
    )
def full_graph_compact_routed_qkv_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether routed rows use complementary mapped projections."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_COMPACT_ROUTED_QKV_ENV)
def _strict_bool(
    values: Mapping[str, str], name: str, default: str = "0"
) -> bool:
    value = str(values.get(name, default) or default).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean value")
def full_graph_fused_router_norm_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether the router's RMSNorm uses SGLang's fused kernel.

    Defaults to FALSE, which keeps the local ``FDRMSNorm`` and therefore keeps
    routing bit-identical to the released FlexiDepth checkpoint. Enabling it
    trades that identity for ~0.32 ms/step of decode tax: the fused kernel's
    parallel variance reduction differs from PyTorch's in the last bit, which
    moves rows sitting within an epsilon of the 0.5 RUN/PROJECT threshold
    (measured: one row of 735,888). Deterministic, but not route-identical.
    Fails closed on a non-boolean value.
    """

    # design constant. OFF keeps routing bit-identical to the released
    # checkpoint; the fused kernel moves rows within epsilon of the 0.5 threshold.
    raw = "1" if SERVED_FUSED_ROUTER_NORM else "0"
    true_values = {"1", "true", "yes", "on"}
    false_values = {"0", "false", "no", "off", ""}
    if raw not in true_values | false_values:
        raise ValueError(f"{FD_FUSED_ROUTER_NORM_ENV} must be a boolean value")
    return raw in true_values


def full_graph_gate_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """Return the routed-MLP gate arithmetic the loaded checkpoint was trained with.

    This is a property of the CHECKPOINT, never a tuning knob. `released` is the
    published FlexiDepth gate (`w * MLP` on RUN, `(1 - w) * PROJECT` otherwise);
    `hard_mask` is the straight-through gate, whose forward selects branches by
    the hard mask with NO `w` scaling. Defaults to `released` so every existing
    deployment is unchanged, and fails closed on any other value.
    """

    # design constant -- a property of the CHECKPOINT, never a tuning knob.
    # [v1.5] Carried as an ARM field (``gate_mode``): a Qwen3 ste_hard checkpoint declares
    # ``hard_mask``; every arm that declares nothing gets the served design's ``released``.
    mode = arm_gate_mode()
    if mode not in _VALID_GATE_MODES:
        choices = ", ".join(sorted(_VALID_GATE_MODES))
        raise ValueError(
            f"{FD_GATE_MODE_ENV} must be one of {choices}; got {mode!r}"
        )
    return mode


def full_graph_low_row_policy(
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the routed-MLP body posture: ``off`` or ``native_dense``.

    [lane-2 knob cleanup] ``full_dual`` and
    ``SGLANG_FD_FULL_GRAPH_LOW_ROW_MAX_ROWS`` are gone. That pair was the third
    occupancy threshold in the design -- a per-layer body swap below 128 rows,
    sitting outside the two admission legs -- and the the HPC cluster duo A/B measured its
    removal as parity on gsm8k (knee TPS -0.1 %, TTFT +1.7 %, TPOT -0.2 %,
    E2E -0.2 %; overload TPS +0.1 %, TTFT -0.8 %, E2E -0.4 %; GR-1a PASS both
    cells). ``native_dense`` is NOT a threshold and stays: it is a whole-model
    posture  that the non-released gate modes require for their
    reference arms, and it is unbounded by construction.
    """

    # "off" IS the design . The environment may not change it;
    # an override here is what silently ran full_dual for eighteen hours.
    values = os.environ if environ is None else environ
    policy = str(values.get(FD_LOW_ROW_POLICY_ENV, "off")).strip().lower()
    if policy != "off" and FD_LOW_ROW_POLICY_ENV in values:
        raise ValueError(
            f"{FD_LOW_ROW_POLICY_ENV} is not configurable: the served "
            f'design is "off". Got {policy!r}'
        )
    if policy not in _VALID_LOW_ROW_POLICIES:
        choices = ", ".join(sorted(_VALID_LOW_ROW_POLICIES))
        raise ValueError(
            f"{FD_LOW_ROW_POLICY_ENV} must be one of {choices}; got {policy!r}"
        )
    return policy
def full_graph_compact_phases(
    environ: Optional[Mapping[str, str]] = None,
) -> frozenset[str]:
    """Return request phases allowed to use a calibrated compact policy."""

    # The served design compacts in BOTH phases. This defaulted to
    # "decode", so a clean deployment silently differed from every measured cell.
    values = os.environ if environ is None else environ
    value = str(values.get(FD_COMPACT_PHASES_ENV, "both") or "both")
    value = value.strip().lower()
    if value not in _VALID_ACTIVE_PHASES:
        choices = ", ".join(sorted(_VALID_ACTIVE_PHASES))
        raise ValueError(
            f"{FD_COMPACT_PHASES_ENV} must be one of {choices}; got {value!r}"
        )
    return (
        frozenset(("decode", "prefill"))
        if value == "both"
        else frozenset((value,))
    )
@dataclass
class FullGraphDeviceRouteTape:
    """Graph-stable per-layer actions and optional row identity metadata."""

    actions: torch.Tensor
    valid_rows: torch.Tensor
    layer_ids: torch.Tensor
    layer_order: tuple[int, ...]
    request_slots: Optional[torch.Tensor]
    token_epochs: Optional[torch.Tensor]
    cache_positions: Optional[torch.Tensor]
    row_ids: Optional[torch.Tensor]
    adapter_name: str
    branch_weights: Optional[torch.Tensor] = None
    logical_request_ids: Optional[torch.Tensor] = None
    compact_evidence_specs: Optional[torch.Tensor] = None
    inline_kv_ready: Optional[torch.Tensor] = None
    structural_inline_kv_ready: bool = False
    recorded_layers: int = 0
    kv_recorded_layers: int = 0

    def route_mask(
        self,
        layer_id: int,
        route_weights: torch.Tensor,
        *,
        forced_route: Optional[bool] = None,
    ) -> torch.Tensor:
        """Write the exact routed or sealed-control decision into the tape."""

        if self.recorded_layers >= len(self.layer_order):
            raise RuntimeError("FlexiDepth device route tape overflow")
        expected_layer = self.layer_order[self.recorded_layers]
        if layer_id != expected_layer:
            raise RuntimeError(
                "FlexiDepth device route tape layer order changed: "
                f"expected {expected_layer}, observed {layer_id}"
            )
        route_index = self.recorded_layers
        target = self.actions[route_index].view(-1, 1)
        if route_weights.shape != target.shape:
            raise RuntimeError(
                "FlexiDepth route weights do not match the device tape rows"
            )
        if self.branch_weights is not None:
            weight_target = self.branch_weights[route_index].view(-1, 1)
            if weight_target.shape != route_weights.shape:
                raise RuntimeError(
                    "FlexiDepth branch-weight tape does not match route rows"
                )
            weight_target.copy_(route_weights)
        if forced_route is None:
            torch.gt(route_weights, 0.5, out=target)
        else:
            target.fill_(forced_route)
        self.recorded_layers += 1
        return target

    def action_mask(
        self,
        layer_id: int,
        action_batch: FullGraphActionBatch,
    ) -> torch.Tensor:
        """Write a common logical-action batch into the legacy bool tape."""

        if self.recorded_layers >= len(self.layer_order):
            raise RuntimeError("full-graph device route tape overflow")
        if action_batch.adapter_name != self.adapter_name:
            raise RuntimeError(
                "full-graph skipper adapter changed inside a captured batch: "
                f"expected {self.adapter_name}, observed {action_batch.adapter_name}"
            )
        expected_layer = self.layer_order[self.recorded_layers]
        if layer_id != expected_layer:
            raise RuntimeError(
                "full-graph device route tape layer order changed: "
                f"expected {expected_layer}, observed {layer_id}"
            )
        route_index = self.recorded_layers
        target = self.actions[route_index].view(-1, 1)
        if self.branch_weights is not None:
            weight_target = self.branch_weights[route_index].view(-1, 1)
            if weight_target.shape != action_batch.branch_weights.shape:
                raise RuntimeError(
                    "full-graph branch-weight tape does not match action rows"
                )
            weight_target.copy_(action_batch.branch_weights)
        action_batch.write_run_storage(out=target)
        self.recorded_layers += 1
        return target

    def action_mask_fused(
        self,
        layer_id: int,
        action_batch: FullGraphActionBatch,
        valid_rows: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """`action_mask` for a router-driven RUN/PROJECT batch, fused (F1).

        Same checks and the same tape writes as `action_mask` (branch-weight
        row copy + `branch_weights > threshold` into the bool action row), in
        one launch that also returns the Block 1B-1 route maps and counts for
        the executing body. Fails closed on forced or explicit-mask batches --
        those keep `action_mask`.
        """

        from sglang.srt.vpipe.routing import route_decide_and_maps

        if action_batch.execution_kind != RUN_PROJECT_EXECUTION:
            raise RuntimeError("fused route decision needs a RUN/PROJECT action batch")
        if action_batch.forced_action is not None or action_batch.explicit_run_mask is not None:
            raise RuntimeError(
                "fused route decision is only for router-driven batches"
            )
        if self.recorded_layers >= len(self.layer_order):
            raise RuntimeError("full-graph device route tape overflow")
        if action_batch.adapter_name != self.adapter_name:
            raise RuntimeError(
                "full-graph skipper adapter changed inside a captured batch: "
                f"expected {self.adapter_name}, observed {action_batch.adapter_name}"
            )
        expected_layer = self.layer_order[self.recorded_layers]
        if layer_id != expected_layer:
            raise RuntimeError(
                "full-graph device route tape layer order changed: "
                f"expected {expected_layer}, observed {layer_id}"
            )
        route_index = self.recorded_layers
        target = self.actions[route_index].view(-1, 1)
        branch_weights = action_batch.branch_weights
        if branch_weights.shape != target.shape:
            raise RuntimeError(
                "full-graph action weights do not match the route-tape rows"
            )
        weight_target = None
        if self.branch_weights is not None:
            weight_target = self.branch_weights[route_index].view(-1, 1)
            if weight_target.shape != branch_weights.shape:
                raise RuntimeError(
                    "full-graph branch-weight tape does not match action rows"
                )
            weight_target = weight_target.reshape(-1)
        maps = route_decide_and_maps(
            branch_weights.reshape(-1),
            float(action_batch.threshold),
            valid_rows,
            target.reshape(-1),
            weight_target,
        )
        self.recorded_layers += 1
        return target, maps

    def require_complete(self) -> None:
        if self.recorded_layers != len(self.layer_order):
            raise RuntimeError(
                "FlexiDepth device route tape is incomplete: "
                f"expected {len(self.layer_order)} layers, "
                f"recorded {self.recorded_layers}"
            )
        if (
            self.branch_weights is not None
            and self.branch_weights.shape != self.actions.shape
        ):
            raise RuntimeError(
                "full-graph branch-weight and action tapes have different shapes"
            )

    def reserve_inline_kv_layer(self, layer_id: int) -> Optional[int]:
        """Reserve one readiness row while planning a routed layer."""

        if self.inline_kv_ready is None and not self.structural_inline_kv_ready:
            return None
        if self.kv_recorded_layers >= len(self.layer_order):
            raise RuntimeError("FlexiDepth inline K/V readiness tape overflow")
        index = self.kv_recorded_layers
        expected_layer = self.layer_order[index]
        if layer_id != expected_layer:
            raise RuntimeError(
                "FlexiDepth inline K/V layer order changed: "
                f"expected {expected_layer}, observed {layer_id}"
            )
        self.kv_recorded_layers += 1
        return index

    def record_inline_kv_ready(
        self,
        layer_id: int,
        *,
        index: Optional[int] = None,
    ) -> None:
        """Mark own-layer K/V complete only after attention returns."""

        if self.inline_kv_ready is None and not self.structural_inline_kv_ready:
            return
        if index is None:
            index = self.reserve_inline_kv_layer(layer_id)
        if index is None or index < 0 or index >= len(self.layer_order):
            raise RuntimeError("FlexiDepth inline K/V readiness index is invalid")
        expected_layer = self.layer_order[index]
        if layer_id != expected_layer:
            raise RuntimeError(
                "FlexiDepth inline K/V readiness index changed: "
                f"expected layer {expected_layer}, observed {layer_id}"
            )
        if self.inline_kv_ready is not None:
            self.inline_kv_ready[index].copy_(self.valid_rows)

    def require_inline_kv_complete(self) -> None:
        if self.inline_kv_ready is None and not self.structural_inline_kv_ready:
            return
        if self.kv_recorded_layers != len(self.layer_order):
            raise RuntimeError(
                "FlexiDepth inline K/V readiness tape is incomplete: "
                f"expected {len(self.layer_order)} layers, "
                f"recorded {self.kv_recorded_layers}"
            )
def full_graph_prefill_cublas_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """P3: run the binary_cohort branch GEMMs through cuBLAS on eager prefill
    passes (RUN/PROJECT counts read to the host once per layer). Captured
    passes (single-request prefill graphs, decode graphs) are untouched."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_PREFILL_CUBLAS_ENV, "0") or "0").strip().lower()
    if value not in ("0", "1"):
        raise ValueError(f"{FD_PREFILL_CUBLAS_ENV} must be 0 or 1; got {value!r}")
    return value == "1"
def full_graph_prefill_fallback_min_project(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[float]:
    """: minimum PROJECT share (fraction of valid rows) below which a
    routed layer on an EAGER prefill pass runs the exact dense full-dual body
    instead of a compaction body. None = off. Must lie in (0, 1]; fail closed
    on anything else (a typo must not silently disable the fallback)."""

    values = os.environ if environ is None else environ
    raw = str(values.get(FD_PREFILL_FALLBACK_ENV, "") or "").strip()
    if not raw:
        return None
    try:
        share = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{FD_PREFILL_FALLBACK_ENV} must be a float share in (0, 1]; got {raw!r}"
        ) from exc
    if not (0.0 < share <= 1.0):
        raise ValueError(
            f"{FD_PREFILL_FALLBACK_ENV} must lie in (0, 1]; got {raw!r}"
        )
    return share
def vp_decode_coverage_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """P5 coverage-as-code switch (default ON). Only consulted when FlexiDepth
    full_graph serving is active; production (no routed layers) never sees it."""

    values = os.environ if environ is None else environ
    value = str(values.get(VP_DECODE_COVERAGE_ENV, "1") or "1").strip().lower()
    if value not in ("0", "1"):
        raise ValueError(f"{VP_DECODE_COVERAGE_ENV} must be 0 or 1; got {value!r}")
    return value == "1"
def vp_decode_coverage_max_bs(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[int]:
    """Explicit ceiling for the coverage endpoint (None = the built-in bound of
    4x the CLI-configured decode max_bs, the largest capture the A100 T2
    emulation proved). Capture memory is finite: the RTX 8000 box OOM'd at
    its full admission cap (2026-09-05)."""

    values = os.environ if environ is None else environ
    raw = str(values.get(VP_DECODE_COVERAGE_MAX_BS_ENV, "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{VP_DECODE_COVERAGE_MAX_BS_ENV} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{VP_DECODE_COVERAGE_MAX_BS_ENV} must be positive")
    return value
def coverage_capture_bs(
    capture_bs: list[int], target: int, generate
) -> tuple[list[int], int]:
    """Extend a decode capture list so its largest bucket reaches ``target``
    (the scheduler admission cap), using the SAME bucket generator as the stock
    CLI (``ServerArgs._generate_decode_cuda_graph_batch_sizes``) so the added
    buckets have the stock spacing. Returns (new_list, buckets_added)."""

    if target <= 0:
        raise ValueError(f"coverage target must be positive; got {target}")
    current_max = max(capture_bs)
    if current_max >= target:
        return list(capture_bs), 0
    extra = sorted({int(bs) for bs in generate(int(target)) if int(bs) > current_max})
    if not extra or extra[-1] != target:
        extra = [bs for bs in extra if bs < target] + [int(target)]
    return sorted(set(capture_bs) | set(extra)), len(extra)
def vp_decode_coverage_target(model_runner) -> Optional[int]:
    """The admission cap the decode graphs must cover, or None when the
    coverage rule does not apply (not FlexiDepth full_graph serving, no routed
    weights, or explicitly disabled). Reads model_runner.max_running_requests
    (set by the KV-pool configurator before graph capture; fails loudly if
    absent — never a silent default)."""

    if not vp_decode_coverage_enabled():
        return None
    if flexidepth_execution_mode() != FD_EXECUTION_FULL_GRAPH:
        return None
    # the skipper is deployed iff the active arm names one; host path
    # comes from the committed host config, never from a shell export.
    if not skipper_deployed():
        return None
    target = int(model_runner.max_running_requests)
    if target <= 0:
        raise ValueError(f"decode coverage: max_running_requests={target}")
    return target
def _route_tape_request_slots(
    forward_batch: Any, row_ids: torch.Tensor
) -> torch.Tensor:
    request_slots = getattr(forward_batch, "req_pool_indices", None)
    if request_slots is None:
        raise RuntimeError("FlexiDepth device route digest requires request slots")
    if full_graph_decode_enabled(forward_batch):
        if request_slots.shape != row_ids.shape:
            raise RuntimeError(
                "FlexiDepth decode request slots do not match the route rows"
            )
        return request_slots

    extend_starts = getattr(forward_batch, "extend_start_loc", None)
    if extend_starts is None:
        raise RuntimeError(
            "FlexiDepth prefill route digest requires extend_start_loc"
        )
    request_rows = torch.bucketize(
        row_ids.to(dtype=extend_starts.dtype), extend_starts[1:], right=True
    )
    return request_slots.index_select(0, request_rows)
def _accumulate_device_route_digest(
    tape: FullGraphDeviceRouteTape, counters: torch.Tensor
) -> None:
    """Accumulate two 64-bit action fingerprints without a host readback."""

    if counters.shape != (6,):
        raise RuntimeError("FlexiDepth device route digest counter shape changed")
    tape.require_complete()
    valid = tape.valid_rows.view(1, -1)
    if tape.logical_request_ids is not None:
        if tape.token_epochs is None:
            raise RuntimeError(
                "logical device route digest requires token epochs"
            )
        action_planes = tape.actions
        layer_ids = tape.layer_ids.to(dtype=torch.int64).view(-1, 1)
        component_ids = torch.zeros_like(layer_ids)
        action_codes = action_planes.to(dtype=torch.int64)
        logical_request_ids = tape.logical_request_ids.to(
            dtype=torch.int64
        ).view(1, -1)
        token_epochs = tape.token_epochs.to(dtype=torch.int64).view(1, -1)
        metadata_codes = (
            (layer_ids + 1) * 1_000_003
            + (component_ids + 1) * 1_000_033
            + (logical_request_ids + 1) * 1_000_037
            + (token_epochs + 1) * 1_000_081
        )
        cell_weights = (
            (layer_ids + 1) * 65_537
            + (component_ids + 1) * 65_539
            + (logical_request_ids + 1) * 65_543
            + (token_epochs + 1) * 65_551
        )
        digest_uses_dispatch_order = False
    else:
        metadata = (
            tape.request_slots,
            tape.token_epochs,
            tape.cache_positions,
            tape.row_ids,
        )
        if any(value is None for value in metadata):
            raise RuntimeError(
                "FlexiDepth device route digest metadata is incomplete"
            )
        request_slots, token_epochs, cache_positions, row_ids = metadata
        assert request_slots is not None
        assert token_epochs is not None
        assert cache_positions is not None
        assert row_ids is not None
        layer_ids = tape.layer_ids.to(dtype=torch.int64).view(-1, 1)
        row_ids = row_ids.to(dtype=torch.int64).view(1, -1)
        request_slots = request_slots.to(dtype=torch.int64).view(1, -1)
        token_epochs = token_epochs.to(dtype=torch.int64).view(1, -1)
        cache_positions = cache_positions.to(dtype=torch.int64).view(1, -1)
        action_planes = tape.actions
        action_codes = action_planes.to(dtype=torch.int64)
        metadata_codes = (
            (layer_ids + 1) * 1_000_003
            + (row_ids + 1) * 1_000_033
            + (request_slots + 1) * 1_000_037
            + (token_epochs + 1) * 1_000_081
            + (cache_positions + 1) * 1_000_099
        )
        cell_weights = (layer_ids + 1) * 65_537 + (row_ids + 1) * 65_539
        digest_uses_dispatch_order = True

    payload = torch.where(
        valid,
        metadata_codes * 2 + action_codes,
        torch.zeros((), dtype=torch.int64, device=tape.actions.device),
    )
    logical_rows = valid.sum(dtype=torch.int64) * action_planes.shape[0]
    run_rows = (action_planes & valid).sum(dtype=torch.int64)
    digest_sum = payload.sum(dtype=torch.int64)
    digest_weighted = (payload * cell_weights).sum(dtype=torch.int64)
    dispatch_ordinal = counters[0] + 1

    counters[0].add_(1)
    counters[1].add_(logical_rows)
    counters[2].add_(run_rows)
    counters[3].add_(digest_sum)
    counters[4].add_(
        digest_weighted
        + digest_sum * dispatch_ordinal * int(digest_uses_dispatch_order)
    )
    counters[5].add_(valid.sum(dtype=torch.int64))
def _route_derived_compact_stats(
    routes: torch.Tensor,
    valid_rows: torch.Tensor,
    compact_specs: torch.Tensor,
    *,
    compact_active: bool,
    capacity_multiple: int,
) -> torch.Tensor:
    """Derive compact coverage and overflow from graph-stable route state."""

    if routes.ndim != 2 or valid_rows.shape != (routes.shape[1],):
        raise RuntimeError("FlexiDepth compact accounting route shape changed")
    if compact_specs.shape != (routes.shape[0], 3):
        raise RuntimeError("FlexiDepth compact accounting metadata changed")
    if capacity_multiple <= 0:
        raise RuntimeError("FlexiDepth compact accounting multiple is invalid")

    valid_count = valid_rows.sum(dtype=torch.int64)
    zero = torch.zeros((), dtype=torch.int64, device=routes.device)
    if not compact_active:
        return torch.stack((zero, zero, zero))

    run_counts = (routes & valid_rows.unsqueeze(0)).sum(
        dim=1, dtype=torch.int64
    )
    project_counts = valid_count - run_counts
    modes = compact_specs[:, 0]
    run_fractions = compact_specs[:, 1]
    project_fractions = compact_specs[:, 2]
    row_count = routes.shape[1]

    def capacities(fractions: torch.Tensor) -> torch.Tensor:
        values = torch.ceil(
            fractions * (float(row_count) / float(capacity_multiple))
        ).to(dtype=torch.int64)
        return (values * capacity_multiple).clamp(min=1, max=row_count)

    run_capacities = capacities(run_fractions)
    project_capacities = capacities(project_fractions)
    explicit_policy = modes == 1.0
    fallback_policy = (modes == 2.0) & (
        ((run_fractions > 0.0) & (run_capacities < row_count))
        | (
            (project_fractions > 0.0)
            & (project_capacities < row_count)
        )
    )
    policy_active = explicit_policy | fallback_policy
    run_overflow = torch.where(
        policy_active & (run_fractions > 0.0),
        torch.clamp(run_counts - run_capacities, min=0),
        zero,
    )
    project_overflow = torch.where(
        policy_active & (project_fractions > 0.0),
        torch.clamp(project_counts - project_capacities, min=0),
        zero,
    )
    return torch.stack(
        (
            policy_active.sum(dtype=torch.int64) * valid_count,
            run_overflow.sum(dtype=torch.int64),
            project_overflow.sum(dtype=torch.int64),
        )
    )
def full_graph_decode_enabled(forward_batch: Any) -> bool:
    if flexidepth_execution_mode() != FD_EXECUTION_FULL_GRAPH:
        return False
    return (
        flexidepth_forward_phase(forward_batch) == "decode"
        and flexidepth_phase_enabled(forward_batch)
    )
def flexidepth_execution_mode(
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the explicit execution mode and reject misspelled modes."""

    # The design applies to arms that SERVE A SKIPPER. The stock arm serves
    # none, so it must not demand the full-graph body (which requires weights). The
    # ARM decides whether the mechanism runs at all; the design decides how it runs.
    mode = SERVED_EXECUTION_MODE if skipper_deployed() else FD_EXECUTION_DIRECT_EAGER
    if mode not in _VALID_EXECUTION_MODES:
        choices = ", ".join(sorted(_VALID_EXECUTION_MODES))
        raise ValueError(
            f"{FD_EXECUTION_MODE_ENV} must be one of {choices}; got {mode!r}"
        )
    return mode
def flexidepth_forward_phase(forward_batch: Any) -> Optional[str]:
    mode = getattr(forward_batch, "forward_mode", None)
    is_decode = getattr(mode, "is_decode", None)
    if callable(is_decode) and is_decode():
        return "decode"
    is_extend = getattr(mode, "is_extend", None)
    if callable(is_extend) and is_extend():
        return "prefill"
    return None
def flexidepth_phase_enabled(forward_batch: Any) -> bool:
    phase = flexidepth_forward_phase(forward_batch)
    if phase is None or phase not in flexidepth_active_phases():
        return False
    # W1 regime switch (prefill leg): a prefill pass the predicate routed to the
    # dense body disables FlexiDepth for this pass, so the grouped routed path
    # (via full_graph_prefill_enabled), the eager router path, and
    # prepare_full_graph_batch all fall through to the untouched base-Llama
    # dense body. Only prefill passes consult the stamp.
    if phase == "prefill" and _regime_switch_prefill_forced_dense(forward_batch):
        return False
    # (c3) coverage stamp: a decode pass outside captured graph coverage
    # (stamped at the model-runner dispatch seam from the runner's own
    # can_run_graph predicate) disables FlexiDepth for this pass through the
    # SAME single lever the W1 legs use. Deliberately a standalone clause
    # OUTSIDE _regime_switch_decode_forced_dense's W1-config guard: it is
    # intrinsic and default-on ("the cap becomes code"), active with W1 off.
    # Strict priority coverage > band (the band only ever selects among
    # CAPTURED bodies). No counter here — this predicate is evaluated ~32-48x
    # per pass (F2); the positive witness lives at the dense body (site B).
    if phase == "decode" and forward_batch.vp_fd_decode_coverage_dense:
        return False
    # W1 regime switch (decode leg, I6b): a low-band ("prod_allrun") decode pass
    # disables FlexiDepth for this pass. Because full_graph_decode_enabled AND
    # the eager router branch both gate on this predicate, returning False here
    # is the single lever that yields the base-Llama dense fall-through — the
    # STOCK production decode body (paired with force_production_attention so
    # every layer uses the flashinfer decode backend). Only decode passes consult
    # the stamp; the high "skip" band leaves it False -> unchanged M3.38 body.
    if phase == "decode" and _regime_switch_decode_forced_dense(forward_batch):
        return False
    return True
class RegimeSwitchPrefillConfig(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Prefill-leg sub-threshold gate parameters."""

    enabled: bool
    min_tokens: int
    row_correction_alpha: float
    include_mixed: bool
    # `max_tokens` (the upper token gate) is DELETED. Its 2026-08-20
    # premise -- the routed body loses on large packed passes -- was falsified by
    # the rebuilt body and by the per-token profile, which INVERTS it (cost falls
    # with pass size). Measured removal : +9.5 % of gsm8k overload passes
    # move dense->routed for TTFT -6.1 % / E2E -2.8 %, and it is provably inert on
    # coqa (byte-identical prefill counters, both arms). The engagement escape
    # below is the mechanism that decides this properly, from measured routing
    # share rather than a token bracket. Setting the key is now refused by
    # forbid_unknown_fields, so a stale config fails the boot instead of silently
    # reinstating the gate.
    # [P8 v2,] Engagement-keyed escape: token thresholds cannot
    # separate engaged large packs (gsm8k 2-4K: routed WINS) from
    # low-engagement ones (mmlu_pro 2.5K: routed loses). When set, the
    # dispatch decision uses a running engagement estimate (EMA of the
    # binary-cohort PROJECT-row share, read per routed pass): routed while
    # EMA >= engagement_min (or unknown — optimistic start), else dense,
    # with one routed probe pass every engagement_probe_every dense passes
    # so the estimate can recover. Token brackets still apply first as
    # hard bounds. None = pure token-bracket behavior (v1, unchanged).
    engagement_min: Optional[float] = None
    engagement_probe_every: int = 64

    def validate(self) -> None:
        if self.engagement_min is not None and not (
            0.0 <= self.engagement_min <= 1.0
        ):
            raise ValueError(
                "regime switch prefill.engagement_min must be in [0, 1]"
            )
        if self.engagement_probe_every < 1:
            raise ValueError(
                "regime switch prefill.engagement_probe_every must be >= 1"
            )
        if self.min_tokens < 0:
            raise ValueError(
                "regime switch prefill.min_tokens must be non-negative"
            )
        if self.row_correction_alpha < 0.0:
            raise ValueError(
                "regime switch prefill.row_correction_alpha must be "
                "non-negative"
            )
class RegimeSwitchDecodeConfig(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Decode-leg hysteresis band parameters and body labels."""

    enabled: bool
    enter_rows: int
    exit_rows: int
    low_body: str
    high_body: str
    # c2 : require this many CONSECUTIVE sub-exit_rows decode passes before
    # flipping skip->dense, so a transient occupancy dip does not withhold skip
    # (the switch tax). Default 1 == byte-identical to the un-smoothed band.
    exit_dwell: int = 1
    # Lane-2 cut1 (2026-09-02): KV-volume criterion. When enter_kv_tokens > 0
    # the band is keyed on the decode batch's resident KV tokens
    # (ForwardBatch.seq_lens_sum, a host int -- no device sync) instead of its
    # rows: the the HPC cluster ladders put the skip body's crossover at ~512 rows x 256
    # tokens (131k KV tokens, parity) vs 256 rows x 1k (262k, -4%), i.e. the
    # saving scales with rows x context while the floor is fixed, so rows alone
    # mis-key the band. 0/0 (default) == the rows criterion, byte-identical.
    enter_kv_tokens: int = 0
    exit_kv_tokens: int = 0

    @property
    def kv_criterion(self) -> bool:
        return self.enter_kv_tokens > 0

    def validate(self) -> None:
        if self.enter_rows < 0 or self.exit_rows < 0:
            raise ValueError(
                "regime switch decode enter/exit rows must be non-negative"
            )
        if self.enter_kv_tokens < 0 or self.exit_kv_tokens < 0:
            raise ValueError(
                "regime switch decode enter/exit kv tokens must be non-negative"
            )
        if (self.enter_kv_tokens > 0) != (self.exit_kv_tokens > 0):
            raise ValueError(
                "regime switch decode.enter_kv_tokens and exit_kv_tokens must "
                "be set together (both 0 selects the rows criterion)"
            )
        if self.enter_kv_tokens and self.enter_kv_tokens <= self.exit_kv_tokens:
            raise ValueError(
                "regime switch decode.enter_kv_tokens must exceed "
                "decode.exit_kv_tokens (hysteresis band requires enter > exit)"
            )
        if self.enter_rows <= self.exit_rows:
            raise ValueError(
                "regime switch decode.enter_rows must exceed decode.exit_rows "
                "(hysteresis band requires enter > exit)"
            )
        if self.low_body not in _DECODE_BODIES:
            raise ValueError(
                "regime switch decode.low_body must be one of "
                f"{sorted(_DECODE_BODIES)}; got {self.low_body!r}"
            )
        if self.high_body not in _DECODE_BODIES:
            raise ValueError(
                "regime switch decode.high_body must be one of "
                f"{sorted(_DECODE_BODIES)}; got {self.high_body!r}"
            )
        if self.low_body != DECODE_BODY_LOW:
            raise ValueError(
                f"regime switch decode.low_body must be {DECODE_BODY_LOW!r}"
            )
        if self.high_body != DECODE_BODY_HIGH:
            raise ValueError(
                f"regime switch decode.high_body must be {DECODE_BODY_HIGH!r}"
            )
        if self.exit_dwell < 1:
            raise ValueError(
                "regime switch decode.exit_dwell must be >= 1; got "
                f"{self.exit_dwell}"
            )
@dataclass
class FDLayerRoute:
    """A FlexiDepth decision plus the normalized state consumed by the layer."""

    hidden: torch.Tensor
    residual: torch.Tensor
    weights: torch.Tensor
    mask: torch.Tensor
    run_rows: torch.Tensor
    kept: torch.Tensor
    skip: Optional[torch.Tensor]
    kept_rows: int
    branch: str
    mask_key: Optional[str]
class RegimeSwitchAdmissionConfig(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """[D-849] Per-request body pinning at admission (version 2 of the switch).

    ``enabled``: pin one body per request for its lifetime. ``criterion``: what
    decides the pin (the scheduler's mirror of the decode band state).
    ``cold_start``: the pin before the band has ever engaged. ``prefill_demotion``:
    under pinning the prefill engagement escape only OBSERVES (it may not hand
    a routed-pinned request a dense prefill). ``mixed_step``: a decode step whose
    running batch carries both pins is PARTITIONED into two uniform sub-passes.
    Every string has exactly one legal value: the field exists so the served
    design is attested and diffable, not so it can be tuned.
    """

    enabled: bool
    criterion: str = ADMISSION_CRITERION_BAND_STATE
    cold_start: str = REQUEST_BODY_STOCK
    prefill_demotion: str = ADMISSION_PREFILL_DEMOTION_OBSERVE_ONLY
    mixed_step: str = ADMISSION_MIXED_STEP_PARTITION

    def validate(self) -> None:
        if self.criterion != ADMISSION_CRITERION_BAND_STATE:
            raise ValueError(
                "regime switch admission.criterion must be "
                f"{ADMISSION_CRITERION_BAND_STATE!r}; got {self.criterion!r}"
            )
        if self.cold_start != REQUEST_BODY_STOCK:
            raise ValueError(
                "regime switch admission.cold_start must be "
                f"{REQUEST_BODY_STOCK!r}; got {self.cold_start!r}"
            )
        if self.prefill_demotion != ADMISSION_PREFILL_DEMOTION_OBSERVE_ONLY:
            raise ValueError(
                "regime switch admission.prefill_demotion must be "
                f"{ADMISSION_PREFILL_DEMOTION_OBSERVE_ONLY!r}; got "
                f"{self.prefill_demotion!r}"
            )
        if self.mixed_step != ADMISSION_MIXED_STEP_PARTITION:
            raise ValueError(
                "regime switch admission.mixed_step must be "
                f"{ADMISSION_MIXED_STEP_PARTITION!r}; got {self.mixed_step!r}"
            )


class RegimeSwitchConfig(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Top-level regime-switch config with a pinned version."""

    version: int
    prefill: RegimeSwitchPrefillConfig
    decode: RegimeSwitchDecodeConfig
    # [D-849] version 2: REQUIRED, so a version-1 literal (no admission block)
    # fails closed at decode instead of silently serving per-pass selection.
    admission: RegimeSwitchAdmissionConfig

    def validate(self) -> None:
        if self.version != REGIME_SWITCH_CONFIG_VERSION:
            raise ValueError(
                "regime switch config version must be "
                f"{REGIME_SWITCH_CONFIG_VERSION}; got {self.version}"
            )
        self.prefill.validate()
        self.decode.validate()
        self.admission.validate()
        if self.admission.enabled and not self.decode.enabled:
            raise ValueError(
                "regime switch admission.enabled requires decode.enabled: the "
                "pin is decided from the decode band state"
            )

    @property
    def pinning(self) -> bool:
        """Per-request body pinning is in force for this served config."""

        return self.admission.enabled and self.decode.enabled
def fdvp_fused_project_input_enabled():
    # design constant, not an environment read
    return mechanism(SERVED_FUSED_PROJECT_INPUT)
def fdvp_fused_project_input_shared_storage_enabled():
    return False  # design constant, not an environment read
def _fdvp_router_graph_enabled():
    return False  # pinned off; trace knob, no served purpose
def _fdvp_timing_enabled():
    return False  # pinned off; trace knob, no served purpose
def fd_parity_trace_target() -> str:
    return ""  # pinned off; trace knob, no served purpose
def resolve_full_graph_skipper(
    environ: Optional[Mapping[str, str]] = None,
) -> FullGraphSkipperAdapter:
    """Resolve the process policy and fail before serving on unsupported names."""

    values = os.environ if environ is None else environ
    name = full_graph_skipper_name(values)
    # reading env here only to REFUSE a stale export : the values come from the arm
    mock_config_present = any(
        str(values.get(key, "")).strip() for key in _MOCK_CONFIG_ENVS
    )
    if name != DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER and mock_config_present:
        raise ValueError(
            "deterministic mock settings require "
            f"{FULL_GRAPH_SKIPPER_ENV}={DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER}"
        )
    # One registry lookup for every policy. A parameterised policy reads its own
    # values out of the arm inside its factory , so the resolver no longer
    # carries a branch per skipper -- which is what made a third skipper an edit
    # in two places of library code .
    arm = dict(active_arm())
    arm.setdefault("name", active_arm_name())
    return build_skipper(name, arm)
def flexidepth_active_phases(
    environ: Optional[Mapping[str, str]] = None,
) -> frozenset[str]:
    """Return the explicitly enabled request phases.

    ``both`` preserves the released FlexiDepth behavior and is the default.
    Phase isolation exists only to build decode-only and prefill-only systems
    comparisons without changing the route equations inside an active phase.
    """

    values = os.environ if environ is None else environ
    # phases come from the ACTIVE ARM, not the environment.
    # An arm serving NO skipper routes no phases. Reporting "both" there was a
    # true-looking field that is false, and the launch gate cannot catch it: intended and
    # resolved would both state the same wrong thing.
    if not skipper_deployed():
        return frozenset()
    value = str(active_arm().get("phases") or "both").strip().lower()
    if value not in _VALID_ACTIVE_PHASES:
        choices = ", ".join(sorted(_VALID_ACTIVE_PHASES))
        raise ValueError(
            f"{FD_ACTIVE_PHASES_ENV} must be one of {choices}; got {value!r}"
        )
    return frozenset(("decode", "prefill")) if value == "both" else frozenset((value,))
def _regime_switch_prefill_forced_dense(forward_batch: Any) -> bool:
    """Return True when the W1 prefill leg stamped this prefill pass dense.

    Consulted only for prefill-phase passes.  The ``vp_fd_prefill_dense`` stamp
    is read exclusively under the switch-on + ``prefill.enabled`` guard:
    ``LlamaModel.forward`` sets it on every prefill pass whenever that guard
    holds, so the attribute is always present here (no defensive ``getattr``).
    When the switch is off — or its prefill leg is disabled — the guard
    short-circuits before the read, so an unstamped batch keeps the
    byte-identical FlexiDepth path.
    """

    cfg = regime_switch_config()
    if cfg is None or not cfg.prefill.enabled:
        return False
    return forward_batch.vp_fd_prefill_dense
def _regime_switch_decode_forced_dense(forward_batch: Any) -> bool:
    """Return True when the W1 decode leg stamped this decode pass dense (I6b).

    Consulted only for decode-phase passes.  The ``vp_fd_decode_dense`` stamp is
    read exclusively under the switch-on + ``decode.enabled`` guard: it is a
    ForwardBatch field defaulting to False (always present, so no defensive
    ``getattr``), stamped True only on the dummies the decode cuda-graph backend
    captures for a ``prod_allrun`` low-band bucket.  When the switch is off — or
    its decode leg is disabled — the guard short-circuits before the read, so an
    unstamped batch keeps the byte-identical FlexiDepth path.
    """

    cfg = regime_switch_config()
    if cfg is None or not cfg.decode.enabled:
        return False
    return forward_batch.vp_fd_decode_dense
def full_graph_skipper_name(
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    values = os.environ if environ is None else environ
    name = configured_full_graph_skipper_name(values)
    known = frozenset(available_skippers())
    if name not in known:
        choices = ", ".join(sorted(known))
        raise ValueError(
            f"{FULL_GRAPH_SKIPPER_ENV} must be one of {choices}; got {name!r}"
        )
    return name
# The served design lives in vpipe/design.py -- ONE definition, imported
# here rather than duplicated, so it cannot drift between modules.


def regime_switch_config(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[RegimeSwitchConfig]:
    """Return the served regime-switch design.

    The design is a CONSTANT, not an environment read. The only value
    the environment may still carry is the explicit string ``"off"``, which the
    byte-identical-baseline arm uses to disable the mechanism outright; it cannot
    change any parameter. An absent variable now yields THE DESIGN, not ``None``
    -- the reverse of the old behaviour, and the point of the change.
    """

    # The ARM decides whether the mechanism engages: the stock arm carries
    # regime_switch=False and gets the byte-identical baseline path.
    if not active_arm().get("regime_switch", True):
        return None
    values = os.environ if environ is None else environ
    raw = str(values.get(REGIME_SWITCH_ENV, "") or "").strip()
    if raw.lower() == "off":
        return None
    if raw:
        raise ValueError(
            f"{REGIME_SWITCH_ENV} may only be unset (= the served design) or "
            f'"off" (= baseline arm). Configuring the design by environment is '
            f"forbidden; it is a constant in vpipe/common.py. Got {raw!r}"
        )
    # An arm routes only the phases it has. The regime switch carries BOTH
    # admission legs, but a prefill-only arm must not run the decode leg: its stock low
    # band is dispatched through the FlexiDepth conditional decode backend, which does
    # not exist without a decode phase (decode_cuda_graph_runner asserts exactly that).
    # The deleted arm_env_vpre_binarycohort.sh expressed this by exporting a prefill-only
    # config; with the scripts gone, the arm's phases have to say it.
    design = dict(SERVED_REGIME_SWITCH)
    phases = arm_phases()
    for leg in ("prefill", "decode"):
        if leg not in phases:
            design[leg] = {**design[leg], "enabled": False}
    # [D-849] The pin is decided from the decode band state, so an arm without a
    # decode leg (prefill-only) cannot pin: its admission block resolves disabled
    # and the version-1 per-pass prefill decision governs that arm unchanged.
    if not design["decode"]["enabled"]:
        design["admission"] = {**design["admission"], "enabled": False}
    # The K/V band is declared per device (design.SERVED_DECODE_KV_BAND_BY_DEVICE) and
    # asserted against the roofline rule at boot (model_runner). On the A100 the entry equals
    # the base declaration, so the served config is byte-identical to before this line.
    if design["decode"].get("enter_kv_tokens", 0) > 0 and torch.cuda.is_available():
        from sglang.srt.vpipe.design import SERVED_DECODE_KV_BAND_BY_DEVICE
        from sglang.srt.vpipe.kernel import canonical_device_key

        from sglang.srt.vpipe.roofline import band_device_key

        key = band_device_key(
            canonical_device_key(torch.cuda.get_device_name(0)),
            torch.cuda.get_device_properties(0).total_memory,
        )
        # [v1.5] an arm may declare its own per-device band (a different checkpoint's rule inputs);
        # the boot assertion checks it against the rule with that arm's inputs.
        band = dict(active_arm().get("decode_kv_band", {})).get(key) or SERVED_DECODE_KV_BAND_BY_DEVICE.get(key)
        if band is not None:
            design["decode"] = {**design["decode"], "exit_kv_tokens": band[0], "enter_kv_tokens": band[1]}
    raw = _json.dumps(design)
    # Memoized on the raw string: the gate (flexidepth_phase_enabled)
    # consults this ~32-48x per pass, and the config is boot-constant —
    # re-decoding the JSON per call taxed every regime-on pass.
    cached = _REGIME_SWITCH_CONFIG_CACHE.get(raw)
    if cached is not None:
        return cached
    try:
        config = msgspec.json.decode(raw, type=RegimeSwitchConfig)
    except msgspec.MsgspecError as error:
        raise ValueError(
            f"{REGIME_SWITCH_ENV} is not a valid regime-switch config: {error}"
        ) from error
    config.validate()
    _REGIME_SWITCH_CONFIG_CACHE[raw] = config
    return config


_REGIME_SWITCH_CONFIG_CACHE: dict[str, RegimeSwitchConfig] = {}


def resolved_design_attestation() -> dict[str, Any]:
    """What this process ACTUALLY resolved -- never what design.py declares.

    [, Codex F8] The first version of the launch gate published
    ``design_attestation()``: the ``SERVED_*`` constants, echoed back. Comparing the
    tree's constants against the tree's own constants is a tautology -- it cannot fail.
    An environment override changes what the RESOLVERS return while the declaration
    still reads correct, which is precisely the failure the gate exists to catch.

    So this calls the resolvers and reports their answers. The gate then compares
    INTENDED (design.py constants) against RESOLVED (this), and a divergence -- from an
    inherited export, a stale process, a mis-selected arm -- shows up as a mismatch.
    """

    switch = regime_switch_config()
    return {
        "source": "resolvers, at boot (D-611)",
        "arm": active_arm_name(),
        # stock serves no skipper; reporting the fallback adapter name there would be a
        # true-looking field that is false -- the shape of defect this file exists to end
        "skipper": full_graph_skipper_name() if skipper_deployed() else None,
        "regime_switch": (
            None
            if switch is None
            else {
                "version": switch.version,
                "prefill": {
                    "enabled": switch.prefill.enabled,
                    "min_tokens": switch.prefill.min_tokens,
                    "row_correction_alpha": switch.prefill.row_correction_alpha,
                    "include_mixed": switch.prefill.include_mixed,
                    "engagement_min": switch.prefill.engagement_min,
                    "engagement_probe_every": switch.prefill.engagement_probe_every,
                },
                "decode": {
                    "enabled": switch.decode.enabled,
                    "enter_rows": switch.decode.enter_rows,
                    "exit_rows": switch.decode.exit_rows,
                    "low_body": switch.decode.low_body,
                    "high_body": switch.decode.high_body,
                    "enter_kv_tokens": switch.decode.enter_kv_tokens,
                    "exit_kv_tokens": switch.decode.exit_kv_tokens,
                },
                # [D-849] per-request body pinning, as RESOLVED (an arm without a
                # decode leg resolves enabled=False; the gate diffs this block).
                "admission": {
                    "enabled": switch.admission.enabled,
                    "criterion": switch.admission.criterion,
                    "cold_start": switch.admission.cold_start,
                    "prefill_demotion": switch.admission.prefill_demotion,
                    "mixed_step": switch.admission.mixed_step,
                },
            }
        ),
        "low_row_policy": full_graph_low_row_policy(),
        "execution_mode": flexidepth_execution_mode(),
        "active_phases": sorted(flexidepth_active_phases()),
        # [v1.5] `compact_phases` is the set of phases compaction MAY use (a design constant);
        # `compact_enabled` is the mechanism switch the arm controls (off for the no-compaction
        # ablation arm and for Qwen3). Without it here, an arm served with compaction on while
        # declaring it off would pass the served-design gate on phases alone.
        "compact_enabled": mechanism(arm_compact_enabled()),
        "compact_phases": sorted(full_graph_compact_phases()),
        "gate_mode": full_graph_gate_mode(),
        "fused_router_norm": full_graph_fused_router_norm_enabled(),
    }
