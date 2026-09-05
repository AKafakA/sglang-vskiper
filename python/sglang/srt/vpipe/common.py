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
import os
import torch
import torch.nn.functional as F
from sglang.srt.vpipe.env import (
    FD_COMPACT_O_PROJ_MIN_ROWS_ENV,
    FD_COMPACT_ROUTED_QKV_ENV,
    FD_CONTIGUOUS_ROUTED_QKV_ENV,
    FD_ROUTED_QKV_CAPACITIES_ENV,
    FD_ROUTED_QKV_CAPACITY_MULTIPLE_ENV,
    FD_ROUTED_QKV_MIN_ROWS_ENV,
    FD_COMPACT_PHASES_ENV,
    FD_PREFILL_CUBLAS_ENV,
    VP_DECODE_COVERAGE_ENV,
    VP_DECODE_COVERAGE_MAX_BS_ENV,
    _VP_DECODE_COVERAGE,
    FD_COMPACT_Q_PROJ_ENV,
    FD_LOW_ROW_MAX_ROWS_ENV,
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
    _LOGICAL_ACTION_CODES,
)
from sglang.srt.vpipe.skipper import (
    _ADAPTERS,
    _deterministic_mock_adapter,
    _parse_mock_config,
    configured_full_graph_skipper_name,
)
from sglang.srt.vpipe.kv_commit import (
    _device_key,
    _trace_counter,
    _fdvp_state,
    _fdvp_trace_enabled,
    _FDVP_TRACE_STATE,
)


REGIME_SWITCH_CONFIG_VERSION = 1
DECODE_BODY_LOW = "prod_allrun"
DECODE_BODY_HIGH = "skip"
_DECODE_BODIES = frozenset((DECODE_BODY_LOW, DECODE_BODY_HIGH))
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
def full_graph_compact_o_proj_min_rows(
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Return the measured graph-bucket crossover for compact output."""

    values = os.environ if environ is None else environ
    try:
        min_rows = int(values.get(FD_COMPACT_O_PROJ_MIN_ROWS_ENV, "128"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{FD_COMPACT_O_PROJ_MIN_ROWS_ENV} must be an integer"
        ) from error
    if min_rows <= 0:
        raise ValueError(f"{FD_COMPACT_O_PROJ_MIN_ROWS_ENV} must be positive")
    return min_rows
def full_graph_compact_q_proj_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether RUN-only Q projection shares the compact output route."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_COMPACT_Q_PROJ_ENV)
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

    values = os.environ if environ is None else environ
    raw = str(values.get(FD_FUSED_ROUTER_NORM_ENV, "0")).strip().lower()
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

    values = os.environ if environ is None else environ
    mode = str(values.get(FD_GATE_MODE_ENV, "released")).strip().lower()
    if mode not in _VALID_GATE_MODES:
        choices = ", ".join(sorted(_VALID_GATE_MODES))
        raise ValueError(
            f"{FD_GATE_MODE_ENV} must be one of {choices}; got {mode!r}"
        )
    return mode


def full_graph_low_row_policy(
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[str, int]:
    """Return the graph-static execution body for decode batches below compaction."""

    values = os.environ if environ is None else environ
    policy = str(values.get(FD_LOW_ROW_POLICY_ENV, "off")).strip().lower()
    if policy not in _VALID_LOW_ROW_POLICIES:
        choices = ", ".join(sorted(_VALID_LOW_ROW_POLICIES))
        raise ValueError(
            f"{FD_LOW_ROW_POLICY_ENV} must be one of {choices}; got {policy!r}"
        )
    if policy == "native_dense":
        # Lane-2 cut3 item 1 (D-358): the routed MLP is the model's OWN dense
        # feed-forward at every occupancy, plus the projector, selected by the
        # route mask. Measured on A100: dense-all is faster than the
        # count-adaptive kernel at 32-128 rows and within 0.4 ms at 256, while
        # a routed layer is weight-bound below ~208 rows, so exact-count
        # compute cannot pay there. Unbounded by construction — a row bound
        # would reintroduce the very dispatch this policy removes.
        if str(values.get(FD_LOW_ROW_MAX_ROWS_ENV, "") or "").strip():
            raise ValueError(
                f"{FD_LOW_ROW_MAX_ROWS_ENV} must not be set with "
                f"{FD_LOW_ROW_POLICY_ENV}=native_dense (it applies to every "
                "occupancy)"
            )
        return policy, _NATIVE_DENSE_UNBOUNDED_ROWS
    try:
        max_rows = int(values.get(FD_LOW_ROW_MAX_ROWS_ENV, "16"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{FD_LOW_ROW_MAX_ROWS_ENV} must be a positive integer"
        ) from error
    if max_rows <= 0:
        raise ValueError(f"{FD_LOW_ROW_MAX_ROWS_ENV} must be a positive integer")
    if (
        policy == "off"
        and str(values.get(FD_LOW_ROW_MAX_ROWS_ENV, "") or "").strip()
    ):
        raise ValueError(
            f"{FD_LOW_ROW_MAX_ROWS_ENV} requires {FD_LOW_ROW_POLICY_ENV}"
        )
    return policy, max_rows
def full_graph_compact_phases(
    environ: Optional[Mapping[str, str]] = None,
) -> frozenset[str]:
    """Return request phases allowed to use a calibrated compact policy."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_COMPACT_PHASES_ENV, "decode") or "decode")
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
    if not str(os.environ.get("SGLANG_FD_WEIGHTS", "")).strip():
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

    values = os.environ if environ is None else environ
    mode = values.get(FD_EXECUTION_MODE_ENV, FD_EXECUTION_DIRECT_EAGER)
    mode = str(mode or FD_EXECUTION_DIRECT_EAGER).strip().lower()
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
    # Upper token gate (2026-08-20, length-controlled prefill A/B): routed
    # prefill WINS on small passes (gsm8k@10, pass p50 ~1.0K tokens,
    # TTFT −9.6/−13.4% over two reps) and LOSES on large packed passes
    # (coqa@40, pass p50 ~7.0K, TTFT +29%) — the thin-projector cohort
    # lanes eat the savings at large shapes. Passes with effective tokens
    # ABOVE this gate run the dense body. None = no upper gate, byte-
    # identical to the pre-existing sub-threshold-only behavior.
    max_tokens: Optional[int] = None
    # [P8 v2, D-339] Engagement-keyed escape: token thresholds cannot
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
        if self.max_tokens is not None:
            if self.max_tokens <= 0:
                raise ValueError(
                    "regime switch prefill.max_tokens must be positive"
                )
            if self.max_tokens < self.min_tokens:
                raise ValueError(
                    "regime switch prefill.max_tokens must not undercut "
                    "min_tokens (the FD window would be empty)"
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
    # c2 (D-156): require this many CONSECUTIVE sub-exit_rows decode passes before
    # flipping skip->dense, so a transient occupancy dip does not withhold skip
    # (the D-138 switch tax). Default 1 == byte-identical to the un-smoothed band.
    exit_dwell: int = 1
    # Lane-2 cut1 (2026-09-02): KV-volume criterion. When enter_kv_tokens > 0
    # the band is keyed on the decode batch's resident KV tokens
    # (ForwardBatch.seq_lens_sum, a host int -- no device sync) instead of its
    # rows: the CSD3 ladders put the skip body's crossover at ~512 rows x 256
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
class RegimeSwitchConfig(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Top-level regime-switch config with a pinned version."""

    version: int
    prefill: RegimeSwitchPrefillConfig
    decode: RegimeSwitchDecodeConfig

    def validate(self) -> None:
        if self.version != REGIME_SWITCH_CONFIG_VERSION:
            raise ValueError(
                "regime switch config version must be "
                f"{REGIME_SWITCH_CONFIG_VERSION}; got {self.version}"
            )
        self.prefill.validate()
        self.decode.validate()
def fdvp_fused_project_input_enabled():
    return os.environ.get(
        "SGLANG_FD_VP_FUSED_PROJECT_INPUT", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
def fdvp_fused_project_input_shared_storage_enabled():
    return os.environ.get(
        "SGLANG_FD_VP_FUSED_PROJECT_INPUT_SHARED_STORAGE", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
def _fdvp_router_graph_enabled():
    return os.environ.get("SGLANG_FD_VP_ROUTER_GRAPH", "0") == "1"
def _fdvp_timing_enabled():
    return os.environ.get("SGLANG_FD_VP_TRACE_TIMING") == "1"
def fd_parity_trace_target() -> str:
    return os.environ.get("SGLANG_FD_PARITY_TRACE_RID", "")
def resolve_full_graph_skipper(
    environ: Optional[Mapping[str, str]] = None,
) -> FullGraphSkipperAdapter:
    """Resolve the process policy and fail before serving on unsupported names."""

    values = os.environ if environ is None else environ
    name = full_graph_skipper_name(values)
    mock_config_present = any(
        str(values.get(key, "")).strip() for key in _MOCK_CONFIG_ENVS
    )
    if name != DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER:
        if mock_config_present:
            raise ValueError(
                "deterministic mock settings require "
                f"{FULL_GRAPH_SKIPPER_ENV}={DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER}"
            )
        return _ADAPTERS[name]

    missing = [
        key
        for key in (
            FULL_GRAPH_MOCK_TOKEN_SKIP_RATE_ENV,
            FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO_ENV,
        )
        if not str(values.get(key, "")).strip()
    ]
    if missing:
        raise ValueError("deterministic mock requires " + ", ".join(missing))
    return _deterministic_mock_adapter(*_parse_mock_config(values))
def forced_route_action(value: Optional[bool]) -> Optional[LogicalAction]:
    if value is None:
        return None
    return LogicalAction.RUN if value else LogicalAction.PROJECT_ONLY
def flexidepth_active_phases(
    environ: Optional[Mapping[str, str]] = None,
) -> frozenset[str]:
    """Return the explicitly enabled request phases.

    ``both`` preserves the released FlexiDepth behavior and is the default.
    Phase isolation exists only to build decode-only and prefill-only systems
    comparisons without changing the route equations inside an active phase.
    """

    values = os.environ if environ is None else environ
    value = str(values.get(FD_ACTIVE_PHASES_ENV, "both") or "both").strip().lower()
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
    known = frozenset(
        (
            *_ADAPTERS,
            DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER,
        )
    )
    if name not in known:
        choices = ", ".join(sorted(known))
        raise ValueError(
            f"{FULL_GRAPH_SKIPPER_ENV} must be one of {choices}; got {name!r}"
        )
    return name
def regime_switch_config(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[RegimeSwitchConfig]:
    """Resolve the fail-closed regime-switch config from the environment.

    Absent, empty, or ``"off"`` yields ``None`` (switch off = current
    byte-identical behavior).  Any other value is strict JSON decoded into a
    :class:`RegimeSwitchConfig` (unknown keys rejected natively) and validated;
    malformed configs raise ``ValueError`` before serving.
    """

    values = os.environ if environ is None else environ
    raw = str(values.get(REGIME_SWITCH_ENV, "") or "").strip()
    if not raw or raw.lower() == "off":
        return None
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
