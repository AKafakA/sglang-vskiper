"""Cohort packing and mapped-linear primitives.

Shared building blocks for the compact and virtual cohort paths: route-map
construction, packed-prefix gathers, and the mapped linear used when routed rows
are addressed through an index vector rather than compacted.
"""

from __future__ import annotations

import triton.language as tl
import triton
import msgspec
import os
from typing import Any, Callable, Optional, Sequence
import torch
import torch.nn.functional as F
from sglang.srt.utils.async_probe import maybe_detect_oob
from sglang.srt.vpipe.kv_commit import (
    _fdvp_trace_enabled,
)
from sglang.srt.vpipe.kv_commit import (
    _device_key,
    _trace_counter,
)
from sglang.srt.vpipe.common import (
    FDLayerRoute,
    _FD_PARITY_EPOCHS,
    fd_parity_trace_target,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_state,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_record_cache_counter,
)


_FDVP_FULL_METADATA_STALE = False
# Import the CANONICAL containers rather than defining a second pair. They were
# duplicated here, so tracker_for_device() (the only producer) registered into
# cohort's dict while drain_request_kv_work()/reset_kv_readiness_trackers() in
# kv_commit.py iterated kv_commit's -- writer and reader on different objects.
# Same defect class as the _graph_lifecycle_depth split fixed earlier.
# Safe to bind by reference: both are mutated in place (setdefault/pop/clear)
# and never rebound.
_MATMUL_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32},
        num_stages=3,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
        num_stages=4,
        num_warps=8,
    ),
]
@triton.jit
def _build_row_map_kernel(
    active_ptr,
    row_map_ptr,
    count_ptr,
    row_count: tl.constexpr,
    block_rows: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    valid = rows < row_count
    active = tl.load(active_ptr + rows, mask=valid, other=0).to(tl.int1)
    counter_lanes = tl.zeros((block_rows,), dtype=tl.int32)
    slots = tl.atomic_add(
        count_ptr + counter_lanes, 1, mask=valid & active
    )
    tl.store(row_map_ptr + slots, rows, mask=valid & active)
@triton.jit
def _pack_cohort_kernel(
    hidden_ptr,
    weights_ptr,
    row_map_ptr,
    count_ptr,
    packed_hidden_ptr,
    packed_weights_ptr,
    hidden_size: tl.constexpr,
    row_offset: tl.constexpr,
    block_hidden: tl.constexpr,
):
    slot = tl.program_id(0)
    columns = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
    count = tl.load(count_ptr)
    active = slot + row_offset < count
    source_row = tl.load(
        row_map_ptr + slot + row_offset, mask=active, other=0
    ).to(tl.int64)
    hidden = tl.load(
        hidden_ptr + source_row * hidden_size + columns,
        mask=active & (columns < hidden_size),
        other=0.0,
    )
    tl.store(
        packed_hidden_ptr + slot * hidden_size + columns,
        hidden,
        mask=columns < hidden_size,
    )
    weight = tl.load(weights_ptr + source_row, mask=active, other=0.0)
    tl.store(
        packed_weights_ptr + slot,
        weight,
        mask=tl.program_id(1) == 0,
    )
@triton.jit
def _scatter_cohort_kernel(
    packed_output_ptr,
    row_map_ptr,
    count_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    row_offset: tl.constexpr,
    block_hidden: tl.constexpr,
):
    slot = tl.program_id(0)
    columns = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
    count = tl.load(count_ptr)
    active = slot + row_offset < count
    destination_row = tl.load(
        row_map_ptr + slot + row_offset, mask=active, other=0
    ).to(tl.int64)
    values = tl.load(
        packed_output_ptr + slot * hidden_size + columns,
        mask=active & (columns < hidden_size),
        other=0.0,
    )
    tl.store(
        output_ptr + destination_row * hidden_size + columns,
        values,
        mask=active & (columns < hidden_size),
    )
@triton.jit
def _scatter_weighted_cohort_kernel(
    packed_output_ptr,
    weights_ptr,
    row_map_ptr,
    count_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    row_offset: tl.constexpr,
    block_hidden: tl.constexpr,
):
    slot = tl.program_id(0)
    columns = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
    count = tl.load(count_ptr)
    active = slot + row_offset < count
    destination_row = tl.load(
        row_map_ptr + slot + row_offset, mask=active, other=0
    ).to(tl.int64)
    values = tl.load(
        packed_output_ptr + slot * hidden_size + columns,
        mask=active & (columns < hidden_size),
        other=0.0,
    )
    weight = tl.load(weights_ptr + destination_row, mask=active, other=0.0)
    tl.store(
        output_ptr + destination_row * hidden_size + columns,
        values * weight,
        mask=active & (columns < hidden_size),
    )
@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _mapped_linear_kernel(
    input_ptr,
    weight_ptr,
    row_map_ptr,
    count_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_M + ROW_OFFSET
    count = tl.load(count_ptr).to(tl.int32)

    if start_m < count:
        slots = start_m + tl.arange(0, BLOCK_M)
        columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        active_rows = slots < count
        physical_rows = tl.load(
            row_map_ptr + slots, mask=active_rows, other=0
        ).to(tl.int64)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in range(0, K, BLOCK_K):
            offsets_k = start_k + tl.arange(0, BLOCK_K)
            values = tl.load(
                input_ptr
                + physical_rows[:, None] * K
                + offsets_k[None, :],
                mask=active_rows[:, None] & (offsets_k[None, :] < K),
                other=0.0,
            )
            weight = tl.load(
                weight_ptr
                + columns[None, :] * K
                + offsets_k[:, None],
                mask=(offsets_k[:, None] < K) & (columns[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(values, weight)

        tl.store(
            output_ptr
            + physical_rows[:, None] * N
            + columns[None, :],
            accumulator,
            mask=active_rows[:, None] & (columns[None, :] < N),
        )
@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _mapped_gate_up_kernel(
    hidden_ptr,
    gate_up_weight_ptr,
    row_map_ptr,
    count_ptr,
    packed_gate_up_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    count = tl.load(count_ptr).to(tl.int32)

    # The grid is fixed at the graph bucket, while count changes every replay.
    # Uniform control flow prevents inactive row tiles from issuing the GEMM.
    if start_m < count:
        slots = start_m + tl.arange(0, BLOCK_M)
        columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        active_rows = slots < count
        source_rows = tl.load(
            row_map_ptr + slots, mask=active_rows, other=0
        ).to(tl.int64)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in range(0, K, BLOCK_K):
            offsets_k = start_k + tl.arange(0, BLOCK_K)
            hidden = tl.load(
                hidden_ptr
                + source_rows[:, None] * K
                + offsets_k[None, :],
                mask=active_rows[:, None] & (offsets_k[None, :] < K),
                other=0.0,
            )
            weight = tl.load(
                gate_up_weight_ptr
                + columns[None, :] * K
                + offsets_k[:, None],
                mask=(offsets_k[:, None] < K) & (columns[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(hidden, weight)

        tl.store(
            packed_gate_up_ptr + slots[:, None] * N + columns[None, :],
            accumulator,
            mask=active_rows[:, None] & (columns[None, :] < N),
        )
@triton.autotune(configs=_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _mapped_swiglu_down_kernel(
    packed_gate_up_ptr,
    down_weight_ptr,
    row_map_ptr,
    count_ptr,
    branch_weights_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    INVERT_WEIGHT: tl.constexpr,
    SCALE_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    count = tl.load(count_ptr).to(tl.int32)

    if start_m < count:
        slots = start_m + tl.arange(0, BLOCK_M)
        columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        active_rows = slots < count
        destination_rows = tl.load(
            row_map_ptr + slots, mask=active_rows, other=0
        ).to(tl.int64)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start_k in range(0, K, BLOCK_K):
            offsets_k = start_k + tl.arange(0, BLOCK_K)
            gate = tl.load(
                packed_gate_up_ptr
                + slots[:, None] * (2 * K)
                + offsets_k[None, :],
                mask=active_rows[:, None] & (offsets_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                packed_gate_up_ptr
                + slots[:, None] * (2 * K)
                + K
                + offsets_k[None, :],
                mask=active_rows[:, None] & (offsets_k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            activated = (gate * tl.sigmoid(gate) * up).to(
                packed_gate_up_ptr.dtype.element_ty
            )
            weight = tl.load(
                down_weight_ptr
                + columns[None, :] * K
                + offsets_k[:, None],
                mask=(offsets_k[:, None] < K) & (columns[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(activated, weight)

        if SCALE_WEIGHT:
            route_weight = tl.load(
                branch_weights_ptr + destination_rows,
                mask=active_rows,
                other=0.0,
            ).to(tl.float32)
            if INVERT_WEIGHT:
                route_weight = 1.0 - route_weight
            accumulator *= route_weight[:, None]
        # else: hard_mask gate (v1.5) -- hard selection by the row map, no w scaling
        tl.store(
            output_ptr + destination_rows[:, None] * N + columns[None, :],
            accumulator,
            mask=active_rows[:, None] & (columns[None, :] < N),
        )
@triton.jit
def _begin_route_evidence_dispatch_kernel(
    valid_rows_ptr,
    route_digest_ptr,
    readiness_ptr,
    row_count: tl.constexpr,
    block_rows: tl.constexpr,
):
    row = tl.arange(0, block_rows)
    valid = (row < row_count) & tl.load(
        valid_rows_ptr + row, mask=row < row_count, other=0
    ).to(tl.int1)
    metadata_rows = tl.sum(valid.to(tl.int64), axis=0)

    dispatch_ordinal = tl.load(route_digest_ptr) + 1
    tl.store(route_digest_ptr, dispatch_ordinal)
    previous = tl.load(route_digest_ptr + 5)
    tl.store(route_digest_ptr + 5, previous + metadata_rows)

    readiness_dispatch = tl.load(readiness_ptr) + 1
    tl.store(readiness_ptr, readiness_dispatch)
    previous = tl.load(readiness_ptr + 4)
    tl.store(readiness_ptr + 4, previous + metadata_rows)
@triton.jit
def _accumulate_per_layer_route_evidence_kernel(
    actions_ptr,
    valid_rows_ptr,
    layer_ids_ptr,
    request_slots_ptr,
    logical_request_ids_ptr,
    token_epochs_ptr,
    cache_positions_ptr,
    compact_specs_ptr,
    route_counters_ptr,
    layer_counters_ptr,
    route_digest_ptr,
    readiness_ptr,
    row_count: tl.constexpr,
    capacity_multiple: tl.constexpr,
    compact_active: tl.constexpr,
    logical_digest: tl.constexpr,
    block_rows: tl.constexpr,
):
    layer = tl.program_id(0)
    row = tl.arange(0, block_rows)
    row_valid = row < row_count
    valid = tl.load(valid_rows_ptr + row, mask=row_valid, other=0).to(tl.int1)
    action = tl.load(
        actions_ptr + layer * row_count + row,
        mask=row_valid,
        other=0,
    ).to(tl.int1)
    valid_count = tl.sum((row_valid & valid).to(tl.int64), axis=0)
    run_count = tl.sum((row_valid & valid & action).to(tl.int64), axis=0)

    tl.atomic_add(route_counters_ptr, valid_count)
    tl.atomic_add(route_counters_ptr + 1, run_count)
    tl.atomic_add(route_counters_ptr + 2, valid_count - run_count)

    counter_base = layer * 3
    previous = tl.load(layer_counters_ptr + counter_base)
    tl.store(layer_counters_ptr + counter_base, previous + valid_count)
    previous = tl.load(layer_counters_ptr + counter_base + 1)
    tl.store(layer_counters_ptr + counter_base + 1, previous + run_count)
    previous = tl.load(layer_counters_ptr + counter_base + 2)
    tl.store(
        layer_counters_ptr + counter_base + 2,
        previous + valid_count - run_count,
    )

    layer_id = tl.load(layer_ids_ptr + layer).to(tl.int64)
    request_slot = tl.load(
        request_slots_ptr + row, mask=row_valid, other=0
    ).to(tl.int64)
    logical_request_id = tl.load(
        logical_request_ids_ptr + row, mask=row_valid, other=0
    ).to(tl.int64)
    token_epoch = tl.load(token_epochs_ptr + row, mask=row_valid, other=0).to(
        tl.int64
    )
    cache_position = tl.load(
        cache_positions_ptr + row, mask=row_valid, other=0
    ).to(tl.int64)
    row_i64 = row.to(tl.int64)
    included = row_valid & valid
    if logical_digest:
        metadata_code = (
            (layer_id + 1) * 1_000_003
            + 1_000_033
            + (logical_request_id + 1) * 1_000_037
            + (token_epoch + 1) * 1_000_081
        )
        route_cell_weight = (
            (layer_id + 1) * 65_537
            + 65_539
            + (logical_request_id + 1) * 65_543
            + (token_epoch + 1) * 65_551
        )
    else:
        metadata_code = (
            (layer_id + 1) * 1_000_003
            + (row_i64 + 1) * 1_000_033
            + (request_slot + 1) * 1_000_037
            + (token_epoch + 1) * 1_000_081
            + (cache_position + 1) * 1_000_099
        )
        route_cell_weight = (
            (layer_id + 1) * 65_537 + (row_i64 + 1) * 65_539
        )
    action_payload = tl.where(
        included, metadata_code * 2 + action.to(tl.int64), 0
    )
    action_digest_sum = tl.sum(action_payload, axis=0)
    action_digest_weighted = tl.sum(
        action_payload * route_cell_weight, axis=0
    )
    dispatch_ordinal = tl.load(route_digest_ptr)
    tl.atomic_add(route_digest_ptr + 1, valid_count)
    tl.atomic_add(route_digest_ptr + 2, run_count)
    tl.atomic_add(route_digest_ptr + 3, action_digest_sum)
    if logical_digest:
        tl.atomic_add(route_digest_ptr + 4, action_digest_weighted)
    else:
        tl.atomic_add(
            route_digest_ptr + 4,
            action_digest_weighted + action_digest_sum * dispatch_ordinal,
        )

    readiness_payload = tl.where(
        included,
        (request_slot + 1) * 1_000_003
        + (token_epoch + 1) * 1_000_033
        + (layer_id + 1) * 1_000_037
        + (cache_position + 1) * 1_000_081,
        0,
    )
    readiness_digest_sum = tl.sum(readiness_payload, axis=0)
    readiness_cell_weight = (
        (layer_id + 1) * 65_537 + (row_i64 + 1) * 65_539
    )
    readiness_digest_weighted = tl.sum(
        readiness_payload * readiness_cell_weight, axis=0
    )
    readiness_dispatch = tl.load(readiness_ptr)
    tl.atomic_add(readiness_ptr + 1, valid_count)
    tl.atomic_add(readiness_ptr + 2, valid_count)
    tl.atomic_add(readiness_ptr + 5, readiness_digest_sum)
    tl.atomic_add(
        readiness_ptr + 6,
        readiness_digest_weighted
        + readiness_digest_sum * readiness_dispatch,
    )

    if compact_active:
        spec_base = layer * 3
        mode = tl.load(compact_specs_ptr + spec_base).to(tl.int32)
        run_fraction = tl.load(compact_specs_ptr + spec_base + 1)
        project_fraction = tl.load(compact_specs_ptr + spec_base + 2)
        run_capacity = (
            tl.ceil(row_count * run_fraction / capacity_multiple).to(tl.int64)
            * capacity_multiple
        )
        project_capacity = (
            tl.ceil(row_count * project_fraction / capacity_multiple).to(
                tl.int64
            )
            * capacity_multiple
        )
        run_capacity = tl.minimum(row_count, tl.maximum(1, run_capacity))
        project_capacity = tl.minimum(
            row_count, tl.maximum(1, project_capacity)
        )
        fallback_has_work = (run_capacity < row_count) | (
            project_capacity < row_count
        )
        policy_active = (mode == 1) | ((mode == 2) & fallback_has_work)
        run_overflow = tl.where(
            policy_active & (run_fraction > 0),
            tl.maximum(0, run_count - run_capacity),
            0,
        )
        project_count = valid_count - run_count
        project_overflow = tl.where(
            policy_active & (project_fraction > 0),
            tl.maximum(0, project_count - project_capacity),
            0,
        )
        tl.atomic_add(
            route_counters_ptr + 3,
            tl.where(policy_active, valid_count, 0),
        )
        tl.atomic_add(route_counters_ptr + 4, run_overflow)
        tl.atomic_add(route_counters_ptr + 5, project_overflow)
def accumulate_fused_route_evidence(
    *,
    actions: torch.Tensor,
    valid_rows: torch.Tensor,
    layer_ids: torch.Tensor,
    request_slots: torch.Tensor,
    logical_request_ids: torch.Tensor | None,
    token_epochs: torch.Tensor,
    cache_positions: torch.Tensor,
    compact_specs: torch.Tensor,
    route_counters: torch.Tensor,
    layer_counters: torch.Tensor,
    route_digest_counters: torch.Tensor,
    readiness_counters: torch.Tensor,
    compact_active: bool,
    capacity_multiple: int,
) -> None:
    """Accumulate complete replay evidence with two fixed-shape kernels."""

    tensors = (
        actions,
        valid_rows,
        layer_ids,
        request_slots,
        (
            logical_request_ids
            if logical_request_ids is not None
            else request_slots
        ),
        token_epochs,
        cache_positions,
        compact_specs,
        route_counters,
        layer_counters,
        route_digest_counters,
        readiness_counters,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused route evidence requires CUDA tensors")
    if actions.ndim != 2 or valid_rows.shape != (actions.shape[1],):
        raise ValueError("fused route evidence has incompatible action rows")
    layer_count, row_count = map(int, actions.shape)
    if layer_ids.shape != (layer_count,):
        raise ValueError("fused route evidence has incompatible layer metadata")
    if any(
        value.shape != (row_count,)
        for value in (request_slots, token_epochs, cache_positions)
    ):
        raise ValueError("fused route evidence has incompatible row metadata")
    if logical_request_ids is not None and logical_request_ids.shape != (
        row_count,
    ):
        raise ValueError(
            "fused route evidence has incompatible logical request IDs"
        )
    if compact_specs.shape != (layer_count, 3):
        raise ValueError("fused route evidence has incompatible compact policies")
    if route_counters.shape != (6,):
        raise ValueError("fused route evidence requires six route counters")
    if layer_counters.shape != (layer_count, 3):
        raise ValueError("fused route evidence has incompatible layer counters")
    if route_digest_counters.shape != (6,):
        raise ValueError("fused route evidence requires six digest counters")
    if readiness_counters.shape != (7,):
        raise ValueError("fused route evidence requires seven readiness counters")
    if row_count <= 0 or layer_count <= 0:
        raise ValueError("fused route evidence requires nonempty actions")

    block_rows = triton.next_power_of_2(row_count)
    if block_rows > 1024:
        raise ValueError("fused route evidence currently supports at most 1024 rows")
    _begin_route_evidence_dispatch_kernel[(1,)](
        valid_rows,
        route_digest_counters,
        readiness_counters,
        row_count=row_count,
        block_rows=block_rows,
        num_warps=4,
    )
    _accumulate_per_layer_route_evidence_kernel[(layer_count,)](
        actions,
        valid_rows,
        layer_ids,
        request_slots,
        (
            logical_request_ids
            if logical_request_ids is not None
            else request_slots
        ),
        token_epochs,
        cache_positions,
        compact_specs,
        route_counters,
        layer_counters,
        route_digest_counters,
        readiness_counters,
        row_count=row_count,
        capacity_multiple=capacity_multiple,
        compact_active=compact_active,
        logical_digest=logical_request_ids is not None,
        block_rows=block_rows,
        num_warps=4,
    )
def build_row_map(active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one fixed-size device row map and its device count."""

    if not active.is_cuda:
        raise ValueError("compact cohort row maps require CUDA tensors")
    if active.ndim != 1:
        raise ValueError("compact cohort row masks must be 1D tensors")
    rows = int(active.numel())
    count = torch.zeros(1, dtype=torch.int32, device=active.device)
    row_map = torch.empty(rows, dtype=torch.int32, device=active.device)
    block_rows = 256
    _build_row_map_kernel[(triton.cdiv(rows, block_rows),)](
        active,
        row_map,
        count,
        row_count=rows,
        block_rows=block_rows,
    )
    return row_map, count
def pack_cohort(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    row_map: torch.Tensor,
    count: torch.Tensor,
    *,
    capacity: int,
    row_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one fixed-capacity cohort directly from its device row map."""

    if hidden.ndim != 2 or weights.shape != (hidden.shape[0], 1):
        raise ValueError("compact cohort hidden/weight shapes are incompatible")
    if capacity <= 0 or row_offset < 0 or capacity + row_offset > len(row_map):
        raise ValueError("compact cohort capacity is outside its row map")
    hidden_size = int(hidden.shape[1])
    packed_hidden = torch.empty(
        (capacity, hidden_size), dtype=hidden.dtype, device=hidden.device
    )
    packed_weights = torch.empty(
        (capacity, 1), dtype=weights.dtype, device=weights.device
    )
    block_hidden = 256
    _pack_cohort_kernel[
        (capacity, triton.cdiv(hidden_size, block_hidden))
    ](
        hidden,
        weights,
        row_map,
        count,
        packed_hidden,
        packed_weights,
        hidden_size=hidden_size,
        row_offset=row_offset,
        block_hidden=block_hidden,
    )
    return packed_hidden, packed_weights
def scatter_cohort(
    output: torch.Tensor,
    packed_output: torch.Tensor,
    row_map: torch.Tensor,
    count: torch.Tensor,
    *,
    row_offset: int = 0,
) -> None:
    """Scatter one fixed-capacity branch output through its device row map."""

    if output.ndim != 2 or packed_output.ndim != 2:
        raise ValueError("compact cohort outputs must be 2D")
    if output.shape[1] != packed_output.shape[1]:
        raise ValueError("compact cohort output hidden sizes do not match")
    capacity, hidden_size = (int(value) for value in packed_output.shape)
    if row_offset < 0 or capacity + row_offset > len(row_map):
        raise ValueError("compact cohort scatter is outside its row map")
    block_hidden = 256
    _scatter_cohort_kernel[
        (capacity, triton.cdiv(hidden_size, block_hidden))
    ](
        packed_output,
        row_map,
        count,
        output,
        hidden_size=hidden_size,
        row_offset=row_offset,
        block_hidden=block_hidden,
    )
def scatter_weighted_cohort(
    output: torch.Tensor,
    packed_output: torch.Tensor,
    weights: torch.Tensor,
    row_map: torch.Tensor,
    count: torch.Tensor,
    *,
    row_offset: int = 0,
) -> None:
    """Apply source-row weights while scattering a fixed-capacity cohort."""

    if output.ndim != 2 or packed_output.ndim != 2:
        raise ValueError("compact cohort outputs must be 2D")
    if output.shape[1] != packed_output.shape[1]:
        raise ValueError("compact cohort output hidden sizes do not match")
    if weights.shape != (output.shape[0], 1):
        raise ValueError("compact cohort weights must align with output rows")
    capacity, hidden_size = (int(value) for value in packed_output.shape)
    if row_offset < 0 or capacity + row_offset > len(row_map):
        raise ValueError("compact cohort scatter is outside its row map")
    block_hidden = 256
    _scatter_weighted_cohort_kernel[
        (capacity, triton.cdiv(hidden_size, block_hidden))
    ](
        packed_output,
        weights,
        row_map,
        count,
        output,
        hidden_size=hidden_size,
        row_offset=row_offset,
        block_hidden=block_hidden,
    )
def _validate_mapped_swiglu(
    hidden: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    row_map: torch.Tensor,
    count: torch.Tensor,
    branch_weights: torch.Tensor,
    output: torch.Tensor,
) -> tuple[int, int, int]:
    tensors = (
        hidden,
        gate_up_weight,
        down_weight,
        row_map,
        count,
        branch_weights,
        output,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("VP virtual cohorts require CUDA tensors")
    if any(tensor.device != hidden.device for tensor in tensors):
        raise ValueError("VP virtual cohort tensors must share one CUDA device")
    if hidden.ndim != 2 or output.ndim != 2:
        raise ValueError("VP virtual cohort hidden and output tensors must be 2D")
    if hidden.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("VP virtual cohorts support only FP16 and BF16")
    if gate_up_weight.dtype != hidden.dtype or down_weight.dtype != hidden.dtype:
        raise ValueError("VP virtual cohort weights must match the hidden dtype")
    if output.dtype != hidden.dtype:
        raise ValueError("VP virtual cohort output must match the hidden dtype")
    if not hidden.is_contiguous() or not output.is_contiguous():
        raise ValueError(
            "VP virtual cohort hidden and output tensors must be contiguous"
        )
    if not gate_up_weight.is_contiguous() or not down_weight.is_contiguous():
        raise ValueError("VP virtual cohort weights must be contiguous")
    rows, hidden_size = (int(value) for value in hidden.shape)
    if output.shape != (rows, hidden_size):
        raise ValueError("VP virtual cohort output shape must match hidden")
    if gate_up_weight.ndim != 2 or gate_up_weight.shape[1] != hidden_size:
        raise ValueError("VP virtual cohort gate/up weight has an invalid shape")
    gate_up_size = int(gate_up_weight.shape[0])
    if gate_up_size % 2:
        raise ValueError("VP virtual cohort gate/up width must be even")
    intermediate_size = gate_up_size // 2
    if down_weight.shape != (hidden_size, intermediate_size):
        raise ValueError("VP virtual cohort down weight has an invalid shape")
    if row_map.ndim != 1 or row_map.numel() != rows:
        raise ValueError("VP virtual cohort row map must cover the graph bucket")
    if row_map.dtype not in {torch.int32, torch.int64}:
        raise ValueError("VP virtual cohort row map must be int32 or int64")
    if not row_map.is_contiguous():
        raise ValueError("VP virtual cohort row map must be contiguous")
    if count.numel() != 1 or count.dtype not in {torch.int32, torch.int64}:
        raise ValueError("VP virtual cohort count must be one integer tensor")
    if not count.is_contiguous():
        raise ValueError("VP virtual cohort count must be contiguous")
    if branch_weights.numel() != rows:
        raise ValueError("VP virtual cohort weights must provide one scalar per row")
    if not branch_weights.is_contiguous():
        raise ValueError("VP virtual cohort route weights must be contiguous")
    return rows, hidden_size, intermediate_size
def mapped_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    row_map: torch.Tensor,
    count: torch.Tensor,
    output: torch.Tensor,
    *,
    row_offset: int = 0,
) -> torch.Tensor:
    """Apply one bias-free linear layer through a dynamic device row map."""

    tensors = (input_tensor, weight, row_map, count, output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("VP mapped linear requires CUDA tensors")
    if any(tensor.device != input_tensor.device for tensor in tensors):
        raise ValueError("VP mapped linear tensors must share one CUDA device")
    if input_tensor.ndim != 2 or output.ndim != 2 or weight.ndim != 2:
        raise ValueError("VP mapped linear input, weight, and output must be 2D")
    if input_tensor.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("VP mapped linear supports only FP16 and BF16")
    if weight.dtype != input_tensor.dtype or output.dtype != input_tensor.dtype:
        raise ValueError("VP mapped linear tensors must share one floating dtype")
    if not input_tensor.is_contiguous() or not output.is_contiguous():
        raise ValueError("VP mapped linear input and output must be contiguous")
    if not weight.is_contiguous():
        raise ValueError("VP mapped linear weight must be contiguous")
    rows, input_size = (int(value) for value in input_tensor.shape)
    output_size = int(weight.shape[0])
    if weight.shape[1] != input_size or output.shape != (rows, output_size):
        raise ValueError("VP mapped linear shapes are incompatible")
    if row_map.shape != (rows,) or row_map.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise ValueError("VP mapped linear row map has an invalid shape or dtype")
    if not row_map.is_contiguous():
        raise ValueError("VP mapped linear row map must be contiguous")
    if count.numel() != 1 or count.dtype not in {torch.int32, torch.int64}:
        raise ValueError("VP mapped linear count must be one integer tensor")
    if not count.is_contiguous():
        raise ValueError("VP mapped linear count must be contiguous")
    if row_offset < 0 or row_offset > rows:
        raise ValueError("VP mapped linear row offset is outside its row map")

    mapped_rows = rows - row_offset
    if mapped_rows == 0:
        return output

    def grid(meta):
        return (
            triton.cdiv(mapped_rows, meta["BLOCK_M"]),
            triton.cdiv(output_size, meta["BLOCK_N"]),
        )

    _mapped_linear_kernel[grid](
        input_tensor,
        weight,
        row_map,
        count,
        output,
        M=mapped_rows,
        N=output_size,
        K=input_size,
        ROW_OFFSET=row_offset,
    )
    return output
def mapped_swiglu(
    hidden: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    row_map: torch.Tensor,
    count: torch.Tensor,
    branch_weights: torch.Tensor,
    output: torch.Tensor,
    *,
    invert_weight: bool = False,
    scale_weight: bool = True,
) -> torch.Tensor:
    """Execute one exact routed SwiGLU branch through a device row map.

    ``scale_weight=False`` = the hard_mask gate (v1.5): no route-weight multiply.

    ``row_map[:count]`` identifies source and destination rows. Inactive output
    rows are left untouched so this operation can overwrite a dominant branch
    or compose two disjoint virtual cohorts in one destination tensor.
    """

    rows, hidden_size, intermediate_size = _validate_mapped_swiglu(
        hidden,
        gate_up_weight,
        down_weight,
        row_map,
        count,
        branch_weights,
        output,
    )
    gate_up_size = 2 * intermediate_size
    packed_gate_up = torch.empty(
        (rows, gate_up_size), dtype=hidden.dtype, device=hidden.device
    )

    def gate_up_grid(meta):
        return (
            triton.cdiv(rows, meta["BLOCK_M"]),
            triton.cdiv(gate_up_size, meta["BLOCK_N"]),
        )

    _mapped_gate_up_kernel[gate_up_grid](
        hidden,
        gate_up_weight,
        row_map,
        count,
        packed_gate_up,
        M=rows,
        N=gate_up_size,
        K=hidden_size,
    )

    def down_grid(meta):
        return (
            triton.cdiv(rows, meta["BLOCK_M"]),
            triton.cdiv(hidden_size, meta["BLOCK_N"]),
        )

    _mapped_swiglu_down_kernel[down_grid](
        packed_gate_up,
        down_weight,
        row_map,
        count,
        branch_weights,
        output,
        M=rows,
        N=hidden_size,
        K=intermediate_size,
        INVERT_WEIGHT=bool(invert_weight),
        SCALE_WEIGHT=bool(scale_weight),
    )
    return output
def fd_parity_trace_advance(layer_id: int, forward_batch) -> None:
    target = fd_parity_trace_target()
    rids = getattr(forward_batch, "rids", None) or ()
    mode = getattr(forward_batch, "forward_mode", None)
    if (
        target
        and target in rids
        and layer_id == 31
        and mode is not None
        and mode.is_decode()
    ):
        _FD_PARITY_EPOCHS[target] = int(_FD_PARITY_EPOCHS.get(target, 0)) + 1
def device_key(device=None) -> int:
    if isinstance(device, str):
        device = torch.device(device)
    if device is None or getattr(device, "type", None) != "cuda":
        return -1
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return int(index)
def _current_stream(device):
    if device is None or getattr(device, "type", None) != "cuda":
        return None
    return torch.cuda.current_stream(device)


@triton.jit
def _fd_batched_commit_kv_kernel(
    table_ptr,  # int64 [L, 4]: k_src, v_src, k_dst, v_dst byte addresses
    row_stride_ptr,  # int64 [L, 2]: k/v source row strides (elements)
    loc_ptr,  # [N] cache locations, shared across layers within a step
    mask_ptr,  # bool [L, N] per-layer PROJECT masks (layer-contiguous)
    L: tl.constexpr,
    N: tl.constexpr,
    HD_K: tl.constexpr,  # flat elements per K row (heads * head_dim)
    HD_V: tl.constexpr,  # flat elements per V row
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= L * N:
        return
    layer = pid // N
    row = pid % N
    if tl.load(mask_ptr + layer * N + row) == 0:
        return
    loc = tl.load(loc_ptr + row).to(tl.int64)
    k_src = tl.load(table_ptr + layer * 4 + 0).to(tl.pointer_type(tl.uint16))
    v_src = tl.load(table_ptr + layer * 4 + 1).to(tl.pointer_type(tl.uint16))
    k_dst = tl.load(table_ptr + layer * 4 + 2).to(tl.pointer_type(tl.uint16))
    v_dst = tl.load(table_ptr + layer * 4 + 3).to(tl.pointer_type(tl.uint16))
    k_row_stride = tl.load(row_stride_ptr + layer * 2 + 0)
    v_row_stride = tl.load(row_stride_ptr + layer * 2 + 1)
    for chunk in range(tl.cdiv(HD_K, CHUNK)):
        idx = chunk * CHUNK + tl.arange(0, CHUNK)
        live = idx < HD_K
        key = tl.load(k_src + row * k_row_stride + idx, mask=live)
        tl.store(k_dst + loc * HD_K + idx, key, mask=live)
    for chunk in range(tl.cdiv(HD_V, CHUNK)):
        idx = chunk * CHUNK + tl.arange(0, CHUNK)
        live = idx < HD_V
        value = tl.load(v_src + row * v_row_stride + idx, mask=live)
        tl.store(v_dst + loc * HD_V + idx, value, mask=live)


class BatchedCommitPlan(msgspec.Struct, frozen=True):
    """Capture-frozen launch arguments for the batched commit kernel."""

    table: torch.Tensor  # int64 [L, 4] device
    row_strides: torch.Tensor  # int64 [L, 2] device
    cache_locations: torch.Tensor  # captured out_cache_loc buffer
    mask_backing: torch.Tensor  # bool [L, N], layer-contiguous
    num_layers: int
    num_tokens: int
    hd_k: int
    hd_v: int
    total_slots: int  # pool size + page_size: the OOB bound for locations
def _require_flat_rows(view: torch.Tensor, name: str) -> int:
    """Return the row stride of a (N, H, D) view whose rows are flat."""

    if view.dim() != 3:
        raise RuntimeError(f"batched K/V commit: {name} must be 3-D")
    if view.stride(2) != 1 or view.stride(1) != view.shape[2]:
        raise RuntimeError(
            f"batched K/V commit: {name} rows are not flat-contiguous"
        )
    return int(view.stride(0))
def build_batched_commit_plan(
    *,
    routed_layers: Sequence[int],
    llama: Any,
    kv_pool: Any,
    repair_k_buffers: Sequence[torch.Tensor],
    repair_v_buffers: Sequence[torch.Tensor],
    mask_backing: torch.Tensor,
    cache_locations: torch.Tensor,
    num_tokens: int,
) -> BatchedCommitPlan:
    """Validate every precondition and freeze the launch tables.

    Raises instead of degrading: any layout this builder cannot prove
    equivalent to the per-layer loop is a configuration error, not a
    fallback case.
    """

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    if not routed_layers:
        raise RuntimeError("batched K/V commit requires routed layers")
    # Review finding 3: prove the pool IS the plain-NHD _store_kv_layer
    # behavior being replaced — exact type (no subclass overrides) plus
    # the plain layout string; store_dtype == dtype below excludes
    # quantized stores.
    if type(kv_pool) is not MHATokenToKVPool:
        raise RuntimeError(
            "batched K/V commit requires exactly MHATokenToKVPool, got "
            f"{type(kv_pool).__name__}"
        )
    if kv_pool.kv_cache_layout != "nhd":
        raise RuntimeError(
            "batched K/V commit requires the plain NHD layout, got "
            f"{kv_pool.kv_cache_layout!r}"
        )
    if len(repair_k_buffers) != len(routed_layers) or len(
        repair_v_buffers
    ) != len(routed_layers):
        raise RuntimeError(
            "batched K/V commit requires one repair K/V view per routed layer"
        )
    if kv_pool.store_dtype != kv_pool.dtype:
        raise RuntimeError(
            "batched K/V commit does not support store-dtype quantization"
        )
    if getattr(kv_pool, "use_hnd", False):
        raise RuntimeError("batched K/V commit requires the NHD cache layout")
    itemsize = torch.empty((), dtype=kv_pool.dtype).element_size()
    if itemsize != 2:
        raise RuntimeError(
            "batched K/V commit supports 2-byte element types only"
        )
    if mask_backing.dtype != torch.bool or mask_backing.shape != (
        len(routed_layers),
        num_tokens,
    ):
        raise RuntimeError("batched K/V commit mask backing shape mismatch")
    if not mask_backing.is_contiguous():
        raise RuntimeError("batched K/V commit mask backing must be contiguous")
    if cache_locations.shape != (num_tokens,):
        raise RuntimeError(
            "batched K/V commit requires aligned cache locations"
        )
    # Review finding 2: the kernel does raw contiguous integer loads —
    # prove dtype, contiguity, and device colocation instead of assuming.
    if cache_locations.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            "batched K/V commit cache locations must be int32/int64, got "
            f"{cache_locations.dtype}"
        )
    if not cache_locations.is_contiguous():
        raise RuntimeError(
            "batched K/V commit cache locations must be contiguous"
        )
    if cache_locations.device != mask_backing.device:
        raise RuntimeError(
            "batched K/V commit tensors must share one device"
        )

    first_attn = llama.layers[routed_layers[0]].self_attn.attn
    hd_k = int(first_attn.tp_k_head_num) * int(first_attn.qk_head_dim)
    hd_v = int(first_attn.tp_v_head_num) * int(first_attn.v_head_dim)
    table_rows: list[list[int]] = []
    stride_rows: list[list[int]] = []
    for stage_index, layer_id in enumerate(routed_layers):
        radix_attention = llama.layers[layer_id].self_attn.attn
        if (
            radix_attention.k_scale is not None
            or radix_attention.v_scale is not None
        ):
            raise RuntimeError(
                "batched K/V commit does not support per-layer K/V scales"
            )
        layer_hd_k = int(radix_attention.tp_k_head_num) * int(
            radix_attention.qk_head_dim
        )
        layer_hd_v = int(radix_attention.tp_v_head_num) * int(
            radix_attention.v_head_dim
        )
        if layer_hd_k != hd_k or layer_hd_v != hd_v:
            raise RuntimeError(
                "batched K/V commit requires uniform K/V head geometry"
            )
        repair_k = repair_k_buffers[stage_index]
        repair_v = repair_v_buffers[stage_index]
        expected_k = (
            num_tokens,
            radix_attention.tp_k_head_num,
            radix_attention.qk_head_dim,
        )
        expected_v = (
            num_tokens,
            radix_attention.tp_v_head_num,
            radix_attention.v_head_dim,
        )
        if repair_k.shape != expected_k or repair_v.shape != expected_v:
            raise RuntimeError(
                "batched K/V commit repair buffer shape mismatch"
            )
        if repair_k.dtype != kv_pool.dtype or repair_v.dtype != kv_pool.dtype:
            raise RuntimeError("batched K/V commit repair dtype mismatch")
        k_row_stride = _require_flat_rows(repair_k, "repair K view")
        v_row_stride = _require_flat_rows(repair_v, "repair V view")
        pool_index = int(layer_id) - int(kv_pool.start_layer)
        k_dst = kv_pool.k_buffer[pool_index]
        v_dst = kv_pool.v_buffer[pool_index]
        for dst, dst_hd, name in (
            (k_dst, hd_k, "pool K buffer"),
            (v_dst, hd_v, "pool V buffer"),
        ):
            if dst.dim() != 3 or not dst.is_contiguous():
                raise RuntimeError(
                    f"batched K/V commit: {name} must be contiguous 3-D"
                )
            if int(dst.shape[1]) * int(dst.shape[2]) != dst_hd:
                raise RuntimeError(
                    f"batched K/V commit: {name} row width mismatch"
                )
        table_rows.append(
            [
                repair_k.data_ptr(),
                repair_v.data_ptr(),
                k_dst.data_ptr(),
                v_dst.data_ptr(),
            ]
        )
        stride_rows.append([k_row_stride, v_row_stride])

    device = mask_backing.device
    return BatchedCommitPlan(
        table=torch.tensor(table_rows, dtype=torch.int64, device=device),
        row_strides=torch.tensor(
            stride_rows, dtype=torch.int64, device=device
        ),
        cache_locations=cache_locations,
        mask_backing=mask_backing,
        num_layers=len(routed_layers),
        num_tokens=num_tokens,
        hd_k=hd_k,
        hd_v=hd_v,
        total_slots=int(kv_pool.size) + int(kv_pool.page_size),
    )
def run_batched_commit(plan: BatchedCommitPlan) -> None:
    """Launch the single cross-layer commit kernel (capture-safe).

    DECLARED delta vs the per-layer writer (review finding, 2026-08-28):
    the per-layer path maps non-PROJECT rows to the reserved padding
    slot 0 and overwrites it; this kernel SKIPS masked rows instead, so
    padding slot 0 keeps stale bytes. Slot 0 is dead storage by
    contract (never read as cache), so live-slot semantics are
    identical — but pool BYTE equality at slot 0 does not hold.
    """

    # Review finding 1: the replaced set_kv_buffer path recorded an
    # async OOB probe on the replay-varying locations; keep that
    # invariant — recorded here, it replays with the commit graph
    # (once per step; locations are shared across layers).
    maybe_detect_oob(
        plan.cache_locations, 0, plan.total_slots, "batched K/V commit"
    )
    _fd_batched_commit_kv_kernel[(plan.num_layers * plan.num_tokens,)](
        plan.table,
        plan.row_strides,
        plan.cache_locations,
        plan.mask_backing,
        plan.num_layers,
        plan.num_tokens,
        plan.hd_k,
        plan.hd_v,
        256,
    )
