"""Cohort packing and mapped-linear primitives.

Shared building blocks for the compact and virtual cohort paths: route-map
construction, packed-prefix gathers, and the mapped linear used when routed rows
are addressed through an index vector rather than compacted.
"""

from __future__ import annotations

import triton.language as tl
import triton
import os
from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
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

        route_weight = tl.load(
            branch_weights_ptr + destination_rows,
            mask=active_rows,
            other=0.0,
        ).to(tl.float32)
        if INVERT_WEIGHT:
            route_weight = 1.0 - route_weight
        accumulator *= route_weight[:, None]
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
) -> torch.Tensor:
    """Execute one exact routed SwiGLU branch through a device row map.

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
