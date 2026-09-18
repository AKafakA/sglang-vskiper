"""Compact-cohort MLP bodies.

Pack the routed cohort into a contiguous prefix and run the grouped path over
it. `dual_compact` packs BOTH cohorts with a per-bucket minimum row count.
Superseded for prefill by the binary-cohort kernel, retained because the
integrated and decode arms still select these policies."""

from __future__ import annotations

from typing import Any, Mapping, Optional
from vskipper.runtime.config import (
    full_graph_virtual_cohort_enabled,
    full_graph_weighted_scatter_enabled,
)
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch
from vskipper.runtime.env import (
    FD_DUAL_COMPACT_MIN_ROWS_ENV,
)


def full_graph_dual_compact_min_rows(
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Per-bucket threshold: dual_compact layers run the FUSED
    project_filtered_run body when the pass has fewer rows than this
    (0 = always dual_compact). rows is a capture-time constant per
    graph bucket (prefill AND decode when compact_phases includes
    decode), so the choice freezes per bucket."""

    values = os.environ if environ is None else environ
    raw = str(values.get(FD_DUAL_COMPACT_MIN_ROWS_ENV, "0")).strip()
    try:
        threshold = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{FD_DUAL_COMPACT_MIN_ROWS_ENV} must be an integer"
        ) from error
    if threshold < 0:
        raise ValueError(
            f"{FD_DUAL_COMPACT_MIN_ROWS_ENV} must be non-negative"
        )
    return threshold
def _compact_capacity(rows: int, fraction: float, multiple: int) -> int:
    capacity = math.ceil(rows * fraction / multiple) * multiple
    return min(rows, max(1, capacity))
def _bounded_compact_branch(
    module: Any,
    hidden_states: torch.Tensor,
    branch_weights: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    capacity: int,
    overflow_w1: torch.Tensor,
    overflow_w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute a fixed-capacity cuBLAS cohort plus an exact filtered overflow."""

    rows = int(hidden_states.shape[0])
    base_rows, base_active, overflow_rows, overflow_active = (
        _aligned_branch_partition(active_rows, capacity)
    )
    base_hidden = hidden_states.index_select(0, base_rows)
    base_weights = branch_weights.index_select(0, base_rows)
    base_output = module(base_hidden) * base_weights
    base_output = torch.where(
        base_active.unsqueeze(-1), base_output, torch.zeros_like(base_output)
    )

    output = torch.empty_like(hidden_states)
    output.index_copy_(0, base_rows, base_output)
    if capacity < rows:
        overflow_hidden = hidden_states.index_select(0, overflow_rows)
        overflow_weights = branch_weights.index_select(0, overflow_rows)
        overflow_output = _one_expert_mlp(
            overflow_hidden,
            overflow_w1,
            overflow_w2,
            overflow_weights,
            overflow_active.unsqueeze(-1),
        )
        overflow_output = torch.where(
            overflow_active.unsqueeze(-1),
            overflow_output,
            torch.zeros_like(overflow_output),
        )
        output.index_copy_(0, overflow_rows, overflow_output)
    overflow_count = torch.clamp(
        active_rows.sum(dtype=torch.int64) - capacity, min=0
    )
    return output, overflow_count
def _aligned_branch_partition(
    active_rows: torch.Tensor, capacity: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-size active-first row maps without a device-wide sort."""

    rows = int(active_rows.shape[0])
    if not active_rows.is_cuda:
        order = torch.argsort(
            active_rows.to(torch.int8), descending=True, stable=True
        )
        return (
            order[:capacity],
            active_rows.index_select(0, order[:capacity]),
            order[capacity:],
            active_rows.index_select(0, order[capacity:]),
        )

    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    block_size = 16
    topk_ids = active_rows.to(torch.int32).sub(1).unsqueeze(-1)
    sorted_rows, _, _ = moe_align_block_size(topk_ids, block_size, 1)
    active_count = active_rows.sum(dtype=torch.int64)
    inactive_count = rows - active_count
    active_start = torch.div(
        inactive_count + block_size - 1,
        block_size,
        rounding_mode="floor",
    ) * block_size

    def select(offset: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        slots = torch.arange(count, dtype=torch.int64, device=active_rows.device)
        global_slots = slots + offset
        slot_active = global_slots < active_count
        # Real filtered rows occupy the beginning of the alignment output;
        # real active rows begin after the filtered block padding. Use the
        # filtered rows as unique fillers so base+overflow remain a permutation
        # of the input and index_copy never receives duplicate destinations.
        positions = torch.clamp(
            torch.where(
                slot_active,
                active_start + global_slots,
                global_slots - active_count,
            ),
            min=0,
            max=sorted_rows.numel() - 1,
        )
        selected = sorted_rows.index_select(0, positions).long()
        return selected, slot_active

    base_rows, base_active = select(0, capacity)
    overflow_rows, overflow_active = select(capacity, rows - capacity)
    return base_rows, base_active, overflow_rows, overflow_active
def _bounded_compact_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
    *,
    capacity: int,
    valid_rows: Optional[torch.Tensor],
    compact_stats: Optional[list[torch.Tensor]],
) -> torch.Tensor:
    """Use fixed-shape standard GEMMs for the common RUN/PROJECT cohorts."""

    if valid_rows is None:
        valid_rows = torch.ones(
            hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device
        )
    run_active = run_mask.squeeze(-1) & valid_rows
    project_active = (~run_mask.squeeze(-1)) & valid_rows
    if hidden_states.is_cuda:
        return _mapped_bounded_compact_mlp(
            layer,
            proj,
            hidden_states,
            route_weights,
            run_active,
            project_active,
            capacity=capacity,
            compact_stats=compact_stats,
        )
    run_output, run_overflow = _bounded_compact_branch(
        layer.mlp,
        hidden_states,
        route_weights,
        run_active,
        capacity=capacity,
        overflow_w1=layer.mlp.gate_up_proj.weight.unsqueeze(0),
        overflow_w2=layer.mlp.down_proj.weight.unsqueeze(0),
    )
    project_output, project_overflow = _bounded_compact_branch(
        proj,
        hidden_states,
        1.0 - route_weights,
        project_active,
        capacity=capacity,
        overflow_w1=proj._fused_gate_down_weight().unsqueeze(0),
        overflow_w2=proj.up_proj.weight.unsqueeze(0),
    )
    if compact_stats is not None:
        compact_stats.append(
            torch.stack(
                (
                    valid_rows.sum(dtype=torch.int64),
                    run_overflow,
                    project_overflow,
                )
            )
        )
    return run_output + project_output
def _mapped_bounded_compact_branch(
    module: Any,
    hidden_states: torch.Tensor,
    branch_weights: torch.Tensor,
    row_map: torch.Tensor,
    count: torch.Tensor,
    output: torch.Tensor,
    *,
    capacity: int,
    overflow_w1: torch.Tensor,
    overflow_w2: torch.Tensor,
) -> torch.Tensor:
    if full_graph_virtual_cohort_enabled():
        from vskipper.runtime.cohort import (
            mapped_swiglu,
        )

        if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
            gate_up_weight = module.gate_up_proj.weight
            down_weight = module.down_proj.weight
        elif all(
            hasattr(module, name)
            for name in ("_fused_gate_down_weight", "up_proj")
        ):
            gate_up_weight = module._fused_gate_down_weight()
            down_weight = module.up_proj.weight
        else:
            raise TypeError(
                "VP virtual cohorts require a Llama MLP or FlexiDepth projector"
            )
        from vskipper.runtime.common import full_graph_gate_mode

        mapped_swiglu(
            hidden_states,
            gate_up_weight,
            down_weight,
            row_map,
            count,
            branch_weights,
            output,
            scale_weight=full_graph_gate_mode() == "released",
        )
        return torch.clamp(
            count.to(torch.int64).reshape(()) - capacity, min=0
        )

    from vskipper.runtime.cohort import (
        pack_cohort,
        scatter_cohort,
        scatter_weighted_cohort,
    )
    from vskipper.kernels.kernel import (
        pack_rows,
    )

    rows = int(hidden_states.shape[0])
    if full_graph_weighted_scatter_enabled():
        base_hidden = pack_rows(
            hidden_states,
            row_map,
            count,
            capacity=capacity,
        )
        base_output = module(base_hidden)
        scatter_weighted_cohort(
            output,
            base_output,
            branch_weights,
            row_map,
            count,
        )
    else:
        base_hidden, base_weights = pack_cohort(
            hidden_states,
            branch_weights,
            row_map,
            count,
            capacity=capacity,
        )
        base_output = module(base_hidden) * base_weights
        scatter_cohort(output, base_output, row_map, count)

    overflow_capacity = rows - capacity
    if overflow_capacity:
        overflow_hidden, overflow_weights = pack_cohort(
            hidden_states,
            branch_weights,
            row_map,
            count,
            capacity=overflow_capacity,
            row_offset=capacity,
        )
        overflow_active = (
            torch.arange(
                overflow_capacity, dtype=count.dtype, device=count.device
            )
            + capacity
            < count.reshape(())
        )
        overflow_output = _one_expert_mlp(
            overflow_hidden,
            overflow_w1,
            overflow_w2,
            overflow_weights,
            overflow_active.unsqueeze(-1),
        )
        scatter_cohort(
            output,
            overflow_output,
            row_map,
            count,
            row_offset=capacity,
        )
    return torch.clamp(count.to(torch.int64).reshape(()) - capacity, min=0)
def _mapped_bounded_compact_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_active: torch.Tensor,
    project_active: torch.Tensor,
    *,
    capacity: int,
    compact_stats: Optional[list[torch.Tensor]],
) -> torch.Tensor:
    return _mapped_asymmetric_compact_mlp(
        layer,
        proj,
        hidden_states,
        route_weights,
        run_active,
        project_active,
        run_capacity=capacity,
        project_capacity=capacity,
        compact_stats=compact_stats,
    )
def _mapped_asymmetric_compact_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_active: torch.Tensor,
    project_active: torch.Tensor,
    *,
    run_capacity: int,
    project_capacity: int,
    compact_stats: Optional[list[torch.Tensor]],
) -> torch.Tensor:
    """Execute exact RUN/PROJECT cohorts with independent fixed capacities."""

    from vskipper.runtime.routing import (
        build_route_maps,
    )

    run_rows, project_rows, counts = build_route_maps(
        run_active, project_active
    )
    output = torch.zeros_like(hidden_states)
    run_overflow = _mapped_bounded_compact_branch(
        layer.mlp,
        hidden_states,
        route_weights,
        run_rows,
        counts[0:1],
        output,
        capacity=run_capacity,
        overflow_w1=layer.mlp.gate_up_proj.weight.unsqueeze(0),
        overflow_w2=layer.mlp.down_proj.weight.unsqueeze(0),
    )
    project_overflow = _mapped_bounded_compact_branch(
        proj,
        hidden_states,
        1.0 - route_weights,
        project_rows,
        counts[1:2],
        output,
        capacity=project_capacity,
        overflow_w1=proj._fused_gate_down_weight().unsqueeze(0),
        overflow_w2=proj.up_proj.weight.unsqueeze(0),
    )
    if compact_stats is not None:
        compact_stats.append(
            torch.stack(
                (
                    (run_active | project_active).sum(dtype=torch.int64),
                    run_overflow,
                    project_overflow,
                )
            )
        )
    return output
def _mapped_single_compact_branch(
    module: Any,
    hidden_states: torch.Tensor,
    branch_weights: torch.Tensor,
    active_rows: torch.Tensor,
    output: torch.Tensor,
    *,
    capacity: int,
    overflow_w1: torch.Tensor,
    overflow_w2: torch.Tensor,
) -> torch.Tensor:
    from vskipper.runtime.routing import (
        build_route_maps,
    )

    row_map, _, counts = build_route_maps(
        active_rows, torch.zeros_like(active_rows)
    )
    return _mapped_bounded_compact_branch(
        module,
        hidden_states,
        branch_weights,
        row_map,
        counts[0:1],
        output,
        capacity=capacity,
        overflow_w1=overflow_w1,
        overflow_w2=overflow_w2,
    )
def _project_filtered_run_compact_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
    *,
    capacity: int,
) -> torch.Tensor:
    """PROJECT dense over all rows; the RUN correction through ONE
    fixed-capacity cuBLAS cohort branch with exact filtered overflow.

    Identical semantics to _project_filtered_run_mlp's virtual-cohort
    branch, but with the cohort bounded at a real capacity instead of
    the full row count, so the dense-MLP GEMM runs at capacity shape.
    """

    output = proj(hidden_states) * (1.0 - route_weights)
    _mapped_single_compact_branch(
        layer.mlp,
        hidden_states,
        route_weights,
        run_mask.squeeze(-1),
        output,
        capacity=capacity,
        overflow_w1=layer.mlp.gate_up_proj.weight.unsqueeze(0),
        overflow_w2=layer.mlp.down_proj.weight.unsqueeze(0),
    )
    return output
def _one_expert_mlp(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    route_weights: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    """Run one filtered expert; inactive rows are represented by expert -1."""

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        outplace_fused_experts,
    )

    topk_ids = active_rows.to(torch.int32).sub(1)
    return outplace_fused_experts(
        hidden_states.contiguous(),
        w1,
        w2,
        route_weights,
        topk_ids,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=False,
        filter_expert=True,
        gate_up_interleaved=False,
    )
