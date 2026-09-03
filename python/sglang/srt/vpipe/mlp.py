"""MLP dispatch and the binary-cohort body.

`fd_conditional_mlp_full_graph` picks the body for a routed layer from its
policy. Precedence matters: binary_cohort is tested BEFORE the grouped/compact
gates, because getting that order wrong silently runs the old body while still
attesting the new policy name.

`_binary_cohort_mlp` drives the count-adaptive kernel. Its scratch is ONE
ceiling-sized allocation sliced per bucket -- a per-bucket allocation OOM'd
graph capture, since allocating inside capture is illegal."""

from __future__ import annotations

from sglang.srt.vpipe.config import (
    full_graph_eager_semantic_debug_enabled,
)
import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
from sglang.srt.vpipe.common import (
    full_graph_gate_mode,
    full_graph_low_row_policy,
)
from sglang.srt.vpipe.env import (
    _BINARY_COHORT_CONFIG_DIGEST,
    _BINARY_COHORT_LAYERS,
    _BINARY_COHORT_SCRATCH,
    _BINARY_COHORT_STATS,
)
from sglang.srt.vpipe.mlp_compact import (
    _compact_capacity,
    full_graph_dual_compact_min_rows,
)
from sglang.srt.vpipe.mlp_compact import (
    _bounded_compact_mlp,
    _mapped_asymmetric_compact_mlp,
    _mapped_single_compact_branch,
    _project_filtered_run_compact_mlp,
)
from sglang.srt.vpipe.config import (
    full_graph_compact_config,
    full_graph_forced_all_run_fastpath_enabled,
    full_graph_layer_policies,
    full_graph_virtual_cohort_enabled,
)
from sglang.srt.vpipe.mlp_compact import (
    _one_expert_mlp,
)


def _direct_eager_semantic_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
    valid_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    """Match released FlexiDepth gather/scatter equations for trace-only gates."""

    def direct_projector(value: torch.Tensor) -> torch.Tensor:
        gate = proj.gate_proj(value)
        down = proj.down_proj(value)
        act_fn = getattr(proj, "act_fn", F.silu)
        return proj.up_proj(act_fn(gate) * down)

    active = run_mask.squeeze(-1)
    if valid_rows is not None:
        active = active & valid_rows
        project_active = (~run_mask.squeeze(-1)) & valid_rows
    else:
        project_active = ~active
    output = torch.zeros_like(hidden_states)
    run_rows = active.nonzero(as_tuple=True)[0]
    project_rows = project_active.nonzero(as_tuple=True)[0]
    if run_rows.numel() > 0:
        output.index_copy_(
            0,
            run_rows,
            layer.mlp(hidden_states.index_select(0, run_rows))
            * route_weights.index_select(0, run_rows),
        )
    if project_rows.numel() > 0:
        output.index_copy_(
            0,
            project_rows,
            direct_projector(hidden_states.index_select(0, project_rows))
            * (1.0 - route_weights.index_select(0, project_rows)),
        )
    return output
def _grouped_prefill_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
    valid_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    """Execute large variable-size prefill cohorts with native GEMMs."""

    rows = int(hidden_states.shape[0])
    if hidden_states.ndim != 2:
        raise RuntimeError("grouped prefill MLP requires a 2D hidden-state batch")
    if route_weights.shape != (rows, 1) or run_mask.shape != (rows, 1):
        raise RuntimeError("grouped prefill MLP route rows do not match the batch")
    if proj is None:
        raise RuntimeError("grouped prefill MLP requires a PROJECT adapter")
    if valid_rows is None:
        valid_rows = torch.ones(
            rows, dtype=torch.bool, device=hidden_states.device
        )
    elif valid_rows.shape != (rows,) or valid_rows.dtype != torch.bool:
        raise RuntimeError("grouped prefill MLP valid rows do not match the batch")

    scale_by_route_weight = full_graph_gate_mode() == "released"
    run_active = run_mask.squeeze(-1) & valid_rows
    project_active = (~run_mask.squeeze(-1)) & valid_rows
    run_rows = run_active.nonzero(as_tuple=True)[0]
    project_rows = project_active.nonzero(as_tuple=True)[0]
    output = torch.zeros_like(hidden_states)
    if run_rows.numel() > 0:
        run_hidden = hidden_states.index_select(0, run_rows)
        run_output = layer.mlp(run_hidden)
        if scale_by_route_weight:
            run_output.mul_(route_weights.index_select(0, run_rows))
        output.index_copy_(0, run_rows, run_output)
    if project_rows.numel() > 0:
        project_hidden = hidden_states.index_select(0, project_rows)
        project_output = proj(project_hidden)
        if scale_by_route_weight:
            project_weights = route_weights.index_select(0, project_rows)
            project_output.mul_(project_weights.neg().add_(1.0))
        output.index_copy_(0, project_rows, project_output)
    return output
def _binary_cohort_stats(device: torch.device) -> torch.Tensor:
    """[calls, run_rows, project_rows] int64 accumulator per device."""

    stats = _BINARY_COHORT_STATS.get(device)
    if stats is None:
        stats = torch.zeros(3, dtype=torch.int64, device=device)
        _BINARY_COHORT_STATS[device] = stats
    return stats
def _binary_cohort_scratch(
    device: torch.device,
    rows: int,
    hidden_size: int,
    intermediate: int,
    dtype: torch.dtype,
) -> dict:
    """ONE shared scratch set per (device, shapes, dtype) — sized at the
    deployment's prefill ceiling and SLICED per bucket.

    Review finding 10 demanded reuse across branches, layers AND GRAPH
    BUCKETS. Keying by bucket rows (the first implementation) allocated
    a full set per captured bucket: the 58-bucket prefill ladder needed
    ~6 GB of scratch and OOM'd capture. One ceiling-sized allocation
    plus row-slices costs the size of the largest bucket alone.

    DAG contract (kernel design v2.1): every consumer sequence within a
    pass is single-stream ordered (RUN branch fully drains before the
    PROJECT branch reuses the buffers; the returned ``out`` is consumed
    by the layer's residual add before the next routed layer's zeroing
    write). No side-stream may touch these buffers.
    """

    key = (device, hidden_size, intermediate, dtype)
    scratch = _BINARY_COHORT_SCRATCH.get(key)
    if scratch is None:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        ceiling = max(
            int(server_args.chunked_prefill_size or 0),
            int(server_args.max_prefill_tokens or 0),
            int(rows),
        )
        if ceiling < rows:
            raise RuntimeError(
                f"binary_cohort scratch ceiling {ceiling} below the live "
                f"bucket {rows}"
            )

        def zeros(*shape):
            return torch.zeros(shape, device=device, dtype=dtype)

        scratch = {
            "ceiling": ceiling,
            "compact": zeros(ceiling, hidden_size),
            "gate_up": zeros(ceiling, 2 * intermediate),
            "activated": zeros(ceiling, intermediate),
            "final": zeros(ceiling, hidden_size),
            "out": zeros(ceiling, hidden_size),
        }
        _BINARY_COHORT_SCRATCH[key] = scratch
    if rows > scratch["ceiling"]:
        raise RuntimeError(
            f"binary_cohort bucket {rows} exceeds the scratch ceiling "
            f"{scratch['ceiling']} (allocation during capture is illegal; "
            "raise chunked_prefill_size/max_prefill_tokens at boot)"
        )
    return {
        name: buffer[:rows]
        for name, buffer in scratch.items()
        if name != "ceiling"
    }
def _binary_cohort_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
    valid_rows: Optional[torch.Tensor],
) -> torch.Tensor:
    """Count-adaptive binary-cohort MLP (D-302/D-303; design v2.1).

    pack -> tuned count-GEMM -> silu_mul -> tuned count-GEMM ->
    route-weighted scatter, per branch, over one shared scratch set.
    Output rows: RUN rows get w*MLP, PROJECT rows (1-w)*FDProj, invalid
    rows ZERO (exactly one write per valid row). Per-op tuned configs
    load fail-closed from the committed per-device artifact; the config
    is selected per bucket-rows at first use (capture-frozen there-
    after). PROJECT GEMMs currently reuse the nearest RUN-op configs
    (declared; per-op proj tuning is follow-up polish).
    """

    from sglang.srt.vpipe.kernel import (
        count_matmul_gridexit,
        count_silu_mul,
        load_tuned_configs,
        pack_rows,
        select_config,
        weighted_scatter,
    )
    from sglang.srt.vpipe.routing import (
        build_route_maps,
    )

    rows, hidden_size = hidden_states.shape
    # Review P1: normalize EVERY per-row input to exactly [rows] before
    # any mask algebra — a [rows, 1] operand would broadcast the AND to
    # [rows, rows] and derive maps beyond the scratch capacity.
    if run_mask.numel() != rows or run_mask.dtype != torch.bool:
        raise ValueError(
            "binary_cohort run_mask must be one bool per row; got "
            f"shape {tuple(run_mask.shape)} dtype {run_mask.dtype}"
        )
    if route_weights.numel() != rows:
        raise ValueError(
            "binary_cohort route_weights must be one scalar per row; "
            f"got shape {tuple(route_weights.shape)}"
        )
    run_active = run_mask.reshape(rows)
    project_active = ~run_active
    if valid_rows is not None:
        if valid_rows.numel() != rows or valid_rows.dtype != torch.bool:
            raise ValueError(
                "binary_cohort valid_rows must be one bool per row; got "
                f"shape {tuple(valid_rows.shape)} dtype {valid_rows.dtype}"
            )
        valid_flat = valid_rows.reshape(rows)
        run_active = run_active & valid_flat
        project_active = project_active & valid_flat
    weights_flat = route_weights.reshape(rows)
    run_map, project_map, counts = build_route_maps(
        run_active, project_active
    )

    mlp = layer.mlp
    gate_up_width = int(mlp.gate_up_proj.weight.shape[0])
    if gate_up_width % 2:
        raise ValueError("binary_cohort gate_up weight width must be even")
    intermediate = gate_up_width // 2
    proj_gate_down = proj._fused_gate_down_weight()
    proj_bottleneck = int(proj.up_proj.weight.shape[1])
    if int(proj_gate_down.shape[0]) != 2 * proj_bottleneck:
        raise ValueError(
            "binary_cohort fused project weight width "
            f"{int(proj_gate_down.shape[0])} != 2*bottleneck "
            f"{2 * proj_bottleneck}"
        )
    if proj_bottleneck > intermediate:
        raise ValueError(
            "binary_cohort project bottleneck exceeds the RUN "
            "intermediate — narrow scratch views would be undersized"
        )
    if (
        mlp.gate_up_proj.weight.dtype != hidden_states.dtype
        or proj_gate_down.dtype != hidden_states.dtype
    ):
        raise ValueError("binary_cohort weight/hidden dtype mismatch")
    scratch = _binary_cohort_scratch(
        hidden_states.device,
        int(rows),
        int(hidden_size),
        intermediate,
        hidden_states.dtype,
    )
    out = scratch["out"]
    out.zero_()

    device_name = torch.cuda.get_device_name(hidden_states.device)
    tuned = load_tuned_configs(device_name)
    if device_name not in _BINARY_COHORT_CONFIG_DIGEST:
        import hashlib

        _BINARY_COHORT_CONFIG_DIGEST[device_name] = hashlib.sha256(
            json.dumps(tuned, sort_keys=True).encode()
        ).hexdigest()
    _BINARY_COHORT_LAYERS.add(int(getattr(layer, "layer_id", -1)))
    stats = _binary_cohort_stats(hidden_states.device)
    stats[0].add_(1)
    stats[1:3].add_(counts.to(torch.int64))
    config_gate_up = select_config(tuned, "gateup", int(rows))
    config_down = select_config(tuned, "down", int(rows))

    run_count = counts[0:1]
    pack_rows(hidden_states, run_map, run_count, scratch["compact"])
    count_matmul_gridexit(
        scratch["compact"],
        mlp.gate_up_proj.weight,
        run_count,
        scratch["gate_up"],
        **config_gate_up,
    )
    count_silu_mul(scratch["gate_up"], run_count, scratch["activated"])
    count_matmul_gridexit(
        scratch["activated"],
        mlp.down_proj.weight,
        run_count,
        scratch["final"],
        **config_down,
    )
    weighted_scatter(
        scratch["final"], run_map, weights_flat, run_count, out,
        invert_weight=False,
    )

    project_count = counts[1:2]
    pack_rows(hidden_states, project_map, project_count, scratch["compact"])
    gate_up_view = scratch["gate_up"][:, : 2 * proj_bottleneck]
    count_matmul_gridexit(
        scratch["compact"],
        proj_gate_down,
        project_count,
        gate_up_view,
        **config_gate_up,
    )
    activated_view = scratch["activated"][:, :proj_bottleneck]
    count_silu_mul(gate_up_view, project_count, activated_view)
    count_matmul_gridexit(
        activated_view,
        proj.up_proj.weight,
        project_count,
        scratch["final"],
        **config_down,
    )
    weighted_scatter(
        scratch["final"], project_map, weights_flat, project_count, out,
        invert_weight=True,
    )
    return out
def fd_conditional_mlp_full_graph(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
    *,
    valid_rows: Optional[torch.Tensor] = None,
    compact_stats: Optional[list[torch.Tensor]] = None,
    compact_phase_enabled: bool = True,
    prefill_grouped_enabled: bool = False,
    force_dense_all_run: Optional[bool] = None,
    force_filtered_all_run: bool = False,
) -> torch.Tensor:
    """Fixed-topology conditional MLP with device-resident complementary routes."""

    if force_dense_all_run is None:
        force_dense_all_run = full_graph_forced_all_run_fastpath_enabled()
    if force_dense_all_run and force_filtered_all_run:
        raise ValueError("dense and filtered all-RUN bodies are mutually exclusive")
    if force_dense_all_run:
        return layer.mlp(hidden_states) * route_weights
    if force_filtered_all_run:
        active_rows = run_mask
        if valid_rows is not None:
            active_rows = active_rows & valid_rows.view(-1, 1)
        # Lane-2 cut2: the conditional graph's all-RUN branch replayed the
        # one-expert fused-MoE kernel (11.6 launches / 5.75 ms per step at 4
        # rows even with the low-row body in place). Below the low-row bound
        # decode is weight-bound, so the dense MLP on all rows costs exactly
        # production's MLP; same output (active rows w*MLP, others zero).
        lr_policy, lr_max_rows = full_graph_low_row_policy()
        if lr_policy == "full_dual" and int(hidden_states.shape[0]) <= lr_max_rows:
            run_output = layer.mlp(hidden_states) * route_weights
        else:
            run_output = _one_expert_mlp(
                hidden_states,
                layer.mlp.gate_up_proj.weight.unsqueeze(0),
                layer.mlp.down_proj.weight.unsqueeze(0),
                route_weights,
                active_rows,
            )
        return torch.where(active_rows, run_output, torch.zeros_like(run_output))

    if full_graph_eager_semantic_debug_enabled():
        return _direct_eager_semantic_mlp(
            layer,
            proj,
            hidden_states,
            route_weights,
            run_mask,
            valid_rows,
        )

    # Lane-2 cut2 (2026-09-02): the low-row `full_dual` body resolves BEFORE
    # every routed-MLP dispatch, binary_cohort included. Below ~32 rows decode
    # is weight-bandwidth-bound, so partial-row skipping saves no weight reads
    # and the count-adaptive / grouped machinery is pure cost (measured on the
    # CSD3 ladder: +8.5 ms/step at 1-16 rows, 1,142 vs 376 launches). The
    # dense-both-branches body costs the production MLP plus the small
    # projector GEMMs. Invalid (padded) rows are zeroed exactly as the
    # binary-cohort path does. Same env gate as the compact-path low-row body.
    compact_enabled, min_rows, fraction, multiple = full_graph_compact_config()
    compact_enabled = compact_enabled and compact_phase_enabled
    rows = int(hidden_states.shape[0])
    low_row_policy, low_row_max_rows = full_graph_low_row_policy()
    # The body needs no compaction, so it is keyed on the policy + rows only
    # (validation already requires COMPACT=1 with the policy); tying it to the
    # per-batch compact-phase flag left graph-captured passes on the
    # fused-MoE fallback (cut2 profile: 11.6 fused_moe launches/step at 4 rows).
    if low_row_policy in ("full_dual", "native_dense") and rows <= low_row_max_rows:
        output = _full_dual_mlp(
            layer, proj, hidden_states, route_weights, run_mask
        )
        if valid_rows is not None:
            output = torch.where(
                valid_rows.view(-1, 1), output, torch.zeros_like(output)
            )
        return output

    # binary_cohort resolves BEFORE the grouped and compact gates
    # (kernel design v2.1 item 5) and fail-closes on conflicts.
    binary_policies = full_graph_layer_policies()
    binary_layer_id = int(getattr(layer, "layer_id", -1))
    if binary_layer_id < 0 and any(
        policy[0] == "binary_cohort" for policy in binary_policies.values()
    ):
        raise RuntimeError(
            "binary_cohort policies are configured but this layer has no "
            "layer_id — refusing to fail open past the binary dispatch"
        )
    binary_policy = binary_policies.get(binary_layer_id)
    if binary_policy is not None and binary_policy[0] == "binary_cohort":
        if prefill_grouped_enabled:
            raise RuntimeError(
                "binary_cohort conflicts with PREFILL_GROUPED_MLP — "
                "remove one (fail-closed, v2.1 item 5)"
            )
        return _binary_cohort_mlp(
            layer, proj, hidden_states, route_weights, run_mask, valid_rows
        )

    if prefill_grouped_enabled:
        return _grouped_prefill_mlp(
            layer,
            proj,
            hidden_states,
            route_weights,
            run_mask,
            valid_rows,
        )

    if compact_enabled and rows >= min_rows:
        policies = full_graph_layer_policies()
        layer_id = int(getattr(layer, "layer_id", -1))
        policy = policies.get(layer_id)
        if policy is not None:
            policy_name, run_fraction, project_fraction = policy
            if policy_name == "full_dual":
                return _full_dual_mlp(
                    layer, proj, hidden_states, route_weights, run_mask
                )
            if policy_name == "dense_filtered_project":
                return _dense_filtered_project_mlp(
                    layer, proj, hidden_states, route_weights, run_mask
                )
            if policy_name == "project_filtered_run":
                return _project_filtered_run_mlp(
                    layer, proj, hidden_states, route_weights, run_mask
                )
            if policy_name == "project_filtered_run_compact":
                if rows < int(project_fraction):
                    return _project_filtered_run_mlp(
                        layer, proj, hidden_states, route_weights, run_mask
                    )
                return _project_filtered_run_compact_mlp(
                    layer,
                    proj,
                    hidden_states,
                    route_weights,
                    run_mask,
                    capacity=_compact_capacity(
                        rows, float(run_fraction), multiple
                    ),
                )
            if policy_name == "project_base":
                capacity = _compact_capacity(
                    rows, float(run_fraction), multiple
                )
                output, run_overflow = _mapped_project_base_mlp(
                    layer,
                    proj,
                    hidden_states,
                    route_weights,
                    run_mask,
                    capacity=capacity,
                    valid_rows=valid_rows,
                )
                if compact_stats is not None:
                    valid_count = (
                        valid_rows.sum(dtype=torch.int64)
                        if valid_rows is not None
                        else torch.tensor(
                            rows, dtype=torch.int64, device=hidden_states.device
                        )
                    )
                    compact_stats.append(
                        torch.stack(
                            (
                                valid_count,
                                run_overflow,
                                torch.zeros_like(run_overflow),
                            )
                        )
                    )
                return output
            if policy_name == "run_base":
                capacity = _compact_capacity(
                    rows, float(project_fraction), multiple
                )
                output, project_overflow = _mapped_run_base_mlp(
                    layer,
                    proj,
                    hidden_states,
                    route_weights,
                    run_mask,
                    capacity=capacity,
                    valid_rows=valid_rows,
                )
                if compact_stats is not None:
                    valid_count = (
                        valid_rows.sum(dtype=torch.int64)
                        if valid_rows is not None
                        else torch.tensor(
                            rows, dtype=torch.int64, device=hidden_states.device
                        )
                    )
                    compact_stats.append(
                        torch.stack(
                            (
                                valid_count,
                                torch.zeros_like(project_overflow),
                                project_overflow,
                            )
                        )
                    )
                return output
            if policy_name == "dual_compact":
                if rows < full_graph_dual_compact_min_rows():
                    return _project_filtered_run_mlp(
                        layer, proj, hidden_states, route_weights, run_mask
                    )
                if valid_rows is None:
                    valid_rows = torch.ones(
                        rows, dtype=torch.bool, device=hidden_states.device
                    )
                run_active = run_mask.squeeze(-1) & valid_rows
                project_active = (~run_mask.squeeze(-1)) & valid_rows
                return _mapped_asymmetric_compact_mlp(
                    layer,
                    proj,
                    hidden_states,
                    route_weights,
                    run_active,
                    project_active,
                    run_capacity=_compact_capacity(
                        rows, float(run_fraction), multiple
                    ),
                    project_capacity=_compact_capacity(
                        rows, float(project_fraction), multiple
                    ),
                    compact_stats=compact_stats,
                )
        elif not policies:
            capacity = _compact_capacity(rows, fraction, multiple)
            if capacity < rows:
                return _bounded_compact_mlp(
                    layer,
                    proj,
                    hidden_states,
                    route_weights,
                    run_mask,
                    capacity=capacity,
                    valid_rows=valid_rows,
                    compact_stats=compact_stats,
                )

    dense_w1 = layer.mlp.gate_up_proj.weight.unsqueeze(0)
    dense_w2 = layer.mlp.down_proj.weight.unsqueeze(0)
    project_w1 = proj._fused_gate_down_weight().unsqueeze(0)
    project_w2 = proj.up_proj.weight.unsqueeze(0)
    run_output = _one_expert_mlp(
        hidden_states,
        dense_w1,
        dense_w2,
        route_weights,
        run_mask,
    )
    project_output = _one_expert_mlp(
        hidden_states,
        project_w1,
        project_w2,
        1.0 - route_weights,
        ~run_mask,
    )
    # Filtered MoE output is unspecified for inactive rows. Select the active
    # branch on device instead of relying on either kernel to zero its tail.
    return torch.where(run_mask, run_output, project_output)
def _mapped_project_base_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
    *,
    capacity: int,
    valid_rows: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a full cheap PROJECT base and overwrite exact compact RUN rows."""

    if valid_rows is None:
        valid_rows = torch.ones(
            hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device
        )
    output = proj(hidden_states) * (1.0 - route_weights)
    run_active = run_mask.squeeze(-1) & valid_rows
    overflow = _mapped_single_compact_branch(
        layer.mlp,
        hidden_states,
        route_weights,
        run_active,
        output,
        capacity=capacity,
        overflow_w1=layer.mlp.gate_up_proj.weight.unsqueeze(0),
        overflow_w2=layer.mlp.down_proj.weight.unsqueeze(0),
    )
    return output, overflow
def _mapped_run_base_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
    *,
    capacity: int,
    valid_rows: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a full dense base and overwrite exact compact PROJECT rows."""

    if valid_rows is None:
        valid_rows = torch.ones(
            hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device
        )
    output = layer.mlp(hidden_states) * route_weights
    project_active = (~run_mask.squeeze(-1)) & valid_rows
    overflow = _mapped_single_compact_branch(
        proj,
        hidden_states,
        1.0 - route_weights,
        project_active,
        output,
        capacity=capacity,
        overflow_w1=proj._fused_gate_down_weight().unsqueeze(0),
        overflow_w2=proj.up_proj.weight.unsqueeze(0),
    )
    return output, overflow
def _full_dual_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
) -> torch.Tensor:
    """Exact fixed-shape fallback that computes both branches for every row."""

    run_output = layer.mlp(hidden_states)
    project_output = proj(hidden_states)
    if full_graph_gate_mode() == "released":
        # Published FlexiDepth gate: both branches carry the router weight.
        run_output = run_output * route_weights
        project_output = project_output * (1.0 - route_weights)
    # `hard_mask` (straight-through checkpoints): the forward is the hard
    # selection below with NO `w` scaling on either branch.
    return torch.where(run_mask, run_output, project_output)
def _dense_filtered_project_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
) -> torch.Tensor:
    """Run dense for every row and filter only the sparse PROJECT correction."""

    run_output = layer.mlp(hidden_states) * route_weights
    project_output = _one_expert_mlp(
        hidden_states,
        proj._fused_gate_down_weight().unsqueeze(0),
        proj.up_proj.weight.unsqueeze(0),
        1.0 - route_weights,
        ~run_mask,
    )
    return torch.where(run_mask, run_output, project_output)
def _project_filtered_run_mlp(
    layer: Any,
    proj: Any,
    hidden_states: torch.Tensor,
    route_weights: torch.Tensor,
    run_mask: torch.Tensor,
) -> torch.Tensor:
    """Run PROJECT for every row and filter only the sparse dense correction."""

    if full_graph_virtual_cohort_enabled() and hidden_states.is_cuda:
        output = proj(hidden_states) * (1.0 - route_weights)
        _mapped_single_compact_branch(
            layer.mlp,
            hidden_states,
            route_weights,
            run_mask.squeeze(-1),
            output,
            capacity=int(hidden_states.shape[0]),
            overflow_w1=layer.mlp.gate_up_proj.weight.unsqueeze(0),
            overflow_w2=layer.mlp.down_proj.weight.unsqueeze(0),
        )
        return output

    project_output = proj(hidden_states) * (1.0 - route_weights)
    run_output = _one_expert_mlp(
        hidden_states,
        layer.mlp.gate_up_proj.weight.unsqueeze(0),
        layer.mlp.down_proj.weight.unsqueeze(0),
        route_weights,
        run_mask,
    )
    return torch.where(run_mask, run_output, project_output)
