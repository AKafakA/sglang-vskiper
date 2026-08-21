#!/usr/bin/env python3
"""Compare V3 conditional execution with compact and cohort oracles.

This is a synthetic CUDA-graph kernel diagnostic. It is not serving or
performance evidence. The compact/cohort paths use a frozen route partition;
they estimate the execution opportunity available to a layer-aware scheduler.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn.functional as F

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.vpipe.cohort import (
    pack_cohort,
    scatter_cohort,
)
from sglang.srt.vpipe.routing import (
    build_route_maps,
)
from sglang.srt.vpipe.mlp import (
    fd_conditional_mlp_full_graph,
)
from sglang.srt.vpipe.mlp_compact import (
    _aligned_branch_partition,
)
from sglang.srt.vpipe.mlp_compact import (
    _bounded_compact_mlp,
)


class _Projector:
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        gate = _weight((intermediate_size, hidden_size), device, dtype)
        down = _weight((intermediate_size, hidden_size), device, dtype)
        self.gate_proj = SimpleNamespace(weight=gate)
        self.down_proj = SimpleNamespace(weight=down)
        self.up_proj = SimpleNamespace(
            weight=_weight((hidden_size, intermediate_size), device, dtype)
        )
        self._fused = torch.cat((gate, down), dim=0).contiguous()

    def _fused_gate_down_weight(self) -> torch.Tensor:
        return self._fused

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        return _project_mlp(self, hidden)


class _DenseMLP:
    def __init__(
        self,
        hidden_size: int,
        dense_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.gate_up_proj = SimpleNamespace(
            weight=_weight((dense_size * 2, hidden_size), device, dtype)
        )
        self.down_proj = SimpleNamespace(
            weight=_weight((hidden_size, dense_size), device, dtype)
        )

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, up = F.linear(hidden, self.gate_up_proj.weight).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.down_proj.weight)


def _weight(
    shape: tuple[int, ...], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=dtype).normal_(std=0.01)


def _build_modules(
    hidden_size: int,
    dense_size: int,
    project_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[SimpleNamespace, _Projector]:
    layer = SimpleNamespace(
        mlp=_DenseMLP(
            hidden_size,
            dense_size,
            device=device,
            dtype=dtype,
        )
    )
    return layer, _Projector(
        hidden_size, project_size, device=device, dtype=dtype
    )


def _dense_mlp(layer: SimpleNamespace, hidden: torch.Tensor) -> torch.Tensor:
    gate, up = F.linear(hidden, layer.mlp.gate_up_proj.weight).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, layer.mlp.down_proj.weight)


def _project_mlp(project: _Projector, hidden: torch.Tensor) -> torch.Tensor:
    gate, down = F.linear(hidden, project._fused_gate_down_weight()).chunk(
        2, dim=-1
    )
    return F.linear(F.silu(gate) * down, project.up_proj.weight)


def _route_partition(
    rows: int, run_fraction: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.arange(rows, device=device, dtype=torch.int64)
    scores = ((indices * 1103515245 + 12345) % 2147483647).float()
    mask = scores < (run_fraction * 2147483647)
    if rows > 1:
        mask[0] = True
        mask[-1] = False
    run_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    project_indices = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
    weights = torch.where(
        mask,
        torch.full_like(scores, 0.8),
        torch.full_like(scores, 0.2),
    ).to(torch.bfloat16).reshape(rows, 1)
    return mask.reshape(rows, 1), weights, run_indices, project_indices


def _compact_mlp(
    layer: SimpleNamespace,
    project: _Projector,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    run_indices: torch.Tensor,
    project_indices: torch.Tensor,
) -> torch.Tensor:
    run_hidden = hidden.index_select(0, run_indices)
    project_hidden = hidden.index_select(0, project_indices)
    run_output = _dense_mlp(layer, run_hidden) * weights.index_select(
        0, run_indices
    )
    project_output = _project_mlp(project, project_hidden) * (
        1.0 - weights.index_select(0, project_indices)
    )
    output = torch.empty_like(hidden)
    output.index_copy_(0, run_indices, run_output)
    output.index_copy_(0, project_indices, project_output)
    return output


def _bounded_no_overflow_branch(
    module: Callable[[torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    weights: torch.Tensor,
    active: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Frozen-route ceiling for the fixed-capacity path without overflow work."""

    order = torch.argsort(active.to(torch.int8), descending=True, stable=True)
    rows = order[:capacity]
    selected_active = active.index_select(0, rows)
    selected_output = module(hidden.index_select(0, rows))
    selected_output = selected_output * weights.index_select(0, rows)
    selected_output = torch.where(
        selected_active.unsqueeze(-1),
        selected_output,
        torch.zeros_like(selected_output),
    )
    output = torch.zeros_like(hidden)
    output.index_copy_(0, rows, selected_output)
    return output


def _aligned_no_overflow_branch(
    module: Callable[[torch.Tensor], torch.Tensor],
    hidden: torch.Tensor,
    weights: torch.Tensor,
    active: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    rows, selected_active, _, _ = _aligned_branch_partition(active, capacity)
    selected_output = module(hidden.index_select(0, rows))
    selected_output = selected_output * weights.index_select(0, rows)
    selected_output = torch.where(
        selected_active.unsqueeze(-1),
        selected_output,
        torch.zeros_like(selected_output),
    )
    output = torch.zeros_like(hidden)
    output.index_copy_(0, rows, selected_output)
    return output


def _bounded_no_overflow_mlp(
    layer: SimpleNamespace,
    project: _Projector,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    run_mask: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    run_active = run_mask.squeeze(-1)
    project_active = ~run_active
    return _bounded_no_overflow_branch(
        layer.mlp, hidden, weights, run_active, capacity
    ) + _bounded_no_overflow_branch(
        project, hidden, 1.0 - weights, project_active, capacity
    )


def _aligned_no_overflow_mlp(
    layer: SimpleNamespace,
    project: _Projector,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    run_mask: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    run_active = run_mask.squeeze(-1)
    project_active = ~run_active
    return _aligned_no_overflow_branch(
        layer.mlp, hidden, weights, run_active, capacity
    ) + _aligned_no_overflow_branch(
        project, hidden, 1.0 - weights, project_active, capacity
    )


def _mapped_no_overflow_mlp(
    layer: SimpleNamespace,
    project: _Projector,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    run_mask: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    run_active = run_mask.squeeze(-1)
    project_active = ~run_active
    run_rows, project_rows, counts = build_route_maps(run_active, project_active)
    run_hidden, run_weights = pack_cohort(
        hidden, weights, run_rows, counts[0:1], capacity=capacity
    )
    project_hidden, project_weights = pack_cohort(
        hidden,
        1.0 - weights,
        project_rows,
        counts[1:2],
        capacity=capacity,
    )
    run_output = layer.mlp(run_hidden) * run_weights
    project_output = project(project_hidden) * project_weights
    output = torch.zeros_like(hidden)
    scatter_cohort(output, run_output, run_rows, counts[0:1])
    scatter_cohort(output, project_output, project_rows, counts[1:2])
    return output


def _capture_and_time(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmups: int,
    replays: int,
    trials: int,
) -> tuple[float, object]:
    side_stream = torch.cuda.Stream(device=device)
    side_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side_stream):
        output = None
        for _ in range(warmups):
            output = function()
    side_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool, stream=side_stream):
        output = function()
    side_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize(device)

    timings = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / replays)
    latency_ms = statistics.median(timings)
    del graph, pool, side_stream
    torch.cuda.empty_cache()
    return latency_ms, output


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[8, 32, 64, 128, 256, 384, 512]
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dense-size", type=int, default=14336)
    parser.add_argument("--project-size", type=int, default=896)
    parser.add_argument("--run-fraction", type=float, default=0.486)
    parser.add_argument(
        "--compact-fractions",
        type=float,
        nargs="+",
        default=[0.5625, 0.625, 0.6875, 0.75],
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 < args.run_fraction < 1.0:
        raise ValueError("run fraction must be strictly between zero and one")
    if any(rows < 2 for rows in args.rows):
        raise ValueError("all row counts must be at least two")
    if any(not 0.0 < value < 1.0 for value in args.compact_fractions):
        raise ValueError("compact fractions must be in (0, 1)")

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    torch.manual_seed(20260713)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    layer, project = _build_modules(
        args.hidden_size,
        args.dense_size,
        args.project_size,
        device,
        dtype,
    )

    results = []
    for rows in args.rows:
        hidden = torch.empty(
            (rows, args.hidden_size), device=device, dtype=dtype
        ).normal_(std=0.2)
        run_mask, weights, run_indices, project_indices = _route_partition(
            rows, args.run_fraction, device
        )
        run_hidden = hidden.index_select(0, run_indices)
        project_hidden = hidden.index_select(0, project_indices)
        run_weights = weights.index_select(0, run_indices)
        project_weights = weights.index_select(0, project_indices)

        production_ms, _ = _capture_and_time(
            lambda: _dense_mlp(layer, hidden),
            device=device,
            warmups=args.warmups,
            replays=args.replays,
            trials=args.trials,
        )
        current_ms, current_output = _capture_and_time(
            lambda: fd_conditional_mlp_full_graph(
                layer, project, hidden, weights, run_mask
            ),
            device=device,
            warmups=args.warmups,
            replays=args.replays,
            trials=args.trials,
        )
        compact_ms, compact_output = _capture_and_time(
            lambda: _compact_mlp(
                layer,
                project,
                hidden,
                weights,
                run_indices,
                project_indices,
            ),
            device=device,
            warmups=args.warmups,
            replays=args.replays,
            trials=args.trials,
        )
        cohort_ms, _ = _capture_and_time(
            lambda: (
                _dense_mlp(layer, run_hidden) * run_weights,
                _project_mlp(project, project_hidden) * (1.0 - project_weights),
            ),
            device=device,
            warmups=args.warmups,
            replays=args.replays,
            trials=args.trials,
        )

        bounded_results = {}
        for fraction in args.compact_fractions:
            capacity = min(rows - 1, max(1, int((rows * fraction + 15) // 16 * 16)))
            route_fits_capacity = (
                max(run_indices.numel(), project_indices.numel()) <= capacity
            )
            compact_stats = []
            bounded_ms, bounded_output = _capture_and_time(
                lambda capacity=capacity, compact_stats=compact_stats: _bounded_compact_mlp(
                    layer,
                    project,
                    hidden,
                    weights,
                    run_mask,
                    capacity=capacity,
                    valid_rows=torch.ones_like(run_mask.squeeze(-1)),
                    compact_stats=compact_stats,
                ),
                device=device,
                warmups=args.warmups,
                replays=args.replays,
                trials=args.trials,
            )
            bounded_abs = _max_abs(bounded_output, compact_output)
            if bounded_abs > args.atol:
                raise AssertionError(
                    f"rows={rows} fraction={fraction} bounded/compact "
                    f"max_abs={bounded_abs} exceeds {args.atol}"
                )
            result = {
                "capacity": capacity,
                "route_fits_capacity": route_fits_capacity,
                "latency_ms": bounded_ms,
                "reduction_vs_production_pct": 100.0
                * (1.0 - bounded_ms / production_ms),
                "compact_max_abs": bounded_abs,
            }
            if route_fits_capacity:
                ceilings = (
                    ("no_overflow", _bounded_no_overflow_mlp),
                    ("aligned_no_overflow", _aligned_no_overflow_mlp),
                    ("mapped_no_overflow", _mapped_no_overflow_mlp),
                )
                for name, function in ceilings:
                    ceiling_ms, ceiling_output = _capture_and_time(
                        lambda function=function, capacity=capacity: function(
                            layer,
                            project,
                            hidden,
                            weights,
                            run_mask,
                            capacity,
                        ),
                        device=device,
                        warmups=args.warmups,
                        replays=args.replays,
                        trials=args.trials,
                    )
                    ceiling_abs = _max_abs(ceiling_output, compact_output)
                    if ceiling_abs > args.atol:
                        raise AssertionError(
                            f"rows={rows} fraction={fraction} {name}/compact "
                            f"max_abs={ceiling_abs} exceeds {args.atol}"
                        )
                    result[f"{name}_latency_ms"] = ceiling_ms
                    result[f"{name}_reduction_vs_production_pct"] = 100.0 * (
                        1.0 - ceiling_ms / production_ms
                    )
                    result[f"{name}_compact_max_abs"] = ceiling_abs
            bounded_results[str(fraction)] = result

        max_abs = _max_abs(current_output, compact_output)
        if max_abs > args.atol:
            raise AssertionError(
                f"rows={rows} current/compact max_abs={max_abs} exceeds {args.atol}"
            )
        row = {
            "rows": rows,
            "run_rows": int(run_indices.numel()),
            "project_rows": int(project_indices.numel()),
            "current_compact_max_abs": max_abs,
            "latency_ms": {
                "production_dense": production_ms,
                "current_filtered_moe": current_ms,
                "static_compact_gather_scatter": compact_ms,
                "ideal_cohort_no_movement": cohort_ms,
            },
            "bounded_compact": bounded_results,
            "reduction_vs_production_pct": {
                "current_filtered_moe": 100.0 * (1.0 - current_ms / production_ms),
                "static_compact_gather_scatter": 100.0
                * (1.0 - compact_ms / production_ms),
                "ideal_cohort_no_movement": 100.0
                * (1.0 - cohort_ms / production_ms),
            },
        }
        results.append(row)
        print("FD_CONDITIONAL_ORACLE_ROW " + json.dumps(row, sort_keys=True))

    payload = {
        "status": "passed",
        "scope": "synthetic-conditional-execution-oracle",
        "performance_evidence_eligible": False,
        "device": torch.cuda.get_device_name(device),
        "dtype": str(dtype),
        "run_fraction": args.run_fraction,
        "rows": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
