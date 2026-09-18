#!/usr/bin/env python3
"""Profile exact per-layer majority policies at observed FlexiDepth ratios."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from bench_fd_conditional_oracle import (
    _build_modules,
    _capture_and_time,
    _compact_mlp,
    _dense_mlp,
    _max_abs,
    _route_partition,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from vskipper.kernels.mlp import (
    _dense_filtered_project_mlp,
    _full_dual_mlp,
    _mapped_project_base_mlp,
    _mapped_run_base_mlp,
    _project_filtered_run_mlp,
    fd_conditional_mlp_full_graph,
)
from vskipper.kernels.mlp_compact import (
    _mapped_asymmetric_compact_mlp,
)


def _capacity(rows: int, fraction: float, multiple: int = 16) -> int:
    return min(rows - 1, max(1, math.ceil(rows * fraction / multiple) * multiple))


def _asymmetric_capacity_fraction(
    route_fraction: float, margin: float, grid: float = 0.0625
) -> float:
    return min(1.0, math.ceil((route_fraction + margin) / grid) * grid)


def _full_capacity(rows: int, fraction: float, multiple: int = 16) -> int:
    return min(rows, max(1, math.ceil(rows * fraction / multiple) * multiple))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[256, 384, 512])
    parser.add_argument(
        "--run-fractions",
        type=float,
        nargs="+",
        default=[0.03, 0.06, 0.10, 0.325, 0.586, 0.90, 0.98],
    )
    parser.add_argument(
        "--branch-capacities",
        type=float,
        nargs="+",
        default=[0.125, 0.25, 0.375, 0.625],
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dense-size", type=int, default=14336)
    parser.add_argument("--project-size", type=int, default=896)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--asymmetric-margin", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if any(not 0.0 < value < 1.0 for value in args.run_fractions):
        raise ValueError("run fractions must be in (0, 1)")
    if any(not 0.0 < value < 1.0 for value in args.branch_capacities):
        raise ValueError("branch capacities must be in (0, 1)")
    if not 0.0 <= args.asymmetric_margin < 1.0:
        raise ValueError("asymmetric margin must be in [0, 1)")

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
            (rows, args.hidden_size), dtype=dtype, device=device
        ).normal_(std=0.2)
        production_ms, _ = _capture_and_time(
            lambda: _dense_mlp(layer, hidden),
            device=device,
            warmups=args.warmups,
            replays=args.replays,
            trials=args.trials,
        )
        for run_fraction in args.run_fractions:
            run_mask, weights, run_indices, project_indices = _route_partition(
                rows, run_fraction, device
            )
            reference_ms, reference = _capture_and_time(
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
            current_ms, current = _capture_and_time(
                lambda: fd_conditional_mlp_full_graph(
                    layer, project, hidden, weights, run_mask
                ),
                device=device,
                warmups=args.warmups,
                replays=args.replays,
                trials=args.trials,
            )
            dual_ms, dual = _capture_and_time(
                lambda: _full_dual_mlp(
                    layer, project, hidden, weights, run_mask
                ),
                device=device,
                warmups=args.warmups,
                replays=args.replays,
                trials=args.trials,
            )
            dense_filtered_ms, dense_filtered = _capture_and_time(
                lambda: _dense_filtered_project_mlp(
                    layer, project, hidden, weights, run_mask
                ),
                device=device,
                warmups=args.warmups,
                replays=args.replays,
                trials=args.trials,
            )
            project_filtered_ms, project_filtered = _capture_and_time(
                lambda: _project_filtered_run_mlp(
                    layer, project, hidden, weights, run_mask
                ),
                device=device,
                warmups=args.warmups,
                replays=args.replays,
                trials=args.trials,
            )
            run_capacity_fraction = _asymmetric_capacity_fraction(
                run_fraction, args.asymmetric_margin
            )
            project_capacity_fraction = _asymmetric_capacity_fraction(
                1.0 - run_fraction, args.asymmetric_margin
            )
            run_capacity = _full_capacity(rows, run_capacity_fraction)
            project_capacity = _full_capacity(rows, project_capacity_fraction)
            asymmetric_ms, asymmetric = _capture_and_time(
                lambda: _mapped_asymmetric_compact_mlp(
                    layer,
                    project,
                    hidden,
                    weights,
                    run_mask.squeeze(-1),
                    ~run_mask.squeeze(-1),
                    run_capacity=run_capacity,
                    project_capacity=project_capacity,
                    compact_stats=None,
                ),
                device=device,
                warmups=args.warmups,
                replays=args.replays,
                trials=args.trials,
            )
            candidates = {}
            for capacity_fraction in args.branch_capacities:
                capacity = _capacity(rows, capacity_fraction)
                for name, function in (
                    ("project_base", _mapped_project_base_mlp),
                    ("run_base", _mapped_run_base_mlp),
                ):
                    latency_ms, output = _capture_and_time(
                        lambda function=function, capacity=capacity: function(
                            layer,
                            project,
                            hidden,
                            weights,
                            run_mask,
                            capacity=capacity,
                        )[0],
                        device=device,
                        warmups=args.warmups,
                        replays=args.replays,
                        trials=args.trials,
                    )
                    max_abs = _max_abs(output, reference)
                    if max_abs > args.atol:
                        raise AssertionError(
                            f"rows={rows} run={run_fraction} {name} "
                            f"capacity={capacity} max_abs={max_abs}"
                        )
                    candidates[f"{name}_{capacity_fraction}"] = {
                        "capacity": capacity,
                        "latency_ms": latency_ms,
                        "reduction_vs_production_pct": 100.0
                        * (1.0 - latency_ms / production_ms),
                        "compact_max_abs": max_abs,
                    }
            row = {
                "rows": rows,
                "run_fraction_target": run_fraction,
                "run_rows": int(run_indices.numel()),
                "project_rows": int(project_indices.numel()),
                "production_ms": production_ms,
                "reference_compact_ms": reference_ms,
                "current_filtered_ms": current_ms,
                "current_filtered_reduction_pct": 100.0
                * (1.0 - current_ms / production_ms),
                "current_compact_max_abs": _max_abs(current, reference),
                "full_dual_ms": dual_ms,
                "full_dual_reduction_pct": 100.0
                * (1.0 - dual_ms / production_ms),
                "full_dual_compact_max_abs": _max_abs(dual, reference),
                "dense_filtered_project_ms": dense_filtered_ms,
                "dense_filtered_project_reduction_pct": 100.0
                * (1.0 - dense_filtered_ms / production_ms),
                "dense_filtered_project_max_abs": _max_abs(
                    dense_filtered, reference
                ),
                "project_filtered_run_ms": project_filtered_ms,
                "project_filtered_run_reduction_pct": 100.0
                * (1.0 - project_filtered_ms / production_ms),
                "project_filtered_run_max_abs": _max_abs(
                    project_filtered, reference
                ),
                "asymmetric_compact_ms": asymmetric_ms,
                "asymmetric_compact_reduction_pct": 100.0
                * (1.0 - asymmetric_ms / production_ms),
                "asymmetric_compact_max_abs": _max_abs(asymmetric, reference),
                "asymmetric_run_capacity_fraction": run_capacity_fraction,
                "asymmetric_project_capacity_fraction": (
                    project_capacity_fraction
                ),
                "candidates": candidates,
            }
            if row["asymmetric_compact_max_abs"] > args.atol:
                raise AssertionError(
                    f"rows={rows} run={run_fraction} asymmetric "
                    f"max_abs={row['asymmetric_compact_max_abs']}"
                )
            if row["dense_filtered_project_max_abs"] > args.atol:
                raise AssertionError(
                    f"rows={rows} run={run_fraction} dense-filtered "
                    f"max_abs={row['dense_filtered_project_max_abs']}"
                )
            if row["project_filtered_run_max_abs"] > args.atol:
                raise AssertionError(
                    f"rows={rows} run={run_fraction} project-filtered "
                    f"max_abs={row['project_filtered_run_max_abs']}"
                )
            results.append(row)
            print("FD_LAYER_POLICY_ROW " + json.dumps(row, sort_keys=True))

    payload = {
        "status": "passed",
        "scope": "synthetic-layer-policy-oracle",
        "performance_evidence_eligible": False,
        "device": torch.cuda.get_device_name(device),
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
