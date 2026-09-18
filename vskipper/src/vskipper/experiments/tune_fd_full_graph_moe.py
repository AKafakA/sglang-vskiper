#!/usr/bin/env python3
"""Tune the two V3 one-expert kernels on an existing GPU environment.

This is a kernel diagnostic used to select Triton launch configurations. Its
timings are not serving-performance evidence and must not be used as headline
results.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import triton

from sglang.srt.layers.moe.moe_runner.triton_utils import (
    get_config_file_name,
    override_config,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
    get_default_config,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from vskipper.kernels.mlp_compact import (
    _one_expert_mlp,
)


def _candidate_configs() -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    for block_m in (16, 32, 64):
        for block_n in (64, 128):
            for block_k in (32, 64):
                for num_warps in (4, 8):
                    configs.append(
                        {
                            "BLOCK_SIZE_M": block_m,
                            "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_K": block_k,
                            "GROUP_SIZE_M": 1,
                            "num_warps": num_warps,
                            "num_stages": 3,
                        }
                    )
    for block_m in (16, 32, 64):
        for num_warps in (4, 8):
            configs.append(
                {
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                    "num_warps": num_warps,
                    "num_stages": 2,
                }
            )
    for block_m in (16, 32):
        configs.append(
            {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 8,
                "num_stages": 3,
            }
        )
    configs.append(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    )
    # Large-M candidates for prefill cohort shapes (M >= ~1K rows). The space
    # above caps BLOCK_SIZE_M at 64 (decode-era), which starves prefill-shaped
    # launches into small tiles. Invalid combinations (e.g. shared-memory
    # overflow) fail per-config and are recorded in `failures` by the caller.
    for block_m in (128, 256):
        for block_n in (128, 256):
            for block_k in (32, 64):
                for group_m in (1, 8, 16):
                    for num_stages in (3, 4):
                        configs.append(
                            {
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": group_m,
                                "num_warps": 8,
                                "num_stages": num_stages,
                            }
                        )
    return configs


def _weight(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.bfloat16).normal_(std=0.01)


def _make_inputs(
    *,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    active_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.empty(
        (rows, hidden_size), device=device, dtype=torch.bfloat16
    ).normal_(std=0.2)
    w1 = _weight((1, intermediate_size * 2, hidden_size), device)
    w2 = _weight((1, hidden_size, intermediate_size), device)
    route_weights = torch.linspace(
        0.51, 0.89, rows, device=device, dtype=torch.bfloat16
    ).reshape(rows, 1)
    active_count = min(rows, max(1, int(round(rows * active_fraction))))
    active_rows = torch.zeros((rows, 1), device=device, dtype=torch.bool)
    active_rows[:active_count] = True
    return hidden, w1, w2, route_weights, active_rows


def _run(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    route_weights: torch.Tensor,
    active_rows: torch.Tensor,
) -> torch.Tensor:
    return _one_expert_mlp(
        hidden, w1, w2, route_weights, active_rows
    )


def _time_config(
    config: dict[str, int],
    inputs: tuple[torch.Tensor, ...],
    *,
    warmups: int,
    replays: int,
    trials: int,
) -> tuple[float, torch.Tensor]:
    timings = []
    output = None
    with override_config(config):
        for _ in range(warmups):
            output = _run(*inputs)
        torch.cuda.synchronize(inputs[0].device)
        for _ in range(trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                output = _run(*inputs)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) / replays)
    assert output is not None
    return statistics.median(timings), output


def _active_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
    active_rows: torch.Tensor,
) -> float:
    """Max abs error over active rows, scale-normalized.

    The absolute component alone is too weak: with std=0.01 weights the
    reference output can be small enough that a candidate silently zeroing
    rows would pass an absolute atol. The relative-Frobenius term catches
    whole-output dropping regardless of scale; it is scaled to atol so the
    single caller-side atol gate covers both.
    """
    active = active_rows.squeeze(-1)
    a = actual[active].float()
    e = expected[active].float()
    max_abs = float((a - e).abs().max().item())
    rel_fro = float(((a - e).norm() / e.norm().clamp_min(1e-12)).item())
    if rel_fro > 0.05:
        # Fail via the caller's existing isfinite gate, independent of atol.
        return float("inf")
    return max_abs


def _tune_branch(
    *,
    name: str,
    hidden_size: int,
    intermediate_size: int,
    rows: list[int],
    active_fraction: float,
    configs: list[dict[str, int]],
    warmups: int,
    replays: int,
    trials: int,
    confirmation_replays: int,
    confirmation_trials: int,
    atol: float,
    device: torch.device,
) -> tuple[dict[str, dict[str, int]], dict[str, Any]]:
    selected: dict[str, dict[str, int]] = {}
    row_results: dict[str, Any] = {}
    for row_count in rows:
        inputs = _make_inputs(
            rows=row_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            active_fraction=active_fraction,
            device=device,
        )
        default_config = get_default_config(
            row_count,
            1,
            intermediate_size,
            hidden_size,
            1,
            None,
            False,
        )
        _default_ms, reference = _time_config(
            default_config,
            inputs,
            warmups=warmups,
            replays=replays,
            trials=trials,
        )
        candidates = []
        failures = []
        coarse_configs = [default_config, *configs]
        unique_configs = []
        for config in coarse_configs:
            if config not in unique_configs:
                unique_configs.append(config)
        for index, config in enumerate(unique_configs):
            try:
                latency_ms, output = _time_config(
                    config,
                    inputs,
                    warmups=warmups,
                    replays=replays,
                    trials=trials,
                )
                max_abs = _active_error(output, reference, inputs[-1])
                if not math.isfinite(max_abs) or max_abs > atol:
                    raise AssertionError(
                        f"active-row max_abs={max_abs} exceeds atol={atol}"
                    )
                candidates.append(
                    {
                        "index": index,
                        "latency_ms": latency_ms,
                        "max_abs": max_abs,
                        "config": config,
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "index": index,
                        "config": config,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        if not candidates:
            raise RuntimeError(f"{name} rows={row_count} has no valid configuration")
        candidates.sort(key=lambda value: value["latency_ms"])
        confirmation_configs = [default_config]
        for candidate in candidates[:5]:
            if candidate["config"] not in confirmation_configs:
                confirmation_configs.append(candidate["config"])
        confirmed = []
        for config in confirmation_configs:
            latency_ms, output = _time_config(
                config,
                inputs,
                warmups=warmups,
                replays=confirmation_replays,
                trials=confirmation_trials,
            )
            max_abs = _active_error(output, reference, inputs[-1])
            if not math.isfinite(max_abs) or max_abs > atol:
                raise AssertionError(
                    f"confirmation active-row max_abs={max_abs} exceeds atol={atol}"
                )
            confirmed.append(
                {
                    "latency_ms": latency_ms,
                    "max_abs": max_abs,
                    "config": config,
                }
            )
        confirmed.sort(key=lambda value: value["latency_ms"])
        best = confirmed[0]
        default_confirmed = next(
            value for value in confirmed if value["config"] == default_config
        )
        speedup_pct = 100.0 * (
            1.0 - best["latency_ms"] / default_confirmed["latency_ms"]
        )
        selected[str(row_count)] = best["config"]
        row_results[str(row_count)] = {
            "active_rows": int(inputs[-1].sum().item()),
            "default": default_confirmed,
            "selected": best,
            "selected_speedup_pct": speedup_pct,
            "coarse_top_five": candidates[:5],
            "confirmed": confirmed,
            "valid_config_count": len(candidates),
            "failed_config_count": len(failures),
            "failures": failures,
        }
        del inputs, reference
        torch.cuda.empty_cache()
        print(
            "FD_MOE_TUNE_ROW "
            f"branch={name} rows={row_count} latency_ms={best['latency_ms']:.6f} "
            f"speedup_pct={speedup_pct:.3f} "
            f"config={json.dumps(best['config'], sort_keys=True)}",
            flush=True,
        )
    return selected, row_results


def _write_configs(
    output_dir: Path,
    *,
    intermediate_size: int,
    selected: dict[str, dict[str, int]],
) -> list[str]:
    version_dir = f"triton_{triton.__version__.replace('.', '_')}"
    config_dir = output_dir / "configs" / version_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for down_moe in (False, True):
        name = get_config_file_name(
            1, intermediate_size, None, down_moe=down_moe
        )
        path = config_dir / name
        path.write_text(
            json.dumps(selected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(str(path))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32]
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dense-size", type=int, default=14336)
    parser.add_argument("--project-size", type=int, default=896)
    parser.add_argument("--dense-active-fraction", type=float, default=0.44)
    parser.add_argument("--project-active-fraction", type=float, default=0.56)
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=("dense", "project"),
        default=["dense", "project"],
        help="Kernel branches to tune; omitted branches keep SGLang defaults.",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--confirmation-replays", type=int, default=100)
    parser.add_argument("--confirmation-trials", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.02)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output_dir.exists():
        raise ValueError(f"refusing to reuse output directory {args.output_dir}")
    if any(rows <= 0 for rows in args.rows):
        raise ValueError("all row buckets must be positive")
    if min(
        args.warmups,
        args.replays,
        args.trials,
        args.confirmation_replays,
        args.confirmation_trials,
    ) <= 0:
        raise ValueError("warmup, replay, and trial counts must be positive")
    for value in (args.dense_active_fraction, args.project_active_fraction):
        if not 0.0 < value <= 1.0:
            raise ValueError("active fractions must be in (0, 1]")

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    torch.manual_seed(20260712)
    device = torch.device("cuda")
    configs = _candidate_configs()
    branch_specs = {
        "dense": (args.dense_size, args.dense_active_fraction),
        "project": (args.project_size, args.project_active_fraction),
    }
    branches = tuple(
        (name, *branch_specs[name]) for name in dict.fromkeys(args.branches)
    )
    payload: dict[str, Any] = {
        "status": "passed",
        "scope": "kernel-configuration-diagnostic-only",
        "performance_evidence_eligible": False,
        "device": torch.cuda.get_device_name(device),
        "triton_version": triton.__version__,
        "dtype": str(torch.bfloat16),
        "hidden_size": args.hidden_size,
        "row_buckets": args.rows,
        "candidate_config_count": len(configs),
        "branches": {},
    }
    args.output_dir.mkdir(parents=True)
    for name, intermediate_size, active_fraction in branches:
        selected, results = _tune_branch(
            name=name,
            hidden_size=args.hidden_size,
            intermediate_size=intermediate_size,
            rows=args.rows,
            active_fraction=active_fraction,
            configs=configs,
            warmups=args.warmups,
            replays=args.replays,
            trials=args.trials,
            confirmation_replays=args.confirmation_replays,
            confirmation_trials=args.confirmation_trials,
            atol=args.atol,
            device=device,
        )
        payload["branches"][name] = {
            "intermediate_size": intermediate_size,
            "active_fraction": active_fraction,
            "selected": selected,
            "rows": results,
            "config_files": _write_configs(
                args.output_dir,
                intermediate_size=intermediate_size,
                selected=selected,
            ),
        }
    output = args.output_dir / "tuning_summary.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"FD_MOE_TUNE=PASSED output={output}", flush=True)


if __name__ == "__main__":
    main()
