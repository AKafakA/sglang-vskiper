#!/usr/bin/env python3
"""Microbenchmark: cohort RUN-branch execution paths at prefill shapes.

Path A: the production one-expert Triton grouped kernel
        (`_one_expert_mlp`), with the tuned config for the row bucket.
Path B: gather -> cuBLAS GEMM pair (gate_up + down, silu-gated) -> scatter
        on the SAME hidden/masks — the compact-GEMM candidate execution.
        Timed twice: at the exact active-row count, and at the
        capacity-padded row count (fixed-shape, graph-capturable posture).

Kernel diagnostic only — timings are not serving-performance evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.vpipe.mlp_compact import (
    _one_expert_mlp,
)


def _time(fn, *, warmups: int, replays: int, trials: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / replays)
    return statistics.median(timings)


def _cublas_path(hidden, w1m, w2m, route_weights, idx, out):
    gathered = hidden.index_select(0, idx)
    g = gathered @ w1m
    inter = g.shape[-1] // 2
    act = torch.nn.functional.silu(g[:, :inter]) * g[:, inter:]
    down = act @ w2m
    down = down * route_weights.index_select(0, idx)
    out.index_copy_(0, idx, down)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 2048, 3072, 4096, 5120])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--active-fraction", type=float, default=0.53)
    parser.add_argument("--capacity-fraction", type=float, default=0.625)
    parser.add_argument("--tuned-config-json", type=Path, default=None,
                        help="Optional per-row-bucket config dict (tuner output format) for path A")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.02)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    if any(r <= 0 for r in args.rows):
        raise ValueError("rows must be positive")
    if not (0.0 < args.active_fraction <= 1.0 and 0.0 < args.capacity_fraction <= 1.0):
        raise ValueError("fractions must be in (0, 1]")
    device = torch.device("cuda")
    torch.manual_seed(0)
    tuned = json.loads(args.tuned_config_json.read_text()) if args.tuned_config_json else None

    for rows in args.rows:
        hidden = torch.empty((rows, args.hidden_size), device=device, dtype=torch.bfloat16).normal_(std=0.2)
        w1 = torch.empty((1, args.intermediate_size * 2, args.hidden_size), device=device, dtype=torch.bfloat16).normal_(std=0.01)
        w2 = torch.empty((1, args.hidden_size, args.intermediate_size), device=device, dtype=torch.bfloat16).normal_(std=0.01)
        route_weights = torch.linspace(0.51, 0.89, rows, device=device, dtype=torch.bfloat16).reshape(rows, 1)
        active_count = min(rows, max(1, int(round(rows * args.active_fraction))))
        active_rows = torch.zeros((rows, 1), device=device, dtype=torch.bool)
        active_rows[:active_count] = True

        # Path A — production grouped kernel, tuned config if provided
        def run_a():
            return _one_expert_mlp(hidden, w1, w2, route_weights, active_rows)

        cfg = tuned.get(str(rows)) if tuned else None
        if cfg is not None:
            with override_config(cfg):
                a_ms = _time(run_a, warmups=args.warmups, replays=args.replays, trials=args.trials)
                ref = run_a()
        else:
            a_ms = _time(run_a, warmups=args.warmups, replays=args.replays, trials=args.trials)
            ref = run_a()

        # Path B — gather + cuBLAS + scatter (weights pre-transposed once,
        # as a real execution mode would hold them)
        w1m = w1[0].t().contiguous()
        w2m = w2[0].t().contiguous()
        mask = active_rows.squeeze(-1)

        # Full path: index construction + fresh output every replay. nonzero()
        # host-syncs, so this OVERSTATES B (a graph-capturable compaction is
        # cheaper) — a conservative bound for the go/no-go decision.
        def run_b():
            idx_t = mask.nonzero(as_tuple=False).squeeze(-1)
            return _cublas_path(hidden, w1m, w2m, route_weights, idx_t, torch.empty_like(hidden))

        b_ms = _time(run_b, warmups=args.warmups, replays=args.replays, trials=args.trials)

        # Kernel-only variant: precomputed index, fresh output per replay.
        idx = mask.nonzero(as_tuple=False).squeeze(-1)

        def run_bk():
            return _cublas_path(hidden, w1m, w2m, route_weights, idx, torch.empty_like(hidden))

        bk_ms = _time(run_bk, warmups=args.warmups, replays=args.replays, trials=args.trials)
        got = _cublas_path(hidden, w1m, w2m, route_weights, idx, torch.empty_like(hidden))
        max_abs = float((got[idx].float() - ref[idx].float()).abs().max().item())
        if not math.isfinite(max_abs) or max_abs > args.atol:
            raise AssertionError(f"rows={rows}: path B mismatch max_abs={max_abs} > atol={args.atol}")

        # Path B at capacity padding (fixed-shape posture): pad with UNIQUE
        # inactive-row indices, matching production's fixed-capacity mapping.
        cap_count = min(rows, max(active_count, int(round(rows * args.capacity_fraction))))
        pad_needed = cap_count - idx.numel()
        if pad_needed > 0:
            inactive = (~mask).nonzero(as_tuple=False).squeeze(-1)
            if inactive.numel() < pad_needed:
                raise AssertionError(f"rows={rows}: not enough inactive rows to pad capacity")
            idx_cap = torch.cat([idx, inactive[:pad_needed]])
        else:
            idx_cap = idx[:cap_count]

        def run_c():
            return _cublas_path(hidden, w1m, w2m, route_weights, idx_cap, torch.empty_like(hidden))

        c_ms = _time(run_c, warmups=args.warmups, replays=args.replays, trials=args.trials)

        print(
            "PATH_BENCH rows=%d active=%d cap=%d triton_ms=%.4f cublas_full_ms=%.4f "
            "cublas_kernel_ms=%.4f cublas_cap_ms=%.4f speedup_full=%.3f speedup_kernel=%.3f "
            "cap_speedup=%.3f max_abs=%.5f"
            % (rows, active_count, cap_count, a_ms, b_ms, bk_ms, c_ms, a_ms / b_ms, a_ms / bk_ms, a_ms / c_ms, max_abs),
            flush=True,
        )


if __name__ == "__main__":
    main()
