#!/usr/bin/env python3
"""Where each cell's decode occupancy sits relative to the engagement band — and the ridge.

Why this file exists. The regime switch's decode leg is `enter_rows=176` / `exit_rows=144`,
and those are the constants a reviewer will call tuning: chosen by profiling on the same
hardware and workloads the paper reports. The defence is not prose. It is (a) showing the
band separates a regime the workload actually visits, and (b) showing where the band sits
relative to the DEVICE's machine balance.

The arithmetic, which is why row count is the right axis at all: for a weight-stationary
decode GEMM at batch M with bf16 weights, FLOPs = 2*M*K*N and bytes read = 2*K*N, so the
arithmetic intensity is **exactly M**. Below the device ridge (peak FLOP/s / peak GB/s) the
batch is weight-bandwidth-bound and routing rows away saves no traffic while adding overhead;
above it, removed rows are removed work.

**This tool reports; it does not conclude.** The device peaks are DECLARED on the command
line, never guessed from a table inside the tool, because a ridge computed from a number the
tool invented is exactly the manufactured justification this project has already logged. And
the band is declared too — read it off the served design, do not let this file hold a second
copy of a served constant.

  engagement_analysis.py <dir-of-load-jsonl> --enter-rows 176 --exit-rows 144 \\
      --peak-tflops 312 --peak-bw-gbs 1935 [--json OUT]

A cell whose MAXIMUM occupancy never reaches --enter-rows cannot engage the decode leg at
all; that is a property of the cell, and it is printed as such rather than averaged away.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def occupancy(path: Path) -> list[int]:
    """Non-idle decode occupancy samples. Idle samples are not a low-occupancy regime."""
    values = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            running = int(json.loads(line).get("running_requests", 0))
            if running > 0:
                values.append(running)
    return sorted(values)


def quantile(values: list[int], q: float) -> int:
    """Nearest-rank on a pre-sorted list; no interpolation on an integer count."""
    if not values:
        raise ValueError("no samples")
    index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[index]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="directory containing *.load.jsonl")
    ap.add_argument("--enter-rows", type=int, required=True,
                    help="the SERVED design's value; this file must not hold a copy")
    ap.add_argument("--exit-rows", type=int, required=True)
    ap.add_argument("--peak-tflops", type=float, required=True,
                    help="device dense bf16 peak, declared — never guessed here")
    ap.add_argument("--peak-bw-gbs", type=float, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if args.exit_rows > args.enter_rows:
        print("REFUSED: exit_rows above enter_rows is not a hysteresis band", file=sys.stderr)
        return 2
    ridge = args.peak_tflops * 1e12 / (args.peak_bw_gbs * 1e9)
    paths = sorted(args.root.rglob("*.load.jsonl"))
    if not paths:
        print(f"REFUSED: no *.load.jsonl under {args.root}", file=sys.stderr)
        return 2

    print(f"  ridge = {args.peak_tflops:g} TFLOP/s / {args.peak_bw_gbs:g} GB/s "
          f"= {ridge:.0f} rows      band [exit {args.exit_rows}, enter {args.enter_rows}], "
          f"midpoint {(args.enter_rows + args.exit_rows) / 2:.0f}\n")
    print(f"  {'rep/cell':<38} {'arm':<16} {'p50':>5} {'p90':>5} {'max':>6} "
          f"{'>=enter':>8} {'<exit':>7}  engages")
    report: list[dict[str, Any]] = []
    for path in paths:
        values = occupancy(path)
        if not values:
            print(f"  {path.name:<32} {'':<16} {'NO NON-IDLE SAMPLES — refused'}")
            return 2
        # EVERY rep's cell file is named `..._rep1.jsonl` -- the suffix is the runner's
        # per-arm-run repetition index, which is always 1. The CAMPAIGN rep lives only in the
        # directory (`rep3/<dataset>/<arm>/cells/`). Pooling across reps on the filename would
        # give six rows all labelled the same, and either collide or silently keep one. Take
        # the rep from the path, and refuse to pretend it is in the name.
        parts = path.parts
        arm = parts[-3] if len(parts) >= 3 else "?"
        rep = next((x for x in reversed(parts) if x.startswith("rep") and x[3:].isdigit()), "")
        cell = path.name.split("_qps")[0]
        if rep:
            cell = f"{rep}/{cell}"
        above = sum(1 for v in values if v >= args.enter_rows) / len(values) * 100
        below = sum(1 for v in values if v < args.exit_rows) / len(values) * 100
        engages = max(values) >= args.enter_rows
        print(f"  {cell:<38} {arm:<16} {quantile(values, 0.5):>5} "
              f"{quantile(values, 0.9):>5} {max(values):>6} {above:>7.1f}% {below:>6.1f}%"
              f"  {'yes' if engages else 'NEVER'}")
        report.append({"cell": cell, "arm": arm, "rep": rep, "samples": len(values),
                       "p50": quantile(values, 0.5), "p90": quantile(values, 0.9),
                       "max": max(values), "pct_at_or_above_enter": above,
                       "pct_below_exit": below, "can_engage_decode": engages,
                       "ridge_rows": ridge})
    never = [r for r in report if not r["can_engage_decode"]]
    if never:
        print(f"\n  {len(never)} cell(s) NEVER reach enter_rows — their decode leg is inert "
              "by construction, so any effect there is prefill or nothing:")
        for row in never:
            print(f"    {row['cell']} / {row['arm']} (max {row['max']})")
    if args.json:
        args.json.write_text(json.dumps(
            {"enter_rows": args.enter_rows, "exit_rows": args.exit_rows,
             "ridge_rows": ridge, "peak_tflops": args.peak_tflops,
             "peak_bw_gbs": args.peak_bw_gbs, "cells": report}, indent=2) + "\n")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
