#!/usr/bin/env python3
"""Prefill crossover implied by a device's tile artifact alone (no serving measurement, no workload).

At a routed layer the dense body runs gate_up + down over all N chunk tokens (cuBLAS, the artifact's reference
timings); the routed body runs the count-bounded gate_up + down over the (1-s)N RUN rows and the projector's
gate-down + up over the sN PROJECT rows (the artifact's tuned timings). Both are read from
binary_cohort_configs/<device>.json (count -> ms per band, piecewise-linear between bands), so the crossover is a
property of the device's kernels as measured once by the tuner. `--extras` adds a per-layer allowance for the
routed body's non-GEMM launches (router GEMM + norm, pack/scatter, silu, run-mask multiply), which the artifact
does not time.

usage: prefill_crossover_from_artifact.py <artifact.json> [--skip 0.5] [--extras-ms 0.1] [--chunks 256,...]
"""
import argparse, bisect, json

def points(a, op, key):
    return sorted((int(k.split("@")[1]), a[k][key]) for k in a if k.startswith(op + "@"))

def interp(p, x):
    xs = [c for c, _ in p]; ys = [v for _, v in p]
    if x <= xs[0]: return ys[0] * x / xs[0]
    if x >= xs[-1]: return ys[-1] * x / xs[-1]
    i = bisect.bisect_left(xs, x); x0, x1 = xs[i - 1], xs[i]; y0, y1 = ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact"); ap.add_argument("--skip", type=float, default=0.5)
    ap.add_argument("--extras-ms", type=float, default=0.1)
    ap.add_argument("--chunks", default="384,512,640,768,896,1024,1280,1536,2048,3072,4096")
    a = ap.parse_args(); art = json.load(open(a.artifact))
    dense = {op: points(art, op, "cublas_ms") for op in ("gateup", "down")}
    routed = {op: points(art, op, "best_ms") for op in ("gateup", "down", "projgd", "projup")}
    first = None; first_x = None
    print(f"{'N':>6s} {'dense ms':>9s} {'routed ms':>10s} {'ratio':>6s} {'+extras':>8s}")
    for N in (int(c) for c in a.chunks.split(",")):
        m, mp = N * (1 - a.skip), N * a.skip
        td = sum(interp(dense[o], N) for o in dense)
        tr = sum(interp(routed[o], m) for o in ("gateup", "down")) + sum(interp(routed[o], mp) for o in ("projgd", "projup"))
        r, rx = tr / td, (tr + a.extras_ms) / td
        first = first or (N if r < 1 else None); first_x = first_x or (N if rx < 1 else None)
        print(f"{N:6d} {td:9.3f} {tr:10.3f} {r:6.2f} {rx:8.2f}")
    print(f"crossover (routed < dense): MLP-only from {first}; with {a.extras_ms} ms/layer extras from {first_x}")

if __name__ == "__main__":
    main()
