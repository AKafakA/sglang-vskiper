#!/usr/bin/env python3
"""Tune the binary-cohort count-GEMM tiles for THIS device (D-842, RTX A6000 / sm_86).

Kernel diagnostic only: its timings select Triton launch configurations and are never
serving evidence. For every artifact key (op @ count band) it times ``count_matmul_gridexit``
on the op's real Llama-3-8B shapes over the candidate tiles that fit the device's shared
memory, keeps the fastest, and gates it on ``torch.equal`` against the artifact's current
config so the served numerics do not move (BLOCK_K is held at the current config's value:
the accumulation order is what changes numerics; BLOCK_M/N, warps and stages do not).
cuBLAS (``torch.mm`` on the packed rows) is recorded beside it as ``cublas_ms``.

usage: tune_count_gemm_tiles.py --artifact python/sglang/srt/vpipe/binary_cohort_configs/NVIDIA_RTX_A6000.json
       [--hidden 4096 --intermediate 14336 --bottleneck 896] [--replays 20] [--min-gain 0.03] --out <json>
"""
from __future__ import annotations

import argparse, json, statistics, time
from pathlib import Path

import torch

from sglang.srt.vpipe.kernel import count_matmul_gridexit

OPS = {  # op -> (K, N) for Llama-3-8B + FlexiDepth (bottleneck 896)
    "gateup": lambda h, i, b: (h, 2 * i),
    "down": lambda h, i, b: (i, h),
    "projgd": lambda h, i, b: (h, 2 * b),
    "projup": lambda h, i, b: (b, h),
}


def smem_bytes(bm, bn, bk, stages, elt=2):
    return stages * (bm * bk + bk * bn) * elt


def time_ms(fn, warmups, replays):
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(3):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(replays):
            fn()
        e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / replays)
    return statistics.median(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--intermediate", type=int, default=14336)
    ap.add_argument("--bottleneck", type=int, default=896)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.03, help="replace only if faster by this fraction")
    ap.add_argument("--smem-limit", type=int, default=None, help="bytes per block; default = device property")
    a = ap.parse_args()
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    limit = a.smem_limit or int(getattr(props, "shared_memory_per_block_optin", 0) or 101376)
    name = torch.cuda.get_device_name(dev)
    art = json.loads(a.artifact.read_text())
    torch.manual_seed(0)
    cands = [(bm, bn, w, st) for bm in (32, 64, 128) for bn in (64, 128, 256) for w in (4, 8) for st in (2, 3, 4)]
    result = {}
    t0 = time.time()
    for key, rec in art.items():
        op, _, count_text = key.partition("@")
        rows = int(count_text)
        K, N = OPS[op](a.hidden, a.intermediate, a.bottleneck)
        cur = list(rec["config"]); bk = cur[2]
        A = torch.empty((rows, K), device=dev, dtype=torch.float16).normal_(std=0.2)
        W = torch.empty((N, K), device=dev, dtype=torch.float16).normal_(std=0.02)
        count = torch.tensor([rows], device=dev, dtype=torch.int32)
        out = torch.zeros((rows, N), device=dev, dtype=torch.float16)

        def run(cfg, o=out):
            bm, bn, w, st = cfg
            count_matmul_gridexit(A, W, count, o, block_m=bm, block_n=bn, block_k=bk,
                                  num_warps=w, num_stages=st)
        ref = torch.zeros_like(out); run((cur[0], cur[1], cur[3], cur[4]), ref); torch.cuda.synchronize()
        cur_ms = time_ms(lambda: run((cur[0], cur[1], cur[3], cur[4])), a.warmups, a.replays)
        cublas_ms = time_ms(lambda: torch.mm(A, W.t()), a.warmups, a.replays)
        best = (cur_ms, tuple(cur[:2] + cur[3:]))
        tried = 0
        for bm, bn, w, st in cands:
            if smem_bytes(bm, bn, bk, st) > limit or (bm, bn, w, st) == best[1]:
                continue
            try:
                probe = torch.zeros_like(out); run((bm, bn, w, st), probe); torch.cuda.synchronize()
            except Exception:
                continue
            if not torch.equal(probe, ref):
                continue
            ms = time_ms(lambda: run((bm, bn, w, st)), a.warmups, a.replays); tried += 1
            if ms < best[0]:
                best = (ms, (bm, bn, w, st))
        gain = 1 - best[0] / cur_ms
        keep = gain >= a.min_gain
        cfg = [best[1][0], best[1][1], bk, best[1][2], best[1][3]] if keep else cur
        result[key] = {
            "cublas_ms": round(cublas_ms, 4), "best_ms": round(best[0] if keep else cur_ms, 4),
            "ratio": round((best[0] if keep else cur_ms) / cublas_ms, 3), "config": cfg,
            "tuner": f"tune_count_gemm_tiles.py on {name} ({props.multi_processor_count} SMs, smem/block {limit} B), "
                     f"BLOCK_K held at {bk}, torch.equal-gated vs the carried config; {tried} candidates; "
                     f"carried {cur} {cur_ms:.3f} ms -> {cfg} {(best[0] if keep else cur_ms):.3f} ms "
                     f"({'replaced' if keep else 'kept'}, gain {gain*100:+.1f} %)",
        }
        print(f"{key:14s} K={K:5d} N={N:5d} carried {cur_ms:7.3f} ms  best {best[0]:7.3f} ms ({gain*100:+5.1f} %) "
              f"cuBLAS {cublas_ms:7.3f} ms  -> {cfg} {'REPLACED' if keep else 'kept'}", flush=True)
        del A, W, out, ref
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {a.out} ({len(result)} keys) in {time.time()-t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
