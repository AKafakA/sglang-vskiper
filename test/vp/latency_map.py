#!/usr/bin/env python3
"""Latency-below-the-knee / throughput-above-it map (D-753, paper appendix).

From a paired campaign root, per (dataset, rate): pair every request across the two arms by request id
(the work is identical by construction, GR-1a), bin the requests by pinned output tokens, and report the
per-bin mean E2E reduction (negative = treatment faster) with the request count; plus the per-row TPS
gain. Reps are pooled (every rep contributes its per-request pairs). Output: JSON + a tab table.

usage: latency_map.py ROOT --baseline upstream --treatment integrated_it4 [--bins 0,64,128,256,512,1024,2048,8192] [--json out.json]
"""
import argparse, glob, json, os, statistics as st

def cells(root, arm):
    out = {}
    for f in glob.glob(os.path.join(root, "rep*", "*", arm, "cells", "*_rep*.jsonl")):
        if "arrival" in f or ".load." in f: continue
        base = os.path.basename(f); ds = f.split(os.sep)[-4]; rep = f.split(os.sep)[-5]
        suite = base.split("_qps")[0]
        try: d = json.load(open(f))
        except Exception: continue
        out[(ds, suite, rep)] = d
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--baseline", required=True); ap.add_argument("--treatment", required=True)
    ap.add_argument("--bins", default="0,64,128,256,512,1024,2048,8192"); ap.add_argument("--json", default="")
    a = ap.parse_args(); edges = [int(x) for x in a.bins.split(",")]
    B = cells(a.root, a.baseline); T = cells(a.root, a.treatment)
    rows = []
    for key in sorted(set(B) & set(T)):
        ds, suite, rep = key; b, t = B[key], T[key]
        bi = dict(zip(b["request_ids"], zip(b["e2e_latencies"], b["output_lens"], b["ttfts"])))
        ti = dict(zip(t["request_ids"], zip(t["e2e_latencies"], t["output_lens"], t["ttfts"])))
        bins = {}
        for rid, (be, bl, bt) in bi.items():
            if rid not in ti: continue
            te, tl, tt = ti[rid]
            if tl != bl: continue  # GR-1a guarantees equality; skip defensively
            k = next(i for i in range(len(edges) - 1) if edges[i] <= bl < edges[i + 1]) if bl < edges[-1] else len(edges) - 2
            bins.setdefault(k, []).append((be, te, bt, tt))
        for k, v in sorted(bins.items()):
            e2e = 100 * (sum(x[1] for x in v) - sum(x[0] for x in v)) / sum(x[0] for x in v)
            ttft = 100 * (sum(x[3] for x in v) - sum(x[2] for x in v)) / sum(x[2] for x in v)
            rows.append({"dataset": ds, "suite": suite, "rep": rep, "bin_lo": edges[k], "bin_hi": edges[k + 1], "n": len(v),
                         "e2e_reduction_pct": round(e2e, 2), "ttft_reduction_pct": round(ttft, 2)})
        rows.append({"dataset": ds, "suite": suite, "rep": rep, "bin_lo": None, "bin_hi": None, "n": len(bi),
                     "tps_gain_pct": round(100 * (t["output_throughput"] - b["output_throughput"]) / b["output_throughput"], 2),
                     "makespan_reduction_pct": round(100 * (t["duration"] - b["duration"]) / b["duration"], 2),
                     "e2e_mean_reduction_pct": round(100 * (t["mean_e2e_latency_ms"] - b["mean_e2e_latency_ms"]) / b["mean_e2e_latency_ms"], 2)})
    # pooled view across reps per (dataset, suite, bin)
    pooled = {}
    for r in rows:
        if r["bin_lo"] is None: continue
        k = (r["dataset"], r["suite"], r["bin_lo"], r["bin_hi"]); p = pooled.setdefault(k, {"n": 0, "w": 0.0})
        p["n"] += r["n"]; p["w"] += r["e2e_reduction_pct"] * r["n"]
    print(f"{'dataset':8s} {'suite':20s} {'tokens':>12s} {'n':>6s} {'E2E red %':>10s}")
    for (ds, suite, lo, hi), p in sorted(pooled.items()):
        print(f"{ds:8s} {suite:20s} {f'{lo}-{hi}':>12s} {p['n']:6d} {p['w']/p['n']:+10.2f}")
    tps = {}
    for r in rows:
        if r["bin_lo"] is None: tps.setdefault((r["dataset"], r["suite"]), []).append(r["tps_gain_pct"])
    mk = {}
    for r in rows:
        if r["bin_lo"] is None: mk.setdefault((r["dataset"], r["suite"]), []).append(r["makespan_reduction_pct"])
    print("\nmakespan change % (negative = faster; TPS gain in brackets) per row, mean over reps:")
    for k, v in sorted(mk.items()): print(f"  {k[0]:8s} {k[1]:20s} {st.mean(v):+7.2f}  [TPS {st.mean(tps[k]):+6.2f}]  (reps {len(v)})")
    if a.json: json.dump({"rows": rows, "bins": edges}, open(a.json, "w"), indent=1); print("wrote", a.json)

if __name__ == "__main__": main()
