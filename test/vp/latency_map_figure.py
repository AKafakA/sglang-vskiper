#!/usr/bin/env python3
"""Two-panel figure for the appendix "latency below the knee, throughput above it" (D-753).

Input: the JSON written by latency_map.py. Left: mean E2E reduction (%) vs offered load (0.75/0.95/1.25 x Q*),
one line per dataset, pooled over reps (negative = faster than upstream). Right: output-throughput gain (%) vs
load, one line per dataset. Also emits macros: the per-row E2E reduction spread across output-length bins
(max - min over bins with >= 50 requests), the evidence that the gain is uniform per request.

usage: latency_map_figure.py MAP_JSON --pdf out.pdf --macros out.tex
"""
import argparse, json, statistics as st
NAMES = {"gsm8k": "GSM8K", "bbh_cot": "BBH-CoT", "coqa": "CoQA"}
MULT = {"r8p25": 0.75, "r10p45": 0.95, "r13p75": 1.25, "r18p75": 0.75, "r23p75": 0.95, "r31p25": 1.25, "r20p25": 0.75, "r25p65": 0.95, "r33p75": 1.25}
MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}; WORD = {0.75: "Low", 0.95: "Mid", 1.25: "High"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("map"); ap.add_argument("--pdf", required=True); ap.add_argument("--macros", required=True)
    a = ap.parse_args(); rows = json.load(open(a.map))["rows"]
    e2e, tps, bins = {}, {}, {}
    for r in rows:
        lbl = r["suite"].rsplit("_r", 1)[1]; m = MULT["r" + lbl]; k = (r["dataset"], m)
        if r["bin_lo"] is None:
            e2e.setdefault(k, []).append(r["e2e_mean_reduction_pct"]); tps.setdefault(k, []).append(r.get("makespan_reduction_pct", -r["tps_gain_pct"] / (1 + r["tps_gain_pct"] / 100)))
        elif r["n"] >= 50:
            bins.setdefault(k, {}).setdefault((r["bin_lo"], r["bin_hi"]), []).append((r["e2e_reduction_pct"], r["n"]))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=200)
    for ds in ("gsm8k", "bbh_cot", "coqa"):
        xs = [m for (d, m) in sorted(e2e) if d == ds]
        ax1.plot(xs, [st.mean(e2e[(ds, m)]) for m in xs], marker="o", label=NAMES[ds])
        ax2.plot(xs, [st.mean(tps[(ds, m)]) for m in xs], marker="o", label=NAMES[ds])
    for ax, yl, t in ((ax1, "mean E2E latency change vs upstream (%)", "latency: grows with load"), (ax2, "makespan change vs upstream (%)", "makespan: no resolved difference below the knee")):
        ax.axhline(0, color="gray", lw=0.6); ax.set_xticks([0.75, 0.95, 1.25]); ax.set_xticklabels(["0.75", "0.95", "1.25"])
        ax.set_xlabel("offered load ($\\times Q^*$)"); ax.set_ylabel(yl, fontsize=8); ax.set_title(t, fontsize=9); ax.grid(alpha=0.3)
    ax1.legend(fontsize=8); fig.tight_layout(); fig.savefig(a.pdf); print("wrote", a.pdf)
    macros = []
    for (ds, m), bb in sorted(bins.items()):
        vals = [sum(v * n for v, n in lst) / sum(n for _, n in lst) for lst in bb.values()]
        macros.append(f"\\newcommand{{\\vpMapSpread{MW[ds]}{WORD[m]}}}{{{max(vals) - min(vals):.1f}}}")
        macros.append(f"\\newcommand{{\\vpMapBins{MW[ds]}{WORD[m]}}}{{{len(vals)}}}")
    for (ds, m) in sorted(e2e):
        macros.append(f"\\newcommand{{\\vpMapEtoE{MW[ds]}{WORD[m]}}}{{{st.mean(e2e[(ds, m)]):+.1f}}}")
        macros.append(f"\\newcommand{{\\vpMapMakespan{MW[ds]}{WORD[m]}}}{{{st.mean(tps[(ds, m)]):+.1f}}}")
    open(a.macros, "w").write("\n".join(macros) + "\n"); print("wrote", a.macros, len(macros), "macros")

if __name__ == "__main__": main()
