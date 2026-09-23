#!/usr/bin/env python3
"""RandomSkip sweep as an E2E heatmap (D-754; layout after the Astrolabe SYSTOR'26 sensitivity figure).

Reads a sweep root's per-arm reports (`paired_report.integrated_randomskip_r{R}_d{D}.json`, written by
run_paired_campaign.py for a spec with several treatments) and renders two panels — E2E mean (left) and
E2E p95 (right) reduction vs the in-session upstream anchor at one (dataset, rate) — x = token skip rate,
y = skipped-depth ratio, cells annotated with the signed percentage (negative = faster than upstream).
The served arm's value at the same row (from the headline report) is marked on each colour bar.

usage: sweep_heatmap.py SWEEP_ROOT --dataset gsm8k --suite gsm8k_eqw_r10p45 [--headline paired_report.json]
                        --png out.png --macros out.tex [--title "..."]
"""
import argparse, glob, json, os, re, sys

def load_rows(path):
    d = json.load(open(path)); return d["rows"] if isinstance(d, dict) else d

def value(rows, dataset, suite, metric):
    for r in rows:
        if r["dataset"] == dataset and r["suite"] == suite and r["metric"] == metric:
            return r["mean_pct"], r.get("ci95_half"), r.get("n")
    return None, None, None

def word(n):
    return {10: "Ten", 25: "Twentyfive", 50: "Fifty", 75: "Seventyfive"}.get(n, str(n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--dataset", required=True); ap.add_argument("--suite", required=True)
    ap.add_argument("--headline", default=""); ap.add_argument("--png", required=True); ap.add_argument("--macros", required=True)
    ap.add_argument("--title", default=""); ap.add_argument("--metrics", default="E2E mean,E2E p95")
    ap.add_argument("--macro-prefix", default="vpSweep", help="macro name prefix (v1.4 map: vpSweep; the per-arm-band map: vpSweepTwo)")
    a = ap.parse_args()
    pat = re.compile(r"paired_report\.integrated_randomskip_r(\d+)_d(\d+)(?:_alwaysroute)?\.json$")  # the ungated (always-route) twins of step 10b share the grid
    grid = {}
    for f in sorted(glob.glob(os.path.join(a.root, "paired_report.*.json"))):
        m = pat.search(f)
        if not m: continue
        r, d = int(m.group(1)), int(m.group(2))
        rows = load_rows(f)
        grid[(r, d)] = {met: value(rows, a.dataset, a.suite, met) for met in a.metrics.split(",")}
    if not grid: sys.exit(f"no sweep reports under {a.root}")
    rates = sorted({k[0] for k in grid}); depths = sorted({k[1] for k in grid})
    served = {}
    if a.headline:
        hrows = load_rows(a.headline)
        for met in a.metrics.split(","):
            served[met] = value(hrows, a.dataset, a.suite, met)[0]
    import matplotlib; matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt, numpy as np
    mets = a.metrics.split(",")
    # v1.7: a single-metric panel is drawn compact and large-typed (it is placed three-across in the paper); no text tag on the
    # colour bar (the served arm's value is only a line there, named in the caption).
    single = len(mets) == 1
    fig, axes = plt.subplots(1, len(mets), figsize=((3.4, 3.1) if single else (4.6 * len(mets), 3.9)), dpi=200)
    fs_cell, fs_tick, fs_label, fs_title = (14, 11, 11, 11) if single else (9, 9, 10, 10)
    axes = np.atleast_1d(axes)
    macros = []
    for ax, met in zip(axes, mets):
        M = np.full((len(depths), len(rates)), np.nan)
        for i, dp in enumerate(depths):
            for j, rt in enumerate(rates):
                v = grid.get((rt, dp), {}).get(met, (None,))[0]
                if v is not None: M[i, j] = v
        lo = min(np.nanmin(M), served.get(met, 0) or 0, 0.0); hi = max(np.nanmax(M), 0.0)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-max(abs(lo), abs(hi)), vmax=max(abs(lo), abs(hi)), aspect="auto")
        ax.set_xticks(range(len(rates))); ax.set_xticklabels([f"{r}%" for r in rates], fontsize=fs_tick)
        ax.set_yticks(range(len(depths))); ax.set_yticklabels([f"{d}%" for d in depths], fontsize=fs_tick)
        ax.set_xlabel("rows routed $r$", fontsize=fs_label); ax.set_ylabel("routed layers skipped $d$", fontsize=fs_label)
        ax.set_title(a.title if (single and a.title) else f"{met} change vs upstream (%)", fontsize=fs_title)
        for i in range(len(depths)):
            for j in range(len(rates)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=fs_cell,
                            color="white" if abs(M[i, j]) > 0.55 * max(abs(lo), abs(hi)) else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=fs_tick - 1)
        if met in served and served[met] is not None:
            cb.ax.axhline(served[met], color="#1f5fbf", lw=2)
        tag = (met.replace("E2E ", "EtoE").replace("TTFT ", "Ttft").replace("TPOT ", "Tpot")
               .replace("mean", "Mean").replace("p95", "Pninetyfive").replace("p99", "Pninetynine"))
        if not tag.isalpha(): sys.exit(f"FATAL: metric {met!r} has no macro-safe name")
        for (rt, dp), vals in grid.items():
            v = vals[met][0]
            if v is not None: macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}R{word(rt)}D{word(dp)}}}{{{v:+.1f}}}")
        finite = [(grid[k][met][0], k) for k in grid if grid[k][met][0] is not None]
        if finite:
            best = min(finite); worst = max(finite)
            macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}Best}}{{{best[0]:+.1f}}}")
            macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}BestCell}}{{r{best[1][0]}\\,d{best[1][1]}}}")
            macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}Worst}}{{{worst[0]:+.1f}}}")
            macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}WorstCell}}{{r{worst[1][0]}\\,d{worst[1][1]}}}")
        if met in served and served[met] is not None:
            macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}Served}}{{{served[met]:+.1f}}}")
    if a.title and not single: fig.suptitle(a.title, fontsize=10)
    fig.tight_layout(); fig.savefig(a.png, **({"metadata": {"CreationDate": None}} if str(a.png).endswith(".pdf") else {})); print("wrote", a.png)
    open(a.macros, "w").write("\n".join(macros) + "\n"); print("wrote", a.macros, f"({len(macros)} macros)")

if __name__ == "__main__": main()
