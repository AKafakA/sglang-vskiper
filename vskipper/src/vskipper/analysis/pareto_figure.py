#!/usr/bin/env python3
"""Three Pareto panels (v1.6, D-829): per dataset, task quality at the knee (block-B lm-eval score, y) against the mean end-to-end
latency at the same knee (x, seconds), one point per served arm: upstream (base model), the hybrid we serve, always-route.
Latency comes from the paired reports (the headline report for upstream and the hybrid, n = 6; the ablation report for
always-route, n = 1: its own paired baseline's absolute is reported as that cell's upstream latency), quality from the
block-B scores (one repetition, per-document interval). Emits the PDF and macros \vpPareto<Ds><Arm>{Lat,Q}.

usage: pareto_figure.py --headline paired_report.json --ablation-dir <dir with <ds>/paired_report.upstream_g1024__integrated_alwaysskip.json>
                        --scores loaded_all_v16.json --knee gsm8k=r12p35 --knee bbh_cot=r32p3 --knee coqa=r23p75 --pdf out.pdf --macros out.tex
"""
import argparse, json, os, sys
DS = {"gsm8k": ("GSM8K", "Gsm", "exact_match,marker-composite"), "bbh_cot": ("BBH", "Bbh", "exact_match,get-answer"), "coqa": ("CoQA", "Coqa", "f1")}
ARMS = [("upstream_g1024", "Up", "base + upstream", "o"), ("vskipper", "Hyb", "hybrid (served)", "s"), ("integrated_alwaysskip", "Alw", "always-route", "^")]


def e2e(rep, ds, lbl):
    for r in rep["rows"]:
        if r["dataset"] == ds and r["suite"].endswith(lbl) and r["metric"] == "E2E mean":
            return r["baseline_abs"] / 1000.0, r["treatment_abs"] / 1000.0
    sys.exit(f"FATAL: no E2E mean row for {ds} {lbl}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--headline", required=True); ap.add_argument("--ablation-dir", required=True)
    ap.add_argument("--scores", required=True); ap.add_argument("--knee", action="append", default=[]); ap.add_argument("--pdf", required=True); ap.add_argument("--macros", required=True)
    a = ap.parse_args(); knees = dict(x.split("=") for x in a.knee)
    head = json.load(open(a.headline)); sc = json.load(open(a.scores)); pts = {}; macros = []
    for ds, (name, mac, metric) in DS.items():
        lbl = knees[ds]; up, hyb = e2e(head, ds, lbl)
        abl = json.load(open(os.path.join(a.ablation_dir, ds, "paired_report.upstream_g1024__integrated_alwaysskip.json")))
        _, alw = e2e(abl, ds, lbl)
        lat = {"upstream_g1024": up, "vskipper": hyb, "integrated_alwaysskip": alw}
        q = {}
        for arm, _, _, _ in ARMS:
            hits = [k for k, v in sc.items() if k != "__per_row__" and f"loaded-{arm}-{ds}-{lbl}/" in k]
            if len(hits) != 1: sys.exit(f"FATAL: {len(hits)} block-B cells for {arm} {ds} {lbl}")
            q[arm] = 100 * sc[hits[0]]["scores"][metric]
        pts[ds] = (lat, q)
        for arm, am, _, _ in ARMS:
            macros += [f"\\newcommand{{\\vpPareto{mac}{am}Lat}}{{{lat[arm]:.1f}}}", f"\\newcommand{{\\vpPareto{mac}{am}Q}}{{{q[arm]:.1f}}}"]
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # v1.7: one shared legend above the three panels, nothing inside a panel; per-task y labels (the scores differ in metric).
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    colors = {"upstream_g1024": "#2171b5", "vskipper": "#cb181d", "integrated_alwaysskip": "#fd8d3c"}
    ylab = {"gsm8k": "GSM8K accuracy (%)", "bbh_cot": "BBH exact match (%)", "coqa": "CoQA F1 (%)"}
    for ax, (ds, (name, mac, metric)) in zip(axes, DS.items()):
        lat, q = pts[ds]
        for arm, am, label, marker in ARMS:
            ax.scatter(lat[arm], q[arm], marker=marker, s=44, color=colors[arm], edgecolor="black", linewidth=0.5, label=label, zorder=3)
        ax.set_title(f"{name} at the knee", fontsize=8); ax.set_xlabel("mean E2E latency (s)", fontsize=7); ax.set_ylabel(ylab[ds], fontsize=7)
        ax.tick_params(labelsize=6.5); ax.grid(alpha=0.3)
        ys = list(q.values()); pad = max(1.0, 0.15 * (max(ys) - min(ys) + 1)); ax.set_ylim(min(ys) - pad, max(ys) + pad)
        xs = list(lat.values()); xpad = max(0.5, 0.12 * (max(xs) - min(xs) + 1)); ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(a.pdf); open(a.macros, "w").write("\n".join(macros) + "\n")
    print(f"3 panels -> {a.pdf}; {len(macros)} macros -> {a.macros}")
    for ds, (lat, q) in pts.items(): print(" ", ds, {k: (round(lat[k], 1), round(q[k], 1)) for k in lat})


if __name__ == "__main__": main()
