#!/usr/bin/env python3
"""sweep_prediction.py -- the roofline rule's falsification plot and the per-cell table for the RandomSkip sweep(s).

usage: sweep_prediction.py --sweep fixed=generated/sweep --occupancy $DATA/sweep-v1/occupancy.json
                           [--sweep rule=generated/sweep_v2 --occupancy2 $DATA/sweep-v2/occupancy.json]
                           --suite gsm8k_eqw_r10p45 --device NVIDIA_A100 --png figures/sweep_prediction.png
                           --rows generated/sweep_cells_rows.tex --macros generated/sweep_prediction_macros.tex

Per arm r{rate}_d{depth}: the rule's crossover V* = tau*BW/(s*L_r*b) with s*L_r = r*d*16 (tau 2.67 ms per 16 routed layers,
b 4 KB), the fixed band every v1 arm was served under (exit 160k / enter 200k), the arm's OWN derived band (10k grid, enter
1.25 V*), the cell's operating occupancy (p90 resident K/V from occupancy.json: running requests x mean tokens per request),
the fraction of decode passes in the skip body, and the measured E2E mean delta (paired report, %). The plot puts V* on the
x axis (log) against the E2E delta, one marker per arm and per map; the occupancy each cell ran at is drawn as its own tick.
Prediction: an arm whose V* lies above the occupancy it ran at cannot pay under the fixed band; the rule band keeps it on the
stock body instead. Every number is read from the artifacts; nothing is fitted.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

TAU_MS = 2.67; KV_BYTES = 2 * 8 * 128 * 2; ROUTED = 16; BW = {"NVIDIA_A100": 1935e9, "NVIDIA_H100_HBM3": 3350e9, "NVIDIA_H100_NVL": 3350e9}
ARM_RE = re.compile(r"integrated_randomskip_r(\d+)_d(\d+)(?:_alwaysroute)?$")


def vstar(r: float, d: float, device: str) -> float:
    return (TAU_MS * 1e-3) * BW[device] / (r * d * ROUTED * KV_BYTES)


def band(v: float) -> tuple[int, int]:
    return int(round(v / 1e4) * 1e4), int(round(1.25 * v / 1e4) * 1e4)


def load_map(root: Path, suite: str, metric: str = "mean_e2e_latency_ms") -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("paired_report.*.json")):
        rep = json.load(open(p)); arm = rep["treatment"]; m = ARM_RE.match(arm)
        if not m:
            continue
        for row in rep["rows"]:
            if row["suite"] == suite and row["field"] == metric:
                out[arm] = {"r": int(m.group(1)) / 100, "d": int(m.group(2)) / 100, "delta": row["mean_pct"], "n": row["n"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", required=True, help="label=dir with paired_report.<arm>.json (fixed=..., rule=...)")
    ap.add_argument("--occupancy", action="append", default=[], help="label=occupancy.json for that sweep")
    ap.add_argument("--suite", default="gsm8k_eqw_r10p45"); ap.add_argument("--device", default="NVIDIA_A100")
    ap.add_argument("--png", type=Path, required=True); ap.add_argument("--rows", type=Path, required=True); ap.add_argument("--macros", type=Path, required=True)
    a = ap.parse_args()
    sweeps = {k: load_map(Path(v), a.suite) for k, v in (x.split("=", 1) for x in a.sweep)}
    if "ungated" in sweeps:   # step 10b twins: integrated_randomskip_r{r}_d{d}_alwaysroute -> the grid arm's key
        sweeps["ungated"] = {k.replace("_alwaysroute", ""): v for k, v in sweeps["ungated"].items()}
    occ = {}
    for k, v in (x.split("=", 1) for x in a.occupancy):
        occ[k] = {kk.replace("_alwaysroute", ""): vv for kk, vv in next(iter(json.load(open(v)).values())).items()}
    if not sweeps.get("fixed"):
        raise SystemExit("no fixed-band sweep rows found")
    lines, macros, points = [], [], {}
    for arm in sorted(sweeps["fixed"], key=lambda n: (sweeps["fixed"][n]["r"], sweeps["fixed"][n]["d"])):
        r, d = sweeps["fixed"][arm]["r"], sweeps["fixed"][arm]["d"]; v = vstar(r, d, a.device); ex, en = band(v)
        cells = []
        for label in [l for l in ("ungated", "fixed", "rule") if l in sweeps]:   # columns only for the maps that exist (no empty placeholders)
            row = sweeps.get(label, {}).get(arm); o = occ.get(label, {}).get(arm, {}).get(a.suite, {})
            if row is None:
                cells += ["--"] if label == "ungated" else ["--", "--", "--"]; continue
            p90 = o.get("resident_kv_p90_est"); sb = o.get("decode_passes_skip"); ar = o.get("decode_passes_allrun"); rs = o.get("decode_rows_skip_share")
            eng = (sb / (sb + ar)) if (sb is not None and ar) else None
            if label == "ungated":   # every pass routed by construction: one column, the E2E change
                cells += [f"{row['delta']:+.1f}"]
            else:   # engaged = share of decode passes on the routed body / share of decode rows (a routed pass is a large batch)
                cells += [f"{p90/1e3:.0f}k" if p90 else "--", f"{100*eng:.0f}/{100*rs:.0f}\\%" if (eng is not None and rs is not None) else "--", f"{row['delta']:+.1f}"]
            points.setdefault(label, []).append((v, row["delta"], p90, arm))
        lines.append(f"{int(r*100)}\\,\\% & {int(d*100)}\\,\\% & {v/1e3:.0f}k & {ex//1000}k/{en//1000}k & " + " & ".join(cells) + r" \\")
    a.rows.write_text("\n".join(lines) + "\n"); a.macros.write_text("\n".join(macros) + "\n")
    # verdict macros: how many fixed-band arms with V* above their operating occupancy lost, etc.
    fx = points.get("fixed", []); above = [p for p in fx if p[2] and p[0] > p[2]]; below = [p for p in fx if p[2] and p[0] <= p[2]]
    WORD = {25: "Twentyfive", 50: "Fifty", 75: "Seventyfive"}
    def occ_of(arm): return occ.get("fixed", {}).get(arm, {}).get(a.suite, {})
    losers = [p for p in fx if p[1] > 0]; winners = [p for p in fx if p[1] < 0]
    def eng(arm):
        o = occ_of(arm); sb, ar = o.get("decode_passes_skip"), o.get("decode_passes_allrun")
        return 100.0 * sb / (sb + ar) if (sb is not None and ar) else None
    upd = occ.get("fixed", {}); up = (upd.get("upstream") or upd.get("upstream_g1024") or {}).get(a.suite, {}).get("resident_kv_p90_est")   # [v1.6] same-ladder control
    tie = [p for p in above if p[1] < 0]
    with open(a.macros, "a") as f:
        f.write(f"\\newcommand{{\\vpPredAboveN}}{{{len(above)}}}\n\\newcommand{{\\vpPredAboveLost}}{{{sum(1 for p in above if p[1] > 0)}}}\n")
        f.write(f"\\newcommand{{\\vpPredBelowN}}{{{len(below)}}}\n\\newcommand{{\\vpPredBelowWon}}{{{sum(1 for p in below if p[1] < 0)}}}\n")
        if losers: f.write(f"\\newcommand{{\\vpPredLoserOccMin}}{{{min(p[2] for p in losers if p[2])/1e3:.0f}}}\n\\newcommand{{\\vpPredLoserOccMax}}{{{max(p[2] for p in losers if p[2])/1e3:.0f}}}\n")
        if winners: f.write(f"\\newcommand{{\\vpPredWinnerOccMin}}{{{min(p[2] for p in winners if p[2])/1e3:.0f}}}\n\\newcommand{{\\vpPredWinnerOccMax}}{{{max(p[2] for p in winners if p[2])/1e3:.0f}}}\n")
        le = [eng(p[3]) for p in losers if eng(p[3]) is not None]; we = [eng(p[3]) for p in winners if eng(p[3]) is not None]
        if le: f.write(f"\\newcommand{{\\vpPredLoserEngMin}}{{{min(le):.0f}}}\n\\newcommand{{\\vpPredLoserEngMax}}{{{max(le):.0f}}}\n")
        if we: f.write(f"\\newcommand{{\\vpPredWinnerEngMin}}{{{min(we):.0f}}}\n\\newcommand{{\\vpPredWinnerEngMax}}{{{max(we):.0f}}}\n")
        # the same two ranges under each arm's OWN band (panel c): the losers are held to the stock body, the
        # winners route a few percent of decode passes and keep most of their gain (its source is prefill's)
        def eng_rule(arm):
            o = occ.get("rule", {}).get(arm, {}).get(a.suite, {}); sb, ar = o.get("decode_passes_skip"), o.get("decode_passes_allrun")
            return 100.0 * sb / (sb + ar) if (sb is not None and ar) else None
        lr = [eng_rule(p[3]) for p in losers if eng_rule(p[3]) is not None]; wr = [eng_rule(p[3]) for p in winners if eng_rule(p[3]) is not None]
        if lr: f.write(f"\\newcommand{{\\vpPredRuleLoserEngMin}}{{{min(lr):.0f}}}\n\\newcommand{{\\vpPredRuleLoserEngMax}}{{{max(lr):.0f}}}\n")
        if wr: f.write(f"\\newcommand{{\\vpPredRuleWinnerEngMin}}{{{min(wr):.0f}}}\n\\newcommand{{\\vpPredRuleWinnerEngMax}}{{{max(wr):.0f}}}\n")
        if up: f.write(f"\\newcommand{{\\vpPredUpstreamOcc}}{{{up/1e3:.0f}}}\n")
        if up:
            au = [p for p in fx if p[0] > up]; bu = [p for p in fx if p[0] <= up]
            f.write(f"\\newcommand{{\\vpPredAboveUpN}}{{{len(au)}}}\n\\newcommand{{\\vpPredAboveUpLost}}{{{sum(1 for p in au if p[1] > 0)}}}\n")
            f.write(f"\\newcommand{{\\vpPredBelowUpN}}{{{len(bu)}}}\n\\newcommand{{\\vpPredBelowUpWon}}{{{sum(1 for p in bu if p[1] < 0)}}}\n")
        if tie:
            m = ARM_RE.match(tie[0][3]); f.write(f"\\newcommand{{\\vpPredTieArm}}{{{int(m.group(1))}\\%$\\times${int(m.group(2))}\\%}}\n\\newcommand{{\\vpPredTieVstar}}{{{tie[0][0]/1e3:.0f}}}\n\\newcommand{{\\vpPredTieOcc}}{{{tie[0][2]/1e3:.0f}}}\n\\newcommand{{\\vpPredTieDelta}}{{{tie[0][1]:+.1f}}}\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.4, 2.35), dpi=200)
    style = {"ungated": dict(marker="^", color="#7f7f7f", label="no band (a)"), "fixed": dict(marker="o", color="#c0392b", label="shared band (b)"), "rule": dict(marker="s", color="#1f77b4", label="own band (c)")}
    for label, pts in points.items():
        ax.scatter([p[0] / 1e3 for p in pts], [p[1] for p in pts], s=22, zorder=3, **style[label])
        if label == "fixed":
            for v_, dl, p90, arm in pts:
                if p90:
                    ax.plot([v_ / 1e3, p90 / 1e3], [dl, dl], color="#c0392b", lw=0.6, alpha=0.5, zorder=2)
                    ax.plot([p90 / 1e3], [dl], marker="|", color="#c0392b", ms=6, alpha=0.8, zorder=2)
    for label, pts in points.items():
        for v_, dl, p90, arm in pts:
            m = ARM_RE.match(arm); ax.annotate(f"{m.group(1)}/{m.group(2)}", (v_ / 1e3, dl), textcoords="offset points", xytext=(4, -2 if label == "fixed" else 3), fontsize=5, color=style[label]["color"])
    ax.axhline(0, color="k", lw=0.6); ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([100, 200, 300, 500, 1000, 1500])); ax.xaxis.set_major_formatter(FixedFormatter(["100k", "200k", "300k", "500k", "1M", "1.5M"])); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("rule crossover $V^*$ (resident K/V tokens); bar to the cell's own occupancy"); ax.set_ylabel("E2E mean $\\Delta$ (%)")
    ax.legend(fontsize=6, frameon=False, loc="lower right"); ax.tick_params(labelsize=6.5); ax.xaxis.label.set_size(6.5); ax.yaxis.label.set_size(7)
    fig.tight_layout(); fig.savefig(a.png); print(f"{len(lines)} arms; fixed: {len(above)} above their occupancy ({sum(1 for p in above if p[1] > 0)} lost), {len(below)} below ({sum(1 for p in below if p[1] < 0)} won) -> {a.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
