#!/usr/bin/env python3
"""Block B for a Qwen row (v1.6, D-830): the three served arms at the model's own g1024 knee on its lm-eval-capped GSM8K suite
(loaded-<arm>-<ds>-<knee>/cell, scored by score_natural_lane_lmeval.py with per-document rows) -> one table row + macros, with the
routed decode share from each fork arm's attestation delta (as loaded_shares.py) and the paired per-document deltas.
usage: qwen_quality_v16.py --scores scores.json --raw /dev/shm/vpipe/campaign --ds gsm8k_q4b --knee r16p15 --arms up=upstream_g1024 hyb=vskipper_qwen3_4b alw=vskipper_qwen3_4b_alwaysroute
                           --label "Qwen3-4B" --macro QwenFour --rows out.tex --macros out.tex
"""
import argparse, glob, json, math, os, sys
from paired_dod_2x2 import t_crit

def counters(path):
    vp = json.load(open(path))["internal_states"][0]["vp_runtime"]; c = vp["fd_c3"]["counters"]; bc = vp["batch_composition"]
    return {"skip": c["fd_tokens_skip_body"], "dense": c["fd_tokens_prod_allrun_band"], "overflow": c["fd_tokens_dense_overflow"], "passes": bc["decode_passes"]}

def routed_share(raw, arm, ds, knee):
    after = glob.glob(os.path.join(raw, f"loaded-{arm}-{ds}-{knee}", "cell", "*_rep1.server_info.after.json"))
    if len(after) != 1: return None
    x = counters(after[0]); y = counters(after[0].replace(".after.", ".before.")); s, d, o = (x[k] - y[k] for k in ("skip", "dense", "overflow"))
    return 100.0 * s / (s + d + o) if (s + d + o) > 0 else None

def paired(a, b):
    d = [x - y for x, y in zip(a, b)]; n = len(d); m = sum(d) / n; sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return 100 * m, 100 * t_crit(n - 1) * sd / math.sqrt(n)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--raw", required=True); ap.add_argument("--ds", required=True)
    ap.add_argument("--knee", required=True); ap.add_argument("--arms", nargs=3, required=True); ap.add_argument("--label", required=True); ap.add_argument("--macro", required=True)
    ap.add_argument("--rows", required=True); ap.add_argument("--macros", required=True); ap.add_argument("--metric", default="exact_match,marker-composite")
    a = ap.parse_args(); arms = dict(x.split("=", 1) for x in a.arms); sc = json.load(open(a.scores)); per = sc.get("__per_row__", {})
    def cell(arm):
        hits = [k for k in sc if k != "__per_row__" and f"loaded-{arm}-{a.ds}-{a.knee}/" in k]
        if len(hits) != 1: sys.exit(f"FATAL: {len(hits)} scored cells for {arm} {a.ds} {a.knee}")
        return hits[0]
    keys = {r: cell(arm) for r, arm in arms.items()}; q = {r: 100 * sc[k]["scores"][a.metric] for r, k in keys.items()}; n = sc[keys["up"]]["n"]
    rows = {r: per[k]["ok"] for r, k in keys.items() if k in per}
    ids = {r: per[k]["ids"] for r, k in keys.items() if k in per}
    if len({tuple(v) for v in ids.values()}) != 1: sys.exit("FATAL: document ids differ across the three arms")
    d_up, ci_up = paired(rows["hyb"], rows["up"]); d_alw, ci_alw = paired(rows["hyb"], rows["alw"]); d_alwup, ci_alwup = paired(rows["alw"], rows["up"])
    sh = {r: routed_share(a.raw, arms[r], a.ds, a.knee) for r in ("hyb", "alw")}
    shs = " / ".join("--" if sh[r] is None else f"{sh[r]:.0f}" for r in ("hyb", "alw"))
    row = f"{a.label} & {q['up']:.2f} & {q['hyb']:.2f} & {q['alw']:.2f} & {shs} & ${d_up:+.2f} \\pm {ci_up:.2f}$ & ${d_alw:+.2f} \\pm {ci_alw:.2f}$ \\\\"   # v1.7: filter and docs named in the table note
    M = a.macro; mac = [f"\\newcommand{{\\vpLq{M}Up}}{{{q['up']:.2f}}}", f"\\newcommand{{\\vpLq{M}Hyb}}{{{q['hyb']:.2f}}}", f"\\newcommand{{\\vpLq{M}Alw}}{{{q['alw']:.2f}}}",
           f"\\newcommand{{\\vpLq{M}DeltaUp}}{{{d_up:+.2f}}}", f"\\newcommand{{\\vpLq{M}DeltaUpCi}}{{{ci_up:.2f}}}", f"\\newcommand{{\\vpLq{M}Delta}}{{{d_alw:+.2f}}}", f"\\newcommand{{\\vpLq{M}DeltaCi}}{{{ci_alw:.2f}}}",
           f"\\newcommand{{\\vpLq{M}AlwDeltaUp}}{{{d_alwup:+.2f}}}", f"\\newcommand{{\\vpLq{M}AlwDeltaUpCi}}{{{ci_alwup:.2f}}}", f"\\newcommand{{\\vpLq{M}Ndocs}}{{{n:,}}}"]
    for r in ("hyb", "alw"):
        if sh[r] is not None: mac.append(f"\\newcommand{{\\vpLq{M}{'Hyb' if r == 'hyb' else 'Alw'}RoutedShare}}{{{sh[r]:.1f}}}")
    open(a.rows, "a").write(row + "\n"); open(a.macros, "a").write("\n".join(mac) + "\n")
    print(f"{a.label}: n={n} up {q['up']:.2f} hyb {q['hyb']:.2f} alw {q['alw']:.2f} | hyb-up {d_up:+.2f}±{ci_up:.2f} hyb-alw {d_alw:+.2f}±{ci_alw:.2f} | routed {shs}")

if __name__ == "__main__": main()
