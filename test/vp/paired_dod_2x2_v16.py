#!/usr/bin/env python3
"""The 2x2 quality gate for the ladder-control campaign (v1.6, D-824): arms A/B are the native PyTorch halves (identity-preserved
from the v1.5 lm-eval runs, `samples_*.jsonl`), arms C/D are the SERVED halves re-measured as block-B knee cells (the frozen
quality suite served at 0.95 x Q* against upstream_g1024, scored per document by score_natural_lane_lmeval.py with lm-eval's own
filters; `__per_row__` vectors keyed by request id). Documents are paired by their dataset index (gsm8k:test:<i> <-> doc_id i;
coqa:validation:<doc>:<turn> <-> doc_id doc; bbh_cot:<task>:test:<i> <-> bbh_cot_fewshot_<task> doc_id i). Same statistic,
same macros and row format as paired_dod_2x2.py.

usage: paired_dod_2x2_v16.py --dataset gsm8k --arm A=<native base samples dir> --arm B=<native FD samples dir>
                             --scores loaded_all_v16.json --cell-c <cell key substring> --cell-d <cell key substring>
                             [--margin 1.0] [--latex rows.tex] [--macros macros.tex] [--json out.json] [--macro-prefix vpQ]
"""
import argparse, json, math, sys
from paired_dod_2x2 import load_arm, t_crit

MACRO = {"gsm8k": "Gsm", "coqa": "Coqa", "bbh_cot": "Bbh"}
DISP = {"gsm8k": "GSM8K", "coqa": "CoQA", "bbh_cot": "BBH"}
METRIC_LABEL = {"gsm8k": "\\texttt{exact\\_match,composite}", "coqa": "\\texttt{f1,none}", "bbh_cot": "\\texttt{exact\\_match,get-answer}"}


def key_of(dataset: str, rid: str):
    p = rid.split(":")
    if dataset == "gsm8k": return ("gsm8k", int(p[2]))
    if dataset == "coqa": return ("coqa", int(p[2]))
    if dataset == "bbh_cot": return ("bbh_cot_fewshot_" + p[1], int(p[3]))
    raise SystemExit(dataset)


def served_arm(scores: dict, dataset: str, needle: str):
    hits = [k for k in scores["__per_row__"] if needle in k]
    if len(hits) != 1: sys.exit(f"FATAL: {len(hits)} per-row vectors match {needle!r} (need exactly one): {hits[:3]}")
    rec = scores["__per_row__"][hits[0]]
    return {key_of(dataset, rid): float(ok) for rid, ok in zip(rec["ids"], rec["ok"])}, hits[0], rec["metric"]


def stats(xs):
    n = len(xs); m = sum(xs) / n; sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean_pp": 100 * m, "ci95_half_pp": 100 * t_crit(n - 1) * sd / math.sqrt(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(MACRO), required=True); ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--scores", required=True); ap.add_argument("--cell-c", required=True); ap.add_argument("--cell-d", required=True)
    ap.add_argument("--margin", type=float, default=None); ap.add_argument("--latex"); ap.add_argument("--macros"); ap.add_argument("--json"); ap.add_argument("--macro-prefix", default="vpQ")
    ap.add_argument("--no-filter-col", action="store_true", help="omit the filter column (the paper names the filters in a table note)")
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arm)
    A, B = load_arm(arms["A"], a.dataset), load_arm(arms["B"], a.dataset)
    sc = json.load(open(a.scores)); C, cpath, cmetric = served_arm(sc, a.dataset, a.cell_c); D, dpath, dmetric = served_arm(sc, a.dataset, a.cell_d)
    common = sorted(set(A) & set(B) & set(C) & set(D))
    if not common: sys.exit("FATAL: no common documents across the four arms")
    dropped = {q: len(v) - len(common) for q, v in (("A", A), ("B", B), ("C", C), ("D", D))}
    ba = [B[k] - A[k] for k in common]; dc = [D[k] - C[k] for k in common]; dod = [d - b for d, b in zip(dc, ba)]
    m = {q: 100 * sum(v[k] for k in common) / len(common) for q, v in (("A", A), ("B", B), ("C", C), ("D", D))}
    z = stats(dod); rep = {"dataset": a.dataset, "n_docs": len(common), "dropped": dropped, "means": m, "B_minus_A": stats(ba), "D_minus_C": stats(dc), "dod": z,
                           "dod_lower_pp": z["mean_pp"] - z["ci95_half_pp"], "arm_C_cell": cpath, "arm_D_cell": dpath, "served_metric": cmetric,
                           "arms_AB": {"A": arms["A"], "B": arms["B"]}, "note": "C/D = block-B knee cells (frozen suite served at 0.95xQ* vs upstream_g1024), per-document lm-eval filter scores; A/B = native lm-eval runs"}
    print(f"{a.dataset}: n={len(common)} A {m['A']:.2f} B {m['B']:.2f} C {m['C']:.2f} D {m['D']:.2f} | B-A {rep['B_minus_A']['mean_pp']:+.2f} D-C {rep['D_minus_C']['mean_pp']:+.2f} | DoD {z['mean_pp']:+.2f} ± {z['ci95_half_pp']:.2f} pp (dropped {dropped})")
    if a.json: json.dump(rep, open(a.json, "w"), indent=1)
    mac = MACRO[a.dataset]; P = a.macro_prefix
    if a.margin is not None:
        upper = z["mean_pp"] + z["ci95_half_pp"]
        gate = "pass" if rep["dod_lower_pp"] > -a.margin else ("FAIL" if upper < -a.margin else "unresolved (crosses $-\\epsilon$)")
    else:
        gate = "--"
    gate = gate.replace("unresolved (crosses $-\\epsilon$)", "unresolved")
    filt = "" if a.no_filter_col else f"{METRIC_LABEL[a.dataset]} & "
    row = (f"{DISP[a.dataset]} & {filt}{m['A']:.2f} & {m['B']:.2f} & {m['C']:.2f} & {m['D']:.2f} & "
           f"{rep['B_minus_A']['mean_pp']:+.2f} & {rep['D_minus_C']['mean_pp']:+.2f} & {z['mean_pp']:+.2f} $\\pm$ {z['ci95_half_pp']:.2f} & {gate} \\\\")
    macros = [f"\\newcommand{{\\{P}{mac}BminusA}}{{{rep['B_minus_A']['mean_pp']:+.2f}}}", f"\\newcommand{{\\{P}{mac}DminusC}}{{{rep['D_minus_C']['mean_pp']:+.2f}}}",
              f"\\newcommand{{\\{P}{mac}Dod}}{{{z['mean_pp']:+.2f}}}", f"\\newcommand{{\\{P}{mac}DodCi}}{{{z['ci95_half_pp']:.2f}}}",
              f"\\newcommand{{\\{P}{mac}Ndocs}}{{{len(common):,}}}", f"\\newcommand{{\\{P}{mac}ArmD}}{{{m['D']/100:.4f}}}", f"\\newcommand{{\\{P}{mac}ArmC}}{{{m['C']/100:.4f}}}",
              f"\\newcommand{{\\{P}{mac}DodLower}}{{{rep['dod_lower_pp']:+.2f}}}", f"\\newcommand{{\\{P}{mac}ArmCPct}}{{{m['C']:.2f}}}",
              f"\\newcommand{{\\{P}{mac}ArmA}}{{{m['A']/100:.4f}}}", f"\\newcommand{{\\{P}{mac}ArmB}}{{{m['B']/100:.4f}}}",
              f"\\newcommand{{\\{P}{mac}ArmAPct}}{{{m['A']:.2f}}}", f"\\newcommand{{\\{P}{mac}ArmBPct}}{{{m['B']:.2f}}}", f"\\newcommand{{\\{P}{mac}ArmDPct}}{{{m['D']:.2f}}}"]
    if a.latex: open(a.latex, "a").write(row + "\n")
    if a.macros: open(a.macros, "a").write("\n".join(macros) + "\n")
    return 0


if __name__ == "__main__": sys.exit(main())
