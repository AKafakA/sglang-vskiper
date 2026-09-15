"""Paired, per-document difference-of-differences for the 2x2 faithfulness gate.

summarize_2x2.py reads lm-eval's aggregate `results_*.json` and reports the d-o-d with an
interval that is the quadrature of the four single-run standard errors -- an UNPAIRED
construction. Every arm scores the same documents, so the honest statistic is paired: for
document i, x_i = (D_i - C_i) - (B_i - A_i); the gate reads mean(x) with a t interval from
sd(x)/sqrt(n). This reads lm-eval's `samples_<task>_*.jsonl` (--log_samples) for the four
arms, matches documents by (task, doc_id), and refuses if any arm is missing a document.

usage: paired_dod_2x2.py --dataset gsm8k --arm A=<dir> --arm B=<dir> --arm C=<dir> --arm D=<dir> [--json out]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

#: (samples-file task prefix, filter, metric key) per dataset -- declared, as in summarize_2x2.
# GSM8K uses BOTH of lm-eval's published filters under one rule (see load_arm): strict-match where
# it parses, flexible extraction where it does not. Neither alone can compare these arms -- strict
# cannot parse ~20 % of the BASE model's generations (no marker, or "#### $21" whose $ is outside
# its [0-9.,] class) while flexible extraction's last-number rule is fooled by the confidence
# epilogue the CHECKPOINT appends (~3 %, and never on the base model). The failures fall on
# opposite arms, so either filter alone biases the comparison.
SPEC = {
    "gsm8k": ("samples_gsm8k_", "__composite__", "exact_match"),
    "bbh_cot": ("samples_bbh_cot_fewshot_", "get-answer", "exact_match"),
    "coqa": ("samples_coqa_", "none", "f1"),
}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980}


def t_crit(df: int) -> float:
    for k in sorted(T95):
        if df <= k:
            return T95[k]
    return 1.960


def load_arm(root: str, dataset: str) -> dict[tuple[str, int], float]:
    prefix, flt, metric = SPEC[dataset]
    files = sorted(glob.glob(os.path.join(root, "**", f"{prefix}*.jsonl"), recursive=True))
    if not files:
        sys.exit(f"FATAL: no {prefix}*.jsonl under {root}")
    # one file per task; if a task was run twice keep the newest (timestamped name)
    newest: dict[str, str] = {}
    for f in files:
        task = os.path.basename(f)[len("samples_"):].rsplit("_20", 1)[0]
        newest[task] = f
    out: dict[tuple[str, int], float] = {}
    if flt == "__composite__":
        # collect every filter's verdict AND its extracted value per document, then combine
        rows: dict[tuple[str, int], dict[str, dict]] = {}
        for task, f in newest.items():
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    fr = r.get("filtered_resps")
                    # fail closed: a row without an extraction is malformed, never "parsed"
                    if fr is None or (isinstance(fr, list) and not fr):
                        sys.exit(f"FATAL {f}: doc {r.get('doc_id')} filter {r.get('filter')!r} has no filtered_resps")
                    val = fr[0] if isinstance(fr, list) else fr
                    rows.setdefault((task, int(r["doc_id"])), {})[r.get("filter", "none")] = {
                        "ok": float(r[metric]), "val": str(val).strip()}
        for key, per in rows.items():
            if not {"strict-match", "flexible-extract"} <= set(per):
                sys.exit(f"FATAL: {key} lacks both GSM8K filters; cannot apply the composite rule")
            parsed = per["strict-match"]["val"] not in ("", "[invalid]")
            out[key] = per["strict-match"]["ok"] if parsed else per["flexible-extract"]["ok"]
        return out
    for task, f in newest.items():
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("filter", "none") != flt:
                    continue
                out[(task, int(r["doc_id"]))] = float(r[metric])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SPEC), required=True)
    ap.add_argument("--arm", action="append", default=[], help="A=<dir> ... (all four required)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--macro-prefix", default="vpQ", help="LaTeX macro prefix (vpQ for the Llama 2x2; e.g. vpQwen for the Qwen row)")
    ap.add_argument("--macros", default=None, help="macros file (default: quality_macros.tex beside --latex)")
    ap.add_argument("--margin", type=float, default=None,
                    help="non-inferiority margin epsilon in pp (owner: 1.0): the gate passes when the "
                         "lower 95%% bound of the paired d-o-d is above -epsilon, i.e. the runtime adds at most "
                         "epsilon on top of the checkpoint's own cost. Without it the old straddle rule is printed.")
    ap.add_argument("--latex", default=None, help="APPEND this dataset's table row and macros (same names as summarize_2x2, plus \\vpQ<Ds>DodCi)")
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arm)
    if set(arms) != {"A", "B", "C", "D"}:
        sys.exit("FATAL: need --arm A=.. B=.. C=.. D=..")
    per = {k: load_arm(v, args.dataset) for k, v in arms.items()}
    keys = set.intersection(*(set(v) for v in per.values()))
    missing = {k: len(set(per["A"]) ^ set(v)) for k, v in per.items() if k != "A"}
    if any(missing.values()):
        sys.exit(f"FATAL: arms do not score the same documents (symmetric differences vs A: {missing})")
    n = len(keys)
    def stats(xs: list[float]) -> dict:
        m = sum(xs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
        se = sd / math.sqrt(n)
        return {"mean_pp": 100 * m, "ci95_half_pp": 100 * t_crit(n - 1) * se, "sd_pp": 100 * sd}
    ba = [per["B"][k] - per["A"][k] for k in keys]
    dc = [per["D"][k] - per["C"][k] for k in keys]
    dod = [d - b for d, b in zip(dc, ba)]
    rep = {"dataset": args.dataset, "n_docs": n, "filter": SPEC[args.dataset][1], "metric": SPEC[args.dataset][2],
           "means": {k: 100 * sum(v[q] for q in keys) / n for k, v in per.items()},
           "B_minus_A": stats(ba), "D_minus_C": stats(dc), "dod": stats(dod)}
    z = rep["dod"]
    if args.margin is not None:
        lower = z["mean_pp"] - z["ci95_half_pp"]
        rep["margin_pp"] = args.margin; rep["dod_lower_pp"] = lower
        rep["verdict"] = (f"non-inferior at eps={args.margin:g} pp (lower bound {lower:+.2f} > {-args.margin:+.2f})"
                          if lower > -args.margin else
                          f"NOT shown non-inferior at eps={args.margin:g} pp (lower bound {lower:+.2f})")
    else:
        rep["verdict"] = "no resolved difference (interval contains zero)" if abs(z["mean_pp"]) <= z["ci95_half_pp"] else "differs"
    print(f"{args.dataset}: n={n} docs | A {rep['means']['A']:.2f} B {rep['means']['B']:.2f} C {rep['means']['C']:.2f} D {rep['means']['D']:.2f} | "
          f"(B-A) {rep['B_minus_A']['mean_pp']:+.2f} ± {rep['B_minus_A']['ci95_half_pp']:.2f} pp | "
          f"(D-C) {rep['D_minus_C']['mean_pp']:+.2f} ± {rep['D_minus_C']['ci95_half_pp']:.2f} pp | "
          f"paired d-o-d {z['mean_pp']:+.2f} ± {z['ci95_half_pp']:.2f} pp -> {rep['verdict']}")
    extra = []   # extra macros, with {macro} filled in below
    if args.dataset == "gsm8k" and SPEC["gsm8k"][1] == "__composite__":
        # The three readings (App. E.2): the same paired d-o-d under each published filter alone. The
        # composite is the reported one; the two others show how much the filter choice moves the result.
        readings = {}
        for flt, tag in (("strict-match", "Strict"), ("flexible-extract", "Flex")):
            saved = SPEC["gsm8k"]; SPEC["gsm8k"] = (saved[0], flt, saved[2])
            try:
                pf = {k: load_arm(v, "gsm8k") for k, v in arms.items()}
            finally:
                SPEC["gsm8k"] = saved
            if set.intersection(*(set(v) for v in pf.values())) != keys:
                sys.exit(f"FATAL: the {flt} reading scores a different document set")
            df = [(pf["D"][k] - pf["C"][k]) - (pf["B"][k] - pf["A"][k]) for k in keys]
            st = stats(df); readings[flt] = {"dod": st, "means": {q: 100 * sum(pf[q][k] for k in keys) / n for q in pf}}
            extra += [f"\\newcommand{{\\{args.macro_prefix}{{macro}}Dod{tag}}}{{{st['mean_pp']:+.2f}}}",
                      f"\\newcommand{{\\{args.macro_prefix}{{macro}}Dod{tag}Ci}}{{{st['ci95_half_pp']:.2f}}}"]
        allm = [rep["dod"]["mean_pp"]] + [r["dod"]["mean_pp"] for r in readings.values()]
        spread = max(allm) - min(allm)
        extra.append(f"\\newcommand{{\\{args.macro_prefix}{{macro}}DodFilterSpread}}{{{spread:.2f}}}")
        rep["readings"] = readings
        print(f"  readings: strict {readings['strict-match']['dod']['mean_pp']:+.2f} ± {readings['strict-match']['dod']['ci95_half_pp']:.2f} | "
              f"flexible {readings['flexible-extract']['dod']['mean_pp']:+.2f} ± {readings['flexible-extract']['dod']['ci95_half_pp']:.2f} | spread {spread:.2f} pp")
    if args.json:
        json.dump(rep, open(args.json, "w"), indent=1)
    if args.latex:
        disp = {"gsm8k": "GSM8K", "coqa": "CoQA", "bbh_cot": "BBH"}[args.dataset]
        macro = {"gsm8k": "Gsm", "coqa": "Coqa", "bbh_cot": "Bbh"}[args.dataset]
        metric_label = {"gsm8k": "\\texttt{exact\\_match,strict$\\vert$flexible}", "coqa": "\\texttt{f1,none}",
                        "bbh_cot": "\\texttt{exact\\_match,get-answer}"}[args.dataset]
        m = rep["means"]
        if args.margin is not None:
            # The standard three-outcome reading of a non-inferiority interval: PASS when the whole
            # interval lies above -margin; FAIL only when the whole interval lies below it (inferiority
            # shown); otherwise UNRESOLVED -- the interval crosses the margin and neither is shown.
            # A point estimate alone never decides (external review, 2026-09-15).
            upper = rep["dod"]["mean_pp"] + rep["dod"]["ci95_half_pp"]
            if rep["dod_lower_pp"] > -args.margin:
                gate = "pass" + (" (served above ref.)" if rep["dod_lower_pp"] > 0 else "")
            elif upper < -args.margin:
                gate = "FAIL"
            else:
                gate = f"unresolved ($n={rep.get('n_reps', 1)}$)"
        else:
            gate = "no resolved diff." if rep["verdict"].startswith("no resolved") else ("served above ref." if z["mean_pp"] > 0 else "served below ref.")
        row = (f"{disp} & {metric_label} & {m['A']:.2f} & {m['B']:.2f} & {m['C']:.2f} & {m['D']:.2f} & "
               f"{rep['B_minus_A']['mean_pp']:+.2f} & {rep['D_minus_C']['mean_pp']:+.2f} & "
               f"{z['mean_pp']:+.2f} $\\pm$ {z['ci95_half_pp']:.2f} & {gate} \\\\\n")
        macros = "\n".join([
            f"\\newcommand{{\\{args.macro_prefix}{macro}BminusA}}{{{rep['B_minus_A']['mean_pp']:+.2f}}}",
            f"\\newcommand{{\\{args.macro_prefix}{macro}DminusC}}{{{rep['D_minus_C']['mean_pp']:+.2f}}}",
            f"\\newcommand{{\\{args.macro_prefix}{macro}Dod}}{{{z['mean_pp']:+.2f}}}",
            f"\\newcommand{{\\{args.macro_prefix}{macro}DodCi}}{{{z['ci95_half_pp']:.2f}}}",
            f"\\newcommand{{\\{args.macro_prefix}{macro}Ndocs}}{{{n:,}}}",
            f"\\newcommand{{\\{args.macro_prefix}{macro}ArmD}}{{{m['D']/100:.4f}}}",
            f"\\newcommand{{\\{args.macro_prefix}{macro}ArmC}}{{{m['C']/100:.4f}}}",
            f"\\newcommand{{\\{args.macro_prefix}{macro}DodLower}}{{{z['mean_pp'] - z['ci95_half_pp']:+.2f}}}",
            f"\\newcommand{{\\{args.macro_prefix}{macro}ArmCPct}}{{{m['C']:.1f}}}",
        ] + [e.replace("{macro}", macro) for e in extra]) + "\n"
        with open(args.latex, "a") as fh:
            fh.write(row)
        mpath = args.macros or os.path.join(os.path.dirname(args.latex), "quality_macros.tex")
        with open(mpath, "a") as fh:
            fh.write(macros)
        print(f"  appended row to {args.latex} and macros to {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
