#!/usr/bin/env python3
"""The loaded block of the quality table: the SERVED configuration scored under load.

The unloaded 2x2 (paired_dod_2x2.py) scores four arms at lm-eval's own client concurrency and emits
a difference-of-differences. It cannot produce this block: it requires exactly four arms and the
loaded lane has three (upstream / vskipper / always-route) at each of three offered rates.

Reads score_natural_lane_lmeval.py's output -- {cell_path: {suite, n, scores}} -- where each cell
path is .../harvest-<arm>-<dataset>-<rate_label>/cell/<suite>_qps<rate>_rep1.jsonl, and emits:

  --rows        the KNEE rows for the main table (0.95xQ* only, one row per dataset)
  --sweep-rows  every rate, for the appendix
  --macros      per-cell macros for the prose

Scores come from the harness's own filter per task, the same one the unloaded block names:
GSM8K marker-composite exact match (strict where it parses, flexible otherwise), BBH-CoT get-answer
exact match, CoQA F1. The scorer must have been run with --eval-split-only: GSM8K and CoQA suites
carry train-split padding to sustain the operating point, and only the held-out rows may be scored.
Both are enforced here: a cell scored with the flag off, or whose n is not the held-out size,
is refused. So is a missing cell -- every arm at every rate is a row of the appendix table.

usage: loaded_quality_table.py scores.json --rows rows.tex --sweep-rows sweep.tex --macros macros.tex
"""
import argparse, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_dod_2x2 import t_crit   # Student t, the same critical values the 2x2 uses

RATES = {"gsm8k": [("r8p25", "0.75"), ("r10p45", "0.95"), ("r13p75", "1.25")],
         "bbh_cot": [("r18p75", "0.75"), ("r23p75", "0.95"), ("r31p25", "1.25")],
         "coqa": [("r20p25", "0.75"), ("r25p65", "0.95"), ("r33p75", "1.25")]}
NAMES = {"gsm8k": "GSM8K", "bbh_cot": "BBH", "coqa": "CoQA"}   # one name per workload across every table; BBH is its CoT split, said once in the paper
MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}
RW = {"0.75": "Low", "0.95": "Mid", "1.25": "High"}
# the held-out document count per dataset: what n must be once the train-split padding is excluded
HELD_OUT = {"gsm8k": 1319, "coqa": 500, "bbh_cot": 6511}
# the harness's own filter per task -- the key score_natural_lane_lmeval.py emits
# GSM8K uses the marker-aware composite: this checkpoint's confidence epilogue makes
# flexible-extract alone an arm-dependent reading (the base model never emits the epilogue, the
# checkpoint does). BBH and CoQA have a single filter each and are unaffected.
METRIC = {"gsm8k": "exact_match,marker-composite",
          "bbh_cot": "exact_match,get-answer",
          "coqa": "f1"}
METRIC_TEX = {"gsm8k": r"\texttt{exact\_match,composite}",
              "bbh_cot": r"\texttt{exact\_match,get-answer}",
              "coqa": r"\texttt{f1,none}"}
ARMS = [("upstream", "Base + upstream"),
        ("integrated_it4", r"FlexiDepth + \sys{} (served)"),
        ("integrated_alwaysskip", "FlexiDepth + always-route")]
ARM_MACRO = {"upstream": "Up", "integrated_it4": "Hyb", "integrated_alwaysskip": "Alw"}

CELL = re.compile(r"harvest-(?P<arm>[a-z0-9_]+)-(?P<ds>gsm8k|bbh_cot|coqa)-(?P<rate>r[0-9p]+)/")
def paired_ci(a, b):
    """Paired per-document 95% interval on the mean difference of two 0/1 vectors.

    Valid at one repetition per cell ONLY because every arm scores the same documents in the same
    order. It quantifies DOCUMENT SAMPLING and nothing else -- run-to-run variation is not in it,
    and the caption must say so.
    """
    if len(a) != len(b):
        sys.exit(f"FATAL: paired vectors differ in length ({len(a)} vs {len(b)})")
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        sys.exit("FATAL: a paired interval needs at least two documents")
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return 100.0 * m, 100.0 * t_crit(n - 1) * math.sqrt(var / n)


def index(scores):
    """cell path -> (dataset, arm, rate_label) -> (score, n). Fails closed on an unparseable path."""
    out = {}
    for cell, rec in scores.items():
        m = CELL.search(cell)
        if not m:
            sys.exit(f"FATAL: cannot parse arm/dataset/rate from cell path {cell!r}")
        ds, arm, rate = m["ds"], m["arm"], m["rate"]
        # NOTE: no execution gate here, deliberately. The "mechanism must be seen executing" rule
        # governs PERFORMANCE claims -- you cannot attribute a speedup to a run where the skipping
        # never happened. This table answers a different question: what does the deployed
        # configuration score at each offered load? At the lowest rung the band declining to engage
        # IS the hybrid's behaviour, and withholding the row would hide exactly what the regime
        # switch does. The routed share is reported beside the score instead of gating it.
        key = METRIC[ds]
        if key not in rec["scores"]:
            sys.exit(f"FATAL {cell}: expected metric {key!r}, got {sorted(rec['scores'])}. "
                     "The loaded block must use the same filter the unloaded block names.")
        # The scorer records the flag; a file from before it did carries the same proof in n, which
        # is 3600/4000 with the padding scored and the held-out count without it.
        if rec.get("eval_split_only") is False:
            sys.exit(f"FATAL {cell}: scored without --eval-split-only; the train-split padding is in "
                     "the score. Rescore.")
        if rec["n"] != HELD_OUT[ds]:
            sys.exit(f"FATAL {cell}: n={rec['n']} but {ds} has {HELD_OUT[ds]} held-out documents")
        out[(ds, arm, rate)] = (100.0 * rec["scores"][key], rec["n"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--rows", required=True, help="knee rows for the main table")
    ap.add_argument("--sweep-rows", default=None, help="all rates, for the appendix")
    ap.add_argument("--macros", required=True)
    ap.add_argument("--macro-prefix", default="vpLq")
    ap.add_argument("--shares", default=None,
                    help="loaded_shares.json: the routed decode share of the served arms per cell, read "
                         "from their attestation; printed beside each sweep row so a cell whose routed "
                         "body never ran (0.0%%) is visible in the table, not only in an appendix macro")
    a = ap.parse_args()
    shares = json.load(open(a.shares)) if a.shares else None

    raw = json.load(open(a.scores))
    per_row = raw.pop("__per_row__", {})
    idx = index(raw)
    # cell key -> per-document score vector, for the paired intervals below
    rows_by_key, ids_by_key = {}, {}
    for cell, rec in per_row.items():
        m = CELL.search(cell)
        if m:
            rows_by_key[(m["ds"], m["arm"], m["rate"])] = rec["ok"]
            ids_by_key[(m["ds"], m["arm"], m["rate"])] = rec.get("ids")
    macros, knee, sweep = [], [], []

    for ds in ("gsm8k", "coqa", "bbh_cot"):
        for rate_lbl, rate_x in RATES[ds]:
            cells = {arm: idx.get((ds, arm, rate_lbl)) for arm, _ in ARMS}
            missing = [arm for arm, v in cells.items() if not v]
            if missing:
                sys.exit(f"FATAL {ds}@{rate_x}: no scored cell for {missing}; every arm at every "
                         "rate is a row of the table, so a missing one is a missing input, not a blank")
            for arm, _ in ARMS:
                if not rows_by_key.get((ds, arm, rate_lbl)):
                    sys.exit(f"FATAL {ds}@{rate_x}: no per-document vector for {arm}; the scores file "
                             "predates the per-row emitter -- rescore")
            present = cells
            n = sorted({v[1] for v in present.values()})
            if len(n) > 1:
                sys.exit(f"FATAL {ds}@{rate_x}: arms scored different row counts {n}; "
                         "a paired comparison needs one document set.")
            def cell(arm):
                v = cells.get(arm)
                return f"{v[0]:.2f}" if v else "--"
            # the sweep carries the rate column; the knee block does not -- it is one rate,
            # named in the caption, so a constant column would be noise in the main table.
            body = (f"{METRIC_TEX[ds]} & {n[0]} & {cell('upstream')} & "
                    f"{cell('integrated_it4')} & {cell('integrated_alwaysskip')}")
            # the sweep rows also carry the hybrid's two paired differences with their intervals
            # (E.4 promises "each cell's interval is reported with it"); the knee block keeps them in prose
            def paired_col(other):
                h, w = cells.get("integrated_it4"), cells.get(other)
                hv, wv = rows_by_key.get((ds, "integrated_it4", rate_lbl)), rows_by_key.get((ds, other, rate_lbl))
                if not (h and w and hv and wv) or len(hv) != len(wv): return "--"
                d, ci = paired_ci(hv, wv); return f"${d:+.2f} \\pm {ci:.2f}$"
            share_col = ""
            if shares is not None:
                def share(arm):
                    rec = shares.get(f"{arm}/{ds}/{rate_lbl}")
                    if rec is None:
                        sys.exit(f"FATAL {ds}@{rate_x}: no routed share for {arm} in {a.shares}")
                    return f"{rec['routed_share_pct']:.0f}"
                share_col = f" & {share('integrated_it4')} / {share('integrated_alwaysskip')}"
            sweep.append(f"{NAMES[ds]} & ${rate_x}\\times Q^*$ & {body}{share_col} & {paired_col('upstream')} & {paired_col('integrated_alwaysskip')} \\\\")
            if rate_x == "0.95":
                knee.append(f"{NAMES[ds]} & {body}{share_col} \\\\")
            for arm, _ in ARMS:
                if cells.get(arm):
                    macros.append(f"\\newcommand{{\\{a.macro_prefix}{MW[ds]}{RW[rate_x]}"
                                  f"{ARM_MACRO[arm]}}}{{{cells[arm][0]:.2f}}}")
            # hybrid minus always-route, and hybrid minus upstream, with PAIRED per-document
            # intervals. Valid at one repetition only because every arm scores the same documents
            # in the same order; they quantify document sampling and NOT run-to-run variance.
            for other, tag in (("integrated_alwaysskip", "Delta"), ("upstream", "DeltaUp")):
                h, w = cells.get("integrated_it4"), cells.get(other)
                if not (h and w):
                    continue
                macros.append(f"\\newcommand{{\\{a.macro_prefix}{MW[ds]}{RW[rate_x]}{tag}}}"
                              f"{{{h[0] - w[0]:+.2f}}}")
                hv = rows_by_key.get((ds, "integrated_it4", rate_lbl))
                wv = rows_by_key.get((ds, other, rate_lbl))
                if hv and wv:
                    # The pairing is by document. When the scorer recorded ids, require them equal;
                    # the length check alone cannot tell two orderings apart.
                    hi, wi = ids_by_key.get((ds, "integrated_it4", rate_lbl)), ids_by_key.get((ds, other, rate_lbl))
                    if hi is not None and wi is not None and hi != wi:
                        sys.exit(f"FATAL {ds} {rate_lbl}: document ids differ between integrated_it4 and {other}")
                    if len(hv) != len(wv):
                        sys.exit(f"FATAL {ds} {rate_lbl}: {len(hv)} vs {len(wv)} scored documents for integrated_it4 vs {other}")
                    d, ci = paired_ci(hv, wv)
                    macros.append(f"\\newcommand{{\\{a.macro_prefix}{MW[ds]}{RW[rate_x]}{tag}Ci}}"
                                  f"{{{ci:.2f}}}")

    # Appendix E.4's mechanism numbers, generated rather than typed: the spread of each arm across
    # the rungs it is reported at. The hybrid's spread is the regime switch's signature and the
    # switch-free arms' spreads are the contrast, so neither may drift from the table above it.
    for ds in ("gsm8k", "coqa", "bbh_cot"):
        for arm, tag in (("upstream", "Up"), ("integrated_it4", "Hyb"),
                         ("integrated_alwaysskip", "Alw")):
            vals = [idx[(ds, arm, r)][0] for r, _ in RATES[ds] if (ds, arm, r) in idx]
            if len(vals) >= 2:
                macros.append(f"\\newcommand{{\\{a.macro_prefix}{MW[ds]}{tag}Spread}}"
                              f"{{{max(vals) - min(vals):.2f}}}")
                macros.append(f"\\newcommand{{\\{a.macro_prefix}{MW[ds]}{tag}Rungs}}{{{len(vals)}}}")

    if not knee:
        sys.exit("FATAL: no 0.95xQ* cells scored; the main-table block would be empty.")
    open(a.rows, "w").write("\n".join(knee) + "\n")
    if a.sweep_rows:
        open(a.sweep_rows, "w").write("\n".join(sweep) + "\n")
    open(a.macros, "w").write("\n".join(macros) + "\n")
    print(f"wrote {a.rows} ({len(knee)} knee rows)"
          + (f", {a.sweep_rows} ({len(sweep)} rows)" if a.sweep_rows else "")
          + f", {a.macros} ({len(macros)} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
