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
GSM8K flexible-extract exact match, BBH-CoT get-answer exact match, CoQA F1. The scorer must have
been run with --eval-split-only: GSM8K and CoQA suites carry train-split padding to sustain the
operating point, and only the held-out rows may be scored.

usage: loaded_quality_table.py scores.json --rows rows.tex --sweep-rows sweep.tex --macros macros.tex
"""
import argparse, json, re, sys

RATES = {"gsm8k": [("r8p25", "0.75"), ("r10p45", "0.95"), ("r13p75", "1.25")],
         "bbh_cot": [("r18p75", "0.75"), ("r23p75", "0.95"), ("r31p25", "1.25")],
         "coqa": [("r20p25", "0.75"), ("r25p65", "0.95"), ("r33p75", "1.25")]}
NAMES = {"gsm8k": "GSM8K", "bbh_cot": "BBH-CoT", "coqa": "CoQA"}
MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}
RW = {"0.75": "Low", "0.95": "Mid", "1.25": "High"}
# the harness's own filter per task -- the key score_natural_lane_lmeval.py emits
# GSM8K uses the marker-aware composite: this checkpoint's confidence epilogue makes
# flexible-extract alone an arm-dependent reading (the base model never emits the epilogue, the
# checkpoint does). BBH and CoQA have a single filter each and are unaffected.
METRIC = {"gsm8k": "exact_match,marker-composite",
          "bbh_cot": "exact_match,get-answer",
          "coqa": "f1"}
METRIC_TEX = {"gsm8k": r"\texttt{exact\_match,marker-composite}",
              "bbh_cot": r"\texttt{exact\_match,get-answer}",
              "coqa": r"\texttt{f1,none}"}
ARMS = [("upstream", "Base + upstream SGLang"),
        ("integrated_it4", r"FlexiDepth + \sys{} (served)"),
        ("integrated_alwaysskip", "FlexiDepth + always-route")]
ARM_MACRO = {"upstream": "Up", "integrated_it4": "Hyb", "integrated_alwaysskip": "Alw"}

CELL = re.compile(r"harvest-(?P<arm>[a-z0-9_]+)-(?P<ds>gsm8k|bbh_cot|coqa)-(?P<rate>r[0-9p]+)/")
# Arms whose quality number is only meaningful if the ROUTED DECODE BODY actually executed. A
# served flag is not a treatment: a hybrid cell that ran the stock body on every decode pass is
# upstream's number wearing the hybrid's label, and printing it would misattribute the baseline.
# Measured instance: the GSM8K and CoQA 0.75xQ* hybrid cells both ran fd_tokens_skip_body = 0.
ROUTED_REQUIRED = {"integrated_it4"}


def routed_decode_share(cell_path):
    """Routed share of decode rows from the cell's own attestation, or None if unavailable."""
    root = cell_path.split("/cell/")[0]
    try:
        d = json.load(open(f"{root}/server_info.after.json"))
    except Exception:
        return None
    v = d.get("internal_states", [{}])[0].get("vp_runtime", {})
    c3 = (v.get("model", {}).get("flexidepth", {}).get("fd_c3", {}) or v.get("fd_c3", {})).get("counters", {})
    skip = c3.get("fd_tokens_skip_body")
    allrun = c3.get("fd_tokens_prod_allrun_band")
    if skip is None or allrun is None or (skip + allrun) == 0:
        return None
    return skip / (skip + allrun)


def index(scores):
    """cell path -> (dataset, arm, rate_label) -> (score, n). Fails closed on an unparseable path."""
    out = {}
    for cell, rec in scores.items():
        m = CELL.search(cell)
        if not m:
            sys.exit(f"FATAL: cannot parse arm/dataset/rate from cell path {cell!r}")
        ds, arm, rate = m["ds"], m["arm"], m["rate"]
        if arm in ROUTED_REQUIRED:
            share = routed_decode_share(cell)
            if share == 0.0:
                print(f"  VOID {ds}@{rate}: routed decode body never executed "
                      f"(fd_tokens_skip_body = 0); this cell is upstream's number, not the "
                      f"hybrid's. Refusing to print it.", file=sys.stderr)
                continue
            if share is None:
                sys.exit(f"FATAL {cell}: no decode body counters in the attestation. A hybrid "
                         "quality number is only meaningful with the routed body seen executing.")
        key = METRIC[ds]
        if key not in rec["scores"]:
            sys.exit(f"FATAL {cell}: expected metric {key!r}, got {sorted(rec['scores'])}. "
                     "The loaded block must use the same filter the unloaded block names.")
        out[(ds, arm, rate)] = (100.0 * rec["scores"][key], rec["n"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("--rows", required=True, help="knee rows for the main table")
    ap.add_argument("--sweep-rows", default=None, help="all rates, for the appendix")
    ap.add_argument("--macros", required=True)
    ap.add_argument("--macro-prefix", default="vpLq")
    a = ap.parse_args()

    idx = index(json.load(open(a.scores)))
    macros, knee, sweep = [], [], []

    for ds in ("gsm8k", "coqa", "bbh_cot"):
        for rate_lbl, rate_x in RATES[ds]:
            cells = {arm: idx.get((ds, arm, rate_lbl)) for arm, _ in ARMS}
            present = {k: v for k, v in cells.items() if v}
            if not present:
                continue
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
            sweep.append(f"{NAMES[ds]} & ${rate_x}\\times Q^*$ & {body} \\\\")
            if rate_x == "0.95":
                knee.append(f"{NAMES[ds]} & {body} \\\\")
            for arm, _ in ARMS:
                if cells.get(arm):
                    macros.append(f"\\newcommand{{\\{a.macro_prefix}{MW[ds]}{RW[rate_x]}"
                                  f"{ARM_MACRO[arm]}}}{{{cells[arm][0]:.2f}}}")
            # hybrid minus always-route, the comparison the block exists to report
            h, w = cells.get("integrated_it4"), cells.get("integrated_alwaysskip")
            if h and w:
                macros.append(f"\\newcommand{{\\{a.macro_prefix}{MW[ds]}{RW[rate_x]}Delta}}"
                              f"{{{h[0] - w[0]:+.2f}}}")

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
