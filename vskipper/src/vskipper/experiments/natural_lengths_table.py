#!/usr/bin/env python3
"""Appendix table: natural-lane generation lengths of both arms per dataset and rate (D-749/D-753).

Reads harvest directories (`harvest-<ds>-<lbl>/bank.json` for the served arm, `harvest-upstream-<ds>-<lbl>/bank.json`
for upstream) and emits LaTeX rows: n, mean tokens, runaway count (>= 4096 tokens, i.e. generations that ran to
the context window), total pinned work; plus macros for the prose, including the share of generations of five tokens
or fewer (\\vpNatLen<Arm>ShortPct<Ds><Rate>).

usage: natural_lengths_table.py ARTIFACT_DIR --rows out_rows.tex --macros out_macros.tex
"""
import argparse, glob, json, os
LABELS = {"gsm8k": [("r8p25", "0.75"), ("r10p45", "0.95"), ("r13p75", "1.25")],
          "bbh_cot": [("r18p75", "0.75"), ("r23p75", "0.95"), ("r31p25", "1.25")],
          "coqa": [("r20p25", "0.75"), ("r25p65", "0.95"), ("r33p75", "1.25")]}
NAMES = {"gsm8k": "GSM8K", "bbh_cot": "BBH", "coqa": "CoQA"}   # one name per workload across every table; BBH is its CoT split, said once in the paper
WORD = {"0.75": "Low", "0.95": "Mid", "1.25": "High"}
MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}
# [v1.6, D-824] ladder-control campaign: g1024 knees; upstream harvests under harvest-upstream_g1024-<ds>-<lbl>, the served arm's
# own-stop cells under natural-vskipper-<ds>-<lbl> (bank.json made from the cell artifact with make_bank.py)
LABELS_V16 = {"gsm8k": [("r9p75", "0.75"), ("r12p35", "0.95"), ("r16p25", "1.25")],
              "bbh_cot": [("r25p5", "0.75"), ("r32p3", "0.95"), ("r42p5", "1.25")],
              "coqa": [("r18p75", "0.75"), ("r23p75", "0.95"), ("r31p25", "1.25")]}

def stats(path):
    L = list(json.load(open(path))["lengths"].values()); n = len(L)
    return {"n": n, "mean": sum(L) / n, "runaway": sum(x >= 4096 for x in L), "sum": sum(L), "short": sum(x <= 5 for x in L),
            "runaway_tok": sum(x for x in L if x >= 4096)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("art"); ap.add_argument("--rows", required=True); ap.add_argument("--macros", required=True); ap.add_argument("--layout", default="v15", choices=["v15", "v16"])
    a = ap.parse_args(); rows = []; macros = []
    global LABELS
    up, served = "harvest-upstream-{ds}-{lbl}", "harvest-{ds}-{lbl}"
    if a.layout == "v16": LABELS, up, served = LABELS_V16, "harvest-upstream_g1024-{ds}-{lbl}", "natural-vskipper-{ds}-{lbl}"
    for ds, labels in LABELS.items():
        for lbl, mult in labels:
            u = os.path.join(a.art, up.format(ds=ds, lbl=lbl), "bank.json"); s = os.path.join(a.art, served.format(ds=ds, lbl=lbl), "bank.json")
            if not (os.path.exists(u) and os.path.exists(s)): raise SystemExit(f"missing bank for {ds} {lbl}: {u if not os.path.exists(u) else s}")
            U, S = stats(u), stats(s)
            rows.append(f"{NAMES[ds]} & {mult}$\\times$ & {U['n']} & {U['mean']:.0f} & {U['runaway']} ({100*U['runaway']/U['n']:.1f}\\%) & {U['sum']/1e6:.2f} & "
                        f"{S['mean']:.0f} & {S['runaway']} ({100*S['runaway']/S['n']:.1f}\\%) & {S['sum']/1e6:.2f} \\\\")
            for arm, st in (("Up", U), ("Served", S)):
                macros.append(f"\\newcommand{{\\vpNatLen{arm}Mean{MW[ds]}{WORD[mult]}}}{{{st['mean']:.0f}}}")
                macros.append(f"\\newcommand{{\\vpNatLen{arm}Runaway{MW[ds]}{WORD[mult]}}}{{{st['runaway']}}}")
                macros.append(f"\\newcommand{{\\vpNatLen{arm}RunawayPct{MW[ds]}{WORD[mult]}}}{{{100*st['runaway']/st['n']:.1f}}}")
                macros.append(f"\\newcommand{{\\vpNatLen{arm}WorkM{MW[ds]}{WORD[mult]}}}{{{st['sum']/1e6:.2f}}}")
                macros.append(f"\\newcommand{{\\vpNatLen{arm}ShortPct{MW[ds]}{WORD[mult]}}}{{{100*st['short']/st['n']:.0f}}}")   # generations of <= 5 tokens (Section 5's CoQA sentence)
                macros.append(f"\\newcommand{{\\vpNatLen{arm}RunawayTokPct{MW[ds]}{WORD[mult]}}}{{{100*st['runaway_tok']/st['sum']:.0f}}}")   # share of the cell's output tokens in runaway generations
    open(a.rows, "w").write("\n".join(rows) + "\n"); open(a.macros, "w").write("\n".join(macros) + "\n")
    print("\n".join(rows)); print(f"wrote {a.rows} ({len(rows)} rows), {a.macros} ({len(macros)} macros)")

if __name__ == "__main__": main()
