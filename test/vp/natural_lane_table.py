#!/usr/bin/env python3
"""Appendix table: the two STACKS served to their own end-of-sequence (natural lane vs natural lane), one repetition per cell.

Reads natural-lane/summary.json ({dataset: {rate_label: {upstream|vskipper: rec}}}, built from the harvest cells' score.json +
bench jsonl) and emits LaTeX rows per (dataset, rate): one row per stack (Base + SGLang, FlexiDepth + vSkipper) and one
stack-level delta row; plus macros for the prose. NO quality column: the labeled-workload scorer is not a third-party
harness (owner rule 2026-09-14: quality only from lm-eval); quality under load is measured by lm-eval at matched occupancy
(step 13). Output TPS = completed x mean output tokens / makespan. A delta here combines the checkpoint's
generation behaviour with the runtime; it is NOT a runtime-effect estimate (that is the matched-work table).

Quality column (lm-eval): --lmeval lmeval_scores.json (score_natural_lane_lmeval.py: the installed lm-eval task filters and
metrics applied to the served generations; GSM8K flexible-extract exact match, BBH-CoT get-answer exact match, CoQA F1).

usage: natural_lane_table.py summary.json --lmeval lmeval_scores.json --rows out_rows.tex --macros out_macros.tex
"""
import argparse, json, re, sys
LABELS = {"gsm8k": [("r8p25", "0.75"), ("r10p45", "0.95"), ("r13p75", "1.25")],
          "bbh_cot": [("r18p75", "0.75"), ("r23p75", "0.95"), ("r31p25", "1.25")],
          "coqa": [("r20p25", "0.75"), ("r25p65", "0.95"), ("r33p75", "1.25")]}
NAMES = {"gsm8k": "GSM8K", "bbh_cot": "BBH-CoT", "coqa": "CoQA"}
WORD = {"0.75": "Low", "0.95": "Mid", "1.25": "High"}
MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}
STACK = {"upstream": "Base + SGLang", "vskipper": r"FlexiDepth + \sys{}"}


def tps(rec):
    return rec["completed"] * rec["mean_out_tokens"] / rec["duration_s"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("summary"); ap.add_argument("--lmeval", required=True); ap.add_argument("--rows", required=True); ap.add_argument("--macros", required=True)
    a = ap.parse_args(); d = json.load(open(a.summary)); rows, macros = [], []
    lm = {}
    for cell, rec in json.load(open(a.lmeval)).items():
        m = re.search(r"harvest-(upstream-)?(gsm8k|bbh_cot|coqa)-(r\d+p\d+)/", cell)
        if not m: continue
        sc = rec["scores"]; q = sc.get("exact_match,flexible-extract", sc.get("exact_match,get-answer", sc.get("f1")))
        lm[("upstream" if m.group(1) else "vskipper", m.group(2), m.group(3))] = q
    def quality(rec, ds, arm, lbl):
        return lm[(arm, ds, lbl)]
    for ds, labels in LABELS.items():
        for lbl, rate in labels:
            cell = d.get(ds, {}).get(lbl)
            if not cell or any("missing" in cell.get(k, {}) for k in ("upstream", "vskipper")):
                continue
            u, v = cell["upstream"], cell["vskipper"]
            if any((arm, ds, lbl) not in lm for arm in ("upstream", "vskipper")):
                print(f"  {ds} {lbl}: no lm-eval score for both arms -> cell skipped", file=sys.stderr); continue
            for arm, r in (("upstream", u), ("vskipper", v)):
                rows.append(f"{NAMES[ds] if arm == 'upstream' else ''} & {rate if arm == 'upstream' else ''} & {STACK[arm]} & {100*quality(r, ds, arm, lbl):.1f} & {r['cap_hits']} & {r['mean_out_tokens']:.0f} & {r['mean_ttft_ms']:.0f} & {r['mean_tpot_ms']:.1f} & {r['mean_e2e_s']:.1f} & {tps(r):.0f} & {r['duration_s']:.0f} \\\\")
            pct = lambda k: 100 * (v[k] - u[k]) / u[k]
            dq = 100 * (quality(v, ds, "vskipper", lbl) - quality(u, ds, "upstream", lbl)); dtps = 100 * (tps(v) - tps(u)) / tps(u)
            rows.append(f" & & \\emph{{stack $\\Delta$}} & {dq:+.1f}~pp & {v['cap_hits']-u['cap_hits']:+d} & {pct('mean_out_tokens'):+.0f}\\% & {pct('mean_ttft_ms'):+.0f}\\% & {pct('mean_tpot_ms'):+.0f}\\% & {pct('mean_e2e_s'):+.0f}\\% & {dtps:+.0f}\\% & {pct('duration_s'):+.0f}\\% \\\\")
            if lbl == labels[-1][0]: rows.append(r"\addlinespace")
            tag = MW[ds] + WORD[rate]
            for name, val in (("OutTok", pct("mean_out_tokens")), ("EtoE", pct("mean_e2e_s")), ("TPOT", pct("mean_tpot_ms")), ("TPS", dtps), ("Qual", dq), ("Makespan", pct("duration_s"))):
                macros.append(f"\\newcommand{{\\vpNat{tag}{name}}}{{{val:+.0f}}}" if name != "Qual" else f"\\newcommand{{\\vpNat{tag}{name}}}{{{val:+.1f}}}")
            macros.append(f"\\newcommand{{\\vpNat{tag}CapUp}}{{{u['cap_hits']}}}\n\\newcommand{{\\vpNat{tag}CapVs}}{{{v['cap_hits']}}}")
            macros.append(f"\\newcommand{{\\vpNat{tag}OutTokUp}}{{{u['mean_out_tokens']:.0f}}}\n\\newcommand{{\\vpNat{tag}OutTokVs}}{{{v['mean_out_tokens']:.0f}}}")
    if rows and rows[-1] == r"\addlinespace": rows.pop()
    open(a.rows, "w").write("\n".join(rows) + "\n"); open(a.macros, "w").write("\n".join(macros) + "\n")
    print(f"natural-lane table: {sum(1 for r in rows if 'stack' in r)} cells -> {a.rows}, {len(macros)} macro lines")


if __name__ == "__main__":
    main()
