#!/usr/bin/env python3
"""Client equivalence for the two blocks of the quality table (Appendix E.3).

Block (a) is driven by lm-eval's own client, block (b) by the serving benchmark's. The two are
printed together only because the difference was measured: the SAME served arm (always-route) on
the SAME held-out documents, lm-eval's archived --log_samples against the bench client's cell at
each load point. Reports, per cell: prompt agreement (the .qual suite's request text against
lm-eval's prompt, where both are available), responses byte-identical, and final-answer agreement
(the text after the last '####' marker, the checkpoint's own answer convention).

  client_equivalence.py --lmeval <arm-D root> --cells <harvest-root>... --macros out.tex --json out.json
"""
import argparse, glob, json, os, re

def lm_samples(root):
    f = sorted(glob.glob(os.path.join(root, "**", "samples_gsm8k_*.jsonl"), recursive=True), key=os.path.getmtime)[-1]
    out = {}
    for line in open(f):
        r = json.loads(line)
        if r.get("filter") != "strict-match": continue
        out[int(r["doc_id"])] = r["resps"][0][0]
    return f, out

def answer(t):
    m = re.findall(r"####\s*([^\n]+)", t)
    return m[-1].strip().replace(",", "").replace("$", "") if m else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lmeval", required=True); ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--macros"); ap.add_argument("--json"); ap.add_argument("--prefix", default="vpClient")
    a = ap.parse_args()
    lf, lm = lm_samples(a.lmeval); rows = []
    for root in a.cells:
        c = [x for x in glob.glob(os.path.join(root, "cell", "gsm8k.qual_*_rep1.jsonl")) if "arrival" not in x and ".load." not in x]
        if not c: continue
        d = json.load(open(c[0])); n = same = ans = unparsed = 0
        if len(d["request_ids"]) != len(d["generated_texts"]):
            raise SystemExit(f"FATAL {c[0]}: {len(d['request_ids'])} ids vs {len(d['generated_texts'])} texts")
        for rid, t in zip(d["request_ids"], d["generated_texts"]):
            m = re.match(r"gsm8k:test:(\d+)$", rid)
            if not m or int(m.group(1)) not in lm: continue
            ref = lm[int(m.group(1))]; n += 1; same += (t == ref)
            x, y = answer(t), answer(ref)
            if x is None or y is None: unparsed += 1      # no marker on one side: not an agreement, counted apart
            else: ans += (x == y)
        if n != len(lm):
            raise SystemExit(f"FATAL {c[0]}: matched {n} of {len(lm)} lm-eval documents")
        rows.append({"cell": os.path.basename(root), "n": n, "verbatim_pct": round(100 * same / n, 1),
                     "answer_pct": round(100 * ans / n, 1), "unparsed_either_side": unparsed})
        print(f"  {os.path.basename(root)}: n={n} verbatim {rows[-1]['verbatim_pct']}% final answer {rows[-1]['answer_pct']}% (no marker on a side: {unparsed})")
    if not rows: raise SystemExit("FATAL: no cells matched")
    v = [r["verbatim_pct"] for r in rows]; q = [r["answer_pct"] for r in rows]
    if a.json: json.dump({"lmeval_samples": lf, "rows": rows}, open(a.json, "w"), indent=1)
    if a.macros:
        with open(a.macros, "w") as fh:
            fh.write(f"\\newcommand{{\\{a.prefix}Docs}}{{{rows[0]['n']:,}}}\n")
            fh.write(f"\\newcommand{{\\{a.prefix}VerbatimLow}}{{{min(v):.1f}}}\n\\newcommand{{\\{a.prefix}VerbatimHigh}}{{{max(v):.1f}}}\n")
            fh.write(f"\\newcommand{{\\{a.prefix}AnswerLow}}{{{min(q):.1f}}}\n\\newcommand{{\\{a.prefix}AnswerHigh}}{{{max(q):.1f}}}\n")
            fh.write(f"\\newcommand{{\\{a.prefix}Cells}}{{{len(rows)}}}\n")
            fh.write(f"\\newcommand{{\\{a.prefix}UnparsedMax}}{{{max(r['unparsed_either_side'] for r in rows)}}}\n")
    print(f"wrote {len(rows)} cells")

if __name__ == "__main__": main()
