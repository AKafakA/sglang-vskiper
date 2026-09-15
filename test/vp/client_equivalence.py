#!/usr/bin/env python3
"""Client equivalence for the two blocks of the quality table (Appendix E.3).

Block (a) is driven by lm-eval's own client, block (b) by the serving benchmark's. The two are
printed together only because the difference was measured: the SAME served arm (always-route) on
the SAME held-out documents, lm-eval's archived --log_samples against the bench client's cell at
each load point. Reports, per cell: (1) VERDICT agreement -- the composite filter's per-document verdict from lm-eval's
samples against the bench client's, paired by request id through the .qual suite (the quantity that
licenses the two blocks: the same filter, the same documents, does the client change the score?),
with the paired score difference and its t interval; (2) responses byte-identical; (3) agreement of the
text after the last '####' marker, with the documents that carry no marker on a side counted apart.

  client_equivalence.py --lmeval <arm-D root> --cells <harvest-root>... --loaded loaded_all.json
                        --suite-meta gsm8k.qual.metadata.jsonl --macros out.tex --json out.json
"""
import argparse, glob, json, math, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_dod_2x2 import load_arm, t_crit

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
    ap.add_argument("--loaded", help="loaded_all.json: the bench client's per-document composite verdicts")
    ap.add_argument("--suite-meta", help="the .qual suite metadata: position -> request id, eval rows only")
    a = ap.parse_args()
    lf, lm = lm_samples(a.lmeval); rows = []
    # per-document composite verdict on lm-eval's side, keyed by doc id
    lm_ok = {doc: ok for (task, doc), ok in load_arm(a.lmeval, "gsm8k").items()}
    bench_ok = {}
    if a.loaded and a.suite_meta:
        meta = [json.loads(l) for l in open(a.suite_meta)]
        ev = [m for m in meta if m.get("evaluator_data", {}).get("source_split") in ("test", "validation")]
        pos_ids = [m["request_id"] for m in ev]
        per = json.load(open(a.loaded)).get("__per_row__", {})
        for cell, rec in per.items():
            if "gsm8k" not in os.path.basename(cell): continue   # this comparison is the GSM8K arm-D lane
            if len(rec["ok"]) != len(pos_ids):
                sys.exit(f"FATAL {cell}: {len(rec['ok'])} verdicts vs {len(pos_ids)} eval rows in the suite")
            if rec.get("ids") and rec["ids"] != pos_ids:
                sys.exit(f"FATAL {cell}: recorded ids differ from the suite's eval rows")
            bench_ok[os.path.basename(os.path.dirname(os.path.dirname(cell)))] = dict(zip(pos_ids, rec["ok"]))
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
        row = {"cell": os.path.basename(root), "n": n, "verbatim_pct": round(100 * same / n, 1),
               "answer_pct": round(100 * ans / n, 1), "unparsed_either_side": unparsed}
        bo = bench_ok.get(os.path.basename(root))
        if bo:
            docs = [int(re.match(r"gsm8k:test:(\d+)$", rid).group(1)) for rid in bo]
            pairs = [(bo[f"gsm8k:test:{doc}"], lm_ok[doc]) for doc in docs if doc in lm_ok]
            if len(pairs) != len(lm_ok):
                sys.exit(f"FATAL {root}: paired {len(pairs)} of {len(lm_ok)} documents")
            agree = sum(1 for b, l in pairs if b == l); diff = [b - l for b, l in pairs]
            m = sum(diff) / len(diff); var = sum((x - m) ** 2 for x in diff) / (len(diff) - 1)
            row.update({"verdict_agree_pct": round(100 * agree / len(pairs), 1),
                        "score_bench_pct": round(100 * sum(b for b, _ in pairs) / len(pairs), 2),
                        "score_lmeval_pct": round(100 * sum(l for _, l in pairs) / len(pairs), 2),
                        "score_diff_pp": round(100 * m, 2), "score_diff_ci_pp": round(100 * t_crit(len(diff) - 1) * math.sqrt(var / len(diff)), 2)})
        rows.append(row)
        extra = f" verdicts agree {row['verdict_agree_pct']}% score bench-lmeval {row['score_diff_pp']:+.2f} ± {row['score_diff_ci_pp']} pp" if bo else ""
        print(f"  {os.path.basename(root)}: n={n} verbatim {row['verbatim_pct']}%{extra} | marker answer {row['answer_pct']}% (no marker on a side: {unparsed})")
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
            if all("verdict_agree_pct" in r for r in rows):
                g = [r["verdict_agree_pct"] for r in rows]; dd = [abs(r["score_diff_pp"]) for r in rows]; cc = [r["score_diff_ci_pp"] for r in rows]
                fh.write(f"\\newcommand{{\\{a.prefix}VerdictLow}}{{{min(g):.1f}}}\n\\newcommand{{\\{a.prefix}VerdictHigh}}{{{max(g):.1f}}}\n")
                fh.write(f"\\newcommand{{\\{a.prefix}ScoreDiffMax}}{{{max(dd):.2f}}}\n\\newcommand{{\\{a.prefix}ScoreDiffCiMax}}{{{max(cc):.2f}}}\n")
    print(f"wrote {len(rows)} cells")

if __name__ == "__main__": main()
