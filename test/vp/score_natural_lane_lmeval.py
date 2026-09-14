#!/usr/bin/env python3
"""Score natural-lane serving cells with lm-eval's OWN filter and metric code (owner 2026-09-14: quality only from the
third-party harness; the prompt is the frozen suite's, the filter + metric are lm-eval's).

GSM8K: the installed `gsm8k` task's filter ensembles (strict-match, flexible-extract) and its exact_match metric config,
applied through `task.process_results` on a synthesized doc {"answer": "#### <gold>"}.
BBH-CoT: one installed `bbh_cot_fewshot_*` task's filter + `process_results` on {"target": gold}.
CoQA: `lm_eval.tasks.coqa.utils.compute_scores(gold_list, pred)` (em, f1).

usage: score_natural_lane_lmeval.py --suites /dev/shm/vpipe/suites --cells <glob>... --out scores.json
Each cell = a benchmark jsonl (first line: request_ids, generated_texts) + its suite metadata (request index -> gold).
"""
import argparse, glob, json, re, sys
from lm_eval.tasks import TaskManager

ap = argparse.ArgumentParser()
ap.add_argument("--suites", required=True); ap.add_argument("--cells", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
tm = TaskManager()
gsm = tm.load_task_or_group("gsm8k")["gsm8k"]
bbh = next(iter(tm.load_task_or_group("bbh_cot_fewshot_boolean_expressions").values()))
from lm_eval.tasks.coqa import utils as coqa_utils
print("lm-eval tasks loaded; gsm8k filters:", [f.name for f in gsm._filters], "| bbh filters:", [f.name for f in bbh._filters], flush=True)


def meta_for(cell):
    name = cell.split("/")[-1]
    suite = re.sub(r"_qps.*$", "", name)  # e.g. gsm8k.d179_qps10p45_rep1.jsonl -> gsm8k.d179
    m = glob.glob(f"{a.suites}/{suite}.metadata.jsonl")
    if not m:
        raise SystemExit(f"no metadata for {cell} (suite {suite})")
    gold = {}
    for line in open(m[0]):
        r = json.loads(line); gold[r["index"]] = (r["dataset"], r["gold"])
    return suite, gold


def apply_filters(task, resps, docs):
    out = {}
    for ens in task._filters:   # FilterEnsemble.apply wants Instance objects; run its filter chain the way it does internally
        r = [[x] for x in resps]
        for f in ens.filters:
            r = f().apply(r, docs)
        out[ens.name] = list(r)
    return out


results = {}
for cell in [c for pat in a.cells for c in sorted(glob.glob(pat))]:
    suite, gold = meta_for(cell)
    rec = json.loads(open(cell).readline())
    ids, texts = rec["request_ids"], rec["generated_texts"]
    idx = [int(i.split(":")[-1]) if False else None for i in ids]
    # request ids look like gsm8k:test:684 -> the suite index is the row order; use the suite's own index via position
    n = len(ids); ds = gold[0][0]
    scores = {}
    if ds == "gsm8k":
        docs = [{"question": "", "answer": f"#### {gold[i][1]}"} for i in range(n)]
        filtered = apply_filters(gsm, texts, docs)
        for fname, fr in filtered.items():
            acc = 0.0
            for i in range(n):
                res = gsm.process_results(docs[i], [fr[i][0] if isinstance(fr[i], list) else fr[i]])
                acc += float(res["exact_match"])
            scores[f"exact_match,{fname}"] = acc / n
    elif ds == "bbh_cot":
        docs = [{"input": "", "target": gold[i][1]} for i in range(n)]
        filtered = apply_filters(bbh, texts, docs)
        for fname, fr in filtered.items():
            acc = 0.0
            for i in range(n):
                res = bbh.process_results(docs[i], [fr[i][0] if isinstance(fr[i], list) else fr[i]])
                acc += float(res["exact_match"])
            scores[f"exact_match,{fname}"] = acc / n
    elif ds == "coqa":
        em = f1 = 0.0
        for i in range(n):
            g = gold[i][1]; g = g if isinstance(g, list) else [g]
            s = coqa_utils.compute_scores(g, texts[i]); em += s["em"]; f1 += s["f1"]
        scores["em"] = em / n; scores["f1"] = f1 / n
    results[cell] = {"suite": suite, "n": n, "scores": scores}
    print(cell.split("/campaign/")[-1], n, {k: round(v, 4) for k, v in scores.items()}, flush=True)
json.dump(results, open(a.out, "w"), indent=1)
print("LMEVAL-SCORE-DONE", len(results))
