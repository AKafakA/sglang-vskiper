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
# Quality-lane suites carry TRAIN-split padding purely to sustain the operating point; only the
# held-out rows may be scored. Keyed on source_split, NEVER on quality_eligible, which is True on
# every row of every suite and would silently admit the padding. Opt-in: without it this scorer
# behaves exactly as before.
ap.add_argument("--eval-split-only", action="store_true",
                help="score only rows whose source_split is test/validation")
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
    rid_by_index = {}
    skipped = 0
    for line in open(m[0]):
        r = json.loads(line)
        if a.eval_split_only and r.get("source_split") not in ("test", "validation"):
            skipped += 1
            continue
        gold[r["index"]] = (r["dataset"], r["gold"])
        rid_by_index[r["index"]] = r.get("request_id")
    if a.eval_split_only:
        print(f"  {suite}: scoring {len(gold)} eval-split rows, skipped {skipped} train-split", flush=True)
    return suite, gold, rid_by_index


def apply_filters(task, resps, docs):
    out = {}
    for ens in task._filters:   # FilterEnsemble.apply wants Instance objects; run its filter chain the way it does internally
        r = [[x] for x in resps]
        for f in ens.filters:
            r = f().apply(r, docs)
        out[ens.name] = list(r)
    return out


results = {}
cells = [c for pat in a.cells for c in sorted(glob.glob(pat))]
if not cells:
    sys.exit(f"FATAL: --cells matched no file: {a.cells}")
for cell in cells:
    suite, gold, rid_by_index = meta_for(cell)
    rec = json.loads(open(cell).readline())
    ids, texts = rec["request_ids"], rec["generated_texts"]
    if a.eval_split_only:
        # eval-split reindex. The scoring loop below pairs response position i with gold[i];
        # filtering makes the suite indices sparse, so subset the responses to the scored rows
        # and renumber densely. The position->request_id correspondence is VERIFIED, not assumed:
        # a mismatch would silently score one document's answer against another's gold.
        if len(texts) < max(gold) + 1 or len(ids) != len(texts):
            raise SystemExit(f"FATAL {cell}: {len(texts)} responses / {len(ids)} ids for a suite whose "
                             f"last scored index is {max(gold)}; refusing to score a truncated cell")
        keep = [i for i in range(len(texts)) if i in gold]
        missing = [i for i in keep if rid_by_index.get(i) is None]
        if missing:
            raise SystemExit(f"FATAL {cell}: the suite metadata carries no request id at {len(missing)} scored "
                             f"positions (first={missing[0]}); identity cannot be verified, refusing")
        bad = [i for i in keep if rid_by_index[i] != ids[i]]
        if bad:
            raise SystemExit(
                f"FATAL {cell}: cell request_ids do not match the suite at {len(bad)} scored "
                f"positions (first={bad[0]}: cell={ids[bad[0]]!r} suite={rid_by_index[bad[0]]!r}). "
                "Refusing to score one document's answer against another's gold.")
        texts = [texts[i] for i in keep]
        ids = [ids[i] for i in keep]
        gold = {new_i: gold[old_i] for new_i, old_i in enumerate(keep)}
    idx = [int(i.split(":")[-1]) if False else None for i in ids]
    # request ids look like gsm8k:test:684 -> the suite index is the row order; use the suite's own index via position
    n = len(ids); ds = gold[0][0]
    scores = {}
    if ds == "gsm8k":
        docs = [{"question": "", "answer": f"#### {gold[i][1]}"} for i in range(n)]
        filtered = apply_filters(gsm, texts, docs)
        per_filter = {}
        for fname, fr in filtered.items():
            acc = 0.0
            ok = []
            for i in range(n):
                res = gsm.process_results(docs[i], [fr[i][0] if isinstance(fr[i], list) else fr[i]])
                ok.append(float(res["exact_match"]))
                acc += ok[-1]
            per_filter[fname] = ok
            scores[f"exact_match,{fname}"] = acc / n
        # Marker-aware composite. NOT a new metric: it selects between lm-eval's OWN two published
        # filters by a stated rule -- strict-match where the generation emits the #### answer marker
        # (GSM8K's own gold format, which is what strict-match keys on), flexible extraction where it
        # does not. Both readings are biased and in opposite directions: this checkpoint appends a
        # confidence epilogue that flexible extraction's last-number rule mistakes for the answer,
        # while the base model often omits the marker under a chat template and strict-match
        # penalises it for that. Reported as a sensitivity check beside the authors' filter.
        if {"strict-match", "flexible-extract"} <= set(per_filter):
            # Key on whether strict-match PARSED, not on whether the text contains "####".
            # strict-match's pattern is `#### (\-?[0-9\.\,]+)`, so a marker followed by anything
            # outside that class -- "#### $21" is the common one -- fails to match and yields
            # [invalid] even though the marker is present. That failure is strongly arm-dependent
            # (measured at 1.25xQ*: upstream 62 of 1109 marker rows, hybrid 3 of 1200, always-route
            # 0 of 1199), so the naive "has ####" rule discards baseline rows that flexible
            # extraction scores correctly and biases the comparison toward the served arms.
            sm_raw = filtered["strict-match"]
            acc = 0.0
            for i in range(n):
                v = sm_raw[i][0] if isinstance(sm_raw[i], list) else sm_raw[i]
                parsed = str(v).strip() not in ("", "[invalid]")
                acc += per_filter["strict-match"][i] if parsed else per_filter["flexible-extract"][i]
            scores["exact_match,marker-composite"] = acc / n
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
    # Per-document score for the headline metric, so a downstream table can compute a PAIRED
    # interval across arms: every arm scores the same documents in the same order, which is the only
    # reason a single-repetition cell can carry an interval at all. It quantifies document sampling,
    # NOT run-to-run variance -- the consumer must say so. Emitted for all three workloads.
    if ds == "gsm8k" and "exact_match,marker-composite" in scores:
        sm = filtered.get("strict-match"); fx = filtered.get("flexible-extract")
        if sm and fx:
            per = []
            for i in range(n):
                v = sm[i][0] if isinstance(sm[i], list) else sm[i]
                src_f = sm if str(v).strip() not in ("", "[invalid]") else fx
                r = gsm.process_results(docs[i], [src_f[i][0] if isinstance(src_f[i], list) else src_f[i]])
                per.append(float(r["exact_match"]))
            results.setdefault("__per_row__", {})[cell] = {"metric": "exact_match,marker-composite",
                                                          "ok": per, "ids": list(ids)}
    elif ds == "bbh_cot":
        fr = filtered.get("get-answer")
        if fr:
            per = [float(bbh.process_results(docs[i],
                        [fr[i][0] if isinstance(fr[i], list) else fr[i]])["exact_match"])
                   for i in range(n)]
            results.setdefault("__per_row__", {})[cell] = {"metric": "exact_match,get-answer",
                                                          "ok": per, "ids": list(ids)}
    elif ds == "coqa":
        per = []
        for i in range(n):
            g = gold[i][1]; g = g if isinstance(g, list) else [g]
            per.append(float(coqa_utils.compute_scores(g, texts[i])["f1"]))
        results.setdefault("__per_row__", {})[cell] = {"metric": "f1", "ok": per, "ids": list(ids)}
    # provenance the table generator checks: whether the train-split padding was excluded
    results[cell] = {"suite": suite, "n": n, "scores": scores, "eval_split_only": bool(a.eval_split_only)}
    print(cell.split("/campaign/")[-1], n, {k: round(v, 4) for k, v in scores.items()}, flush=True)
json.dump(results, open(a.out, "w"), indent=1)
print("LMEVAL-SCORE-DONE", len(results))
