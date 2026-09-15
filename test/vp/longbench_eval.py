"""LongBench quality eval — resolves the task-vs-length confound behind the I-195 depth frontier.

I-194/195: SHORT factual QA (triviaqa, ~20-tok prompt) is destroyed by prefill-skip even at depth-1, while
LONG summarization (arxiv) is robust to depth-3. Confound: is it the TASK (factual recall) or the LENGTH
(short prompt has no redundancy to absorb the project-only approximation)? LongBench gives LONG-context
factual QA (hotpotqa / 2wikimqa, ~5-9k tokens) + long-context summ (gov_report / multi_news) at the SAME
skip depth => if long-ctx QA tolerates skip (unlike short triviaqa) the effect is LENGTH and the paper's
operating envelope is "all long-context"; if it collapses too, the effect is TASK and the envelope is summ-only.

One TASK for one MODE per process (mode chosen by env, same as task_quality.py: SGLANG_VP_MODE=skip_prefill:N
=> prefill-skip, SGLANG_FD_WEIGHTS => flexidepth, else vanilla). Thinking OFF. Middle-truncate context to fit.
QA scored with LongBench qa_f1; summ with ROUGE-L (+ optional BERTScore)."""
import json
import os
import re
import string
import zipfile
from collections import Counter

import datasets
from huggingface_hub import hf_hub_download

MODEL = os.environ.get("VP_AGREE_MODEL", "NousResearch/Meta-Llama-3-8B-Instruct")
TASK = os.environ.get("TASK", "hotpotqa")
N = int(os.environ.get("QN", "100"))
MAXLEN = int(os.environ.get("MAXLEN", "7000"))  # total prompt-token budget (Llama-3-8B 8k, room for gen)
MODE = os.environ.get("QMODE_LABEL") or ("flexidepth" if os.environ.get("SGLANG_FD_WEIGHTS") else "vanilla")
REQUEST_EXTRA_KEY = os.environ.get("VP_REQUEST_EXTRA_KEY")

# LongBench prompt templates + gen budgets + metric (subset we use)
PROMPT = {
    "triviaqa_lc": ("Answer the question based on the given context. Only give me the answer and do not output "
                    "any other words.\n\nContext:\n{context}\n\nQuestion: {input}\nAnswer:"),
    "hotpotqa": ("Answer the question based on the given passages. Only give me the answer and do not output "
                 "any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based "
                 "on the given passages. Only give me the answer and do not output any other words.\n\n"
                 "Question: {input}\nAnswer:"),
    "2wikimqa": ("Answer the question based on the given passages. Only give me the answer and do not output "
                 "any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based "
                 "on the given passages. Only give me the answer and do not output any other words.\n\n"
                 "Question: {input}\nAnswer:"),
    "qasper": ("You are given a scientific article and a question. Answer the question as concisely as you can, "
               "using a single phrase or sentence if possible. If the question cannot be answered based on the "
               "information in the article, write \"unanswerable\". If the question is a yes/no question, answer "
               "\"yes\", \"no\", or \"unanswerable\".\n\nArticle: {context}\n\nAnswer the question based on the "
               "above article as concisely as you can, using a single phrase or sentence if possible.\n\n"
               "Question: {input}\n\nAnswer:"),
    "gov_report": ("You are given a report by a government agency. Write a one-page summary of the report.\n\n"
                   "Report:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"),
    "multi_news": ("You are given several news passages. Write a one-page summary of all news.\n\nNews:\n"
                   "{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:"),
}
GENLEN = {"triviaqa_lc": 32, "hotpotqa": 32, "2wikimqa": 32, "qasper": 128, "gov_report": 512, "multi_news": 512}
METRIC = {"triviaqa_lc": "f1", "hotpotqa": "f1", "2wikimqa": "f1", "qasper": "f1", "gov_report": "rouge", "multi_news": "rouge"}


def load_records(task, n):
    """Return list of {context, input, answers}. triviaqa_lc = SAME triviaqa task as the short-ctx run
    (I-195) but WITH its long evidence documents (rc config) => isolates length from task. LongBench tasks
    load directly from data.zip (datasets>=4.0 refuses the repo's .py loader script)."""
    if task == "triviaqa_lc":
        # STREAMING: rc config is tens of GB; stream so we pull only the ~n examples we consume.
        d = datasets.load_dataset("mandarjoshi/trivia_qa", "rc", split="validation", streaming=True)
        recs = []
        for x in d:
            ep = x.get("entity_pages") or {}
            sr = x.get("search_results") or {}
            parts = list(ep.get("wiki_context", []) or []) + list(sr.get("search_context", []) or [])
            ctx = "\n\n".join(p for p in parts if p)
            if not ctx:
                continue  # skip questions with no evidence (would collapse to the short-ctx case)
            recs.append({"context": ctx, "input": x["question"],
                         "answers": list(x["answer"]["aliases"]) + [x["answer"]["value"]]})
            if len(recs) >= n:
                break
        return recs
    # LongBench now resolves to a repo with data.zip + LongBench.py only; there is no
    # refs/convert/parquet revision, and datasets>=4.0 rejects the script. Read the
    # JSONL member directly so the eval is independent of datasets script support.
    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    member = f"data/{task}.jsonl"
    recs = []
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        for line in f:
            x = json.loads(line.decode("utf-8"))
            recs.append({"context": x["context"], "input": x["input"], "answers": x["answers"]})
            if len(recs) >= n:
                break
    return recs


def _norm(s):
    s = s.lower().strip()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred, golds):
    best = 0.0
    for gold in golds:
        p, g = _norm(pred).split(), _norm(gold).split()
        if not p or not g:
            best = max(best, 0.0)
            continue
        common = sum((Counter(p) & Counter(g)).values())
        if common == 0:
            continue
        prec, rec = common / len(p), common / len(g)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else (dp[j] if dp[j] >= dp[j - 1] else dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(pred, ref):
    p, r = pred.split(), ref.split()
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    prec, rec = lcs / len(p), lcs / len(r)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def main():
    import sglang as sgl
    from transformers import AutoTokenizer

    d = load_records(TASK, N)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tmpl = PROMPT[TASK]
    genlen = GENLEN[TASK]
    metric = METRIC[TASK]

    prompts, refs, ctxlens = [], [], []
    for x in d:
        # middle-truncate the CONTEXT so the fully-rendered chat prompt fits MAXLEN tokens (LongBench protocol)
        budget = MAXLEN - genlen - len(tok(tmpl.format(context="", input=x["input"]),
                                            add_special_tokens=False)["input_ids"]) - 64
        cids = tok(x["context"], add_special_tokens=False)["input_ids"]
        if len(cids) > budget:
            half = budget // 2
            cids = cids[:half] + cids[-(budget - half):]
        ctxlens.append(len(cids))
        context = tok.decode(cids)
        content = tmpl.format(context=context, input=x["input"])
        msgs = [{"role": "user", "content": content}]
        try:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompts.append(p)
        refs.append(x["answers"])

    avg_ctx = sum(ctxlens) / len(ctxlens)
    e = sgl.Engine(model_path=MODEL, mem_fraction_static=0.85, disable_cuda_graph=True,
                   disable_radix_cache=True, disable_overlap_schedule=True,
                   max_running_requests=32, tp_size=1)
    generate_kwargs = {}
    if REQUEST_EXTRA_KEY:
        generate_kwargs["extra_key"] = [REQUEST_EXTRA_KEY] * len(prompts)
    outs = e.generate(
        prompts,
        {"temperature": 0.0, "max_new_tokens": genlen},
        **generate_kwargs,
    )
    e.shutdown()
    gens = [o["text"] for o in outs]

    if metric == "f1":
        s = sum(qa_f1(g.strip().split("\n")[0], golds) for g, golds in zip(gens, refs)) / len(gens)
        print(f"LONGBENCH [{MODE}] {TASK}: F1={s:.3f} (n={len(gens)} avg_ctx={avg_ctx:.0f}tok)", flush=True)
    else:
        s = sum(max(rouge_l(g, r) for r in golds) for g, golds in zip(gens, refs)) / len(gens)
        print(f"LONGBENCH [{MODE}] {TASK}: ROUGE-L={s:.3f} (n={len(gens)} avg_ctx={avg_ctx:.0f}tok)", flush=True)


if __name__ == "__main__":
    main()
