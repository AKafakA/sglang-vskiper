#!/usr/bin/env python3
"""Mixed quality gate for a (FlexiDepth-)Qwen3 checkpoint: gsm8k + bbh (raw 3-shot CoT) + coqa_mt + coqa_flat (2026-09-16)
(fixed file mixed_val_400.json), one user message per item under the model's own
chat template, thinking disabled, greedy. Scores: gsm8k = last number equals gold;
mmlu_pro = last 'answer is (X)' letter equals gold. Writes JSON + prints one line.

usage: gate_mixed.py --model DIR --val mixed_val_400.json --output gate.json
       [--dtype bfloat16] [--gsm8k-n 200] [--mmlu-n 200] [--batch 8]
"""
import argparse, json, re, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
ANS = re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE)


def score_gsm8k(text, gold):
    m = NUM.findall(text)
    return bool(m) and m[-1].replace(",", "").rstrip(".") == gold


def _norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def score_f1(text, gold):
    p = _norm(text.strip().split("\n")[0]); g = _norm(gold)
    if not p or not g:
        return float(p == g)
    common = 0
    gg = list(g)
    for w in p:
        if w in gg:
            gg.remove(w); common += 1
    if common == 0:
        return 0.0
    pr = common / len(p); rc = common / len(g)
    return 2 * pr * rc / (pr + rc)


BBH_ANS = re.compile(r"(?<=the answer is )(.*)(?=.)")


def score_bbh(text, gold):
    """lm-eval bbh cot_fewshot: cut at the few-shot separator, take the FIRST 'the answer is ' capture, exact match."""
    for sep in ("\n\n", "\nQ:"):
        text = text.split(sep)[0]
    m = BBH_ANS.findall(text)
    if not m:
        return False
    return m[0].strip().rstrip(".").strip() == str(gold).strip().rstrip(".").strip()


def score_mmlu(text, gold):
    m = ANS.findall(text)
    if m:
        return m[-1].upper() == gold
    m2 = re.findall(r"\(([A-J])\)", text)
    return bool(m2) and m2[-1].upper() == gold


def run(model, tok, items, max_new, batch):
    outs = []
    for b in range(0, len(items), batch):
        chunk = items[b:b + batch]
        prompts = []
        stop = None
        for q in chunk:
            if "raw_prompt" in q:  # raw continuation (BBH CoT: the paper's protocol; chat kills the answer marker)
                prompts.append(q["raw_prompt"]); stop = q.get("stop") or stop
                continue
            msgs = q["messages"] if "messages" in q else [{"role": "user", "content": q["question"]}]
            try:
                p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(p)
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        extra = {"stop_strings": list(stop), "tokenizer": tok} if stop else {}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id, **extra)
        for i in range(len(chunk)):
            outs.append(tok.decode(gen[i][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gsm8k-n", type=int, default=100)
    ap.add_argument("--bbh-n", type=int, default=100)
    ap.add_argument("--tasks", default=None, help="comma-separated subset of the task table")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    val = json.load(open(a.val))
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype=getattr(torch, a.dtype), device_map="cuda")
    model.eval()
    t0 = time.time(); res = {"model": a.model, "tasks": {}}
    # Owner 2026-09-16: the gate is the paper's three datasets -- gsm8k (chat), bbh (raw 3-shot CoT), coqa (chat, multi-turn
    # and flat). mmlu_pro and ifeval removed. `--tasks` narrows further (e.g. stock reference re-gate of bbh only).
    table = (("gsm8k", a.gsm8k_n, 512, score_gsm8k), ("bbh", a.bbh_n, 512, score_bbh),
             ("coqa_mt", 100, 48, score_f1), ("coqa_flat", 100, 48, score_f1))
    for task, n, max_new, scorer in table:
        if a.tasks and task not in a.tasks.split(","):
            continue
        items = val.get(task, [])[:n]
        if not items:
            continue
        outs = run(model, tok, items, max_new, a.batch)
        empties = sum(1 for o in outs if not o.strip())
        if scorer is None:  # generation-only tasks scored off-box (ifeval checker / isolated humaneval scorer)
            lens = sorted(len(o) for o in outs)
            res["tasks"][task] = {"n": len(items), "accuracy": None, "correct": None, "empties": empties,
                                  "median_len": lens[len(lens) // 2], "scored": False,
                                  "rows": [{"i": q["idx"], "response": o, **{k: q[k] for k in ("key", "instruction_id_list", "kwargs", "task_id", "prompt", "test", "entry_point") if k in q}} for o, q in zip(outs, items)]}
            continue
        ok = [scorer(o, q["gold"]) for o, q in zip(outs, items)]
        lens = sorted(len(o) for o in outs)
        res["tasks"][task] = {"n": len(items), "accuracy": sum(ok) / max(1, len(items)), "correct": int(sum(1 for k in ok if k >= 0.999)),
                              "empties": empties, "median_len": lens[len(lens) // 2],
                              "rows": [{"i": q["idx"], "gold": q["gold"], "ok": bool(k), "head": o[:100]} for o, q, k in zip(outs, items, ok)]}
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.output).write_text(json.dumps(res, indent=1))
    parts = [(f"{k}={v['accuracy']:.3f}(e{v['empties']})" if v["accuracy"] is not None else f"{k}=gen-only(e{v['empties']},len{v['median_len']})") for k, v in res["tasks"].items()]
    print("GATE model=" + a.model + " " + " ".join(parts) + f" t={res['seconds']}s")


if __name__ == "__main__":
    main()
