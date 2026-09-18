#!/usr/bin/env python3
"""Behavioural probe for a trained FlexiDepth checkpoint.

Everything proven so far about FlexiDepth-Qwen3-8B-final-r1 is STRUCTURAL: the
frozen base is unmodified, the trainable set is as declared, and the compact form
reconstructs bit-exactly.  None of that says the model WORKS.

This asks the two behavioural questions:
  1. does the trained router actually skip?  (skip rate > 0 and < 100%)
  2. does the model still generate coherent text?

`router_masks` comes straight out of the model (modeling_ddqwen3.py:283).
mask == 1 -> layer RAN for that token; mask == 0 -> layer was SKIPPED.

usage: behavioural_probe.py --model DIR --output JSON [--max-new-tokens 48]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# A layer that fires rarely will not show up in a handful of short prompts, so
# "always skipped" is only meaningful against a wide, long, diverse sample.
_SHORT = [
    "Question: What is 17 plus 26? Answer:",
    "Explain in one sentence why the sky appears blue.",
    "Write a Python function that returns the maximum of a list.",
    "The capital of France is",
    "Translate to French: The weather is nice today.",
    "What is the derivative of x^3 + 2x?",
    "Name three prime numbers greater than 50.",
    "Who wrote Pride and Prejudice?",
    "Convert 45 degrees Celsius to Fahrenheit.",
    "What is the chemical symbol for gold?",
    "Summarise the plot of Hamlet in two sentences.",
    "Is 91 a prime number? Explain briefly.",
    "Write a SQL query selecting all users older than 30.",
    "What causes tides on Earth?",
    "Give a one-line definition of recursion.",
    "Sort this list mentally and give the median: 8, 3, 91, 12, 45.",
    "What year did the Berlin Wall fall?",
    "Explain the difference between TCP and UDP.",
    "Write a haiku about winter.",
    "What is 12 factorial?",
]
_REASONING = [
    "A train leaves at 14:35 and arrives at 19:10. Journey time?",
    "If a shirt costs $40 after a 20% discount, what was the original price?",
    "Alice has twice as many apples as Bob. Together they have 27. How many does Bob have?",
    "A rectangle has perimeter 34 and width 6. What is its area?",
    "Prove that the sum of two even integers is even.",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
    "What is the probability of rolling two sixes with two fair dice?",
    "Solve for x: 3x + 7 = 2x + 19.",
]
_LONG = [
    "Read the following and answer. The Industrial Revolution began in Britain in the "
    "late 18th century, driven by innovations in textile manufacturing, steam power and "
    "iron production. It spread to continental Europe and North America during the 19th "
    "century, transforming agrarian societies into industrial ones and causing mass "
    "urbanisation, new labour patterns and significant environmental change. "
    "Question: name two technological drivers and one social consequence.",
    "You are given a Python function that is supposed to reverse a string but returns "
    "the original instead:\n\ndef rev(s):\n    out = ''\n    for ch in s:\n        "
    "out = out + ch\n    return out\n\nExplain the bug and give the corrected code.",
    "Consider a distributed key-value store replicating data across three datacentres "
    "with asynchronous replication and last-write-wins conflict resolution. Describe two "
    "failure modes that can cause a client to read stale data, and one mitigation for each.",
    "Below is a short dialogue.\nA: I think we should ship on Friday.\nB: The tests are "
    "still failing on ARM.\nA: Those are flaky though.\nB: Two of them reproduce every "
    "run.\nQuestion: summarise the disagreement and state what evidence would settle it.",
]
PROMPTS = _SHORT + _REASONING + _LONG


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--chat-template", action="store_true",
                    help="apply the tokenizer chat template (instruct models)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), trust_remote_code=True, dtype=torch.bfloat16
    ).cuda().eval()

    cfg = model.config
    routing_layers = list(getattr(cfg, "routing_layers", []) or [])
    report: dict = {
        "model": str(args.model),
        "routing_layers": routing_layers,
        "router_threshold": getattr(cfg, "router_threshold", None),
        "prompts": [],
    }

    total_run = 0
    total_slots = 0
    per_layer_run: dict[int, list[int]] = {}

    report["chat_template"] = bool(args.chat_template)
    for prompt in PROMPTS:
        if args.chat_template:
            text_in = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        else:
            text_in = prompt
        ids = tok(text_in, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**ids, output_hidden_states=False)
        masks = getattr(out, "router_masks", None)
        if masks is None:
            # ddllama exposes router_masks on the BASE model output; the CausalLM
            # wrapper consumes them internally (modeling_ddllama.py:527) without
            # re-exposing them.  Read the base directly rather than report ABSENT.
            with torch.no_grad():
                base_out = model.model(**ids)
            masks = getattr(base_out, "router_masks", None)
        entry: dict = {"prompt": prompt}
        if masks:
            run = skipped = 0
            for layer_index, mask in zip(routing_layers, masks):
                m = mask.detach().float()
                r = int(m.sum().item())
                n = int(m.numel())
                per_layer_run.setdefault(layer_index, [0, 0])
                per_layer_run[layer_index][0] += r
                per_layer_run[layer_index][1] += n
                run += r
                skipped += n - r
            entry["routed_slots"] = run + skipped
            entry["run_slots"] = run
            entry["skip_slots"] = skipped
            entry["skip_rate"] = skipped / (run + skipped) if (run + skipped) else None
            total_run += run
            total_slots += run + skipped
        else:
            entry["router_masks"] = "ABSENT"

        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        entry["continuation"] = text
        report["prompts"].append(entry)
        print(f"\n--- {prompt}", flush=True)
        print(f"    skip_rate = {entry.get('skip_rate')}", flush=True)
        print(f"    -> {text[:200]!r}", flush=True)

    report["overall_skip_rate"] = (
        (total_slots - total_run) / total_slots if total_slots else None
    )
    report["per_layer_skip_rate"] = {
        str(layer): (n - r) / n for layer, (r, n) in sorted(per_layer_run.items()) if n
    }

    # The two behavioural questions, answered explicitly rather than left to a reader.
    rate = report["overall_skip_rate"]
    report["router_is_dynamic"] = bool(rate is not None and 0.0 < rate < 1.0)
    report["nonempty_generations"] = all(
        bool((p.get("continuation") or "").strip()) for p in report["prompts"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n=== SUMMARY ===")
    print(f"overall_skip_rate   = {rate}")
    print(f"router_is_dynamic   = {report['router_is_dynamic']}  "
          f"(0 < skip < 1; False means the router degenerated to all-run or all-skip)")
    print(f"nonempty_generations= {report['nonempty_generations']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
