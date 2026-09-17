#!/usr/bin/env python3
"""Add a `bbh` task to the mixed gate: 100 items, the paper's protocol (test/vp/labeled_workload.py load_bbh_cot):
lm-eval bbh cot_fewshot v4 task files verbatim (description + three fixed CoT shots as a RAW continuation prefix),
"Q: {input}\nA: Let's think step by step.\n", tasks interleaved round-robin, targets = the dataset's `target`.
usage: build_bbh_items.py --val mixed_val.json --yaml-dir bbh_cot_fewshot --out mixed_val_v2.json [--n 100]"""
import argparse, json
from pathlib import Path
import yaml, datasets

TASKS = ("boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "dyck_languages",
         "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
         "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation",
         "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table",
         "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks",
         "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
         "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting")

def q(text): return f"Q: {text}\nA: Let's think step by step.\n"

ap = argparse.ArgumentParser(); ap.add_argument("--val", required=True); ap.add_argument("--yaml-dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=100); a = ap.parse_args()
per_task = []
for task in TASKS:
    spec = yaml.safe_load((Path(a.yaml_dir) / f"{task}.yaml").read_text())
    shots = spec["fewshot_config"]["samples"]
    assert spec["fewshot_config"].get("sampler") == "first_n" and len(shots) == 3, task
    assert spec["doc_to_text"] == "Q: {{input}}\nA: Let's think step by step.\n", task
    prefix = spec["description"] + "\n\n".join(q(s["input"]) + s["target"] for s in shots) + "\n\n"
    rows = datasets.load_dataset("SaylorTwift/bbh", task, split="test")
    per_task.append([{"idx": f"{task}:{i}", "task": task, "raw_prompt": prefix + q(r["input"]), "gold": r["target"],
                      "stop": ["\n\n", "\nQ:"]} for i, r in enumerate(rows)])
items, k = [], 0
while len(items) < a.n:
    t = per_task[k % len(per_task)]; j = k // len(per_task)
    if j < len(t): items.append(t[j])
    k += 1
val = json.load(open(a.val)); val["bbh"] = items
json.dump(val, open(a.out, "w")); print("bbh items", len(items), "tasks", len({i['task'] for i in items}), "->", a.out)
