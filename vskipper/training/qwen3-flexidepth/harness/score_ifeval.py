#!/usr/bin/env python3
"""Score the ifeval rows of a gate JSON in place (prompt-level strict + loose, inst-level strict)
using lm-eval's instruction checker. usage: score_ifeval.py gate.json"""
import json, sys
from lm_eval.tasks.ifeval import instructions_registry
p = sys.argv[1]; d = json.load(open(p)); t = d["tasks"].get("ifeval")
if not t: print("no ifeval rows"); sys.exit(0)
ps = pl = 0; inst_ok = inst_n = 0
for r in t["rows"]:
    resp = r["response"]; ids = r["instruction_id_list"]; kws = r["kwargs"]
    strict_all = True; loose_all = True
    for iid, kw in zip(ids, kws):
        cls = instructions_registry.INSTRUCTION_DICT[iid]; ins = cls(iid)
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        ins.build_description(**kw)
        args = ins.get_instruction_args()
        if args and "prompt" in args: ins.build_description(prompt=r.get("prompt", ""))
        ok = bool(resp.strip()) and ins.check_following(resp)
        inst_n += 1; inst_ok += ok
        strict_all &= ok
        # loose: try common relaxations
        variants = [resp, resp.replace("*", ""), "\n".join(resp.split("\n")[1:]), "\n".join(resp.split("\n")[:-1])]
        loose_all &= any(bool(v.strip()) and ins.check_following(v) for v in variants)
    ps += strict_all; pl += loose_all
n = len(t["rows"])
t.update({"accuracy": ps / n, "correct": ps, "prompt_loose_acc": pl / n, "inst_strict_acc": inst_ok / max(1, inst_n), "scored": True})
json.dump(d, open(p, "w"), indent=1)
print(f"IFEVAL prompt_strict={ps/n:.3f} prompt_loose={pl/n:.3f} inst_strict={inst_ok/max(1,inst_n):.3f} n={n}")
