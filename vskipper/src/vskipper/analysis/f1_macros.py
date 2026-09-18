#!/usr/bin/env python3
"""Figure-1 numbers (Section 2.2): the stock-loop throughput change of the FlexiDepth checkpoint against its
base at batch sizes 1 and 8, from the four run summaries in F1/ (decode_tok_per_s). Macros \vpFOneDropBsOne etc."""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("f1_dir"); ap.add_argument("--macros", required=True)
a = ap.parse_args(); out = []
for bs, w in ((1, "One"), (8, "Eight")):
    fd = json.load(open(os.path.join(a.f1_dir, f"f1_flexidepth_bs{bs}.json"))); va = json.load(open(os.path.join(a.f1_dir, f"f1_vanilla_bs{bs}.json")))
    assert fd["batch_size"] == va["batch_size"] == bs and fd["n_prompts"] == va["n_prompts"] and fd["new_tokens_per_req"] == va["new_tokens_per_req"]
    drop = 100.0 * (va["decode_tok_per_s"] - fd["decode_tok_per_s"]) / va["decode_tok_per_s"]
    out.append(f"\\newcommand{{\\vpFOneDropBs{w}}}{{{drop:.1f}}}"); print(f"  bs{bs}: vanilla {va['decode_tok_per_s']} tok/s, flexidepth {fd['decode_tok_per_s']} -> drop {drop:.1f}%")
out.append(f"\\newcommand{{\\vpFOnePrompts}}{{{fd['n_prompts']}}}"); out.append(f"\\newcommand{{\\vpFOneTokens}}{{{fd['new_tokens_per_req']}}}")
open(a.macros, "w").write("\n".join(out) + "\n"); print("wrote", a.macros)
