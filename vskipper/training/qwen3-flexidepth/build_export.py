#!/usr/bin/env python3
"""Build a served/native export from the frozen base + a router delta (.pt with the trained tensors only).
Base shards are symlinked; the delta becomes one extra safetensors shard; the index maps the trained keys to it; config/code/
tokenizer files come from a reference export of the same run. --verify REF checks every tensor against a real export."""
import argparse, json, os, sys, torch
from safetensors.torch import save_file, load_file
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--delta", required=True)
ap.add_argument("--ref", required=True, help="a real export of the same run (config, model code, tokenizer files)")
ap.add_argument("--out", required=True); ap.add_argument("--verify", default=None, help="a real export of THIS checkpoint to compare tensors")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.delta, map_location="cpu"); d = d.get("state_dict", d)
d = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous() for k, v in d.items()}
idx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))
wm = dict(idx["weight_map"]); shard = "model-router.safetensors"
save_file(d, os.path.join(a.out, shard), metadata={"format": "pt"})
for k in d: wm[k] = shard
for f in sorted(set(idx["weight_map"].values())):
    dst = os.path.join(a.out, f)
    if not os.path.exists(dst): os.symlink(os.path.join(a.base, f), dst)
json.dump({"metadata": idx.get("metadata", {}), "weight_map": wm}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
for f in ("config.json", "configuration_ddqwen3.py", "modeling_ddqwen3.py", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
          "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja"):
    src = os.path.join(a.ref, f)
    if os.path.exists(src):
        dst = os.path.join(a.out, f)
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(os.path.realpath(src), dst)
print(f"built {a.out}: {len(d)} delta tensors in {shard}, {len(set(idx['weight_map'].values()))} base shards linked")
if a.verify:
    ridx = json.load(open(os.path.join(a.verify, "model.safetensors.index.json")))["weight_map"]
    rk = set(ridx); mk = set(wm)
    if rk != mk: sys.exit(f"KEY MISMATCH: only-ref {sorted(rk-mk)[:5]} only-built {sorted(mk-rk)[:5]}")
    bad = 0; cache = {}
    def get(root, wmap, k):
        f = os.path.join(root, wmap[k])
        if f not in cache: cache[f] = load_file(f)
        return cache[f][k]
    built_cache = {}
    for i, k in enumerate(sorted(rk)):
        r = get(a.verify, ridx, k); b = get(a.out, wm, k)
        if r.dtype != b.dtype or r.shape != b.shape or not torch.equal(r, b): bad += 1; print("DIFF", k, r.dtype, b.dtype, tuple(r.shape))
        if len(cache) > 3: cache.clear()
    print(f"verify vs {a.verify}: {len(rk)} tensors, {bad} differ")
    sys.exit(1 if bad else 0)
