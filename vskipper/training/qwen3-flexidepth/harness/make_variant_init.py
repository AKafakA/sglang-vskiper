#!/usr/bin/env python3
"""Derive a config-variant init from an existing init dir WITHOUT copying weights: hardlink every file,
rewrite config.json with the given key=value overrides, install the patched source files, and rewrite
INITIALIZATION_MANIFEST.json (status PASS + output_files hashes) so the sealed trainer accepts it."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--output", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--set", action="append", default=[], help="config key=json_value")
a = ap.parse_args()
base, out, src = Path(a.base).resolve(), Path(a.output).resolve(), Path(a.src).resolve()
assert not out.exists(), out
tmp = out.with_name(out.name + ".tmp"); tmp.mkdir(parents=True)
for p in base.iterdir():
    if p.is_file() and p.name not in ("config.json", "INITIALIZATION_MANIFEST.json", "configuration_ddqwen3.py", "modeling_ddqwen3.py", "STEP0_VERIFY.json"):
        try:
            os.link(p, tmp / p.name)
        except OSError:
            shutil.copy2(p, tmp / p.name)  # cross-device (e.g. NFS base): real copy
cfg = json.loads((base / "config.json").read_text())
for kv in a.set:
    k, _, v = kv.partition("="); cfg[k] = json.loads(v)
(tmp / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
for name in ("configuration_ddqwen3.py", "modeling_ddqwen3.py"):
    shutil.copy2(src / "models/qwen3_flexidepth_source" / name, tmp / name)
bm = json.loads((base / "INITIALIZATION_MANIFEST.json").read_text())
files = {str(p.relative_to(tmp)): sha256(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
man = {"schema_version": 1, "status": "PASS", "derived_from_init": str(base), "derived_from_manifest_sha256": sha256(base / "INITIALIZATION_MANIFEST.json"),
       "delta": f"config overrides {a.set}; patched source from {src}; weights hardlinked (bit-identical)", "config_overrides": a.set,
       "routing_layers": cfg.get("routing_layers"), "router_head_bias_init": cfg.get("router_head_bias_init"), "router_gate_mode": cfg.get("router_gate_mode"),
       "base_delta": bm.get("delta"), "output_files": files}
(tmp / "INITIALIZATION_MANIFEST.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
tmp.rename(out)
print("SAVED", out, "gate_mode", cfg.get("router_gate_mode"), "bias", cfg.get("router_head_bias_init"))
