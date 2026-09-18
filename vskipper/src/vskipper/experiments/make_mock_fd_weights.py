#!/usr/bin/env python3
"""Generate SYNTHETIC FlexiDepth router/projector weights for mock-skipper arms.

The deterministic_mock skipper requires FD weights loaded (its PROJECT payload
runs the projector), so gating a family that has no trained checkpoint yet
(e.g. the qwen3 mock arm before the D-306 training lands) needs a weights file
with that family's dimensions. These weights are RANDOM: mock arms exercise
the machinery and attest routing/accounting — never quality, never
performance. The file must never be confused with a trained checkpoint; the
output name is forced to contain 'mock'.

Shapes mirror vpipe/routing.py exactly:
  FDRouter: router_enc [h/r, h] · router_norm [h/r] · router_dec [h, h/r] ·
            router_head [1, h]
  FDProj:   gate_proj [m/r, h] · down_proj [m/r, h] · up_proj [h, m/r]
Keys mirror seam.init_fd_layer's load_state_dict:
  model.layers.{i}.router.<name>.weight / model.layers.{i}.router_proj.<name>.weight
"""
import argparse
import hashlib
import json
import pathlib

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="model config.json path")
    ap.add_argument("--layer-lo", type=int, required=True)
    ap.add_argument("--layer-hi", type=int, required=True, help="inclusive")
    ap.add_argument("--reduction", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if "mock" not in out.name:
        raise SystemExit("refusing: output filename must contain 'mock'")
    cfg = json.load(open(args.config))
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    r = hidden // args.reduction
    m = inter // args.reduction

    torch.manual_seed(args.seed)
    sd = {}
    for i in range(args.layer_lo, args.layer_hi + 1):
        p = f"model.layers.{i}"
        sd[f"{p}.router.router_enc.weight"] = torch.randn(r, hidden) * 0.02
        sd[f"{p}.router.router_norm.weight"] = torch.ones(r)
        sd[f"{p}.router.router_dec.weight"] = torch.randn(hidden, r) * 0.02
        sd[f"{p}.router.router_head.weight"] = torch.randn(1, hidden) * 0.02
        sd[f"{p}.router_proj.gate_proj.weight"] = torch.randn(m, hidden) * 0.02
        sd[f"{p}.router_proj.down_proj.weight"] = torch.randn(m, hidden) * 0.02
        sd[f"{p}.router_proj.up_proj.weight"] = torch.randn(hidden, m) * 0.02
    torch.save(sd, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(
        f"wrote {out}  layers {args.layer_lo}-{args.layer_hi}  "
        f"hidden={hidden} inter={inter} r={args.reduction} seed={args.seed}\n"
        f"sha256 {sha}"
    )


if __name__ == "__main__":
    main()
