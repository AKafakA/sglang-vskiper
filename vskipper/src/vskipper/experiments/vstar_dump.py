#!/usr/bin/env python3
"""Stage 2 (serving host): dump the roofline rule's crossover V* for every swept arm.

Imports the served tree's own `design` and `roofline` modules, so the numbers the paper prints are the ones the
runtime derives at boot -- not a re-implementation of the rule in the analysis layer. Runs wherever the tree is
importable; no GPU and no server are needed, only the package.

The output JSON is small and is shipped in the paper-minimal evidence tier, so Stage 3 (`vstar_macros.py`) turns
it into LaTeX macros from a clone with no host at all.

usage: vstar_dump.py [--device NVIDIA_A100] [--out sweep_vstar.json]
"""
import argparse, json, sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="NVIDIA_A100", help="device key the band is derived for")
    ap.add_argument("--out", help="write JSON here (default: stdout)")
    args = ap.parse_args()

    from vskipper.runtime import design, roofline

    out = {"device": args.device, f"device_peaks_{args.device.split('_')[-1]}": roofline.device_peaks(args.device),
           "arms": {}}
    for name, arm in sorted(design.ARMS.items()):
        if not (name.startswith("integrated_randomskip_") or name in ("vskipper", "vskipper_qwen3_4b")):
            continue
        r = roofline.arm_kv_rule_inputs(arm)
        out["arms"][name] = {
            "rate": arm.get("mock_token_skip_rate"),
            "depth": arm.get("mock_skipped_depth_ratio"),
            "s": r["skip_ratio"], "Lr": r["routed_layers"], "tau_ms": r["tau_ms"],
            "vstar": round(roofline.kv_crossover_tokens(args.device, **r)),
            "band": arm.get("decode_kv_band", {}).get(args.device),
        }
    text = json.dumps(out, indent=1)
    if args.out:
        open(args.out, "w").write(text + "\n")
        print(f"  {len(out['arms'])} arms -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
