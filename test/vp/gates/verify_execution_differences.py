#!/usr/bin/env python3
"""Execution-difference gate (2026-09-16, incident-graph-ladder-confound-20260916 rule 1).

Gate E compares each arm's LAUNCH ARGUMENTS with the defaults; G1c compares the arms' configs
with each other. Neither sees what a tree EXECUTES when the difference is written in code: the
fork captured decode CUDA graphs to 1,024 rows while upstream captured to 256, both reporting
`cuda_graph_config.decode.max_bs = 256`. This gate reads, from every arm's boot snapshot
(`server_info.conformance.<arm>.<port>.json`), the EFFECTIVE decode ladder (the fork's from
`vp_runtime.fd_c3.ladder`, upstream's from `cuda_graph_config`), the K/V capacity, the chunked
prefill size, memory fraction, dtype and backends, and refuses unless every cross-arm difference
is declared in `deploy/execution_differences.json` with the expected value and a reason.

[plan v3, 2026-09-16] It also checks the fork arms' REGIME-SWITCH BAND: the live decode K/V band
(`vp_runtime.regime_switch.decode`) must equal what the roofline rule gives for the device the gate
is told it runs on (`--device-name`/`--device-memory-mib` from nvidia-smi, the memory class deciding
the A100 key) with that arm's attested rule inputs (`model.design.routed_layers`,
`design_skip_ratio`), unless the arm attests `decode_kv_band_policy == "shared"` AND the declaration
says so. A card whose band was never derived cannot pass.

usage: verify_execution_differences.py --snapshots DIR --arms a,b,c --declaration FILE [--report]
       [--device-name "NVIDIA RTX 5880 Ada Generation" --device-memory-mib 49140]
"""
import argparse, glob, json, sys
from pathlib import Path


def dig(o, *path, default=None):
    for p in path:
        if isinstance(o, dict) and p in o:
            o = o[p]
        elif isinstance(o, list) and isinstance(p, int) and len(o) > p:
            o = o[p]
        else:
            return default
    return o


def effective(info: dict) -> dict:
    vp = dig(info, "internal_states", 0, "vp_runtime")
    if vp:  # fork arm: the ladder the conditional-graph runtime actually captured
        lad = dig(vp, "fd_c3", "ladder", default={}) or {}
        bs = lad.get("capture_bs") or []
        top, n = (max(bs) if bs else lad.get("decode_capture_bs_max")), len(bs)
        source = "vp_runtime.fd_c3.ladder"
    else:   # upstream-served arm: the server's own decode graph config
        dec = dig(info, "cuda_graph_config", "decode", default={}) or {}
        bs = dec.get("bs") or []
        top, n = (max(bs) if bs else dec.get("max_bs")), len(bs)
        source = "cuda_graph_config.decode"
    rs = (dig(vp, "regime_switch", "decode", default={}) or {}) if vp else {}
    md = dig(info, "internal_states", 0, "vp_runtime", "model", "design", default={}) or {}
    dec_cfg = dig(info, "cuda_graph_config", "decode", default={}) or {}
    pre_cfg = dig(info, "cuda_graph_config", "prefill", default={}) or {}
    return {
        "decode_ladder_top": top, "decode_ladder_buckets": n, "decode_ladder_source": source,
        "decode_ladder_bs": list(bs),                       # the LIST, compared across arms (review 09-16 finding 3)
        "decode_graph_backend": dec_cfg.get("backend"), "decode_tc_compiler": dec_cfg.get("tc_compiler"),
        "prefill_graph_backend": pre_cfg.get("backend"), "prefill_graph_bs": list(pre_cfg.get("bs") or []),
        "kv_capacity_tokens": info.get("max_total_num_tokens") or dig(info, "memory_usage", "token_capacity"),
        "chunked_prefill_size": info.get("chunked_prefill_size"),
        "mem_fraction_static": info.get("mem_fraction_static"),
        "dtype": info.get("dtype"),
        "attention_backend": info.get("attention_backend"),
        "sampling_backend": info.get("sampling_backend"),
        "decode_kv_band": [rs.get("exit_kv_tokens"), rs.get("enter_kv_tokens")] if rs else None,
        "decode_kv_band_policy": md.get("decode_kv_band_policy"),
        "routed_layers": md.get("routed_layers"),
        "design_skip_ratio": md.get("design_skip_ratio"),
    }


def rule_band(device_name: str, device_memory_mib: int, routed_layers: list, skip_ratio: float) -> tuple:
    """The band the tree's own rule gives for this device and these arm inputs (no GPU needed)."""
    tree = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(tree / "python"))
    try:
        from sglang.srt.vpipe.kernel import canonical_device_key
        from sglang.srt.vpipe.roofline import DECODE_BODY_TAX_MS, band_device_key, derived_kv_band
    finally:
        sys.path.remove(str(tree / "python"))
    key = band_device_key(canonical_device_key(device_name), int(device_memory_mib) * 1024 * 1024)
    n = len(routed_layers)
    return key, derived_kv_band(key, routed_layers=n, skip_ratio=float(skip_ratio), tau_ms=DECODE_BODY_TAX_MS * n / 16.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True, type=Path)
    ap.add_argument("--arms", required=True, help="comma-separated arm names, baseline first")
    ap.add_argument("--declaration", required=True, type=Path)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--port", default=None, help="only this campaign's snapshots (server_info.conformance.<arm>.<port>.json)")
    ap.add_argument("--since", type=float, default=None, help="refuse snapshots older than this epoch (stale files from a previous run)")
    ap.add_argument("--device-name", default=None, help="CUDA device name of the serving GPU (nvidia-smi); enables the band check")
    ap.add_argument("--device-memory-mib", type=int, default=None, help="its total memory in MiB (memory class selects the A100 key)")
    a = ap.parse_args()
    decl = json.loads(a.declaration.read_text())
    arms = a.arms.split(",")
    eff = {}
    for arm in arms:
        files = sorted(glob.glob(str(a.snapshots / f"server_info.conformance.{arm}.{a.port or '*'}.json")))
        if not files:
            print(f"REFUSED: no boot snapshot for arm {arm!r} under {a.snapshots}"); return 2
        snap = Path(files[-1])
        if a.since is not None and snap.stat().st_mtime < a.since:
            print(f"REFUSED: snapshot for {arm!r} ({snap.name}) predates this campaign (stale)"); return 2
        info = json.loads(snap.read_text())
        eff[arm] = effective(info); eff[arm]["snapshot"] = snap.name
        eff[arm]["is_fork"] = bool(dig(info, "internal_states", 0, "vp_runtime"))
    base = arms[0]
    print(f"execution differences vs {base} ({eff[base]['snapshot']}):")
    undeclared = []
    for arm in arms:
        e = eff[arm]; d = decl["declared"].get(arm)
        if d is None and e["is_fork"] and "fork_default" in decl:
            d = decl["fork_default"]      # every fork-served arm (sweep/ablation/Qwen mocks) shares the design's ladder + K/V rule
        d = d or {}
        print(f"  {arm:16s} ladder top {e['decode_ladder_top']!s:>5} ({e['decode_ladder_buckets']} buckets, {e['decode_ladder_source']}, "
              f"list==base {e['decode_ladder_bs']==eff[base]['decode_ladder_bs']}) decode {e['decode_graph_backend']}/{e['decode_tc_compiler']} "
              f"prefill {e['prefill_graph_backend']}/{len(e['prefill_graph_bs'])} kv {e['kv_capacity_tokens']} chunk {e['chunked_prefill_size']} "
              f"mem {e['mem_fraction_static']} dtype {e['dtype']} attn {e['attention_backend']} samp {e['sampling_backend']} [{e['snapshot']}]")
        for field in decl["fields_compared"]:
            v, vb = e.get(field), eff[base].get(field)
            expected = d.get(field)
            if field == "decode_ladder_top":
                if expected is None or v != expected:
                    undeclared.append(f"{arm}: {field}={v} (declared {expected})")
            elif field == "kv_capacity_tokens":
                if v is None or vb is None:
                    undeclared.append(f"{arm}: {field} unreadable"); continue
                rel = abs(v - vb) / max(1, vb)
                if v > vb and expected != "larger_ok":      # a larger pool FAVOURS the treatment: never passes undeclared
                    undeclared.append(f"{arm}: {field}={v} > baseline {vb} (+{rel*100:.2f} %) undeclared")
                elif v < vb and (expected != "smaller_ok" or rel > decl["tolerances"]["kv_capacity_tokens_rel"]):
                    undeclared.append(f"{arm}: {field}={v} vs {vb} (-{rel*100:.2f} %, declared {expected})")
            elif field == "decode_ladder_buckets":
                continue  # reported; the LIST below is what is gated
            elif field == "decode_ladder_bs":
                # arms declared at the same top must capture the SAME bucket list; a declared different top
                # (the literal-default arm) is allowed to differ
                if d.get("decode_ladder_top") == decl["declared"].get(base, {}).get("decode_ladder_top") and v != vb:
                    undeclared.append(f"{arm}: decode bucket list differs from {base} (n={len(v)} vs {len(vb)}, sets differ: {sorted(set(v)^set(vb))[:8]})")
            elif field in ("prefill_graph_bs",):
                if v != vb:
                    undeclared.append(f"{arm}: {field} differs from {base} (n={len(v)} vs {len(vb)})")
            elif field == "decode_kv_band":
                if not e["is_fork"]:
                    continue                                  # upstream serves no band
                if a.device_name is None or a.device_memory_mib is None:
                    undeclared.append(f"{arm}: {field} check needs --device-name/--device-memory-mib (band {v})"); continue
                if e["decode_kv_band_policy"] == "shared":
                    if expected != "shared":
                        undeclared.append(f"{arm}: {field} served as a declared deviation (policy shared, band {v}) but the declaration says {expected!r}")
                    continue
                if expected != "rule":
                    undeclared.append(f"{arm}: {field} must be declared 'rule' for a fork arm (declared {expected!r})"); continue
                if not v or e["routed_layers"] is None or e["design_skip_ratio"] is None:
                    undeclared.append(f"{arm}: {field} unreadable from the attestation (band {v}, layers {e['routed_layers']}, s {e['design_skip_ratio']})"); continue
                key, want = rule_band(a.device_name, a.device_memory_mib, e["routed_layers"], e["design_skip_ratio"])
                if tuple(v) != tuple(want):
                    undeclared.append(f"{arm}: {field}={v} but the rule gives {list(want)} for {key} (s={e['design_skip_ratio']}, {len(e['routed_layers'])} routed layers)")
                else:
                    print(f"  {arm:16s} decode K/V band {v} == rule for {key} (s={e['design_skip_ratio']}, {len(e['routed_layers'])} routed layers)")
            else:
                if v != vb and expected is None:
                    undeclared.append(f"{arm}: {field}={v!r} vs {base} {vb!r} (undeclared)")
    if undeclared:
        print("REFUSED: undeclared execution difference(s):"); [print("   ! " + u) for u in undeclared]
        return 0 if a.report else 1
    print("OK: every cross-arm execution difference is declared in " + str(a.declaration))
    return 0


if __name__ == "__main__":
    sys.exit(main())
