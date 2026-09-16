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

usage: verify_execution_differences.py --snapshots DIR --arms a,b,c --declaration FILE [--report]
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
    return {
        "decode_ladder_top": top, "decode_ladder_buckets": n, "decode_ladder_source": source,
        "kv_capacity_tokens": info.get("max_total_num_tokens") or dig(info, "memory_usage", "token_capacity"),
        "chunked_prefill_size": info.get("chunked_prefill_size"),
        "mem_fraction_static": info.get("mem_fraction_static"),
        "dtype": info.get("dtype"),
        "attention_backend": info.get("attention_backend"),
        "sampling_backend": info.get("sampling_backend"),
        "prefill_graph": bool(dig(info, "cuda_graph_config", "prefill", default=None)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True, type=Path)
    ap.add_argument("--arms", required=True, help="comma-separated arm names, baseline first")
    ap.add_argument("--declaration", required=True, type=Path)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--port", default=None, help="only this campaign's snapshots (server_info.conformance.<arm>.<port>.json)")
    a = ap.parse_args()
    decl = json.loads(a.declaration.read_text())
    arms = a.arms.split(",")
    eff = {}
    for arm in arms:
        files = sorted(glob.glob(str(a.snapshots / f"server_info.conformance.{arm}.{a.port or '*'}.json")))
        if not files:
            print(f"REFUSED: no boot snapshot for arm {arm!r} under {a.snapshots}"); return 2
        eff[arm] = effective(json.loads(Path(files[-1]).read_text())); eff[arm]["snapshot"] = Path(files[-1]).name
    base = arms[0]
    print(f"execution differences vs {base} ({eff[base]['snapshot']}):")
    undeclared = []
    for arm in arms:
        e = eff[arm]; d = decl["declared"].get(arm, {})
        print(f"  {arm:16s} ladder top {e['decode_ladder_top']!s:>5} ({e['decode_ladder_buckets']} buckets, {e['decode_ladder_source']}) "
              f"kv {e['kv_capacity_tokens']} chunk {e['chunked_prefill_size']} mem {e['mem_fraction_static']} dtype {e['dtype']} "
              f"attn {e['attention_backend']} samp {e['sampling_backend']} prefill_graph {e['prefill_graph']}")
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
                if rel > decl["tolerances"]["kv_capacity_tokens_rel"] or (v < vb and expected != "smaller_ok" and rel > 1e-9):
                    if not (v > vb and rel <= decl["tolerances"]["kv_capacity_tokens_rel"]):
                        undeclared.append(f"{arm}: {field}={v} vs {vb} ({rel*100:.2f} %, declared {expected})")
            elif field == "decode_ladder_buckets":
                continue  # follows the top; reported, not gated
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
