#!/usr/bin/env python3
"""Routed decode share of the served hybrid in each quality-lane cell, from the attestation (Appendix E.4).

The share is row-weighted: decode rows the routed body executed over all decode rows,
fd_tokens_skip_body / (fd_tokens_skip_body + fd_tokens_prod_allrun_band), read from the cell's own
server_info.after.json minus .before.json. Rows per decode pass = all decode rows / decode passes.
A cell whose attestation is missing is refused, never skipped.

  loaded_shares.py --raw <campaign-raw dir> --macros out.tex [--json out.json]
"""
import argparse, glob, json, os, sys

RUNGS = {"gsm8k": [("r8p25", "Low"), ("r10p45", "Mid"), ("r12p65", "Probe"), ("r13p75", "High")],
         "bbh_cot": [("r18p75", "Low"), ("r23p75", "Mid"), ("r31p25", "High")],
         "coqa": [("r20p25", "Low"), ("r25p65", "Mid"), ("r33p75", "High")]}
MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}
ARMS = {"integrated_it4": "Hyb", "integrated_alwaysskip": "Alw"}

def counters(path):
    vp = json.load(open(path))["internal_states"][0]["vp_runtime"]
    c = vp["fd_c3"]["counters"]; bc = vp["batch_composition"]
    return {"skip": c["fd_tokens_skip_body"], "dense": c["fd_tokens_prod_allrun_band"], "passes": bc["decode_passes"]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--macros", required=True); ap.add_argument("--json")
    a = ap.parse_args(); out, macros = {}, []
    for ds, rungs in RUNGS.items():
        for rate, rw in rungs:
            for arm, aw in ARMS.items():
                root = os.path.join(a.raw, f"harvest-{arm}-{ds}-{rate}")
                after = glob.glob(os.path.join(root, "cell", f"{ds}.qual_*_rep1.server_info.after.json"))
                if not after:
                    if rw == "Probe": continue          # the 1.15x probe exists on GSM8K only
                    sys.exit(f"FATAL: no attestation for {arm} {ds} {rate} under {root}")
                x = counters(after[0]); y = counters(after[0].replace(".after.", ".before."))
                skip, dense, passes = x["skip"] - y["skip"], x["dense"] - y["dense"], x["passes"] - y["passes"]
                if skip + dense <= 0 or passes <= 0:
                    sys.exit(f"FATAL {root}: zero decode rows or passes in the attestation delta")
                share = 100.0 * skip / (skip + dense); rows = (skip + dense) / passes
                out[f"{arm}/{ds}/{rate}"] = {"routed_share_pct": round(share, 1), "rows_per_decode_pass": round(rows), "decode_passes": passes}
                macros.append(f"\\newcommand{{\\vpLq{MW[ds]}{rw}{aw}RoutedShare}}{{{share:.1f}}}")
                macros.append(f"\\newcommand{{\\vpLq{MW[ds]}{rw}{aw}RowsPerPass}}{{{rows:.0f}}}")
                print(f"  {arm:22s} {ds:8s} {rate:7s} routed share {share:5.1f}%  rows/pass {rows:5.0f}")
    open(a.macros, "w").write("\n".join(macros) + "\n")
    if a.json: json.dump(out, open(a.json, "w"), indent=1)
    print(f"wrote {len(macros)} macros -> {a.macros}")

if __name__ == "__main__": main()
