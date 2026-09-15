#!/usr/bin/env python3
"""Pin a natural-lane harvest into a bank: {"lengths": {request_id: output_len}} plus a summary.

A bank is what the equal-work replay serves (pin_output_lengths_from_bank.py); this writes one from a
finished harvest cell so a recipient can replay the paper's lengths without re-harvesting, or compare
a fresh harvest against them. Reads only the per-request arrays the runner wrote; never a server.

  python3 bank_from_harvest.py <harvest-root> ... --out <dir>     # one <dir>/<root-name>/{bank.json,summary.json} each
"""
import argparse, glob, json, os, statistics as st, sys

def cell_file(root):
    c = [f for f in glob.glob(os.path.join(root, "cell", "*_rep*.jsonl")) if "arrival" not in f and ".load." not in f]
    return sorted(c)[-1] if c else None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("roots", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); n_ok = 0
    for root in a.roots:
        f = cell_file(root)
        if not f: print(f"  skip {os.path.basename(root)}: no cell jsonl"); continue
        d = json.load(open(f))
        rid, out = d["request_ids"], d["output_lens"]
        if len(rid) != len(out) or len(set(rid)) != len(rid): sys.exit(f"FATAL {f}: ids/lengths mismatch or duplicate ids")
        fin = d.get("finish_reasons") or []
        dst = os.path.join(a.out, os.path.basename(root)); os.makedirs(dst, exist_ok=True)
        json.dump({"lengths": dict(zip(rid, out))}, open(os.path.join(dst, "bank.json"), "w"))
        summ = {"source_cell": os.path.relpath(f, root), "tag": d.get("tag"), "n": len(rid),
                "output_len_mean": round(st.mean(out), 2), "output_len_max": max(out),
                "requested_output_len_max": max(d.get("requested_output_lens") or [0]),
                "finish_reasons": {r: fin.count(r) for r in sorted(set(fin))} if fin else None,
                "duration_s": d.get("duration"), "submitted": d.get("submitted")}
        json.dump(summ, open(os.path.join(dst, "summary.json"), "w"), indent=1); n_ok += 1
    print(f"wrote {n_ok} banks under {a.out}")

if __name__ == "__main__": main()
