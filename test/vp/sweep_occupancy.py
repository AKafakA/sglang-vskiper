#!/usr/bin/env python3
"""sweep_occupancy.py -- per-arm occupancy + attestation counters for a paired sweep root (runs on the box, no venv needed).

usage: sweep_occupancy.py <campaign root> <tag> --out <json>

For each <root>/rep1/<dataset>/<arm>/cells/<suite>_qps*_rep1.jsonl with its .load.summary.json (sample_sglang_load.py) and
.server_info.after.json: running-request percentiles, mean tokens per request (input + output), the resident-K/V estimate
(running p90 x tokens per request), mean E2E, duration, and from the attestation the decode pass counters (prod_allrun,
fd_tokens_skip_body), the attested decode skip ratio and the served band for the device. Output {tag: {arm: {suite: rec}}},
the format sweep_prediction.py reads. Written 2026-09-14 from the inline extraction used for sweep v1/v2 (identical fields).
"""
import argparse, glob, json, re


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("tag"); ap.add_argument("--out", required=True); ap.add_argument("--device", default="NVIDIA_A100")
    a = ap.parse_args()
    out = {}
    for arm in sorted(glob.glob(a.root + "/rep1/*/*/")):
        name = arm.rstrip("/").split("/")[-1]
        for js in glob.glob(arm + "cells/*_rep1.jsonl"):
            suite = js.split("/")[-1].split("_qps")[0]; base = js[: -len(".jsonl")]
            try:
                with open(js) as f:
                    d = json.loads(f.readline())
                L = json.load(open(base + ".load.summary.json"))
            except Exception:
                continue
            tok = (sum(d["input_lens"]) + sum(d["output_lens"])) / len(d["input_lens"])
            rec = {"running_p50": L["p50_running_requests"], "running_p90": L["p90_running_requests"], "running_max": L["max_running_requests"],
                   "tokens_per_request_mean": tok, "resident_kv_p90_est": L["p90_running_requests"] * tok, "resident_kv_max_est": L["max_running_requests"] * tok,
                   "mean_e2e_latency_ms": d["mean_e2e_latency_ms"], "duration_s": d["duration"]}
            try:
                s = json.dumps(json.load(open(base + ".server_info.after.json")))
                pa = re.search(r"\"prod_allrun\": (\d+)", s); sb = re.search(r"\"fd_tokens_skip_body\": (\d+)", s)
                m = re.search(r"\"decode\": \{\"layer_rows\": (\d+), \"(?:run|project)_rows\": (\d+), \"(?:project|run)_rows\": (\d+), \"skip_ratio\": ([0-9.]+)", s)
                band = re.search(r"\"decode_kv_band_by_device\": \{\"%s\": \[(\d+), (\d+)\]" % a.device, s)
                rec.update({"decode_passes_allrun": int(pa.group(1)) if pa else None, "decode_passes_skipbody": int(sb.group(1)) if sb else None,
                            "decode_skip_ratio_attested": float(m.group(4)) if m else None, "band": [int(band.group(1)), int(band.group(2))] if band else None})
            except Exception:
                pass
            out.setdefault(a.tag, {}).setdefault(name, {})[suite] = rec
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.tag}: {len(out.get(a.tag, {}))} arms -> {a.out}")
    for name, v in sorted(out.get(a.tag, {}).items()):
        r = next(iter(v.values())); sb, ar = r.get("decode_passes_skipbody"), r.get("decode_passes_allrun")
        eng = (100 * sb / (sb + ar)) if (sb is not None and ar) else None
        print("  %-44s p90 ~%4.0fk band %-18s engaged %-5s E2E %6.0f ms" % (name, r["resident_kv_p90_est"] / 1e3, r.get("band"), ("%.0f%%" % eng) if eng is not None else "?", r["mean_e2e_latency_ms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
