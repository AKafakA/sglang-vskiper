#!/usr/bin/env python3
"""sweep_occupancy.py -- per-arm occupancy + attestation counters for a paired sweep root (runs on the box, no venv needed).

usage: sweep_occupancy.py <campaign root> <tag> --out <json>

For each <root>/rep1/<dataset>/<arm>/cells/<suite>_qps*_rep1.jsonl with its .load.summary.json (sample_sglang_load.py) and
.server_info.after.json: running-request percentiles, mean tokens per request (input + output), the resident-K/V estimate
(running p90 x tokens per request), mean E2E, duration, and from the attestation the decode pass counters (prod_allrun,
fd_tokens_skip_body), the attested decode skip ratio and the served band for the device. Output {tag: {arm: {suite: rec}}},
the format sweep_prediction.py reads. Written 2026-09-14 from the inline extraction used for sweep v1/v2 (identical fields).
"""
import argparse, glob, json, re, sys


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
                info = json.load(open(base + ".server_info.after.json"))
                st = info["internal_states"][0]
                if "vp_runtime" not in st:
                    # The upstream anchor is a separate tree with no vpipe package: no attestation to read.
                    if name not in ("upstream", "upstream_g1024"):   # [v1.6] the same-ladder control is upstream too
                        sys.exit(f"FATAL: routed arm {name} served without vp_runtime attestation: {base}")
                    rec.update({"decode_passes_allrun": None, "decode_passes_skip": None, "decode_rows_skip_share": None,
                                "decode_skip_ratio_attested": None, "band": None})
                    out.setdefault(a.tag, {}).setdefault(name, {})[suite] = rec
                    continue
                vp = st["vp_runtime"]
                # "engaged" is a share of decode PASSES, from the regime switch's own pass counters --
                # the same quantity Table 13 prints. (An earlier version matched fd_tokens_skip_body, a
                # ROW counter, against the pass counter and printed tokens/(tokens+passes).) Counters are
                # the cell's own: after minus before (the warm-up's 20 passes), as loaded_shares.py reads them.
                vp0 = json.load(open(base + ".server_info.before.json"))["internal_states"][0]["vp_runtime"]
                dec = {k: vp["regime_switch"]["counters"]["decode"][k] - vp0["regime_switch"]["counters"]["decode"][k] for k in ("prod_allrun", "skip")}
                c3 = {k: vp["fd_c3"]["counters"][k] - vp0["fd_c3"]["counters"][k] for k in ("fd_tokens_skip_body", "fd_tokens_prod_allrun_band", "fd_tokens_dense_overflow")}
                rows_total = c3["fd_tokens_skip_body"] + c3["fd_tokens_prod_allrun_band"] + c3["fd_tokens_dense_overflow"]
                bands = vp["model"]["design"].get("decode_kv_band_by_device") or {}
                band = bands.get(a.device)
                rec.update({"decode_passes_allrun": int(dec["prod_allrun"]), "decode_passes_skip": int(dec["skip"]),
                            "decode_rows_skip_share": (c3["fd_tokens_skip_body"] / rows_total) if rows_total else None,
                            "decode_skip_ratio_attested": vp["model"]["flexidepth"]["full_graph_routes"]["by_phase"]["decode"].get("skip_ratio"),
                            "band": [int(band[0]), int(band[1])] if band else None})
            except (OSError, KeyError, IndexError, TypeError) as exc:
                sys.exit(f"FATAL: attestation counters unreadable for {base}: {exc!r}")
            out.setdefault(a.tag, {}).setdefault(name, {})[suite] = rec
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.tag}: {len(out.get(a.tag, {}))} arms -> {a.out}")
    for name, v in sorted(out.get(a.tag, {}).items()):
        r = next(iter(v.values())); sb, ar, rs = r.get("decode_passes_skip"), r.get("decode_passes_allrun"), r.get("decode_rows_skip_share")
        if sb is not None and ar is not None and sb + ar:
            eng = "%.0f%% passes, %.0f%% rows" % (100 * sb / (sb + ar), 100 * rs)
        elif rs is not None:
            eng = "no switch, %.0f%% rows" % (100 * rs)   # always-route: the switch is not consulted; every row is in the routed body
        else:
            eng = "?"
        print("  %-44s p90 ~%4.0fk band %-18s engaged %-24s E2E %6.0f ms" % (name, r["resident_kv_p90_est"] / 1e3, r.get("band"), eng, r["mean_e2e_latency_ms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
