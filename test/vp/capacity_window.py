#!/usr/bin/env python3
"""Throughput under saturation at 1.25xQ*: output tokens (TPS) and completed requests (RPS) per second inside the arrival window.

Owner 2026-09-23. The matched-work cells fix each request's work, so over a whole cell tokens per second follows the
arrival schedule and the drain (D-756). Inside the shared arrival window (first arrival -> last arrival) at 1.25xQ*,
both servers fall behind (fewer requests complete than arrive), so the rates there measure how much work each server
completes per second under overload. Both arms share the window; the drain is excluded.

  capacity_window.py extract <raw-root> <compact.json>   # per cell: token-emission counts in 50 ms bins,
                                                          # request start and end offsets (read from cells/<cell>.jsonl)
  capacity_window.py emit <compact.json> <generated-dir> # capacity_rows.tex, capacity_macros.tex, capacity.json

A cell whose rebuilt token times do not cover every output token is refused; cells/rejected/* is never read. A row is
refused unless every rep of both arms completes fewer requests than arrive inside the window.
"""
import bisect, glob, json, math, os, statistics as st, sys

BIN = 0.05
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
# key, macro stem, header, raw root, upstream arm, vSkipper arm, 1.25xQ* rate tag
ROWS = [
    ("llama_a100_gsm8k", "Gsm", "headline_gsm8k", "upstream_g1024", "vskipper", "r16p25"),
    ("llama_a100_bbh", "Bbh", "headline_bbh_cot", "upstream_g1024", "vskipper", "r42p5"),
    ("llama_a100_coqa", "Coqa", "headline_coqa", "upstream_g1024", "vskipper", "r31p25"),
    ("qwen3_4b_a100_gsm8k", "QwenFour", "qwen4b_gsm8k_q4b", "upstream_g1024", "vskipper_qwen3_4b", "r21p25"),
    ("qwen3_8b_a100_gsm8k", "QwenEight", "qwen8b_c1e4s15000_gsm8k_q8b", "upstream_g1024", "vskipper_qwen3_8b_c1e4s15000", "r17p5"),
    ("llama_h100_gsm8k", "Hundred", "h100v18_gsm8k", "upstream_g1024", "vskipper", "r17p5"),
    ("llama_a6000_gsm8k", "Asix", "a6000v18g256_gsm8k", "upstream_g256", "vskipper", "r7p5"),
]


def _cell(raw, root, rep, arm, rate):
    fs = [f for f in glob.glob(f"{raw}/{root}/{rep}/*/{arm}/cells/*_{rate}_qps*.jsonl")
          if not f.endswith((".load.jsonl", ".arrival.requests.jsonl"))]
    if len(fs) != 1:
        raise SystemExit(f"REFUSED {root}/{rep}/{arm}: {len(fs)} candidate cells")
    res = json.loads(open(fs[0]).read().splitlines()[-1])
    bins, want, got = {}, 0, 0
    for s, ttft, itls, ok, n in zip(res["request_start_offsets_s"], res["ttfts"], res["itls"],
                                    res["successes"], res["output_lens"]):
        if not ok:
            raise SystemExit(f"REFUSED {fs[0]}: failed request")
        t = s + ttft
        times = [t]
        for g in itls:
            t += g
            times.append(t)
        want += n
        got += len(times)
        for x in times:
            k = int(math.floor(x / BIN))
            bins[k] = bins.get(k, 0) + 1
    if got != want:
        raise SystemExit(f"REFUSED {fs[0]}: token times cover {got} of {want}")
    return {"cell": os.path.relpath(fs[0], raw), "starts": res["request_start_offsets_s"],
            "ends": res["request_end_offsets_s"], "token_bins": sorted(bins.items())}


def extract(raw, out):
    data = {"bin_s": BIN, "rows": {}}
    for key, _, root, up_arm, vs_arm, rate in ROWS:
        reps = sorted(os.path.basename(p) for p in glob.glob(f"{raw}/{root}/rep*"))
        data["rows"][key] = [{"rep": r, "upstream": _cell(raw, root, r, up_arm, rate),
                              "vskipper": _cell(raw, root, r, vs_arm, rate)} for r in reps]
    json.dump(data, open(out, "w"), separators=(",", ":"))


def _rates(c, a, b, binw):
    ks = [k for k, _ in c["token_bins"]]
    lo, hi = bisect.bisect_left(ks, int(math.floor(a / binw))), bisect.bisect_right(ks, int(math.floor(b / binw)))
    toks = sum(n for _, n in c["token_bins"][lo:hi])
    arrived = sum(1 for x in c["starts"] if a <= x <= b)
    done = sum(1 for x in c["ends"] if a <= x <= b)
    return toks / (b - a), done / (b - a), done / arrived


def _ci(xs):
    return st.mean(xs), T975[len(xs) - 1] * st.stdev(xs) / math.sqrt(len(xs))


def emit(compact, gen):
    data = json.load(open(compact))
    binw, summary, macros, cols, ratios = data["bin_s"], {}, [], {}, []
    for key, stem, *_ in ROWS:
        rps, tps = [], []
        for r in data["rows"][key]:
            up, vs = r["upstream"], r["vskipper"]
            a, b = min(up["starts"] + vs["starts"]), max(up["starts"])
            if abs(b - max(vs["starts"])) > 1.0:
                raise SystemExit(f"REFUSED {key} {r['rep']}: arrival windows differ")
            tu, ru, fu = _rates(up, a, b, binw)
            tv, rv, fv = _rates(vs, a, b, binw)
            if not (fu < 1 and fv < 1):
                raise SystemExit(f"REFUSED {key} {r['rep']}: an arm keeps up with arrivals ({fu:.2f}, {fv:.2f})")
            ratios += [fu, fv]
            rps.append(100 * (rv / ru - 1))
            tps.append(100 * (tv / tu - 1))
        (mr, hr), (mt, ht) = _ci(rps), _ci(tps)
        summary[key] = {"n": len(rps), "completed_rps_pct": [mr, hr], "output_tps_pct": [mt, ht]}
        cols[key] = (f"${mr:+.1f} \\pm {hr:.1f}$", f"${mt:+.1f} \\pm {ht:.1f}$")
        macros += [f"\\newcommand{{\\vpCap{stem}Rps}}{{{mr:+.1f}}}", f"\\newcommand{{\\vpCap{stem}RpsCi}}{{{hr:.1f}}}",
                   f"\\newcommand{{\\vpCap{stem}RpsAbs}}{{{abs(mr):.1f}}}",
                   f"\\newcommand{{\\vpCap{stem}Tps}}{{{mt:+.1f}}}", f"\\newcommand{{\\vpCap{stem}TpsCi}}{{{ht:.1f}}}"]
    macros += [f"\\newcommand{{\\vpCapDoneArrivedMin}}{{{min(ratios):.2f}}}",
               f"\\newcommand{{\\vpCapDoneArrivedMax}}{{{max(ratios):.2f}}}"]
    order = [k for k, *_ in ROWS]
    rows = ["TPS & " + " & ".join(cols[k][1] for k in order) + " \\\\",
            "RPS & " + " & ".join(cols[k][0] for k in order) + " \\\\"]
    open(os.path.join(gen, "capacity_rows.tex"), "w").write("\n".join(rows) + "\n")
    open(os.path.join(gen, "capacity_macros.tex"), "w").write("\n".join(macros) + "\n")
    json.dump(summary, open(os.path.join(gen, "capacity.json"), "w"), indent=1)


if __name__ == "__main__":
    {"extract": extract, "emit": emit}[sys.argv[1]](*sys.argv[2:])
