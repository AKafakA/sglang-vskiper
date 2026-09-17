#!/usr/bin/env python3
"""graph_coverage.py -- how often each headline cell ran decode batches above a graph-ladder rung (Appendix J).

usage: graph_coverage.py <headline root> --macros out.tex [--json out.json] [--reps 1,2,3,4,5,6] [--rung 256] [--rung 1024]

Upstream at SGLang's default captures decode graphs up to 256 rows and runs larger batches without a graph; the fork
captures to 1,024. The per-cell load sampler (sample_sglang_load.py, one sample per second) records running_requests,
which is the decode batch's row count between prefill passes. For every <root>/rep<k>/<dataset>/<arm>/cells/*.load.jsonl
the share of ACTIVE samples (running or waiting requests > 0) whose running_requests exceeds each rung is computed, then
averaged over the repetitions. Macros: \vpCov<Ds><Rate><Arm>Above<Rung> in percent, e.g. \vpCovGsmMidUpAboveTwoFiftySix.
A cell whose load samples are missing is fatal, never skipped.
"""
import argparse, glob, json, os, sys

DS = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}
ARM = {"upstream": "Up", "upstream_g1024": "Up", "vskipper": "Vs", "integrated_it4": "Vs"}
RUNG = {256: "TwoFiftySix", 1024: "OneKtwentyFour"}


def rate_word(rates: list[float], q: float) -> str:
    return {0: "Low", 1: "Mid", 2: "High"}[sorted(rates).index(q)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--macros", required=True); ap.add_argument("--json")
    ap.add_argument("--reps", default="1,2,3,4,5,6"); ap.add_argument("--rung", type=int, action="append")
    ap.add_argument("--baseline", default="upstream"); ap.add_argument("--treatment", default="integrated_it4")   # [v1.6] the same-ladder control is the baseline (D-824)
    a = ap.parse_args()
    rungs = a.rung or [256, 1024]
    reps = [int(x) for x in a.reps.split(",")]
    shares: dict[tuple[str, str, float, int], list[float]] = {}
    for rep in reps:
        for ds in DS:
            for arm in (a.baseline, a.treatment):
                files = sorted(glob.glob(f"{a.root}/rep{rep}/{ds}/{arm}/cells/*.load.jsonl"))   # every rep directory names its cells _rep1
                if not files:
                    sys.exit(f"FATAL: no load samples for rep{rep}/{ds}/{arm} under {a.root}")
                for f in files:
                    q = float(os.path.basename(f).split("_qps")[1].split("_rep")[0].replace("p", "."))
                    rows = [json.loads(l) for l in open(f) if l.strip()]
                    active = [r["running_requests"] for r in rows if r["running_requests"] > 0 or r["waiting_requests"] > 0]
                    if not active:
                        sys.exit(f"FATAL: no active load samples in {f}")
                    for rung in rungs:
                        shares.setdefault((ds, arm, q, rung), []).append(100.0 * sum(1 for x in active if x > rung) / len(active))
    rates = {ds: sorted({q for (d, _, q, _) in shares if d == ds}) for ds in DS}
    out, lines = {}, []
    for (ds, arm, q, rung), xs in sorted(shares.items()):
        if len(xs) != len(reps):
            sys.exit(f"FATAL: {ds}/{arm} at {q} req/s has {len(xs)} repetitions, expected {len(reps)}")
        mean = sum(xs) / len(xs)
        name = f"\\vpCov{DS[ds]}{rate_word(rates[ds], q)}{ARM[arm]}Above{RUNG[rung]}"
        lines.append(f"\\newcommand{{{name}}}{{{mean:.0f}}}")
        out[f"{ds}/{arm}/{q}/{rung}"] = {"mean_pct": mean, "per_rep_pct": xs}
    with open(a.macros, "w") as f:
        f.write("\n".join(lines) + "\n")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    for k, v in out.items():
        print(f"  {k:32s} {v['mean_pct']:5.1f}%  reps {' '.join('%.0f' % x for x in v['per_rep_pct'])}")
    print(f"{len(lines)} macros -> {a.macros}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
