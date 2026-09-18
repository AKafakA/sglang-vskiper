#!/usr/bin/env python3
"""suite_composition.py -- what the serving suites are made of, read from the natural-lane cells' arrival records.

usage: suite_composition.py <campaign-raw root> --macros out.tex

For each workload's upstream natural-lane harvest at the knee (harvest-upstream-<ds>-<label>/cell/<suite>_qps*.arrival.requests.jsonl,
the serving suite, not the .qual held-out suite), counts the requests by the split their request_id names (gsm8k:train:..., gsm8k:test:..., coqa:...)
and records the declared generation budget, which on this lane is the remaining context window (8,192 minus the
prompt). Macros: \\vpSuite<Ds>Requests, \\vpSuite<Ds>Train, \\vpSuite<Ds>HeldOut (the non-train remainder), and
\\vpSuiteRemainingCtxMin / Max over all three suites. A missing arrival file is fatal.
"""
import argparse, glob, json, sys

KNEE = {"gsm8k": "r10p45", "bbh_cot": "r23p75", "coqa": "r25p65"}
KNEE_V16 = {"gsm8k": "r12p35", "bbh_cot": "r32p3", "coqa": "r23p75"}   # [v1.6, D-824] g1024 knees 13/34/25 x 0.95; harvests under harvest-upstream_g1024-*
MW = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--macros", required=True); ap.add_argument("--layout", default="v15", choices=["v15", "v16"])
    a = ap.parse_args()
    lines, lo, hi = [], None, None
    knees, arm = (KNEE_V16, "upstream_g1024") if a.layout == "v16" else (KNEE, "upstream")
    for ds, lbl in knees.items():
        files = sorted(f for f in glob.glob(f"{a.root}/harvest-{arm}-{ds}-{lbl}/cell/*.arrival.requests.jsonl")
                       if ".qual" not in f.rsplit("/", 1)[1])   # the serving suite, not the held-out quality suite
        if len(files) != 1:
            sys.exit(f"FATAL: expected one serving-suite arrival record for harvest-{arm}-{ds}-{lbl}, found {len(files)}")
        rows = [json.loads(l) for l in open(files[0]) if l.strip()]
        train = sum(1 for r in rows if f"{ds.split('_')[0]}:train:" in r["request_id"])
        budgets = [r["output_len"] for r in rows]
        if any(r["output_len"] + r["prompt_len"] != 8192 for r in rows):
            sys.exit(f"FATAL: {files[0]}: a request's budget is not the remaining 8,192-token window")
        lo = min(budgets) if lo is None else min(lo, min(budgets)); hi = max(budgets) if hi is None else max(hi, max(budgets))
        lines += [f"\\newcommand{{\\vpSuite{MW[ds]}Requests}}{{{len(rows):,}}}",
                  f"\\newcommand{{\\vpSuite{MW[ds]}Train}}{{{train:,}}}",
                  f"\\newcommand{{\\vpSuite{MW[ds]}HeldOut}}{{{len(rows) - train:,}}}"]
        print(f"  {ds:8s} {len(rows)} requests, {train} train-split, budget {min(budgets)}--{max(budgets)}")
    lines += [f"\\newcommand{{\\vpSuiteRemainingCtxMin}}{{{lo:,}}}", f"\\newcommand{{\\vpSuiteRemainingCtxMax}}{{{hi:,}}}"]
    with open(a.macros, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} macros -> {a.macros}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
