#!/usr/bin/env python3
"""The paired headline table: within-rep deltas, t-based 95 % CI, straddle rule.

Why this file exists. The plan requires paired analysis *bound into the run flow, not left as
a tool someone remembers to run* (D-624 #5) — and it was not. `run_paired_campaign.py` writes
54 cells and computes no delta; the phrase "the paired unit is a within-rep delta" appears only
in its docstring. The one tool that implements the right METHOD,
`vPipe-doc/codex/tools/v13_results_table_ci.py`, was written for the v1.3 artifact layout and
cannot read these artifacts: it globs `ovn-<ds>-r<rate>-rep<N>/results/`, looks for
`workgate_*.log` where the driver writes `workgate_*.json`, and — worst — hardcodes
`ROWS = [("BBH", …, 35), ("GSM8K", …, 11), ("CoQA", …, 23)]`, two of which are the VOID knees.
The live knees are BBH 25 / gsm8k 11 / coqa 27. A tool that half-worked would have looked for
rates derived from a system that no longer exists.

Method, unchanged from that tool because the method was never the problem:

  * the paired unit is ONE REP's delta — a rep runs both arms on the same pinned equal-work
    suite at the same rate, so the comparison is already paired within the rep;
  * across reps, a t-based 95 % CI on those deltas;
  * a CI containing zero is a STRADDLE, reported as parity, never as a win.

Gating is inherited and non-negotiable: a rep counts only with a GR-1a work-identity PASS.
**A rep that is MISSING and a rep that FAILED must never look alike** (D-593), so an ungated
rep is refused loudly rather than quietly dropped into the mean.

Usage:  paired_analysis.py <out-dir> [--reps 1,2,3] [--json OUT]
Exit 1 if any (dataset, rate) has no gated rep at all.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

# D-605: the headline is TTFT / TPOT / E2E at p50, mean and p99. Output TPS is ONE line and is
# reported, never headlined — in the equal-work lane it is algebraically the drain time, because
# identical tokens make TPS = tokens/duration with tokens fixed.
LATENCY_METRICS = [
    ("TTFT p50", "median_ttft_ms"), ("TTFT mean", "mean_ttft_ms"), ("TTFT p99", "p99_ttft_ms"),
    ("TPOT p50", "median_tpot_ms"), ("TPOT mean", "mean_tpot_ms"), ("TPOT p99", "p99_tpot_ms"),
    ("E2E p50", "median_e2e_latency_ms"), ("E2E mean", "mean_e2e_latency_ms"),
    ("E2E p99", "p99_e2e_latency_ms"),
]
THROUGHPUT_METRICS = [("output TPS", "output_throughput")]


def last_record(path: Path) -> dict:
    last = None
    with path.open() as handle:
        for line in handle:
            if line.strip():
                last = line
    if last is None:
        sys.exit(f"FATAL: {path} is empty")
    return json.loads(last)


def cell_for(arm_root: Path, suite: str) -> Path | None:
    """The one results .jsonl for a suite, excluding the arrival/load sidecars."""
    matches = [p for p in (arm_root / "cells").rglob(f"{suite}_qps*_rep*.jsonl")
               if "arrival" not in p.name and "load" not in p.name]
    return matches[0] if len(matches) == 1 else None


def gate_verdict(dataset_root: Path, suite: str) -> tuple[bool, str]:
    """GR-1a's verdict for this (dataset, suite, rep). Absent is NOT pass."""
    path = dataset_root / f"workgate_{suite}.json"
    if not path.is_file():
        return False, "no workgate verdict file — GR-1a did not run on this rep"
    verdict = json.loads(path.read_text())
    if verdict.get("pass") is not True:
        return False, f"GR-1a FAILED ({verdict.get('mismatch_count', '?')} mismatches)"
    return True, "gated"


def ci95(values: list[float]) -> tuple[float, float]:
    """Mean and t-based 95 % half-width. One rep has a mean and no interval."""
    mean = st.mean(values)
    if len(values) < 2:
        return mean, float("nan")
    # two-sided t at 95 % for small n; beyond 10 the normal approximation is close enough
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
         9: 2.306, 10: 2.262}.get(len(values), 1.96)
    return mean, t * st.stdev(values) / math.sqrt(len(values))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--baseline", default="upstream")
    ap.add_argument("--treatment", default="integrated_it4")
    ap.add_argument("--reps", default="", help="comma list; default = every rep<N> present")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    reps = ([int(r) for r in args.reps.split(",")] if args.reps
            else sorted(int(p.name[3:]) for p in args.out_dir.glob("rep*") if p.is_dir()))
    if not reps:
        sys.exit(f"FATAL: no rep* directories under {args.out_dir}")

    # (dataset, suite) -> {metric: [per-rep delta %]}
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}
    refused: list[str] = []
    counted: dict[tuple[str, str], list[int]] = {}

    for rep in reps:
        rep_root = args.out_dir / f"rep{rep}"
        for dataset_root in sorted(p for p in rep_root.glob("*") if p.is_dir()):
            dataset = dataset_root.name
            # A MISSING ARM IS REPORTED, NOT SKIPPED. This tool runs at the END of a
            # campaign, where an absent arm directory means that dataset never completed --
            # and a silent skip would make "it never ran" indistinguishable from "it was not
            # in the spec", which is D-593's rule (a missing rep and a failed rep must not
            # look alike) broken one level up. Found by running this against the live
            # headline's partial output instead of only against fixtures.
            absent = [a for a in (args.baseline, args.treatment)
                      if not (dataset_root / a).is_dir()]
            if absent:
                refused.append(
                    f"rep{rep} {dataset}: no directory for {', '.join(absent)} "
                    "-- that arm never ran for this dataset"
                )
            else:
                suites = sorted({p.name.split("_qps")[0] for p in
                                 (dataset_root / args.baseline / "cells").rglob("*_qps*_rep*.jsonl")
                                 if "arrival" not in p.name and "load" not in p.name})
                for suite in suites:
                    ok, why = gate_verdict(dataset_root, suite)
                    if not ok:
                        refused.append(f"rep{rep} {dataset} {suite}: {why}")
                        continue
                    base = cell_for(dataset_root / args.baseline, suite)
                    treat = cell_for(dataset_root / args.treatment, suite)
                    if base is None or treat is None:
                        refused.append(f"rep{rep} {dataset} {suite}: missing a unique cell artifact")
                        continue
                    b, t = last_record(base), last_record(treat)
                    key = (dataset, suite)
                    bucket = cells.setdefault(key, {})
                    for _, field in LATENCY_METRICS + THROUGHPUT_METRICS:
                        if field in b and field in t and b[field]:
                            bucket.setdefault(field, []).append((t[field] - b[field]) / b[field] * 100.0)
                    counted.setdefault(key, []).append(rep)

    if refused:
        print("REFUSED reps (a missing rep and a failed rep must never look alike, D-593):")
        for line in refused:
            print(f"  ! {line}")
        print()

    if not cells:
        print("FATAL: no gated (dataset, rate) pair produced a delta.")
        return 1

    print(f"paired headline — {args.treatment} vs {args.baseline}")
    print("  negative = treatment FASTER for latency; positive = treatment HIGHER for TPS")
    print("  an interval containing zero is PARITY, reported as parity (straddle rule)\n")

    report: dict = {"baseline": args.baseline, "treatment": args.treatment, "rows": []}
    for (dataset, suite), bucket in sorted(cells.items()):
        n = len(counted[(dataset, suite)])
        print(f"  {dataset}  {suite}   (n = {n} gated rep{'s' if n != 1 else ''}: "
              f"{','.join(map(str, counted[(dataset, suite)]))})")
        for label, field in LATENCY_METRICS + THROUGHPUT_METRICS:
            if field not in bucket:
                continue
            mean, half = ci95(bucket[field])
            if math.isnan(half):
                verdict = "n=1, no interval"
                span = ""
            else:
                lo, hi = mean - half, mean + half
                verdict = "PARITY (straddles 0)" if lo <= 0 <= hi else (
                    "treatment better" if (hi < 0 and field != "output_throughput")
                    or (lo > 0 and field == "output_throughput") else "treatment worse")
                span = f"  [{lo:+.2f}, {hi:+.2f}]"
            print(f"      {label:<11} {mean:+7.2f} %{span:<22} {verdict}")
            report["rows"].append({"dataset": dataset, "suite": suite, "metric": label,
                                  "field": field, "n": n, "mean_pct": mean,
                                  "ci95_half": None if math.isnan(half) else half,
                                  "verdict": verdict})
        print()

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
