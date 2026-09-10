#!/usr/bin/env python3
"""Summarise the 2x2 quality gate. The metric is DECLARED per dataset, never guessed.

Why this is a committed tool and not three lines of inline shell. The chain's ad-hoc
summariser tried a list of metric names, found none of them, and fell through to whichever
key came first in the results dict -- `strict-match`. On gsm8k that INVERTS the story:

    strict-match      (B - A) = +5.31 pp   the checkpoint appears to GAIN
    flexible-extract  (B - A) = -5.53 pp   the checkpoint costs

D-630 recorded this exact reversal, which is why flexible-extract is a RULING
(2026-08-17: "flexible extraction ALWAYS under chat template") and not a preference. A
summariser that silently picks a metric can hand over a headline with the wrong sign, so
this one refuses instead: the metric per dataset is a table below, and a missing metric is
an error, never a fallback.

The gate (owner, 2026-09-10), per row, BOTH must hold:
  1. no quality collapse for D against stock
  2. (D - C) comparable to (B - A) -- our gap is the checkpoint's, not ours

Usage:  summarize_2x2.py <root-with-A-B-C-D-subdirs> --dataset gsm8k|coqa|bbh_cot
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

# The metric each dataset is scored on, and its stderr key. Declared, not discovered.
METRICS = {
    "gsm8k": ("exact_match,flexible-extract", "exact_match_stderr,flexible-extract"),
    "bbh_cot": ("exact_match,flexible-extract", "exact_match_stderr,flexible-extract"),
    "coqa": ("f1,none", "f1_stderr,none"),
}

ARMS = {
    "A": "raw base model, PyTorch",
    "B": "released checkpoint, its OWN code",
    "C": "stock base model, OUR serving stack",
    "D": "vSkipper serving the checkpoint",
}


def _load(root: Path, arm: str, dataset: str) -> tuple[float, float]:
    hits = sorted(glob.glob(f"{root}/{arm}/**/results_*.json", recursive=True))
    if not hits:
        sys.exit(f"FATAL: no results_*.json for arm {arm} under {root}")
    results = json.load(open(hits[-1]))["results"]
    keys = [k for k in results if dataset.split("_")[0] in k]
    # A grouped task (bbh_cot_fewshot has 27 subtasks) must be read from its aggregate,
    # never from whichever subtask happens to sort first.
    task = next((k for k in keys if k in (dataset, f"{dataset}_fewshot", "bbh_cot_fewshot")), None)
    if task is None:
        if len(keys) != 1:
            sys.exit(
                f"FATAL: arm {arm} has {len(keys)} {dataset} entries and no aggregate: "
                f"{sorted(keys)[:6]}. Refusing to pick one."
            )
        task = keys[0]
    metric, stderr_key = METRICS[dataset]
    row = results[task]
    if metric not in row:
        sys.exit(
            f"FATAL: arm {arm} task {task} has no '{metric}'. Present: "
            f"{sorted(k for k in row if isinstance(row[k], float))}.\n"
            "       The metric is a ruling (2026-08-17), not a preference -- on gsm8k the "
            "sign flips between extractors (D-630). Not falling back."
        )
    return float(row[metric]), float(row.get(stderr_key, 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--dataset", choices=sorted(METRICS), required=True)
    args = ap.parse_args()

    metric, _ = METRICS[args.dataset]
    vals = {arm: _load(args.root, arm, args.dataset) for arm in ARMS}

    print(f"=== 2x2 quality gate -- {args.dataset} -- metric {metric} ===")
    for arm, (v, se) in vals.items():
        print(f"  {arm}  {v:.4f} +/- {se:.4f}   {ARMS[arm]}")

    (a, sa), (b, sb) = vals["A"], vals["B"]
    (c, sc), (d, sd) = vals["C"], vals["D"]
    ba, dc = b - a, d - c
    se_ba = math.hypot(sa, sb)
    se_dc = math.hypot(sc, sd)
    dod = dc - ba
    se_dod = math.hypot(se_ba, se_dc)

    def line(label: str, value: float, se: float, note: str) -> None:
        verdict = "PARITY (straddles 0)" if abs(value) <= 1.96 * se else "DECIDED"
        print(f"  {label} = {100*value:+6.2f} pp  +/- {100*1.96*se:5.2f}   {verdict:22s} {note}")

    print()
    line("(B - A)", ba, se_ba, "the checkpoint's own cost")
    line("(D - C)", dc, se_dc, "ours")
    line("d-o-d  ", dod, se_dod, "faithful iff this straddles zero")

    print()
    collapse = dc < -0.10  # a >10 pp drop against stock is a collapse by any reading
    faithful = abs(dod) <= 1.96 * se_dod
    print(f"  gate 1  no collapse vs stock : {'PASS' if not collapse else 'FAIL'}"
          f"   (D - C = {100*dc:+.2f} pp)")
    print(f"  gate 2  (D-C) ~ (B-A)        : {'PASS' if faithful else 'FAIL'}"
          f"   (d-o-d = {100*dod:+.2f} pp, CI +/- {100*1.96*se_dod:.2f})")
    print()
    print("  NOTE gate 2 passing means our gap MATCHES the checkpoint's. It does not mean")
    print("  the gap is zero -- report (D - C) itself alongside it.")
    return 0 if (not collapse and faithful) else 1


if __name__ == "__main__":
    raise SystemExit(main())
