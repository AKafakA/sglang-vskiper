#!/usr/bin/env python3
"""Absolute baseline numbers per headline row (reviewer W7), from paired_report.json.

Emits LaTeX rows: dataset, rate label, upstream TTFT mean (ms), TPOT mean (ms), E2E mean (s),
makespan (s), output tok/s — the numbers a systems reader needs beside the relative table.
Usage: absolutes_table.py paired_report.json --knee gsm8k=11 --knee bbh_cot=25 --knee coqa=27 --out rows.tex
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

NAMES = {"gsm8k": "GSM8K", "bbh_cot": "BBH", "coqa": "CoQA"}
COLS = [("TTFT mean", 1.0, "{:.0f}"), ("TPOT mean", 1.0, "{:.1f}"), ("E2E mean", 1e-3, "{:.1f}"),
        ("makespan", 1.0, "{:.0f}"), ("output TPS", 1.0, "{:.0f}")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path)
    ap.add_argument("--knee", action="append", default=[], metavar="DS=QSTAR")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    knees = {k: float(v) for k, v in (x.split("=", 1) for x in a.knee)}
    rows = json.loads(a.report.read_text())["rows"]
    by = {}
    for r in rows:
        by.setdefault((r["dataset"], r["suite"]), {})[r["metric"]] = r
    out = []
    for (ds, suite), m in sorted(by.items(), key=lambda kv: (list(NAMES).index(kv[0][0]) if kv[0][0] in NAMES else 9, kv[0][1])):
        rate = suite.split("_r")[-1].replace("p", ".")
        frac = f"{float(rate)/knees[ds]:.2f}" if ds in knees else "--"
        cells = []
        for metric, scale, fmt in COLS:
            r = m.get(metric)
            if r is None or r.get("baseline_abs") is None:
                sys.exit(f"{ds} {suite}: no absolute for {metric} (re-run paired_analysis)")
            cells.append(fmt.format(r["baseline_abs"] * scale))
        out.append(f"{NAMES.get(ds, ds)} & {frac} & " + " & ".join(cells) + r" \\")
    a.out.write_text("\n".join(out) + "\n")
    print(f"  {len(out)} absolute rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
