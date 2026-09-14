#!/usr/bin/env python3
"""qwen_serving_macros.py HYBRID_REPORT ALWAYSROUTE_REPORT [ALWAYSROUTE_RATES_REPORT]

Prints LaTeX macros for the Qwen3-4B serving rows (App. J): \\vpQwenHyb<Metric><Low|Mid|High> for the rule-governed arm and
\\vpQwenAll<Metric><Low|Mid|High> for the always-route twin, from the paired reports' mean_pct (baseline upstream). Rates are
labelled by the suite's rate suffix against the knee (Q* = 14: r10p5 -> Low, r13p3 -> Mid, r17p5 -> High).
"""
import json, sys

WORD = {"r10p5": "Low", "r13p3": "Mid", "r17p5": "High"}
METRIC = {"E2E mean": "EtoE", "E2E p95": "EtoEpNN", "TTFT mean": "TTFT", "TPOT mean": "TPOT", "makespan": "Makespan", "output TPS": "TPS"}


def emit(prefix: str, path: str) -> None:
    for r in json.load(open(path))["rows"]:
        w = WORD.get(r["suite"].split("_")[-1]); m = METRIC.get(r["metric"])
        if w and m:
            print(f"\\newcommand{{\\vpQwen{prefix}{m}{w}}}{{{r['mean_pct']:+.1f}}}")


emit("Hyb", sys.argv[1])
for p in sys.argv[2:]:
    emit("All", p)
