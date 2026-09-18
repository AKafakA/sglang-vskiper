#!/usr/bin/env python3
"""ablation_table.py -- the mechanism-ablation rows for the paper (plan v5 item 1, D-761 add.).

usage: ablation_table.py --headline generated/paired_report.json --report stock=PATH --report vdec_fd=PATH ...
                         --dataset gsm8k --knee gsm8k=11 --rows out.tex [--macros out.tex] [--metrics mean_e2e_latency_ms,duration]

One row per (dataset, rate) cell; one column pair per arm (E2E mean delta %, makespan delta %), the headline arm first.
Every number is the paired report's `mean_pct` (baseline = upstream in the same session); a `ci95_half` is printed only
for the headline (n = 6), which is the noise floor the single-rep ablation cells are read against. An arm whose report
lacks a cell prints `--`. Rates are labelled by the knee multiple recovered from the suite name (r10p45 -> 0.95 x 11).
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

RATE_RE = re.compile(r"_r(\d+)p(\d+)$")


def load(path: Path) -> dict[tuple[str, str, str], dict]:
    rep = json.load(open(path))
    return {(r["dataset"], r["suite"], r["field"]): r for r in rep["rows"]}


def rate_of(suite: str) -> float:
    m = RATE_RE.search(suite)
    if not m:
        raise SystemExit(f"suite {suite!r} carries no _r<int>p<frac> rate")
    return float(f"{m.group(1)}.{m.group(2)}")


def multiple(rate: float, knee: float) -> str:
    for m in (0.75, 0.95, 1.25):
        if abs(rate - m * knee) < 0.06 * knee:
            return f"{m:.2f}"
    return f"{rate / knee:.2f}"


WORD = {"0.75": "Low", "0.95": "Mid", "1.25": "High"}          # macro names take letters only (as \vpEtoEMeanGsmMid does)
FIELD = {"mean_e2e_latency_ms": "EtoE", "duration": "Makespan", "mean_ttft_ms": "TTFT", "mean_tpot_ms": "TPOT", "output_throughput": "TPS"}


def signed(x: float) -> str:
    """+.1f without a signed zero: a -0.0 cell reads as a loss it is not."""
    return f"{x:+.1f}".replace("-0.0", "+0.0")


def fmt(row: dict | None, ci: bool) -> str:
    if row is None:
        return "--"
    s = signed(row['mean_pct'])
    return f"{s} $\\pm$ {row['ci95_half']:.1f}" if ci else s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline", type=Path, required=True)
    ap.add_argument("--report", action="append", default=[], help="arm=path, in column order")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--knee", action="append", default=[], help="dataset=Q*")
    ap.add_argument("--metrics", default="mean_e2e_latency_ms,duration")
    ap.add_argument("--macro-only-metrics", default="mean_tpot_ms,mean_ttft_ms",
                    help="fields emitted as macros but not as table columns (the prose cites an arm's TPOT/TTFT)")
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--macros", type=Path)
    ap.add_argument("--macro-prefix", default="vpAbl", help="macro name prefix (letters only), e.g. vpAblBbh for a second dataset")
    a = ap.parse_args()
    knees = {k: float(v) for k, v in (x.split("=") for x in a.knee)}
    if a.dataset not in knees:
        raise SystemExit(f"--knee {a.dataset}=Q* is required")
    metrics = a.metrics.split(",")
    head = load(a.headline)
    arms = [(name, load(Path(p))) for name, p in (x.split("=", 1) for x in a.report)]
    suites = sorted({s for (d, s, f) in head if d == a.dataset and f == metrics[0]}, key=rate_of)
    if not suites:
        raise SystemExit(f"no {a.dataset} cells in the headline report")
    lines, macros = [], []
    for suite in suites:
        rate = rate_of(suite)
        cells = [f"{multiple(rate, knees[a.dataset])}$\\times$"]
        for f in metrics:
            cells.append(fmt(head.get((a.dataset, suite, f)), ci=True))
        for name, rep in arms:
            for f in metrics:
                r = rep.get((a.dataset, suite, f))
                cells.append(fmt(r, ci=False))
                if r is not None and a.macros:
                    mult = multiple(rate, knees[a.dataset])
                    tag = re.sub(r"[^A-Za-z]", "", name.title()) + FIELD.get(f, re.sub(r"[^A-Za-z]", "", f.title())) + WORD.get(mult, "X")
                    macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}}}{{{signed(r['mean_pct'])}}}")
                    # the same cell as a difference from the headline arm, in points of the
                    # percent change: what the arm "gives up" (positive) or gains (negative)
                    h = head.get((a.dataset, suite, f))
                    if h is not None:
                        macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}VsHead}}"
                                      f"{{{r['mean_pct'] - h['mean_pct']:+.1f}}}")
                        macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}VsHeadAbs}}"
                                      f"{{{abs(r['mean_pct'] - h['mean_pct']):.1f}}}")
        for name, rep in arms:
            for f in [x for x in a.macro_only_metrics.split(",") if x]:
                r = rep.get((a.dataset, suite, f))
                if r is not None and a.macros:
                    mult = multiple(rate, knees[a.dataset])
                    tag = re.sub(r"[^A-Za-z]", "", name.title()) + FIELD.get(f, re.sub(r"[^A-Za-z]", "", f.title())) + WORD.get(mult, "X")
                    macros.append(f"\\newcommand{{\\{a.macro_prefix}{tag}}}{{{r['mean_pct']:+.1f}}}")
        lines.append(" & ".join(cells) + r" \\")
    a.rows.write_text("\n".join(lines) + "\n")
    if a.macros:
        a.macros.write_text("\n".join(macros) + "\n")
    print(f"{len(lines)} rows x ({1 + len(arms)} arms x {len(metrics)} metrics) -> {a.rows}" + (f"; {len(macros)} macros" if a.macros else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
