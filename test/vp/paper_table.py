#!/usr/bin/env python3
"""Emit the paper's headline table from the gated artifacts — and verify main.tex against it.

Why. `main.tex` carries **996 numeric literals**, every headline one of them typed by hand, and
all of them are currently the VOID v1.3 table. A transcription error between an artifact and a
LaTeX cell is invisible: the paper compiles, the number looks plausible, nothing fails. This
project has already published a number from the wrong campaign.

There WAS a checker for this — `vPipe-doc/codex/tools/verify_paper_numbers.py` — and it is dead
for v1.4, which is worse than absent. It imports `v13_results_table_ci`, which globs
`ovn-<ds>-r<rate>-rep<N>/results/`, looks for `workgate_*.log` where the driver now writes
`workgate_*.json`, and hardcodes `ROWS` with **two void knees** (BBH 35, coqa 23; the live ones
are 25 and 27). Pointed at the new campaign it would either crash or verify the wrong rows. A
gate that cannot fire is the defect this project keeps re-finding (D-614, D-624 #5, D-627).

So this reads the ONE source of truth: `paired_analysis.py --json`, which is bound into the run
flow, applies the GR-1a gating, computes the t-based CI and the straddle rule. **The statistic
is not re-implemented here** — that is how a generator and its verifier drift apart.

  paper_table.py report.json --knee gsm8k=11 --knee bbh_cot=25 --emit
  paper_table.py report.json --knee gsm8k=11 --knee bbh_cot=25 --verify main.tex

`--emit` writes the `tabular` rows and a macro file, so the prose can say `\\vpTPSbbhMid`
instead of a literal. `--verify` re-derives and diffs, exiting non-zero on any mismatch,
on a row main.tex has that the artifacts do not, and on a row the artifacts have that
main.tex does not.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# The table's column order. Same nine latency metrics plus TPS as paired_analysis, in the
# order the paper prints them; a metric absent from the report is an absent COLUMN, refused,
# never a silently short row.
COLUMNS = [
    "TTFT p50", "TTFT mean", "TTFT p99",
    "TPOT p50", "TPOT mean", "TPOT p99",
    "E2E p50", "E2E mean", "E2E p99",
    "output TPS",
]
MULTIPLIERS = (0.75, 0.95, 1.25)
DISPLAY = {"gsm8k": "GSM8K", "bbh_cot": "BBH", "coqa": "CoQA"}


def rate_of(suite: str) -> float:
    """`gsm8k_eqw_r8p25` -> 8.25. The rate is IN the suite name; never inferred."""
    match = re.search(r"_r(\d+)p(\d+)$", suite)
    if not match:
        raise ValueError(f"suite {suite!r} does not end in the _r<rate> form")
    return float(f"{match.group(1)}.{match.group(2)}")


def multiplier_of(suite: str, knee: float) -> float:
    """Resolve a suite to its 0.75/0.95/1.25 rung, or refuse.

    Refusing here is the point: a suite that does not land on a declared rung means the
    campaign ran a rate the table has no column for, and that must stop the emit rather
    than appear as an unlabelled row.
    """
    rate = rate_of(suite)
    for mult in MULTIPLIERS:
        if abs(rate - knee * mult) < 0.02:
            return mult
    raise ValueError(
        f"suite {suite!r} is {rate} which is not 0.75/0.95/1.25 x knee {knee} "
        f"(expected one of {[round(knee * m, 2) for m in MULTIPLIERS]})"
    )


def table_rows(report: dict[str, Any], knees: dict[str, float]) -> list[dict[str, Any]]:
    """One row per (dataset, rung), columns in COLUMNS order. Refuses a short row."""
    by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for row in report["rows"]:
        by_cell.setdefault((row["dataset"], row["suite"]), {})[row["metric"]] = row
    out = []
    for (dataset, suite), metrics in by_cell.items():
        if dataset not in knees:
            raise ValueError(f"no --knee given for dataset {dataset!r}")
        missing = [c for c in COLUMNS if c not in metrics]
        if missing:
            raise ValueError(f"{dataset} {suite}: report is missing columns {missing}")
        out.append({
            "dataset": dataset,
            "suite": suite,
            "multiplier": multiplier_of(suite, knees[dataset]),
            "n": metrics[COLUMNS[0]]["n"],
            "cells": [(c, metrics[c]["mean_pct"], metrics[c]["ci95_half"]) for c in COLUMNS],
        })
    out.sort(key=lambda r: (r["dataset"], r["multiplier"]))
    return out


def latex_rows(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        cells = []
        for _, mean, half in row["cells"]:
            cells.append(f"{mean:+.1f}" if half is None
                         else f"{mean:+.1f} $\\pm$ {half:.1f}")
        lines.append(
            f"{DISPLAY.get(row['dataset'], row['dataset'])} & "
            f"${row['multiplier']:g}\\times Q^*$ & {row['n']} & "
            + " & ".join(cells) + r" \\"
        )
    return "\n".join(lines)


def macro_name(dataset: str, multiplier: float, column: str) -> str:
    """A LaTeX-legal name: letters only, so the prose can reference it."""
    rung = {0.75: "Low", 0.95: "Mid", 1.25: "High"}[multiplier]
    metric = column.replace(" ", "").replace("output", "").replace("p50", "Pfifty")
    metric = metric.replace("p99", "Pninetynine").replace("mean", "Mean")
    # A LaTeX control sequence cannot contain a digit: \vpE"2"E... is a syntax error, not
    # an ugly name. Caught by the test, not by reading.
    metric = metric.replace("E2E", "EtoE")
    name = "vp" + metric + DISPLAY.get(dataset, dataset).replace("_", "") + rung
    if not name.isalpha():
        raise ValueError(f"macro name {name!r} is not LaTeX-legal (letters only)")
    return name


def macros(rows: list[dict[str, Any]]) -> str:
    lines = ["% GENERATED by test/vp/paper_table.py -- do not edit by hand.",
             "% Regenerate from paired_analysis.py --json; verify with paper_table.py --verify."]
    for row in rows:
        for column, mean, _ in row["cells"]:
            lines.append(
                f"\\newcommand{{\\{macro_name(row['dataset'], row['multiplier'], column)}}}"
                f"{{{mean:+.1f}}}"
            )
    return "\n".join(lines) + "\n"


ROW_RE = re.compile(
    r"^(?P<ds>[A-Za-z0-9]+)\s*&\s*\$(?P<mult>[0-9.]+)\\times Q\^\*\$\s*&\s*(?P<n>\d+)\s*&"
    r"(?P<cells>.+?)\\\\\s*$",
    re.M,
)
CELL_RE = re.compile(r"([+-]?\d+\.\d+)(?:\s*\$\\pm\$\s*(\d+\.\d+))?")


def verify(rows: list[dict[str, Any]], tex: str, tol: float) -> list[str]:
    """Diff main.tex's paired-form rows against the artifacts. Both directions."""
    problems: list[str] = []
    seen: set[tuple[str, float]] = set()
    expected = {(DISPLAY.get(r["dataset"], r["dataset"]), r["multiplier"]): r for r in rows}
    for match in ROW_RE.finditer(tex):
        key = (match.group("ds"), float(match.group("mult")))
        if key not in expected:
            problems.append(f"main.tex has row {key} that the artifacts do not")
            continue
        seen.add(key)
        row = expected[key]
        if int(match.group("n")) != row["n"]:
            problems.append(f"{key}: n={match.group('n')} in main.tex, {row['n']} measured")
        found = CELL_RE.findall(match.group("cells"))
        if len(found) != len(row["cells"]):
            problems.append(
                f"{key}: {len(found)} cells in main.tex, {len(row['cells'])} measured")
            continue
        for (column, mean, half), (tex_mean, tex_half) in zip(row["cells"], found):
            if abs(float(tex_mean) - mean) > tol:
                problems.append(
                    f"{key} {column}: main.tex {tex_mean}, measured {mean:+.1f}")
            if half is not None and tex_half and abs(float(tex_half) - half) > tol:
                problems.append(
                    f"{key} {column} CI: main.tex {tex_half}, measured {half:.1f}")
    for key in expected:
        if key not in seen:
            problems.append(f"artifacts have row {key} that main.tex does not")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", type=Path, help="paired_analysis.py --json output")
    ap.add_argument("--knee", action="append", default=[], required=True,
                    help="dataset=Q*, repeatable. Explicit so a suite that does not land "
                         "on a declared rung is REFUSED rather than labelled by guess.")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--macros", type=Path, help="with --emit: write the macro file here")
    ap.add_argument("--verify", type=Path, help="main.tex to check against the artifacts")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="printed to one decimal, so 0.05 is exact-match at that precision")
    args = ap.parse_args()

    if not args.emit and not args.verify:
        ap.error("give --emit, --verify, or both")
    knees: dict[str, float] = {}
    for entry in args.knee:
        name, _, value = entry.partition("=")
        if not value:
            ap.error(f"--knee wants dataset=Q*, got {entry!r}")
        knees[name] = float(value)

    report = json.loads(args.report.read_text())
    try:
        rows = table_rows(report, knees)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("REFUSED: the report has no rows", file=sys.stderr)
        return 2

    if args.emit:
        print(latex_rows(rows))
        if args.macros:
            args.macros.write_text(macros(rows))
            print(f"\n% wrote {len(rows) * len(COLUMNS)} macros to {args.macros}",
                  file=sys.stderr)
    if args.verify:
        problems = verify(rows, args.verify.read_text(), args.tol)
        if problems:
            print(f"\nMISMATCH — {len(problems)} problem(s) between {args.verify} "
                  "and the artifacts:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"\nOK: every headline cell in {args.verify} matches the gated artifacts "
              f"({len(rows)} rows x {len(COLUMNS)} columns)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
