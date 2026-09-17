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
gate that cannot fire is the defect this project keeps re-finding ( #5).

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
    "TTFT p50", "TTFT mean", "TTFT p95", "TTFT p99",
    "TPOT p50", "TPOT mean", "TPOT p95", "TPOT p99",
    "E2E p50", "E2E mean", "E2E p95", "E2E p99",
    "makespan", "output TPS",
]
# [/, owner 2026-09-13] The paper's headline table is latency-led: E2E first, means + p95;
# p50/p99 go to the appendix table. Macros are always emitted for every column.
COLUMN_SETS = {
    "all": COLUMNS,
    "main": ["E2E mean", "E2E p95", "TPOT mean", "TPOT p95", "TTFT mean", "TTFT p95", "makespan"],
    "tails": ["E2E p50", "E2E p99", "TPOT p50", "TPOT p99", "TTFT p50", "TTFT p99"],
    # The served-arm-bank campaign ( backup): E2E and makespan only (owner, 2026-09-13).
    "backup": ["E2E mean", "E2E p95", "makespan"],
}
# Metrics that get MACROS when the report carries them but are never table columns: a report without
# them (an older campaign, the served-bank replay) is not refused.
MACRO_ONLY = ["ITL mean"]
MULTIPLIERS = (0.75, 0.95, 1.25)
DISPLAY = {"gsm8k": "GSM8K", "bbh_cot": "BBH", "coqa": "CoQA", "gsm8k_q4b": "GSM8K", "gsm8k_q8b": "GSM8K"}   # the Qwen rows carry their own dataset keys (v1.6)
# A LaTeX control sequence is letters ONLY -- \vpTPSGSM8KLow and \vpE2E... are syntax
# errors, not ugly names. So macro names get their own digit-free map; the TABLE keeps the
# real display names. Found by the test, not by reading.
MACRO_DISPLAY = {"gsm8k": "Gsm", "bbh_cot": "Bbh", "coqa": "Coqa", "gsm8k_q4b": "QwenFour", "gsm8k_q8b": "QwenEight"}   # v1.6 Qwen rows


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
            "macro_only": [(c, metrics[c]["mean_pct"], metrics[c]["ci95_half"]) for c in MACRO_ONLY if c in metrics],
        })
    out.sort(key=lambda r: (r["dataset"], r["multiplier"]))
    return out


def latex_rows(rows: list[dict[str, Any]], columns: list[str] = COLUMNS) -> str:
    lines = []
    for row in rows:
        cells = []
        by_name = {c[0]: c for c in row["cells"]}
        for _, mean, half in [by_name[name] for name in columns]:  # in the column set's order
            cells.append(f"{mean:+.1f}" if half is None
                         else f"{mean:+.1f} $\\pm$ {half:.1f}")
        lines.append(
            f"{DISPLAY.get(row['dataset'], row['dataset'])} & "
            f"${row['multiplier']:g}\\times Q^*$ & {row['n']} & "
            + " & ".join(cells) + r" \\"
        )
    # Trailing `%` so the file swallows its own final newline. Without it, \input inside a
    # tabular leaves a space token after the last row's `\\`, which opens a new row -- and the
    # `\bottomrule` that follows is a `\noalign`, giving "Misplaced \noalign". Found by
    # building the paper, not by reading it.
    return "\n".join(lines) + "%"


def macro_name(dataset: str, multiplier: float, column: str) -> str:
    """A LaTeX-legal name: letters only, so the prose can reference it."""
    rung = {0.75: "Low", 0.95: "Mid", 1.25: "High"}[multiplier]
    metric = column.replace(" ", "").replace("output", "").replace("makespan", "Makespan").replace("p50", "Pfifty")
    metric = metric.replace("p99", "Pninetynine").replace("p95", "Pninetyfive").replace("mean", "Mean")
    metric = metric.replace("E2E", "EtoE")
    name = "vp" + metric + MACRO_DISPLAY.get(dataset, dataset.replace("_", "").title()) + rung
    if not name.isalpha():
        raise ValueError(f"macro name {name!r} is not LaTeX-legal (letters only)")
    return name


def macros(rows: list[dict[str, Any]], prefix: str = "vp") -> str:
    lines = ["% GENERATED by test/vp/paper_table.py -- do not edit by hand.",
             "% Regenerate from paired_analysis.py --json; verify with paper_table.py --verify."]
    for row in rows:
        for column, mean, half in row["cells"] + row.get("macro_only", []):
            name = prefix + macro_name(row['dataset'], row['multiplier'], column)[2:]
            lines.append(f"\\newcommand{{\\{name}}}{{{mean:+.1f}}}")
            # The half-interval travels with the mean so the text can write "+7.3 ± 17.6" for
            # a cell whose verdict is parity, instead of a bare number that reads as an effect.
            lines.append(f"\\newcommand{{\\{name}Ci}}{{{'--' if half is None else f'{half:.1f}'}}}")
    return "\n".join(lines) + "\n"


ROW_RE = re.compile(
    r"^(?P<ds>[A-Za-z0-9]+)\s*&\s*\$(?P<mult>[0-9.]+)\\times Q\^\*\$\s*&\s*(?P<n>\d+)\s*&"
    # `%?` because the generated file ends its last row with a comment character to swallow
    # the newline that would otherwise open a phantom row inside the tabular.
    r"(?P<cells>.+?)\\\\%?\s*$",
    re.M,
)
CELL_RE = re.compile(r"([+-]?\d+\.\d+)(?:\s*\$\\pm\$\s*(\d+\.\d+))?")


# Both LaTeX's \input and any wrapper around the TeX primitive (the paper defines
# \inputrows, because \input inside a tabular breaks \bottomrule). A verifier that
# tracks only one of them goes blind the moment the paper changes mechanism -- which is
# exactly what happened when this file was first written.
INPUT_RE = re.compile(r"\\input(?:rows)?\{([^}]+)\}")


def expand_inputs(tex: str, base: Path) -> str:
    """Splice in one level of `\\input{...}`.

    The generated rows live in their own file so the paper never carries a hand-typed number,
    which means a verifier that reads only main.tex sees an empty table and reports every row
    as missing -- a false alarm that would train the reader to ignore it.
    """
    def _splice(match: re.Match[str]) -> str:
        name = match.group(1)
        for candidate in (base / name, base / f"{name}.tex"):
            if candidate.is_file():
                return candidate.read_text()
        return match.group(0)
    return INPUT_RE.sub(_splice, tex)


def verify(rows: list[dict[str, Any]], tex: str, tol: float,
           placeholder: set[str] | None = None) -> list[str]:
    """Diff the paper's paired-form rows against the artifacts. Both directions.
    Rows of a `placeholder` dataset are skipped (interim tables only; the caller names them)."""
    problems: list[str] = []
    seen: set[tuple[str, float]] = set()
    placeholder = placeholder or set()
    expected = {(DISPLAY.get(r["dataset"], r["dataset"]), r["multiplier"]): r for r in rows}
    for match in ROW_RE.finditer(tex):
        key = (match.group("ds"), float(match.group("mult")))
        if key[0] in placeholder:
            continue
        if key not in expected:
            problems.append(f"main.tex has row {key} that the artifacts do not")
            continue
        seen.add(key)
        row = expected[key]
        if int(match.group("n")) != row["n"]:
            problems.append(f"{key}: n={match.group('n')} in main.tex, {row['n']} measured")
        found = CELL_RE.findall(match.group("cells"))
        # The paper carries the same (dataset, rate) row in more than one table (the headline
        # with COLUMN_SETS["main"], the tails appendix with COLUMN_SETS["tails"]); a tex row is
        # checked against the column set whose width it has. A width matching no declared set
        # is a mismatch, never a skip.
        by_name = {c[0]: c for c in row["cells"]}
        widths = {name: cols for name, cols in COLUMN_SETS.items() if len(cols) == len(found)}
        if not widths:
            problems.append(
                f"{key}: {len(found)} cells in main.tex match no column set "
                f"({', '.join(f'{n}={len(c)}' for n, c in COLUMN_SETS.items())})")
            continue
        columns = next(iter(widths.values()))
        cells = [by_name[c] for c in columns]
        for (column, mean, half), (tex_mean, tex_half) in zip(cells, found):
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
    ap.add_argument("report", type=Path, nargs="+",
                    help="paired_analysis.py --json output(s); several campaigns (one per "
                         "dataset family) are merged row-wise, each row tagged with its source")
    ap.add_argument("--knee", action="append", default=[], required=True,
                    help="dataset=Q*, repeatable. Explicit so a suite that does not land "
                         "on a declared rung is REFUSED rather than labelled by guess.")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--macros", type=Path, help="with --emit: write the macro file here")
    ap.add_argument("--macro-prefix", default="vp",
                    help="macro name prefix (letters only), e.g. vpHundred for the H100 replicate's cells")
    ap.add_argument("--verify", type=Path, help="main.tex to check against the artifacts")
    ap.add_argument("--exclude-datasets", default="",
                    help="comma list of datasets whose report rows are NOT emitted and NOT "
                         "expected in main.tex (their macros are still emitted). Used for an "
                         "interim table whose rows for that dataset come from elsewhere.")
    ap.add_argument("--placeholder-datasets", default="",
                    help="comma list of datasets whose main.tex rows are placeholders: verify "
                         "skips them instead of refusing. Interim use only; the flag lives in "
                         "the regeneration script so its presence is visible.")
    ap.add_argument("--dataset-suffix", default="",
                    help="appended to the dataset name in emitted rows (e.g. ' (served-arm banks)') so a "
                         "companion table's rows are not read as headline rows by --verify")
    ap.add_argument("--dataset-label", default="",
                    help="replaces the dataset name in emitted rows (v1.7: short row labels such as 'H100' or 'Qwen3-8B')")
    ap.add_argument("--columns", default="all", choices=sorted(COLUMN_SETS),
                    help="column set for the emitted rows (macros always cover every column)")
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

    # One campaign per dataset family is the normal case (gsm8k+bbh_cot ran as one 6-rep
    # campaign, coqa as another after its banks landed); rows are merged, never re-derived.
    report = {"rows": [], "sources": []}
    for path in args.report:
        part = json.loads(path.read_text())
        for row in part["rows"]:
            row["source"] = path.name
        report["rows"].extend(part["rows"])
        report["sources"].append(path.name)
    try:
        rows = table_rows(report, knees)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if not rows:
        print("REFUSED: the report has no rows", file=sys.stderr)
        return 2

    excluded = {d for d in args.exclude_datasets.split(",") if d}
    placeholder = {DISPLAY.get(d, d) for d in args.placeholder_datasets.split(",") if d}
    table_only = [r for r in rows if r["dataset"] not in excluded]
    if args.emit:
        if args.dataset_suffix:
            table_only = [dict(r, dataset=DISPLAY.get(r["dataset"], r["dataset"]) + args.dataset_suffix) for r in table_only]
        if args.dataset_label:
            table_only = [dict(r, dataset=args.dataset_label) for r in table_only]
        print(latex_rows(table_only, COLUMN_SETS[args.columns]))
        if args.macros:
            args.macros.write_text(macros(rows, args.macro_prefix))
            print(f"\n% wrote {len(rows) * len(COLUMNS)} macros to {args.macros}",
                  file=sys.stderr)
    if args.verify:
        problems = verify(
            table_only, expand_inputs(args.verify.read_text(), args.verify.parent), args.tol,
            placeholder,
        )
        if problems:
            print(f"\nMISMATCH — {len(problems)} problem(s) between {args.verify} "
                  "and the artifacts:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        note = f"; placeholder rows skipped for {sorted(placeholder)}" if placeholder else ""
        print(f"\nOK: every headline cell in {args.verify} matches the gated artifacts "
              f"({len(table_only)} rows x {len(COLUMNS)} columns{note})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
