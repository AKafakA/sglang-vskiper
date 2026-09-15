#!/usr/bin/env python3
"""Summarise the 2x2 quality gate. The metric is DECLARED per dataset, never guessed.

Why this is a committed tool and not three lines of inline shell. The chain's ad-hoc
summariser tried a list of metric names, found none of them, and fell through to whichever
key came first in the results dict -- `strict-match`. On gsm8k that INVERTS the story:

    strict-match      (B - A) = +5.31 pp   the checkpoint appears to GAIN
    flexible-extract  (B - A) = -5.53 pp   the checkpoint costs

 recorded this exact reversal, which is why flexible-extract is a RULING
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
    # lm-eval emits `get-answer` for bbh_cot_fewshot, not flexible-extract.: this
    # filter only scores a response containing "the answer is", and the arms differ sharply
    # on that -- stock 76.5% vs vSkipper 95.5% marker rate -- so it measures format
    # compliance and correctness together. (D - C) alone is NOT quotable for BBH; only the
    # difference-of-differences, which cancels a confound both pairs share.
    "bbh_cot": ("exact_match,get-answer", "exact_match_stderr,get-answer"),
    "coqa": ("f1,none", "f1_stderr,none"),
}

ARMS = {
    "A": "raw base model, PyTorch",
    "B": "released checkpoint, its OWN code",
    "C": "base model on UPSTREAM SGLang (D-587; was our fork with the skipper off)",
    "D": "vSkipper serving the checkpoint",
}


def _arm_dir(root: Path, arm: str, mapping: dict[str, str]) -> str:
    """Where this arm's cells live. DECLARED, because the layout changed under us.

    The hand-run 2x2 wrote `<root>/A|B|C|D/`; the committed driver
    (`run_quality_2x2.py`) writes `<root>/<arm-name>/<workload>/`, so a summariser that
    assumes the letter is a directory name dies on every new campaign -- which is the same
    dead-tool defect as a checker that globs a layout the writer no longer produces.
    """
    mapped = mapping.get(arm, arm)
    # An ABSOLUTE mapping is the common case for the corrected campaign: arms A and B are
    # native PyTorch and carry over from the earlier run, while C and D are re-measured into
    # a new root. Forcing one root would mean copying artifacts around to satisfy a tool,
    # which is how a cell ends up somewhere its provenance no longer says.
    if Path(mapped).is_absolute():
        return mapped
    return f"{root}/{mapped}"


def _assert_every_arm_passed(root: Path, mapping: dict[str, str]) -> None:
    """Refuse if ANY arm's cell was refused by its own gates.

    Found 2026-09-12 by reading an emitted row: coqa arm D was REFUSED on `zero_empty` --
    one empty generation, the gate doing exactly its job -- and the summariser read its
    `results_*.json` anyway and published $0.7760$ as arm D. **A gate that fires and a
    consumer that ignores it is the defect this project has logged four times** (
    #5); here it would have put a refused measurement into the paper's
    faithfulness table, which is the one table whose entire purpose is to be trustworthy.

    Checked per ARM rather than per file: a refused cell still writes complete-looking
    results, so nothing downstream can tell by inspection.
    """
    for arm in sorted(ARMS):
        manifests = sorted(glob.glob(f"{_arm_dir(root, arm, mapping)}/**/quality_manifest.json",
                                     recursive=True))
        if not manifests:
            sys.exit(f"FATAL: arm {arm} has no quality_manifest.json under "
                     f"{_arm_dir(root, arm, mapping)} -- cannot verify its cell passed")
        for path in manifests:
            record = json.loads(Path(path).read_text())
            status = record.get("status")
            if status != "passed":
                sys.exit(
                    f"FATAL: arm {arm} cell {path} has status={status!r} "
                    f"failed_gates={record.get('failed_gates')}. Its gates REFUSED this "
                    "measurement; it must not enter the 2x2. Re-run the cell, or drop the "
                    "row and say why -- never publish a refused number."
                )


def _assert_arm_c_is_upstream(root: Path, mapping: dict[str, str]) -> None:
    """Arm C must be GENUINE upstream, because that is what the gate condition names.

    The gate reads "no quality collapse vs UPSTREAM sglang". For months arm C was
    ARMS["stock"] -- this fork with the skipper off -- and every downstream reader saw only
    the arm LABEL, so the violation was invisible. Refuse rather than compute (D - C)
    against a C that was never upstream.
    """
    manifests = sorted(glob.glob(f"{_arm_dir(root, 'C', mapping)}/**/quality_manifest.json",
                                 recursive=True))
    if not manifests:
        sys.exit(f"FATAL: arm C has no quality_manifest.json under "
                 f"{_arm_dir(root, 'C', mapping)} -- cannot verify it was upstream")
    for path in manifests:
        record = json.loads(Path(path).read_text())
        if "upstream_baseline" not in record:
            sys.exit(
                f"FATAL: {path} predates the upstream_baseline field, so whether arm C was "
                "genuine upstream is UNKNOWABLE from the artifact. Re-run arm C rather than "
                "assuming it (D-587)."
            )
        if not record["upstream_baseline"]:
            sys.exit(
                f"FATAL: arm C cell {path} was NOT served by upstream SGLang "
                f"(arm={record.get('arm')!r}). The gate condition is 'no quality collapse "
                "vs UPSTREAM sglang' -- refusing to compute (D - C) against this."
            )


def _load(root: Path, arm: str, dataset: str, mapping: dict[str, str]) -> tuple[float, float]:
    hits = sorted(glob.glob(f"{_arm_dir(root, arm, mapping)}/**/results_*.json",
                            recursive=True))
    if not hits:
        sys.exit(f"FATAL: no results_*.json for arm {arm} under "
                 f"{_arm_dir(root, arm, mapping)}")
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
    ap.add_argument("--arm-dir", action="append", default=[],
                    help="ARM=DIRNAME, repeatable, e.g. --arm-dir C=upstream "
                         "--arm-dir D=integrated_alwaysskip. Defaults to the letter itself.")
    ap.add_argument("--latex", type=Path,
                    help="APPEND this dataset's row and its macros to a generated LaTeX file. "
                         "One tool owns the statistic: the paper must never carry a "
                         "difference-of-differences computed anywhere but here.")
    ap.add_argument("--skip-upstream-check", action="store_true",
                    help="Read a HISTORICAL arm-C directory that predates the "
                         "upstream_baseline field. Never for a new campaign: it disables the "
                         "only check that arm C is what the gate condition names.")
    args = ap.parse_args()

    metric, _ = METRICS[args.dataset]
    mapping: dict[str, str] = {}
    for entry in args.arm_dir:
        arm, _, dirname = entry.partition("=")
        if arm not in ARMS or not dirname:
            ap.error(f"--arm-dir wants one of {sorted(ARMS)}=DIRNAME, got {entry!r}")
        mapping[arm] = dirname
    _assert_every_arm_passed(args.root, mapping)
    if not args.skip_upstream_check:
        _assert_arm_c_is_upstream(args.root, mapping)
    vals = {arm: _load(args.root, arm, args.dataset, mapping) for arm in ARMS}

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

    if args.latex:
        # Signed, in percentage points, to one decimal -- the same precision the paper prints.
        # `\pm` is the 95% half-width, so a reader can apply the straddle rule themselves.
        disp = {"gsm8k": "GSM8K", "coqa": "CoQA", "bbh_cot": "BBH"}[args.dataset]
        # The lm-eval key itself, escaped -- never a prettier name invented here, because the
        # filter IS the measurement (: gsm8k strict-match flips the sign of (B-A)).
        metric_label = "\\texttt{" + metric.replace("_", r"\_") + "}"
        macro = {"gsm8k": "Gsm", "coqa": "Coqa", "bbh_cot": "Bbh"}[args.dataset]
        row = (f"{disp} & {metric_label} & "
               f"{vals['A'][0]:.4f} & {vals['B'][0]:.4f} & "
               f"{vals['C'][0]:.4f} & {vals['D'][0]:.4f} & "
               f"{100*ba:+.2f} & {100*dc:+.2f} & "
               f"{100*dod:+.2f} $\\pm$ {100*1.96*se_dod:.2f} & "
               f"{'parity' if faithful else 'DECIDED'} \\\\")
        macros = "\n".join([
            f"\\newcommand{{\\vpQ{macro}BminusA}}{{{100*ba:+.2f}}}",
            f"\\newcommand{{\\vpQ{macro}DminusC}}{{{100*dc:+.2f}}}",
            f"\\newcommand{{\\vpQ{macro}Dod}}{{{100*dod:+.2f}}}",
            f"\\newcommand{{\\vpQ{macro}ArmD}}{{{vals['D'][0]:.4f}}}",
            f"\\newcommand{{\\vpQ{macro}ArmC}}{{{vals['C'][0]:.4f}}}",
        ])
        with args.latex.open("a") as handle:
            handle.write(row + "\n")
        macro_path = args.latex.with_name(args.latex.name.replace("rows", "macros"))
        with macro_path.open("a") as handle:
            handle.write(macros + "\n")
        print(f"  appended a LaTeX row to {args.latex} and macros to {macro_path}")
    if args.dataset == "bbh_cot":
        print()
        print("  WARNING (D-641) bbh_cot's `get-answer` filter scores only responses that")
        print("  contain 'the answer is'. Measured marker rates differ sharply by arm")
        print("  (stock 76.5% vs vSkipper 95.5%), so this metric mixes FORMAT COMPLIANCE")
        print("  with correctness. Quote the difference-of-differences, which cancels a")
        print("  confound shared by both pairs -- never (D - C) on its own.")
    return 0 if (not collapse and faithful) else 1


if __name__ == "__main__":
    raise SystemExit(main())
