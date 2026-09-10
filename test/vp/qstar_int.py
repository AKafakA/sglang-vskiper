#!/usr/bin/env python3
"""Read a Q* ladder and report the knee. Rebuilt from the rule, then validated against a
curve whose answer is already known.

Q* determines every operating point the campaign reports, so a bug here silently relocates
the whole grid. That is not hypothetical: on ONE unchanged BBH curve, three versions of this
tool returned three different answers --

    38.75   a self-extending loop multiplying the top rung by 1.25
    25      the same rule with one interior rung added
    21      "grew >=1% over the PRECEDING rung", stopped by a single flat pair

-- and the true knee was 35. All three were instrument artifacts, never the system
(D-590/D-591). Hence the rule below, and hence `--self-test`.

THE RULE (D-590 owner definition, D-591 fixes):

  1. Q* is the last INTEGER offered rate whose achieved output tok/s beats the RUNNING
     MAXIMUM by >= 1%. Running maximum = the best value at any lower rung, NOT the value at
     the immediately preceding rung. On the BBH curve rung 22 tied rung 21 (+0.08%) while
     throughput went on rising 16% to rung 35; comparing against the predecessor stopped
     there and returned 21.
  2. There is NO EARLY BREAK. Measure the whole contiguous band, then read off the last
     growing rung. A tie or dip is a non-growing rung, not a terminator.
  3. The band must be CONTIGUOUS integers. Across gaps a per-rung threshold is meaningless
     -- you cannot tell a genuine plateau from an unmeasured stretch.
  4. A cell whose accounting shows completed != submitted is SKIPPED, loudly. BBH rung 36
     was refused by the harness (3,998 of 4,000) and its artifacts were still on disk being
     read as a valid point.
  5. No multipliers. This tool never proposes a non-integer rate.

Usage:
    qstar_int.py <cells-dir>            # read a ladder, print the curve and Q*
    qstar_int.py --self-test            # replay the recorded BBH curve; MUST return 35
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

GROWTH = 0.01  # a rung "grows" when it beats the running maximum by >= 1%


class LadderError(RuntimeError):
    pass


def growing_rungs(curve: list[tuple[float, float]]) -> list[tuple[float, float, bool]]:
    """Mark each rung as growing (beats the running max by >= GROWTH) or not.

    `curve` must be sorted by rate ascending. No early break: every rung is classified.
    """
    out: list[tuple[float, float, bool]] = []
    running_max = 0.0
    for rate, tps in curve:
        grew = running_max <= 0.0 or tps >= running_max * (1.0 + GROWTH)
        if grew:
            running_max = max(running_max, tps)
        out.append((rate, tps, grew))
    return out


def qstar(curve: list[tuple[float, float]]) -> float:
    """The last rung that beat the running maximum. Raises if none did."""
    marked = growing_rungs(curve)
    growers = [rate for rate, _, grew in marked if grew]
    if not growers:
        raise LadderError("no rung grew; the ladder yields NO Q*")
    return growers[-1]


def check_contiguous(rates: list[float]) -> list[str]:
    """Integer rungs with no interior gaps. Returns human-readable complaints."""
    problems = []
    non_integer = [r for r in rates if abs(r - round(r)) > 1e-9]
    if non_integer:
        problems.append(
            f"non-integer rungs {non_integer} — Q* is an INTEGER rate (D-590); a "
            "non-integer here means a multiplier crept in"
        )
    ints = sorted({int(round(r)) for r in rates})
    missing = [n for n in range(ints[0], ints[-1] + 1) if n not in ints]
    if missing:
        problems.append(
            f"band {ints[0]}..{ints[-1]} is NOT contiguous, missing {missing} — a per-rung "
            "threshold cannot distinguish a plateau from an unmeasured stretch"
        )
    return problems


def read_cells(cells: Path) -> tuple[list[tuple[float, float]], list[str]]:
    """(curve, skipped) from a runner output directory.

    Reads the achieved output throughput from each cell's own artifact, and REFUSES any cell
    whose accounting is short.
    """
    curve: list[tuple[float, float]] = []
    skipped: list[str] = []
    for command_file in sorted(cells.glob("*.command.json")):
        record = json.loads(command_file.read_text())
        label = command_file.name.replace(".command.json", "")
        status = record.get("status")
        if status != "completed":
            skipped.append(f"{label}: status={status}")
            continue
        match = re.search(r"_qps([0-9p.]+)_rep", label)
        if not match:
            skipped.append(f"{label}: cannot parse rate from label")
            continue
        rate = float(match.group(1).replace("p", "."))
        output_file = Path(record["output_file"])
        if not output_file.is_file():
            skipped.append(f"{label}: artifact missing")
            continue
        last = json.loads(output_file.read_text().strip().splitlines()[-1])
        submitted = last.get("num_prompts") or last.get("completed")
        completed = last.get("completed")
        if submitted is not None and completed is not None and submitted != completed:
            # D-591 defect 2: the harness refused this cell; its artifacts are still here.
            skipped.append(
                f"{label}: completed {completed} != submitted {submitted} — "
                "accounting short, not a measurement"
            )
            continue
        tps = last.get("output_throughput")
        if tps is None:
            skipped.append(f"{label}: artifact carries no output_throughput")
            continue
        curve.append((rate, float(tps)))
    curve.sort(key=lambda item: item[0])
    return curve, skipped


def verdict(curve: list[tuple[float, float]]) -> tuple[float, list[str]]:
    """(knee, blockers). A non-empty blockers list means the knee is NOT usable yet.

    Separated from report() so --emit answers exactly the question report() prints, rather
    than a second implementation of it that could drift. Raises LadderError if no rung grew.
    """
    knee = qstar(curve)
    blockers = check_contiguous([rate for rate, _ in curve])
    if knee == max(rate for rate, _ in curve):
        blockers.append(
            f"Q* = {knee:g} IS THE TOP RUNG — the curve never plateaued, so the knee is not "
            "bracketed. WIDEN the band upward (next INTEGER rungs, never a multiplier) and "
            "re-measure."
        )
    if len(curve) > 1 and knee == min(rate for rate, _ in curve):
        # The mirror of the top-rung case, and easier to miss: the FIRST rung always counts
        # as growth (there is no lower rung for it to beat), so a curve that is already flat
        # when the band opens returns its own bottom rung. That is not a knee, it is the
        # statement that the knee is at or below where we started looking.
        blockers.append(
            f"Q* = {knee:g} IS THE BOTTOM RUNG — the first rung counts as growth by "
            "construction, so this says only that nothing above it grew. The knee is at or "
            "below the band. WIDEN the band DOWNWARD and re-measure."
        )
    return knee, blockers


def report(curve: list[tuple[float, float]], skipped: list[str]) -> int:
    if skipped:
        print("SKIPPED CELLS (not silently dropped):")
        for line in skipped:
            print(f"  - {line}")
        print()
    if not curve:
        print("FATAL: no valid rungs")
        return 1

    problems = check_contiguous([rate for rate, _ in curve])
    marked = growing_rungs(curve)
    print(f"{'rate':>6}  {'output tok/s':>13}  {'vs running max':>15}  grew")
    running_max = 0.0
    for rate, tps, grew in marked:
        delta = "—" if running_max <= 0 else f"{100 * (tps / running_max - 1):+.2f}%"
        print(f"{rate:>6g}  {tps:>13.1f}  {delta:>15}  {'yes' if grew else 'no'}")
        if grew:
            running_max = max(running_max, tps)

    print()
    try:
        knee, blockers = verdict(curve)
    except LadderError as error:
        print(f"NO Q*: {error}")
        return 1
    for problem in blockers:
        print(f"⚠ {problem}")

    print(f"\nQ* = {knee:g}")
    if blockers:
        return 1
    print(f"  cells at {{0.75, 0.95, 1.25}} x Q* = "
          f"{0.75 * knee:g} / {0.95 * knee:g} / {1.25 * knee:g}")
    return 0


# The recorded BBH curve (D-591). Its answer is known: 35. Rung 22 ties rung 21, which is
# exactly the shape that made an earlier version return 21.
_BBH_CURVE = [
    (12, 2000.0), (20, 2600.0), (21, 2690.0), (22, 2692.1), (23, 2750.0), (24, 2800.0),
    (25, 2805.0), (26, 2810.0), (27, 2860.0), (28, 2920.0), (31, 2930.0), (35, 3088.8),
    (38.75, 3091.6), (48.44, 3090.0),
]


def self_test() -> int:
    print("self-test: replaying the recorded BBH curve (D-591), which must return 35\n")
    knee = qstar(_BBH_CURVE)
    print(f"  Q* = {knee:g}")
    ok = knee == 35
    print(f"  {'PASS' if ok else 'FAIL'} — expected 35")
    if not ok:
        print("  A tool returning 21 reproduces the predecessor-comparison bug;")
        print("  25 or 38.75 reproduce the sparse-rung and multiplier bugs.")
        return 1

    # The tie at 22 must be a non-growing rung, not a terminator.
    marked = {rate: grew for rate, _, grew in growing_rungs(_BBH_CURVE)}
    assert marked[22] is False, "rung 22 ties rung 21 and must NOT count as growth"
    assert marked[35] is True, "rung 35 beats every lower rung and must count"
    print("  PASS — the tie at rung 22 is non-growing but does not stop the search")

    # A curve that never plateaus must not yield a confident knee at its own edge.
    rising = [(float(n), 1000.0 * n) for n in range(5, 12)]
    assert qstar(rising) == 11, "monotone curve should return its top rung"
    print("  PASS — a never-plateauing curve returns its top rung (caller must widen)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="?", type=Path, help="runner --output-dir of the ladder")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--emit",
        action="store_true",
        help="print ONLY the knee, for a drive loop. Exits non-zero -- printing nothing -- "
             "whenever the knee is not usable (no growth, a gappy band, or a knee sitting "
             "on the top rung, which means the curve never plateaued). A caller that reads "
             "stdout therefore cannot mistake an unbracketed edge for a measured knee; the "
             "reason goes to stderr so the loop can log it and widen the band.",
    )
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.cells:
        ap.error("give a cells directory, or --self-test")
    curve, skipped = read_cells(args.cells)
    if args.emit:
        if not curve:
            print("no valid rungs", file=sys.stderr)
            return 1
        try:
            knee, blockers = verdict(curve)
        except LadderError as error:
            print(str(error), file=sys.stderr)
            return 1
        for line in skipped:
            print(f"skipped {line}", file=sys.stderr)
        if blockers:
            for problem in blockers:
                print(problem, file=sys.stderr)
            return 1
        print(f"{knee:g}")
        return 0
    return report(curve, skipped)


if __name__ == "__main__":
    raise SystemExit(main())
