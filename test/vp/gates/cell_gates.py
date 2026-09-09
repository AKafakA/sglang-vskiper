#!/usr/bin/env python3
"""The three per-experiment gates (owner contract, D-622).

    G1  CONFIG IDENTITY   BEFORE the cell  -> refuse to run
    G2  EQUAL WORK        AFTER the cell   -> assert; on failure FIX + RERUN
    G3  ZERO ERRORS       AFTER the cell   -> assert; on failure FIX + RERUN

G1 blocks because a wrong configuration cannot be repaired once the GPU is spent -- that is the
~18 h of A100 time D-609 records. G2 and G3 can only be asserted afterwards, because work
identity and error counts do not exist until the cell has run.

WHAT HAPPENS ON FAILURE IS THE POINT. A failed G2/G3 makes the cell's numbers NOT QUOTABLE: the
cell is fixed and re-run, never reinterpreted, re-labelled, or cited with a caveat. This project
has failed exactly there before -- quarantined numbers kept being cited, and a mechanism
diagnostic was substituted for a full suite. So a failing cell is marked INVALID.* on disk, the
same treatment that made the D-609 cells unreadable to every downstream tool.

    cell_gates.py pre  --tree T --upstream U --workdir W --url URL --arm ARM
    cell_gates.py post --cell <cell dir> [--arm NAME=results.jsonl ...] [--mark-invalid]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = HERE.parents[2]


def run(cmd: list[str], label: str) -> bool:
    print(f"\n--- {label}")
    r = subprocess.run([sys.executable, *cmd], capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:])
    if r.returncode != 0:
        sys.stdout.write(r.stderr[-1500:])
    print(f"--- {label}: {'PASS' if r.returncode == 0 else 'FAIL'} (exit {r.returncode})")
    return r.returncode == 0


def pre(a) -> int:
    """G1 -- the only gate that can still save GPU time."""
    ok = run([str(HERE / "campaign_preflight.py"), "--tree", str(a.tree),
              "--upstream", str(a.upstream), "--workdir", str(a.workdir)],
             "G1a campaign preflight (upstream baseline staged and genuinely stock)")
    ok &= run([str(HERE / "verify_served_design.py"), "--url", a.url,
               "--arm", a.arm, "--tree", str(a.tree)],
              "G1b served design == intended design (INTENDED vs RESOLVED)")
    if not ok:
        print("\nG1 FAILED -- refusing to run the cell. A cell measured against the wrong "
              "configuration cannot be repaired afterwards (D-609).")
        return 1
    print("\nG1 PASS -- configuration verified. Cell may run.")
    return 0


def mark_invalid(cell: Path, why: str) -> None:
    """Rename so no downstream tool's path pattern can match it (the D-609 treatment)."""
    if cell.name.startswith("INVALID."):
        return
    dest = cell.with_name(f"INVALID.{cell.name}")
    cell.rename(dest)
    (dest / "INVALID_REASON.txt").write_text(
        f"{why}\n\nThis cell's numbers are NOT quotable. Fix the cause and RERUN; do not "
        f"reinterpret, re-label, or cite with a caveat (D-622).\n", encoding="utf-8"
    )
    print(f"\nmarked INVALID: {dest}")


def post(a) -> int:
    """G2 + G3 -- assertions on a finished cell.

    G3 is read from the status the RUNNER already computed. `run_qps_evaluation.py:861` runs
    validate_qps_artifact.py after every cell and records
    `status: "completed" | "accounting_failed"` -- but `accounting_failed` appears NOWHERE else
    in the tree: no analysis tool reads it, so `analyze_qps_evaluation.py` will aggregate a
    failed cell into a table exactly like a passing one. The gate ran, recorded the failure, and
    nothing refused the number. That is the D-609 pattern, and closing it is this function's job.
    """
    cell = a.cell
    if not cell.is_dir():
        sys.exit(f"FATAL: no such cell directory: {cell}")

    # ---- G3: zero errors / no filtered tail -----------------------------------------
    commands = sorted(cell.rglob("*.command.json"))
    statuses: list[tuple[str, str]] = []
    for c in commands:
        try:
            st = str(json.loads(c.read_text()).get("status", "MISSING"))
        except Exception as exc:
            st = f"UNREADABLE ({exc})"
        statuses.append((c.name, st))
    ok_g3 = bool(statuses) and all(st == "completed" for _, st in statuses)
    for name, st in statuses:
        print(f"    {name}: status={st}")
    if not statuses:
        print("    no *.command.json found -- a cell with no accounting record is not a cell")
    print(f"--- G3 zero errors / accounting: {'PASS' if ok_g3 else 'FAIL'} "
          f"({len(statuses)} record(s))")

    # ---- G2: cross-arm work identity (GR-1a) ----------------------------------------
    ok_g2 = True
    if a.arm:
        cmd = [str(TREE / "test/vp/cross_arm_work_gate.py"), "--cell", a.cell_name or cell.name]
        if a.reference:
            cmd += ["--reference", a.reference]
        if a.table:
            cmd += ["--table", str(a.table)]
        for spec in a.arm:
            cmd += ["--arm", spec]
        if a.out:
            cmd += ["--out", str(a.out)]
        ok_g2 = run(cmd, "G2 cross-arm work identity (GR-1a)")
    else:
        print("--- G2 SKIPPED: no --arm given. A single-arm cell has no work identity to check; "
              "a PAIRED cell without --arm is an UNCHECKED cell, not a passing one.")

    if not (ok_g2 and ok_g3):
        why = (f"G2 equal-work={'PASS' if ok_g2 else 'FAIL'}, "
               f"G3 zero-errors={'PASS' if ok_g3 else 'FAIL'}")
        if a.mark_invalid:
            mark_invalid(cell, why)
        print(f"\nPOST-CELL GATES FAILED ({why}). These numbers are NOT quotable: fix and rerun.")
        return 1
    print("\nG2 + G3 PASS -- the cell's numbers are quotable.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("pre", help="G1, before the cell")
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--arm", required=True)
    p.set_defaults(fn=pre)

    q = sub.add_parser("post", help="G2 + G3, after the cell")
    q.add_argument("--cell", type=Path, required=True)
    q.add_argument("--arm", action="append", default=[], metavar="NAME=RESULTS",
                   help="repeatable, matching cross_arm_work_gate.py's own interface")
    q.add_argument("--reference", help="reference arm for the length table")
    q.add_argument("--table", type=Path, help="frozen identity table instead of a reference arm")
    q.add_argument("--cell-name", dest="cell_name", help="cell label if it differs from the dir")
    q.add_argument("--out", type=Path, help="where cross_arm_work_gate writes its verdict")
    q.add_argument("--mark-invalid", action="store_true",
                   help="rename a failing cell INVALID.* so no tool reads it as a result")
    q.set_defaults(fn=post)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
