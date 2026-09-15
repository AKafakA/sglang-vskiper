#!/usr/bin/env python3
"""Gate A (work==0): cross-arm work identity for the V1 4-arm perf comparison.

GOLDEN RULE (GR-1a / / SPEC 2026-08-12 "Validation gates" A): a cross-arm
throughput/latency comparison is admissible ONLY if, for every request in the
cell, EVERY arm served the identical input-token count AND generated the
identical output-token count. This gate checks all arms of a cell against a
reference identity (the ``V-dec-rs`` = "vpipe" calibration by default, per SPEC
§2 "length calibration — vpipe FIRST") and against each other, then writes a
signed verdict. It exits nonzero on any mismatch.

Promoted from ``codex/campaigns/2026-08-07-rebuilt-benchmark-pipeline`` into
``test/vp/``  with two corrections:
  1. read the aggregate **LAST** record of the benchmark artifact (reuses
     ``validate_qps_artifact.read_last_record``) — the per-cell rollup is the
     final JSONL line, not the first; the old first-line read returned a warmup
     or partial record.
  2. ``--reference`` names the arm whose per-request (input, output) lengths are
     the identity table — for V1 that is ``V-dec-rs`` (the arm we ship, whose
     NATURAL lengths calibrate the equal-work budgets of the other arms).

Usage:
  cross_arm_work_gate.py --cell gsm8k_qps6_rep1 \
      --reference V-dec-rs \
      --arm P-def=<P-def>/gsm8k_qps6_rep1.jsonl \
      --arm FD-eager=<FD-eager>/gsm8k_qps6_rep1.jsonl \
      --arm V-dec=<V-dec>/gsm8k_qps6_rep1.jsonl \
      --arm V-dec-rs=<V-dec-rs>/gsm8k_qps6_rep1.jsonl \
      --out gate_a.json
  # or a fixed frozen table instead of a reference arm:
  cross_arm_work_gate.py --cell... --table lengths.json --arm... --out...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from validate_qps_artifact import read_last_record


def load_cell(path: str | Path) -> dict[str, tuple[int, int]]:
    """Return {request_id: (input_len, output_len)} from the cell's LAST record."""
    record = read_last_record(Path(path))
    ids = record.get("logical_request_ids") or record.get("request_ids")
    ins = record.get("input_lens")
    outs = record.get("output_lens")
    if not isinstance(ins, list) or not isinstance(outs, list) or not ins or not outs:
        sys.exit(f"{path}: missing input_lens/output_lens arrays in last record")
    if len(ins) != len(outs):
        sys.exit(f"{path}: input_lens/output_lens length mismatch")
    if isinstance(ids, list) and len(ids) == len(ins):
        return {str(i): (int(a), int(b)) for i, a, b in zip(ids, ins, outs)}
    return {str(i): (int(a), int(b)) for i, (a, b) in enumerate(zip(ins, outs))}


def evaluate(
    cell: str,
    arms: dict[str, dict[str, tuple[int, int]]],
    table: dict[str, tuple[int, int]] | None,
    reference: str | None,
) -> dict[str, Any]:
    """Build the work-identity verdict for one cell across all arms."""
    if table is None:
        if reference is None:
            raise ValueError("need a table or a reference arm")
        if reference not in arms:
            raise ValueError(f"reference arm {reference!r} not among arms")
        table = {rid: (io[0], io[1]) for rid, io in arms[reference].items()}
    verdict: dict[str, Any] = {
        "gate": "A_work_identity",
        "cell": cell,
        "reference": reference,
        "arms": sorted(arms),
        "n_table": len(table),
        "mismatches": [],
        "pass": True,
    }
    keys: set[str] | None = None
    for name, arm in arms.items():
        if keys is None:
            keys = set(arm)
        elif set(arm) != keys:
            verdict["mismatches"].append({"arm": name, "kind": "request_set_differs"})
            verdict["pass"] = False
    # [Codex review 2 P1-3] The comparison below iterates the TABLE, so a request present in
    # every arm but ABSENT from the table was never compared: arms agreeing on their key set
    # passed while that request did 30 tokens in one arm and 999 in another. Unchecked work is
    # not equal work. Every request an arm actually served must be in the table.
    table_keys = {str(k) for k in table}
    for name, arm in arms.items():
        extra = sorted(set(arm) - table_keys)
        if extra:
            verdict["mismatches"].append(
                {
                    "arm": name,
                    "kind": "requests_absent_from_table",
                    "n": len(extra),
                    "examples": extra[:5],
                }
            )
            verdict["pass"] = False
    for rid, want in table.items():
        for name, arm in arms.items():
            got = arm.get(str(rid))
            if got is None:
                verdict["mismatches"].append({"arm": name, "rid": rid, "kind": "missing"})
                verdict["pass"] = False
            elif [int(got[0]), int(got[1])] != [int(want[0]), int(want[1])]:
                verdict["mismatches"].append(
                    {
                        "arm": name,
                        "rid": rid,
                        "kind": "length_mismatch",
                        "want_in_out": [int(want[0]), int(want[1])],
                        "got_in_out": [int(got[0]), int(got[1])],
                    }
                )
                verdict["pass"] = False
    verdict["mismatch_count"] = len(verdict["mismatches"])
    verdict["mismatches"] = verdict["mismatches"][:50]
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--table", help="frozen identity table json {rid:[in,out]}")
    ap.add_argument(
        "--reference",
        default="V-dec-rs",
        help="arm name whose per-request lengths are the identity table "
        "(default: V-dec-rs, the shipped 'vpipe' calibration)",
    )
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=RESULTS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms: dict[str, dict[str, tuple[int, int]]] = {}
    for spec in a.arm:
        name, sep, path = spec.partition("=")
        if not sep:
            sys.exit(f"--arm expects NAME=RESULTS, got {spec!r}")
        arms[name] = load_cell(path)

    table: dict[str, tuple[int, int]] | None = None
    reference: str | None = a.reference
    if a.table:
        raw = json.load(open(a.table))
        table = {rid: (int(io[0]), int(io[1])) for rid, io in raw.items()}
        reference = None

    verdict = evaluate(a.cell, arms, table, reference)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(verdict, open(a.out, "w"), indent=1, sort_keys=True)
    print(
        f"{a.cell}: {'PASS' if verdict['pass'] else 'FAIL'} "
        f"({len(arms)} arms, {verdict['n_table']} requests, "
        f"{verdict['mismatch_count']} mismatches)"
    )
    sys.exit(0 if verdict["pass"] else 1)


if __name__ == "__main__":
    main()
