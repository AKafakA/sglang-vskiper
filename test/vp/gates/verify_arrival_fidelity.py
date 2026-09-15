#!/usr/bin/env python3
"""Reject a cell whose REALIZED arrivals departed from the frozen trace.

[Audit #1, the highest-risk finding] `use_trace_timestamps` was never forwarded to
`get_request()`, so every cell drew Poisson arrivals while its summary reported "trace" and its
artifacts hashed the frozen arrival file. Forwarding the flag makes trace replay POSSIBLE. It
does not establish that a run FOLLOWED the trace -- and the audit asked for exactly that:

    "make the frozen arrival trace actually drive request submission, AND REJECT CELLS WHOSE
     REALIZED ARRIVALS MATERIALLY DEPART FROM IT"

Codex review 4 judged the forwarding fix "correct but incomplete: campaign timestamps survive
loading; realized fidelity remains unchecked." This is the missing half.

Why it matters more than it sounds: the campaign is a PAIRED comparison. Two arms handed the
same arrival file must actually see the same traffic, or the TTFT/TPOT/E2E/TPS deltas are
between different workloads. That failure is invisible to every output gate, because each arm's
own accounting is internally consistent.

Both sides already exist:
    intended  <label>.arrival.requests.jsonl -> per-row "timestamp" (ms, cumulative)
    realized  the bench artifact             -> "request_start_offsets_s" (s, from t0)

    verify_arrival_fidelity.py --arrival <f.jsonl> --artifact <bench.jsonl> [--tolerance-s 0.25]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_intended(path: Path) -> list[float]:
    out = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "timestamp" not in row:
            sys.exit(f"FATAL: {path} row has no 'timestamp' -- not an arrival schedule")
        out.append(float(row["timestamp"]) / 1000.0)
    if not out:
        sys.exit(f"FATAL: {path} is empty")
    base = out[0]
    return [t - base for t in out]


def load_realized(path: Path) -> list[float]:
    rec = json.loads(path.read_text().strip().splitlines()[-1])
    starts = rec.get("request_start_offsets_s")
    if not starts:
        sys.exit(
            f"FATAL: {path} carries no request_start_offsets_s. A cell that cannot show WHEN "
            "its requests were submitted cannot be checked against its own trace."
        )
    base = min(starts)
    return [float(t) - base for t in starts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrival", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--tolerance-s", type=float, default=0.25,
                    help="max acceptable MEDIAN absolute departure per request")
    ap.add_argument("--max-tolerance-s", type=float, default=5.0,
                    help="max acceptable WORST-CASE departure")
    a = ap.parse_args()

    intended = load_intended(a.arrival)
    realized = load_realized(a.artifact)

    print(f"intended arrivals : {len(intended)}  span {intended[-1]:.1f}s")
    print(f"realized arrivals : {len(realized)}  span {max(realized):.1f}s")

    if len(intended) != len(realized):
        print(f"\nREFUSING: {len(intended)} intended vs {len(realized)} realized arrivals -- "
              "the cell did not submit the trace it certifies.")
        return 1

    # Compare in submission order: both lists are ordered by submission.
    dev = [abs(r - i) for r, i in zip(sorted(realized), sorted(intended))]
    med = statistics.median(dev)
    worst = max(dev)
    p90 = sorted(dev)[int(len(dev) * 0.9)]
    print(f"departure from trace: median {med:.3f}s  p90 {p90:.3f}s  max {worst:.3f}s")

    # A Poisson-drawn run against a trace-derived schedule diverges progressively; a genuine
    # replay tracks it within scheduling jitter. The median catches systematic divergence, the
    # max catches a stall that a median would average away.
    ok = med <= a.tolerance_s and worst <= a.max_tolerance_s
    if not ok:
        print(f"\nREFUSING: realized arrivals depart from the frozen trace "
              f"(median {med:.3f}s > {a.tolerance_s}s or max {worst:.3f}s > {a.max_tolerance_s}s).\n"
              "This cell measured different traffic from the traffic its artifacts certify, and "
              "a paired comparison against it is not valid.")
        return 1
    print("\nOK: realized arrivals track the frozen trace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
