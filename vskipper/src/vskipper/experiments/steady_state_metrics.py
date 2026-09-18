#!/usr/bin/env python3
"""Steady-state windowed re-analysis of a cell's per-request arrays.

Fixed-N open-loop cells mix ramp-up, (at overload) unbounded queue
growth, and drain; their whole-cell aggregates are not stationary
quantities. This tool recomputes metrics over requests SUBMITTED inside
a declared steady window [warmup_s, injection_end - tail_s], and reports
the sustained in-window completion rate — the honest capacity number at
saturation. Post-analysis only; consumes banked cell jsonl unchanged.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def pct(values, q):
    xs = sorted(values)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q / 100
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-jsonl", type=Path, required=True)
    parser.add_argument("--warmup-s", type=float, default=60.0,
                        help="discard requests submitted before this offset")
    parser.add_argument("--tail-s", type=float, default=30.0,
                        help="discard requests submitted within this many "
                             "seconds of the injection end")
    args = parser.parse_args()

    r = json.loads(args.cell_jsonl.open().readline())
    starts = r["request_start_offsets_s"]
    ends = r["request_end_offsets_s"]
    ttfts = r["ttfts"]
    n = len(starts)
    if not (len(ends) == len(ttfts) == n):
        raise SystemExit("array length mismatch in cell row")

    injection_end = max(starts)
    lo, hi = args.warmup_s, injection_end - args.tail_s
    if hi <= lo:
        raise SystemExit(
            f"window empty: injection spans {injection_end:.1f}s, "
            f"warmup {args.warmup_s} + tail {args.tail_s} leave nothing"
        )
    idx = [i for i in range(n) if lo <= starts[i] <= hi]
    if len(idx) < 50:
        raise SystemExit(f"only {len(idx)} requests in the steady window")

    w_ttft = [ttfts[i] * 1000 for i in idx if ttfts[i] is not None]
    w_e2e = [(ends[i] - starts[i]) * 1000 for i in idx]
    # Sustained completion rate: completions whose END falls inside the window
    done_in_window = sum(1 for i in range(n) if lo <= ends[i] <= hi)
    sustained = done_in_window / (hi - lo)

    out = {
        "cell": str(args.cell_jsonl),
        "window_s": [lo, hi],
        "injection_end_s": injection_end,
        "n_submitted_in_window": len(idx),
        "n_completed_in_window": done_in_window,
        "sustained_completion_rate": sustained,
        "ttft_ms": {
            "p50": pct(w_ttft, 50),
            "p90": pct(w_ttft, 90),
            "p99": pct(w_ttft, 99),
            "mean": statistics.mean(w_ttft) if w_ttft else float("nan"),
        },
        "e2e_ms_p50": pct(w_e2e, 50),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
