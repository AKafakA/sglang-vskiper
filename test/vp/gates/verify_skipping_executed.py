#!/usr/bin/env python3
"""Refuse a quality measurement in which the skipper never skipped.

[D-627] On 2026-09-09 a quality run executed **16,702 decode passes with skip: 0**. The regime
switch's decode leg enters the skip body only above enter_rows=176; lm-eval drives concurrency
16, so the arm sat in `prod_allrun` -- the production all-RUN body -- for the entire run. Every
quality number from it describes a system that never skipped, and none of them was quotable.

Nothing objected, because every gate in the project asks "is the server serving the configuration
we declared?" and the answer was yes. The configuration was declared correctly and produced no
skipping. That is the D-609 failure one level up: an input gate cannot catch an error in the
intent itself.

So this gate does not compare configuration. It reads the ATTESTED COUNTERS and asks a different
question: **did the mechanism actually execute?** A served flag is not an active treatment
(`verify-mechanism-executed-not-just-configured`).

Usage -- capture /server_info BEFORE and AFTER the measurement:
    verify_skipping_executed.py --before before.json --after after.json [--min-skip-share 0.5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def counters(path: Path) -> dict:
    data = json.loads(path.read_text())
    states = data.get("internal_states") or []
    if not states:
        sys.exit(f"FATAL: {path} has no internal_states")
    vp = states[0].get("vp_runtime")
    if vp is None:
        sys.exit(
            f"FATAL: {path} has no vp_runtime. A quality run on a stock/no-skipper arm has "
            "nothing to verify here -- do not run this gate against the baseline arm."
        )
    return ((vp.get("regime_switch") or {}).get("counters") or {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--min-skip-share", type=float, default=0.5,
                    help="minimum share of decode passes that must have run the SKIP body")
    a = ap.parse_args()

    b, c = counters(a.before), counters(a.after)
    bd, cd = (b.get("decode") or {}), (c.get("decode") or {})
    skip = int(cd.get("skip", 0)) - int(bd.get("skip", 0))
    allrun = int(cd.get("prod_allrun", 0)) - int(bd.get("prod_allrun", 0))
    total = skip + allrun

    print(f"decode passes during the run : {total}")
    print(f"  skip body (routed)         : {skip}")
    print(f"  prod_allrun (NO skipping)  : {allrun}")

    if total == 0:
        print("\nREFUSING: zero decode passes recorded between the two snapshots. Either the "
              "counters were captured outside the run window, or nothing ran.")
        return 1

    share = skip / total
    print(f"  skip share                 : {share:.1%}  (floor {a.min_skip_share:.0%})")

    if skip == 0:
        print("\nREFUSING: the skipper NEVER SKIPPED. This measurement describes the production "
              "all-RUN body, not the skipper. Use the always-skip quality arm "
              "(integrated_alwaysskip), or drive load above the admission threshold (D-627).")
        return 1
    if share < a.min_skip_share:
        print(f"\nREFUSING: only {share:.1%} of decode passes routed. A quality number that is "
              "mostly the no-skip body attributes the skipper's quality to a system that "
              "largely was not it.")
        return 1

    print("\nOK: the skipper executed. This measurement describes the routed system.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
