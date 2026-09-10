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


def _vp_runtime(path: Path) -> dict:
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
    return vp


def skip_and_total(path: Path) -> tuple[int, int, str]:
    """(skipped, total, source) -- read the counter block that APPLIES to the served arm.

    There are two, and which one carries the evidence depends on the arm:

      regime_switch.counters   admission-gated arms (integrated_it4). Counts DECODE PASSES
                               as skip vs prod_allrun. This is what D-627 was caught with.
      fd_c3.counters           always-route arms (integrated_alwaysskip), whose defining
                               property is `regime_switch: False` -- so its counters are
                               permanently zero and reading them says "nothing ran". Counts
                               TOKENS as fd_tokens_skip_body vs fd_tokens_prod_allrun_band.

    Reading only the first refused `integrated_alwaysskip` -- the arm the 2x2 quality gate
    depends on -- while fd_c3 showed 4,975 tokens through the skip body against 0 in the
    all-RUN band. Fail-closed was correct; blind to the arm was not.
    """
    vp = _vp_runtime(path)
    regime = vp.get("regime_switch") or {}
    if regime.get("enabled"):
        decode = (regime.get("counters") or {}).get("decode") or {}
        return (
            int(decode.get("skip", 0)),
            int(decode.get("skip", 0)) + int(decode.get("prod_allrun", 0)),
            "regime_switch.counters.decode (passes)",
        )
    c3 = vp.get("fd_c3") or {}
    if c3.get("enabled"):
        counts = c3.get("counters") or {}
        skipped = int(counts.get("fd_tokens_skip_body", 0))
        allrun = int(counts.get("fd_tokens_prod_allrun_band", 0))
        return skipped, skipped + allrun, "fd_c3.counters (tokens)"
    sys.exit(
        f"FATAL: {path} has neither an enabled regime_switch nor fd_c3, so no counter "
        "block attests whether the skipper executed. Refusing rather than assuming."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--min-skip-share", type=float, default=0.5,
                    help="minimum share of routed work that must have run the SKIP body")
    a = ap.parse_args()

    before_skip, before_total, source = skip_and_total(a.before)
    after_skip, after_total, after_source = skip_and_total(a.after)
    if source != after_source:
        sys.exit(
            f"FATAL: counter source changed mid-run ({source} -> {after_source}). The server "
            "was restarted or reconfigured between the snapshots; the delta is meaningless."
        )
    skip = after_skip - before_skip
    total = after_total - before_total
    allrun = total - skip

    print(f"counter source               : {source}")
    print(f"routed work during the run   : {total}")
    print(f"  skip body (routed)         : {skip}")
    print(f"  all-RUN (NO skipping)      : {allrun}")

    if total == 0:
        print("\nREFUSING: zero work recorded between the two snapshots. Either the counters "
              "were captured outside the run window, or nothing ran.")
        return 1

    share = skip / total
    print(f"  skip share                 : {share:.1%}  (floor {a.min_skip_share:.0%})")

    if skip == 0:
        print("\nREFUSING: the skipper NEVER SKIPPED. This measurement describes the production "
              "all-RUN body, not the skipper. Use the always-skip quality arm "
              "(integrated_alwaysskip), or drive load above the admission threshold (D-627).")
        return 1
    if share < a.min_skip_share:
        print(f"\nREFUSING: only {share:.1%} of routed work went through the skip body. A "
              "quality number that is mostly the no-skip body attributes the skipper's "
              "quality to a system that largely was not it.")
        return 1

    print("\nOK: the skipper executed. This measurement describes the routed system.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
