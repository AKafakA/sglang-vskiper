#!/usr/bin/env python3
"""Refuse a quality measurement in which the skipper never skipped.

On 2026-09-09 a quality run executed **16,702 decode passes with skip: 0**. The regime
switch's decode leg enters the skip body only above enter_rows=176; lm-eval drives concurrency
16, so the arm sat in `prod_allrun` -- the production all-RUN body -- for the entire run. Every
quality number from it describes a system that never skipped, and none of them was quotable.

Nothing objected, because every gate in the project asks "is the server serving the configuration
we declared?" and the answer was yes. The configuration was declared correctly and produced no
skipping. That is the failure one level up: an input gate cannot catch an error in the
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
from typing import Optional
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

      regime_switch.counters   admission-gated arms (vskipper, formerly integrated_it4). Counts DECODE PASSES
                               as skip vs prod_allrun. This is what was caught with.
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
        # BOTH PHASES, NOT DECODE ALONE.
        #
        # This read `counters.decode` only, and refused the 0.75 x Q* cell as "never skipped"
        # while the treatment arm routed 4,925 prefill passes over 6,977,016 prefill tokens.
        # The cell it discarded reproduces v1.3's published row within its confidence interval
        # on five of eight metrics.
        #
        # `counters.prefill` is NOT the prefill equivalent: it reads {dense: 0, fd: 0} on that
        # same cell because it counts switch TRANSITIONS, not routed work. The routed prefill
        # work is in `batch_composition`, which is where this now looks.
        #
        # was a gate blind to a mechanism that was OFF. This was the same gate blind to a
        # mechanism that was ON -- and the second is worse, because it throws away real results
        # while looking like diligence.
        # [/] THE ASSUMPTION ABOVE WAS THE BUG'S HIDING PLACE. `counters.prefill`
        # read {dense: 0, fd: 0} not because it "counts transitions" but because the runner's
        # dispatch machinery was gated on an env var had deleted -- OFF in every cell.
        # Taking `batch_composition.prefill_passes` as "all routed" then certified a mechanism
        # that was not running. From now on the per-body counters ARE the prefill evidence,
        # and they must account for every prefill pass the scheduler ran (checked as a delta
        # in main()); an arm that routes prefill with fd == 0 is refused like.
        counters = regime.get("counters") or {}
        decode = counters.get("decode") or {}
        d_skip = int(decode.get("skip", 0))
        d_total = d_skip + int(decode.get("prod_allrun", 0))
        active = set(((vp.get("served_design") or {}).get("active_phases")) or [])
        p_fd = 0
        if "prefill" in active:
            prefill = counters.get("prefill")
            if not isinstance(prefill, dict) or "fd" not in prefill or "dense" not in prefill:
                sys.exit(
                    f"FATAL: {path}: this arm routes prefill but regime_switch.counters.prefill "
                    "carries no per-body counts -- the prefill dispatch machinery is not running "
                    ". Refusing rather than assuming every pass routed."
                )
            p_fd = int(prefill["fd"])
        return (
            d_skip + p_fd,
            d_total + p_fd,
            # The label identifies the SOURCE, never the values: it is compared between the
            # before and after snapshots to catch a server restart, so embedding the counts
            # makes every run look like a reconfiguration.
            "regime_switch.counters.decode (passes) + regime_switch.counters.prefill.fd (passes)"
            + ("" if "prefill" in active else " [prefill not routed by this arm]"),
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


def prefill_evidence(path: Path) -> Optional[dict]:
    """The prefill leg's own metrics, for the delta checks in main(): per-body pass
    counts, the scheduler's prefill pass count they must add up to, and the engagement
    escape's observation count. None when the arm does not route prefill."""
    vp = _vp_runtime(path)
    regime = vp.get("regime_switch") or {}
    active = set(((vp.get("served_design") or {}).get("active_phases")) or [])
    if not regime.get("enabled") or "prefill" not in active:
        return None
    counters = regime.get("counters") or {}
    prefill = counters.get("prefill") or {}
    engagement = counters.get("prefill_engagement") or {}
    comp = vp.get("batch_composition") or {}
    return {
        "fd": int(prefill.get("fd", 0)),
        "dense": int(prefill.get("dense", 0)),
        "passes": int(comp.get("prefill_passes", 0)),
        "engagement_min": (regime.get("prefill") or {}).get("engagement_min"),
        "engagement_samples": int(engagement.get("samples", 0) or 0),
        "engagement_ema": engagement.get("ema"),
    }


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

    # Prefill metrics, checked as deltas over the run window.
    pb, pa = prefill_evidence(a.before), prefill_evidence(a.after)
    if pa is not None and pb is not None:
        d_fd, d_dense = pa["fd"] - pb["fd"], pa["dense"] - pb["dense"]
        d_passes = pa["passes"] - pb["passes"]
        print(f"prefill passes (scheduler)   : {d_passes}")
        print(f"  routed body (fd)           : {d_fd}")
        print(f"  dense body                 : {d_dense}")
        if d_fd + d_dense != d_passes:
            print(f"\nREFUSING: the prefill body counters account for {d_fd + d_dense} passes but the "
                  f"scheduler ran {d_passes}. Counters that do not add up are not evidence.")
            return 1
        if d_passes > 0 and d_fd == 0:
            print("\nREFUSING: this arm routes prefill but the routed body ran ZERO prefill passes in "
                  "the window.")
            return 1
        if pa["engagement_min"] is not None:
            d_samples = pa["engagement_samples"] - pb["engagement_samples"]
            print(f"  engagement escape          : ema={pa['engagement_ema']} samples(+{d_samples})")
            if d_samples <= 0 and d_fd > 0:
                print("\nREFUSING: engagement_min is configured but the escape observed nothing while "
                      "routed passes ran -- the escape is not executing.")
                return 1

    # [D-849] Per-request body pinning invariants, checked on the AFTER snapshot
    # (both counters are monotonic and must be zero for the whole boot).
    vp_after = _vp_runtime(a.after)
    admission = ((vp_after.get("regime_switch") or {}).get("counters") or {}).get("admission") or {}
    pins = vp_after.get("admission_pins") or {}
    violations = int(admission.get("coverage_dense_violation_rows", 0))
    cross_body = int(pins.get("cross_body_prefix_reuse", 0))
    if pins.get("enabled"):
        print(f"admission pins               : stock={pins.get('admitted_stock')} fd={pins.get('admitted_fd')} "
              f"flips={pins.get('band_flips')} mixed_steps={pins.get('mixed_steps')} "
              f"split_passes={admission.get('split_passes')} prefix_hits(fd)={pins.get('prefix_hit_tokens_fd')}")
    if violations > 0:
        print(f"\nREFUSING: {violations} fd-pinned decode rows were served by the coverage-dense "
              "fall-through -- a pin violation (the ladder does not cover the running batch).")
        return 1
    if cross_body > 0:
        print(f"\nREFUSING: {cross_body} prefix-cache hits reused K/V computed under the other body.")
        return 1

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

    # THE FLOOR IS "SOMETHING ROUTED", NOT A SHARE (owner, 2026-09-11: "we need just record but
    # not as the hard gates"). A share floor cannot express the real question. A cell where
    # prefill routes 7 M tokens while decode correctly sits in prod_allrun is a VALID treatment
    # measurement -- the load-aware design behaving as designed below its decode threshold --
    # and refusing it discards a row that reproduces the published table. What must still be
    # refused is the case: NOTHING routed anywhere, so the measurement describes the
    # production body while claiming to describe the skipper.
    if skip == 0:
        print("\nREFUSING: NOTHING ROUTED in either phase. This measurement describes the "
              "production all-RUN body, not the skipper.")
        return 1
    if share < a.min_skip_share:
        print(f"  NOTE: {share:.1%} of routed work went through the skip body, below the "
              f"{a.min_skip_share:.0%} reference floor. RECORDED, not refused: below the decode "
              "threshold the load-aware design runs prod_allrun by construction, and the phase "
              "that routed is named above. Report the share with the cell.")

    print("\nOK: the skipper executed. This measurement describes the routed system.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
