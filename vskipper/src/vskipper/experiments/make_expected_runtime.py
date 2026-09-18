#!/usr/bin/env python3
"""Emit the per-arm expected-runtime file the campaign gate reads back after boot.

Why this is a tool and not twelve hand-written files. `run_paired_campaign.py` refuses any
arm without `<expect_dir>/expect_<arm>.json`, and those files had only ever been written by
hand -- so only four existed, for the four arms that had been run. The moment the RandomSkip
sweep added twelve arms, all twelve would have been REFUSED at preflight, after the headline
had already spent seven hours of GPU, with a message about a missing file rather than about
the design. (`integrated_randomskip` itself has no expect file either, and has therefore never
been run through this driver.)

The content is fully determined by `design.py`, so deriving it removes the failure mode
entirely: a new arm in ARMS gets its expectation for free, and an expectation can no longer
disagree with the design it is supposed to check.

  make_expected_runtime.py --out-dir /opt/vpipe/campaign            # every arm
  make_expected_runtime.py --out-dir DIR --arm vskipper       # one
  make_expected_runtime.py --out-dir DIR --check                    # diff, write nothing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _arms() -> dict:
    """Load ARMS from design.py BY PATH.

    Importing `vskipper.runtime.design` drags in the whole serving package (orjson, torch,
    ...), which is not installed on the code host and must not be -- local is for source and
    checksums only. design.py's arm table is pure data, so the gate tests already load it this
    way and so does this tool.
    """
    import importlib.util

    path = ROOT / "vskipper/src/vskipper/runtime/design.py"
    spec = importlib.util.spec_from_file_location("_vp_design_for_expectations", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ARMS


ARMS = _arms()

# "both" is stored as one token but attested as the sorted phase list the server reports.
_PHASES = {"both": ["decode", "prefill"], "decode": ["decode"], "prefill": ["prefill"],
           None: []}


def expectation(arm: str) -> dict:
    """What `/server_info` must report for this arm, derived from the design."""
    if arm == "upstream" or arm.startswith("upstream_"):  # every upstream-served arm (incl. the same-ladder control) has no vpipe attestation
        # Genuine upstream SGLang has no vpipe package, so there is no attestation block at
        # all. Its ABSENCE is the assertion (D-587/D-646).
        return {"attestation": "absent"}
    spec = ARMS[arm]
    return {
        "attestation": "present",
        "runtime": {
            "served_design": {
                "arm": arm,
                "skipper": spec["skipper"],
                "active_phases": _PHASES[spec["phases"]],
            }
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arm", action="append", default=[],
                    help="limit to these arms (default: every arm, plus upstream)")
    ap.add_argument("--check", action="store_true",
                    help="compare against what is on disk and write nothing")
    args = ap.parse_args()

    arms = args.arm or (["upstream"] + sorted(ARMS))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    differs = 0
    for arm in arms:
        want = expectation(arm)
        path = args.out_dir / f"expect_{arm}.json"
        if path.is_file():
            have = json.loads(path.read_text())
            if have == want:
                print(f"  same    {path.name}")
                continue
            differs += 1
            print(f"  DIFFERS {path.name}\n     on disk: {json.dumps(have)}\n     derived: {json.dumps(want)}")
            if args.check:
                continue
        else:
            print(f"  {'MISSING' if args.check else 'new    '} {path.name}: {json.dumps(want)}")
            differs += 1
        if not args.check:
            path.write_text(json.dumps(want, indent=2) + "\n")
    if args.check and differs:
        print(f"\n{differs} expectation(s) missing or disagreeing with design.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
