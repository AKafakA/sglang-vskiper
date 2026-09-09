#!/usr/bin/env python3
"""THE INPUT GATE: refuse to run a campaign unless the server is serving the design.

[D-609/D-614] Every gate this project owns checks the OUTPUT of a measurement -- GR-1a work
identity, GR-2 tables, Accounting-v5, bank audits, straddle rules. **None of them checks that
the right system was measured.** On 2026-09-09 roughly eighteen hours of A100 time (~$15) went
into a design the owner had rejected: the intended value was pushed in through
``EXTRA_SERVER_ENV``, never reached the server, and every output gate passed anyway. A fully
green campaign on the wrong system is indistinguishable from a fully green campaign on the
right one.

So this runs BEFORE the first cell, against the LIVE server, and compares what it is actually
serving with the design constants in the deployed tree. It reads back ``observed_runtime``
(via ``/server_info``); it never inspects the environment, and it never trusts the value the
launcher set -- that value is exactly what lied.

Fails closed: any mismatch, any missing field, any unreachable server is a non-zero exit. A
campaign script must treat a non-zero exit as "do not run", not as a warning.

Usage:
    verify_served_design.py --url http://127.0.0.1:30000 --arm integrated_it4 --tree <path>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


def load_design(tree: Path):
    """Import the DEPLOYED tree's design.py by path.

    By path, not by package name: the gate must describe the tree that is being served, not
    whichever sglang happens to be importable in the calling interpreter. Those differ exactly
    when it matters most.
    """
    path = tree / "python" / "sglang" / "srt" / "vpipe" / "design.py"
    if not path.is_file():
        sys.exit(f"FATAL: no design.py in the deployed tree: {path}")
    spec = importlib.util.spec_from_file_location("_served_design", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/server_info", timeout=30) as fh:
            return json.load(fh)
    except Exception as exc:  # unreachable server is a FAILURE, never a skip
        sys.exit(f"FATAL: cannot read {url}/server_info -- {exc}")


def find_vp_runtime(info: dict[str, Any]) -> dict[str, Any]:
    states = info.get("internal_states") or []
    if not states:
        sys.exit("FATAL: /server_info carries no internal_states -- cannot verify anything")
    vp = states[0].get("vp_runtime")
    if vp is None:
        sys.exit(
            "FATAL: /server_info has no vp_runtime block. Either this is not a vPipe tree, or "
            "the attestation is not wired -- and an unwired attestation is exactly the hole "
            "this gate exists to close (D-614)."
        )
    return vp


def diff(expected: Any, observed: Any, path: str = "") -> list[str]:
    """Recursive comparison that reports EVERY difference, not just the first."""
    out: list[str] = []
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed)):
            if key not in expected:
                continue  # the server may attest more than the design pins; that is fine
            if key not in observed:
                out.append(f"  {path}/{key}: MISSING from the served attestation")
                continue
            out.extend(diff(expected[key], observed[key], f"{path}/{key}"))
    elif isinstance(expected, (list, tuple)) and isinstance(observed, (list, tuple)):
        if list(expected) != list(observed):
            out.append(f"  {path}: intended {list(expected)!r} != served {list(observed)!r}")
    elif expected != observed:
        out.append(f"  {path}: intended {expected!r} != served {observed!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL of the running server")
    ap.add_argument("--arm", required=True, help="the arm this campaign intends to measure")
    ap.add_argument("--tree", required=True, type=Path, help="the DEPLOYED source tree")
    args = ap.parse_args()

    design = load_design(args.tree)
    design.resolve_arm(args.arm)  # fails closed on an unknown name

    vp = find_vp_runtime(fetch(args.url))
    served_arm = vp.get("arm")
    served_design = vp.get("design")

    problems: list[str] = []
    if served_arm is None or served_design is None:
        problems.append(
            "  the server does not publish 'arm'/'design' -- it predates the input gate "
            "(D-614); redeploy the current tree before running a campaign"
        )
    else:
        if served_arm != args.arm:
            problems.append(f"  arm: intended {args.arm!r} != served {served_arm!r}")
        problems.extend(diff(design.design_attestation(), served_design, "design"))

    print(f"intended arm : {args.arm}")
    print(f"served arm   : {served_arm}")
    print(f"tree         : {args.tree}")
    if problems:
        print("\nSERVED SYSTEM DOES NOT MATCH THE INTENDED DESIGN:")
        print("\n".join(problems))
        print("\nREFUSING. Do not run cells against this server (D-609).")
        return 1
    print("\nOK: the live server is serving the deployed design. Campaign may proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
