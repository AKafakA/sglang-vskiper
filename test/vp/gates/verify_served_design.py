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
    verify_served_design.py --url http://127.0.0.1:30000 --arm vskipper --tree <path>
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


def intended(tree: Path, arm: str) -> dict[str, Any]:
    """What a CLEAN process resolves for this arm, from the deployed tree.

    [Codex F8] The intended state is not "the constants" -- the constants pass through
    arm gating (``mechanism()``, ``arm_phases()``, per-leg pruning) before they become
    what is served. Re-implementing that gating here would create a second copy free to
    drift from the first, which is the same class of bug as the one being fixed.

    So the gate resolves through the DEPLOYED TREE'S OWN resolvers, in an environment
    scrubbed of every ``SGLANG_*`` value. Intended = what a clean process resolves;
    served = what the server actually resolved. A difference means the server's
    environment, arm file, or code diverged -- which is the whole question.
    """

    import os

    scrubbed = {
        k: v
        for k, v in os.environ.items()
        # the host-config POINTER survives: it names a committed file, it is not a value
        if not k.startswith("SGLANG_") or k == "SGLANG_VP_HOST_CONFIG"
    }
    saved = dict(os.environ)
    sys.path.insert(0, str(tree / "python"))
    try:
        os.environ.clear()
        os.environ.update(scrubbed)
        # Select the arm IN MEMORY. A verification tool must never write to the tree it
        # is verifying -- an interrupted run would leave deploy/active_arm rewritten.
        from sglang.srt.vpipe import design as _d

        _d._ARM_CACHE.clear()
        _d._ARM_CACHE["name"] = arm
        from sglang.srt.vpipe.common import resolved_design_attestation

        return resolved_design_attestation()
    finally:
        os.environ.clear()
        os.environ.update(saved)
        sys.path.remove(str(tree / "python"))


def diff(expected: Any, observed: Any, path: str = "") -> list[str]:
    """Recursive comparison that reports EVERY difference, not just the first."""
    out: list[str] = []
    if isinstance(expected, dict) and isinstance(observed, dict):
        for key in sorted(set(expected) | set(observed)):
            if key not in expected:
                # [Codex F11] An EXTRA served field is a mismatch, not a courtesy. The
                # first version skipped unknown keys, so a served prefill.max_tokens=6144
                # -- the exact knob D-596 deleted -- produced ZERO differences.
                out.append(f"  {path}/{key}: SERVED BUT NOT INTENDED = {observed[key]!r}")
                continue
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
    served = vp.get("served_design")
    served_arm = None if served is None else served.get("arm")

    problems: list[str] = []
    if served is None or served_arm is None:
        problems.append(
            "  the server does not publish vp_runtime.served_design -- it predates the "
            "resolved-state gate (D-611); redeploy the current tree before any cell"
        )
    else:
        if served_arm != args.arm:
            problems.append(f"  arm: intended {args.arm!r} != served {served_arm!r}")
        problems.extend(diff(intended(args.tree, args.arm), served, "served_design"))

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
