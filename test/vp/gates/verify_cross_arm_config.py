#!/usr/bin/env python3
"""THE CROSS-ARM GATE: are the two arms the same engine apart from the treatment?

Every other config gate in this project checks ONE arm against its OWN intent.
`verify_served_design` asks "is this server serving the design the tree declares?" and
`campaign_preflight` asks "is the staged tree the tree we think it is". Neither can see the
question a paired table actually rests on: **are the two arms comparable to each other?**

That question used to be nearly free, because both arms were the same binary differing only
by environment -- a field-level diff then found exactly two differing fields (the treatment,
and `max_total_num_tokens` at -0.98%). It stopped being free the moment the baseline became
GENUINE UPSTREAM SGLANG (D-587/D-646): the arms now run from two different source trees, 193
commits apart, and identical CLI flags no longer imply identical resolved configuration.
Upstream may default differently on the scheduler, chunked prefill, the CUDA-graph ladder or
KV capacity, and every one of those moves throughput without touching the treatment.

So this diffs the two arms' `server_identity` blocks and REFUSES on any difference outside a
DECLARED allowlist. The allowlist is evidence, not assumption: run `--report` once, read what
actually differs, and classify each field as

  (a) the treatment            -- `vp_runtime`, present on the routed arm and absent upstream
  (b) a DERIVED consequence    -- e.g. `max_total_num_tokens`, ~1% lower on the treatment arm
                                  because the router/projector weights occupy HBM. Declared,
                                  and it costs the TREATMENT capacity, so it cannot flatter us
  (c) a defect                 -- fix it before measuring, do not allowlist it

Usage:
    verify_cross_arm_config.py --arm upstream=<manifest.json> \\
                               --arm integrated_it4=<manifest.json> \\
                               [--allow vp_runtime --allow server.max_total_num_tokens] \\
                               [--report]

`--report` prints the differences and exits 0; it is for BUILDING the allowlist, never for a
measured cell. Without it, an undeclared difference exits non-zero and names the field.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted-path leaves, so a diff can name the exact field rather than a subtree."""
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in sorted(value):
            out.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        # A list is compared whole: element order is meaningful for a config (a CUDA-graph
        # batch-size ladder is not a set), and per-index paths would be noise.
        out[prefix] = json.dumps(value, sort_keys=True)
    else:
        out[prefix] = value
    return out


def load_identity(spec: str) -> tuple[str, dict[str, Any]]:
    if "=" not in spec:
        sys.exit(f"--arm expects NAME=path/to/deployment_manifest.json, got {spec!r}")
    name, path = spec.split("=", 1)
    manifest = json.loads(Path(path).read_text())
    identity = manifest.get("server_identity")
    if identity is None:
        sys.exit(f"{name}: {path} carries no server_identity -- it cannot be compared")
    return name, flatten(identity)


def covered(field: str, allow: list[str]) -> bool:
    """A prefix allows its whole subtree: `vp_runtime` covers `vp_runtime.served_design.arm`."""
    return any(field == a or field.startswith(a + ".") for a in allow)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=MANIFEST")
    ap.add_argument(
        "--allow",
        action="append",
        default=[],
        help="dotted field (or prefix) whose difference is DECLARED. Each one is a claim you "
             "are making about the comparison; state why in the campaign record.",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="print differences and exit 0. For BUILDING the allowlist from evidence -- "
             "never for a measured cell, where an undeclared difference must refuse.",
    )
    args = ap.parse_args()
    if len(args.arm) != 2:
        ap.error(f"exactly two --arm are required, got {len(args.arm)}")

    (name_a, a), (name_b, b) = (load_identity(s) for s in args.arm)

    differing: list[tuple[str, Any, Any]] = []
    for field in sorted(set(a) | set(b)):
        va, vb = a.get(field, "<absent>"), b.get(field, "<absent>")
        if va != vb:
            differing.append((field, va, vb))

    undeclared = [d for d in differing if not covered(d[0], args.allow)]

    print(f"cross-arm config: {name_a}  vs  {name_b}")
    print(f"  fields compared : {len(set(a) | set(b))}")
    print(f"  differing       : {len(differing)}")
    print(f"  declared        : {len(differing) - len(undeclared)}  (allow: {args.allow or 'none'})")
    print(f"  UNDECLARED      : {len(undeclared)}")
    if differing:
        print()
        for field, va, vb in differing:
            mark = "    " if covered(field, args.allow) else "  ! "
            print(f"{mark}{field}")
            print(f"        {name_a}: {str(va)[:110]}")
            print(f"        {name_b}: {str(vb)[:110]}")

    if args.report:
        print("\n--report: exiting 0 without judging. Classify each field as the treatment, a")
        print("declared derived consequence, or a defect -- then run WITHOUT --report.")
        return 0
    if undeclared:
        print(f"\nREFUSED: {len(undeclared)} undeclared difference(s) between the arms.")
        print("A paired table compares two engines; an undeclared configuration difference is")
        print("an uncontrolled variable in every cell it produces. Either fix it, or declare")
        print("it with --allow and record WHY it cannot flatter the treatment.")
        return 1
    print("\nOK: the arms differ only in declared fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
