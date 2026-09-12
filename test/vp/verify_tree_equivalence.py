#!/usr/bin/env python3
"""Boot the same arm from two trees and prove the SERVED DESIGN is identical.

Why this exists. A refactor that is "behaviour-preserving by construction" is an
argument, and this project's standing rule is that an argument is not a
measurement. D-702 split the projector out of the policy and opened the skipper
registry; it was written so that every consumer keeps evaluating the same
predicate and no attestation field moves. That claim is checkable in about
twenty minutes of card time, and until it is checked the paper is describing an
interface the measured binary does not have.

The comparison basis is `stable_server_identity()` -- the same function the
cross-arm gate uses -- because it already strips the fields that advance with
traffic (route counters, `branch_counts`, `realized`). Two servers booted from
two trees, no requests sent, so ANY difference is a code difference.

  verify_tree_equivalence.py --spec quality_spec.json \\
      --tree-a /opt/vpipe/trees/tree-318bc23929 \\
      --tree-b /opt/vpipe/trees/tree-57434b28d5 \\
      --arm vskipper --arm integrated_randomskip --port 32097

Exits non-zero and PRINTS THE DIFFERING PATHS on any difference. A pass here does
not license re-using a measurement across the change; it licenses saying the two
trees serve the same design, which is a narrower and checkable claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qps_deployment import stable_server_identity  # noqa: E402
from run_paired_campaign import Server, log, server_info  # noqa: E402


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Leaf paths, so a diff names the field rather than dumping two blobs."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value):
            out.update(flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list):
        out = {}
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{index}]"))
        return out or {prefix: "[]"}
    return {prefix: value}


def identity_for(spec: dict[str, Any], tree: str, arm: str, port: int,
                 log_path: Path) -> dict[str, Any]:
    """Boot one arm from one tree, read /server_info, tear down."""
    local = dict(spec)
    local["tree"] = tree
    with Server(local, arm, port, log_path):
        return stable_server_identity(server_info(port))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--tree-a", required=True, help="the tree that was MEASURED")
    ap.add_argument("--tree-b", required=True, help="the tree that was REFACTORED")
    ap.add_argument("--arm", action="append", default=[], required=True)
    ap.add_argument("--port", type=int, default=32097)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--allow", action="append", default=[],
        help="leaf path permitted to differ. Empty is the point: a "
             "behaviour-preserving refactor should need none, and each entry "
             "here is a claim that must be argued in the log, not a knob.",
    )
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    for role, tree in (("a", args.tree_a), ("b", args.tree_b)):
        if not (Path(tree) / "python/sglang/srt/vpipe").is_dir():
            print(f"FATAL: --tree-{role} {tree} has no vpipe package", file=sys.stderr)
            return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    report: dict[str, Any] = {"tree_a": args.tree_a, "tree_b": args.tree_b, "arms": {}}
    for arm in args.arm:
        log(f"=== {arm} ===")
        identities = {}
        for role, tree in (("a", args.tree_a), ("b", args.tree_b)):
            log(f"  booting {arm} from tree-{role}: {tree}")
            identities[role] = identity_for(
                spec, tree, arm, args.port, args.out_dir / f"server.{arm}.{role}.log"
            )
            (args.out_dir / f"identity.{arm}.{role}.json").write_text(
                json.dumps(identities[role], indent=2, sort_keys=True) + "\n"
            )
        flat_a, flat_b = flatten(identities["a"]), flatten(identities["b"])
        allow = set(args.allow)
        differing = sorted(
            path for path in set(flat_a) | set(flat_b)
            if flat_a.get(path, "<absent>") != flat_b.get(path, "<absent>")
            and path not in allow
        )
        report["arms"][arm] = {
            "fields_compared": len(set(flat_a) | set(flat_b)),
            "differing": differing,
            "identical": not differing,
        }
        if differing:
            failures += 1
            log(f"  REFUSED: {len(differing)} field(s) differ between the trees")
            for path in differing[:40]:
                log(f"    {path}: a={flat_a.get(path, '<absent>')!r} "
                    f"b={flat_b.get(path, '<absent>')!r}")
        else:
            log(f"  IDENTICAL across {len(flat_a)} compared fields")

    written = args.out_dir / "tree_equivalence.json"
    written.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"--- {len(args.arm)} arm(s), {failures} differing — {written} ---")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
