#!/usr/bin/env python3
"""THE DEFAULT-CONFORMANCE GATE (Gate E restored;): does ONE server run at
SGLang's DEFAULT configuration for its model on its device, except for knobs DECLARED by name?

The cross-arm gate (`verify_cross_arm_config`) asks whether the two arms are comparable to
EACH OTHER. It cannot see what both arms share: on 2026-09-13 every headline cell since v1.3
turned out to carry `dtype float16` on checkpoints published in bf16 and
`mem_fraction_static 0.8` against a computed default of 0.83, both undeclared, and the
cross-arm gate passed every one of them because both arms carried the same values.
 (owner, 2026-08-12) had ordered a default-conformance gate; it was lost in the
vpipe-core extraction. This is that gate, as code, refusing at boot.

"Default" means: what `ServerArgs(model_path=<the model>)` resolves to, computed by the
BASELINE tree's own code on THIS device (so the memory fraction is the device's computed
value, the dtype is the checkpoint's release dtype, the backends are what upstream would
pick). Every field the identity block records is compared; a field may differ only if it is
listed by name in `design.SERVED_LAUNCH_EXEMPTIONS` with the exact served value and the
decision that exempted it. Model identity (path, revision, served name, host, port) is not a
knob and is skipped.

Usage:
    verify_default_conformance.py --server-info <server_info.json> --model-path <model>
        --defaults-tree <upstream tree root> --design-tree <fork tree root>
        [--python <interpreter for the defaults dump>] [--report]

`--report` prints every non-default and exits 0 -- for reading what a server actually does,
never for a measured cell.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# Model identity and per-boot plumbing: not configuration knobs.
NOT_A_KNOB = frozenset({
    "model_path", "tokenizer_path", "revision", "served_model_name", "host", "port",
    "log_level", "log_level_http", "api_key", "download_dir", "random_seed", "nccl_port",
    "dist_init_addr", "log_requests", "show_time_cost", "enable_metrics", "watchdog_timeout",
})

DUMP = r"""
import dataclasses, json, sys
from sglang.srt.server_args import ServerArgs
args = ServerArgs(model_path=sys.argv[1])
print(json.dumps(dataclasses.asdict(args), default=str, sort_keys=True))
"""


def upstream_defaults(defaults_tree: Path, model_path: str, python: str) -> dict[str, Any]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(defaults_tree / "python")
    for key in list(env):
        if key.startswith("SGLANG_") or key.startswith("VP_"):
            # The defaults are what a bare process resolves; an inherited variable would
            # make the reference carry the very drift this gate exists to catch.
            env.pop(key)
    done = subprocess.run([python, "-c", DUMP, model_path], env=env,
                          capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"defaults dump failed under {defaults_tree}:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def _read_declarations(design_tree: Path) -> dict[str, Any] | None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(design_tree / "python")
    code = ("import json; from sglang.srt.vpipe import design; "
            "e = getattr(design, 'SERVED_LAUNCH_EXEMPTIONS', None); "
            "print(json.dumps(None if e is None else {'exemptions': e, "
            "'profiles': design.SERVED_LAUNCH_PROFILES, "
            "'default_profile': design.SERVED_LAUNCH_PROFILE_DEFAULT}, default=str))")
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"could not read the launch declarations from {design_tree}:\n"
                 f"{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def load_exemptions(design_tree: Path, profile: str | None,
                    fallback_design_tree: Path | None = None) -> dict[str, dict[str, Any]]:
    """The declared exemptions that apply under `profile` (None = the design's default profile).

    An exemption carrying a "profile" key applies only under that launch profile; under any
    other profile the field is held to the default. So `sglang_default` cannot inherit the
    paper profile's fp16, and the paper profile cannot smuggle a value it did not declare."""
    decl = _read_declarations(design_tree)
    if decl is None:
        # A tree from before the declarations existed (the measured 735d451c89, booted for tree
        # equivalence): it hard-coded the paper protocol in its own driver, so it is held to the
        # declarations of the tree that runs this gate -- said out loud, never silently.
        if fallback_design_tree is None:
            sys.exit(f"{design_tree} declares no launch profiles and no fallback tree was given")
        print(f"  legacy design tree {design_tree}: no launch declarations -- using those of "
              f"{fallback_design_tree} (paper protocol)")
        decl = _read_declarations(fallback_design_tree)
        if decl is None:
            sys.exit(f"fallback tree {fallback_design_tree} declares no launch profiles either")
    active = profile or decl["default_profile"]
    if active not in decl["profiles"]:
        sys.exit(f"unknown launch profile {active!r}; declared: {sorted(decl['profiles'])}")
    return resolve_exemptions(decl["exemptions"], active)


def resolve_exemptions(exemptions: dict[str, dict[str, Any]], active: str) -> dict[str, dict[str, Any]]:
    """The exemptions in force under `active`, each with ONE resolved value.

    Two rule forms: {"value": v, "profile": p?} applies under p (or everywhere when p is absent);
    {"profiles": {p1: v1, p2: v2}} applies under p1 with v1, under p2 with v2, and NOT under any
    other profile (: fp16 under "paper", bf16 under "paper_bf16", the default elsewhere)."""
    out: dict[str, dict[str, Any]] = {}
    for field, rule in exemptions.items():
        if "profiles" in rule:
            if active in rule["profiles"]:
                out[field] = {k: v for k, v in rule.items() if k != "profiles"} | {"value": rule["profiles"][active], "profile": active}
        elif rule.get("profile") in (None, active):
            out[field] = rule
    return out


def nested_match(served: Any, spec: dict[str, Any]) -> bool:
    """Every dotted path in `spec` resolves inside `served` to the given value; a path ending in
    `[max]` compares the maximum of a list (e.g. "decode.bs[max]": 1024)."""
    for path, want in spec.items():
        node = served
        take_max = path.endswith("[max]")
        for part in path.removesuffix("[max]").split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if take_max:
            if not isinstance(node, list) or not node:
                return False
            node = max(node)
        if not same(node, want):
            return False
    return True


def same(served: Any, default: Any) -> bool:
    if isinstance(default, (int, float)) and isinstance(served, (int, float))\
            and not isinstance(default, bool) and not isinstance(served, bool):
        return abs(float(served) - float(default)) < 1e-9
    return served == default or str(served) == str(default)


def conformance(server_info: dict[str, Any], defaults: dict[str, Any],
                exemptions: dict[str, dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Returns (declared, undeclared) lines. Pure: the testable core."""
    declared: list[str] = []
    undeclared: list[str] = []
    for field in sorted(defaults):
        if field in NOT_A_KNOB or field not in server_info:
            continue
        served, default = server_info[field], defaults[field]
        if same(served, default):
            continue
        rule = exemptions.get(field)
        if rule is not None and "accept_if" in rule and nested_match(served, rule["accept_if"]):
            # a structured field (e.g. cuda_graph_config) declared by the properties that the
            # decision fixes, not by its whole served value (the bucket list is derived)
            declared.append(f"  = {field}: served satisfies {rule['accept_if']!r} default={default!r} "
                            f"[{rule['decision']}]")
        elif rule is not None and "accept_if" not in rule and same(served, rule["value"]):
            declared.append(f"  = {field}: served={served!r} default={default!r} "
                            f"[{rule['decision']}]")
        else:
            undeclared.append(f"  ! {field}: served={served!r} default={default!r}"
                              + (f" (exempted value is {rule.get('value', rule.get('accept_if'))!r}, not this)"
                                 if rule is not None else ""))
    return declared, undeclared


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-info", required=True, type=Path)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--defaults-tree", required=True, type=Path,
                    help="the BASELINE (upstream) tree whose ServerArgs defines 'default'")
    ap.add_argument("--design-tree", required=True, type=Path,
                    help="the fork tree whose design.py declares the exemptions")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--fallback-design-tree", type=Path, default=None,
                    help="declarations to use when --design-tree predates launch profiles (legacy tree)")
    ap.add_argument("--launch-profile", default=None,
                    help="the launch profile the server was booted under (design.SERVED_LAUNCH_PROFILES); "
                         "default: the design's default profile")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--arm-exemptions", default=None,
                    help="JSON {field: {value|accept_if, decision}} declared for ONE upstream-served arm "
                         "launched with extra arguments (the same-ladder control, 2026-09-16)")
    args = ap.parse_args()

    info = json.loads(args.server_info.read_text())
    defaults = upstream_defaults(args.defaults_tree, args.model_path, args.python)
    exemptions = load_exemptions(args.design_tree, args.launch_profile, args.fallback_design_tree)
    if args.arm_exemptions:
        extra = json.loads(args.arm_exemptions)
        for field, rule in extra.items():
            if "decision" not in rule or not ({"value", "accept_if"} & set(rule)):
                sys.exit(f"arm exemption for {field!r} needs a decision and a value/accept_if")
        exemptions = dict(exemptions) | extra
    declared, undeclared = conformance(info, defaults, exemptions)

    print(f"default conformance: {len(declared)} declared, {len(undeclared)} undeclared "
          f"(profile {args.launch_profile or 'default'}; defaults from {args.defaults_tree}, "
          f"exemptions from design.py)")
    for line in declared + undeclared:
        print(line)
    if undeclared and not args.report:
        print("REFUSED: a served value differs from the default and no decision exempts it")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
