#!/usr/bin/env python3
"""THE DEFAULT-CONFORMANCE GATE (Gate E, D-184 restored; D-757): does ONE server run at
SGLang's DEFAULT configuration for its model on its device, except for knobs DECLARED by name?

The cross-arm gate (`verify_cross_arm_config`) asks whether the two arms are comparable to
EACH OTHER. It cannot see what both arms share: on 2026-09-13 every headline cell since v1.3
turned out to carry `dtype float16` on checkpoints published in bf16 and
`mem_fraction_static 0.8` against a computed default of 0.83, both undeclared, and the
cross-arm gate passed every one of them because both arms carried the same values.
D-184 (owner, 2026-08-12) had ordered a default-conformance gate; it was lost in the
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


def load_exemptions(design_tree: Path, profile: str | None) -> dict[str, dict[str, Any]]:
    """The declared exemptions that apply under `profile` (None = the design's default profile).

    An exemption carrying a "profile" key applies only under that launch profile; under any
    other profile the field is held to the default. So `sglang_default` cannot inherit the
    paper profile's fp16, and the paper profile cannot smuggle a value it did not declare."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(design_tree / "python")
    code = ("import json; from sglang.srt.vpipe import design; "
            "print(json.dumps({'exemptions': design.SERVED_LAUNCH_EXEMPTIONS, "
            "'profiles': design.SERVED_LAUNCH_PROFILES, "
            "'default_profile': design.SERVED_LAUNCH_PROFILE_DEFAULT}, default=str))")
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"could not read the launch declarations from {design_tree}:\n"
                 f"{done.stderr[-2000:]}")
    decl = json.loads(done.stdout.strip().splitlines()[-1])
    active = profile or decl["default_profile"]
    if active not in decl["profiles"]:
        sys.exit(f"unknown launch profile {active!r}; declared: {sorted(decl['profiles'])}")
    return {field: rule for field, rule in decl["exemptions"].items()
            if rule.get("profile") in (None, active)}


def same(served: Any, default: Any) -> bool:
    if isinstance(default, (int, float)) and isinstance(served, (int, float)) \
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
        if rule is not None and same(served, rule["value"]):
            declared.append(f"  = {field}: served={served!r} default={default!r} "
                            f"[{rule['decision']}]")
        else:
            undeclared.append(f"  ! {field}: served={served!r} default={default!r}"
                              + (f" (exempted value is {rule['value']!r}, not this)"
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
    ap.add_argument("--launch-profile", default=None,
                    help="the launch profile the server was booted under (design.SERVED_LAUNCH_PROFILES); "
                         "default: the design's default profile")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    info = json.loads(args.server_info.read_text())
    defaults = upstream_defaults(args.defaults_tree, args.model_path, args.python)
    exemptions = load_exemptions(args.design_tree, args.launch_profile)
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
