"""Ground truth: every vpipe module must actually import.

test_module_import_safety.py approximates this statically because the package's
dependencies are not installed everywhere. That approximation has been wrong in
six distinct ways -- class-body annotations, branch scopes, aug-assign targets,
walrus ordering, try/else paths, lambda parameters -- and each fix exposed the
next. Re-implementing Python's scoping rules is an unbounded surface.

So this is the AUTHORITATIVE gate and the static one is a fast pre-check. It
imports every module for real, in a subprocess, and reports what actually
happened. Nothing here can be wrong about Python's semantics, because it does
not model them.

Requires the package's dependencies, so it runs on a GPU host, not on the
source-edit box.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "python" / "sglang" / "srt" / "vpipe"


def module_names():
    """Every module RECURSIVELY, so a future subpackage cannot escape the sweep.

    A flat glob could never discover modules moved beneath a subpackage, while
    the count guard would still be satisfied by whatever remained at the top.
    Nested __init__ files are included; they execute on import of anything
    beneath them.
    """
    names = []
    for p in sorted(PKG.rglob("*.py")):
        rel = p.relative_to(PKG).with_suffix("")
        parts = [x for x in rel.parts if x != "__init__"]
        names.append(".".join(["sglang.srt.vpipe", *parts]) if parts
                     else "sglang.srt.vpipe")
    return sorted(set(names))


# Two postures. The repository DEFAULT is FlashInfer-available; forcing it off
# for every import would hide a failure confined to the normal deployment path.
# The disabled posture is what this sm75 box can actually run, so it is the
# gating one here, and the default posture is reported alongside with its cause
# distinguished -- an unavailable FlashInfer is an environment limit on this
# host, anything else is a real defect.
POSTURES = (
    ("flashinfer-disabled", {"SGLANG_IS_FLASHINFER_AVAILABLE": "false"}),
    ("repository-default", {}),
)


def import_one(mod, extra_env):
    """Import in a FRESH subprocess so import order cannot mask a failure."""
    env = {"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin",
           "HOME": "/tmp"}
    env.update(extra_env)
    r = subprocess.run(
        [sys.executable, "-c", f"import {mod}"],
        capture_output=True, text=True, env=env,
    )
    return r.returncode, (r.stderr or "").strip().splitlines()[-1:] or [""]


def sweep(extra_env):
    mods = module_names()
    failed = []
    for mod in mods:
        rc, tail = import_one(mod, extra_env)
        if rc != 0:
            failed.append((mod, tail[0][:110]))
    return mods, failed


def main():
    mods = module_names()
    if len(mods) < 20:
        print(f"FATAL: expected the vpipe package at {PKG}, found {len(mods)} modules")
        return 2
    rc = 0
    for label, env in POSTURES:
        mods, failed = sweep(env)
        for mod, err in failed:
            print(f"  [{label}] IMPORT FAILED {mod}: {err}")
        ok = len(mods) - len(failed)
        if not failed:
            print(f"  [{label}] PASS ({ok}/{len(mods)})")
            continue
        flashinfer_only = all("flashinfer" in e.lower() for _, e in failed)
        if label == "repository-default" and flashinfer_only:
            print(f"  [{label}] ENVIRONMENT: FlashInfer unavailable on this host "
                  f"({len(failed)} module(s)); not a code defect, but this posture "
                  f"is UNVERIFIED here")
        else:
            print(f"  [{label}] FAIL ({ok}/{len(mods)})")
            rc = 1
    print(f"PACKAGE IMPORTS: {'FAIL' if rc else 'PASS'} ({len(mods)} modules, "
          f"{len(POSTURES)} postures)")
    return rc


def test_every_vpipe_module_imports():
    mods = module_names()
    assert len(mods) >= 20, f"expected the vpipe package at {PKG}, found {len(mods)}"
    _, failed = sweep(dict(POSTURES[0][1]))
    assert not failed, f"modules that fail to import: {dict(failed)}"


if __name__ == "__main__":
    sys.exit(main())
