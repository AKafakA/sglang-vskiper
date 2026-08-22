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


def flashinfer_available():
    """Probe the DEPENDENCY directly. Never infer it from a message.

    The previous version classified any failure whose final stderr line merely
    CONTAINED "flashinfer" as an environment limitation and still reported PASS.
    A real defect -- `NameError: name 'flashinfer' is not defined` -- matched
    that substring and was exempted. A substring is not evidence about the
    environment; importing the dependency is.
    """
    r = subprocess.run(
        [sys.executable, "-c", "import flashinfer"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    return r.returncode == 0


# Exit statuses. UNVERIFIED is deliberately NOT success: a release gate must not
# treat "we could not check the production default posture" as "we checked it".
OK, FAILED, UNVERIFIED, BROKEN = 0, 1, 3, 2


def run_all(verbose=True):
    mods = module_names()
    if len(mods) < 20:
        if verbose:
            print(f"FATAL: expected the vpipe package at {PKG}, found {len(mods)} modules")
        return BROKEN, mods, {}
    have_fi = flashinfer_available()
    status = OK
    results = {}
    for label, env in POSTURES:
        _, failed = sweep(env)
        results[label] = failed
        ok = len(mods) - len(failed)
        if not failed:
            if verbose:
                print(f"  [{label}] PASS ({ok}/{len(mods)})")
            continue
        if verbose:
            for mod, err in failed:
                print(f"  [{label}] IMPORT FAILED {mod}: {err}")
        if label == "repository-default" and not have_fi:
            # The dependency really is absent on this host, proven by probe, so
            # this posture is UNVERIFIED rather than failed -- and unverified
            # is still not a pass.
            if verbose:
                print(f"  [{label}] UNVERIFIED: flashinfer is not importable on "
                      f"this host, so the production-default posture was not "
                      f"checked ({len(failed)} module(s) affected)")
            status = max(status, UNVERIFIED)
        else:
            if verbose:
                print(f"  [{label}] FAIL ({ok}/{len(mods)})")
            status = FAILED
    if verbose:
        word = {OK: "PASS", FAILED: "FAIL", UNVERIFIED: "UNVERIFIED"}[status]
        print(f"  flashinfer importable on this host: {have_fi}")
        print(f"PACKAGE IMPORTS: {word} ({len(mods)} modules, {len(POSTURES)} postures)")
    return status, mods, results


def main():
    return run_all()[0]


def test_every_vpipe_module_imports_flashinfer_disabled():
    mods = module_names()
    assert len(mods) >= 20, f"expected the vpipe package at {PKG}, found {len(mods)}"
    _, failed = sweep(dict(POSTURES[0][1]))
    assert not failed, f"modules that fail to import: {dict(failed)}"


def test_every_vpipe_module_imports_repository_default():
    """The production-default posture, which pytest previously never ran.

    Skips ONLY when the dependency is provably absent, established by importing
    it -- not by pattern-matching an error message.
    """
    import pytest

    if not flashinfer_available():
        pytest.skip("flashinfer not importable on this host; posture unverifiable")
    _, failed = sweep(dict(POSTURES[1][1]))
    assert not failed, f"modules that fail to import: {dict(failed)}"


if __name__ == "__main__":
    sys.exit(main())
