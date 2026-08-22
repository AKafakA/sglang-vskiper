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
    return sorted(
        f"sglang.srt.vpipe.{p.stem}"
        for p in PKG.glob("*.py")
        if p.stem != "__init__"
    )


def import_one(mod):
    """Import in a FRESH subprocess so import order cannot mask a failure."""
    r = subprocess.run(
        [sys.executable, "-c", f"import {mod}"],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin",
             "HOME": "/tmp", "SGLANG_IS_FLASHINFER_AVAILABLE": "false"},
    )
    return r.returncode, (r.stderr or "").strip().splitlines()[-1:] or [""]


def main():
    mods = module_names()
    if len(mods) < 20:
        print(f"FATAL: expected the vpipe package at {PKG}, found {len(mods)} modules")
        return 2
    failed = []
    for mod in mods:
        rc, tail = import_one(mod)
        if rc != 0:
            failed.append((mod, tail[0][:100]))
    for mod, err in failed:
        print(f"  IMPORT FAILED {mod}: {err}")
    print(f"PACKAGE IMPORTS: {'FAIL' if failed else 'PASS'} "
          f"({len(mods) - len(failed)}/{len(mods)} modules)")
    return 1 if failed else 0


def test_every_vpipe_module_imports():
    mods = module_names()
    assert len(mods) >= 20, f"expected the vpipe package at {PKG}, found {len(mods)}"
    failed = {m: import_one(m)[1][0][:120] for m in mods if import_one(m)[0] != 0}
    assert not failed, f"modules that fail to import: {failed}"


if __name__ == "__main__":
    sys.exit(main())
