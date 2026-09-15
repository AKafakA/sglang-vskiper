#!/usr/bin/env python3
"""The two-tree equivalence gate, pinned on the three ways it could lie.

It exists because "behaviour-preserving by construction" is an argument, and an
argument is not a measurement. Three properties make the check worth
running at all:

  1. it compares `stable_server_identity`, the SAME basis the cross-arm gate
     uses, so the fields that advance with traffic are already stripped and a
     difference is a code difference;
  2. it sends NO requests -- both servers are read at boot, so nothing the
     traffic does can mask or manufacture a difference;
  3. the allowlist DEFAULTS TO EMPTY. A gate whose default is permissive passes
     on the day it matters (: Gate B passed for weeks on lanes differing by
     exactly the token it stripped).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = ROOT / "test/vp/verify_tree_equivalence.py"
GATE = GATE_PATH.read_text()


def _code_only(text: str) -> str:
    module = ast.parse(text)
    body = module.body[1:] if ast.get_docstring(module) else module.body
    first = body[0].lineno - 1 if body else len(text.splitlines())
    return "\n".join(
        l for l in text.splitlines()[first:] if not l.strip().startswith("#")
    )


CODE = _code_only(GATE)


def test_it_compares_the_same_basis_as_the_cross_arm_gate():
    assert "from qps_deployment import stable_server_identity" in CODE
    assert "stable_server_identity(server_info(port))" in CODE


def test_no_traffic_is_sent():
    """A boot-time read; anything that generates load would mask a difference."""
    for forbidden in ("bench_serving", "run_qps_evaluation", "requests.post", "/generate"):
        assert forbidden not in CODE, forbidden


def test_the_allowlist_defaults_to_empty():
    idx = CODE.index('"--allow"')
    body = CODE[idx: idx + 200]
    assert "default=[]" in body


def test_it_boots_the_same_arm_from_both_trees():
    """The arm is held fixed and the TREE varies -- the other way round is a
    different question (that is the cross-arm gate's job)."""
    assert 'local["tree"] = tree' in CODE
    assert 'for role, tree in (("a", args.tree_a), ("b", args.tree_b))' in CODE


def test_a_difference_names_the_FIELD_not_a_blob():
    assert "def flatten" in CODE
    assert "a={flat_a.get(path, '<absent>')!r} " in CODE or "flat_a.get(path" in CODE


def test_it_exits_non_zero_on_any_difference():
    assert "return 1 if failures else 0" in CODE


def test_the_boot_path_is_imported_not_reimplemented():
    assert "from run_paired_campaign import Server" in CODE
    assert "launch_server" not in CODE


def test_it_refuses_a_tree_without_a_vpipe_package(tmp_path: Path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"tree": str(tmp_path), "python": sys.executable}))
    (tmp_path / "empty").mkdir()
    done = subprocess.run(
        [sys.executable, str(GATE_PATH), "--spec", str(spec),
         "--tree-a", str(tmp_path / "empty"), "--tree-b", str(tmp_path / "empty"),
         "--arm", "integrated_it4", "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "has no vpipe package" in done.stderr
    assert not (tmp_path / "out").exists(), "it created output before refusing"
