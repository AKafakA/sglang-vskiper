#!/usr/bin/env python3
"""The preflight validates the host config the RUN names -- never one it chose itself.

`campaign_preflight.py` used to glob `deploy/hosts/*.json`, sort, and check `hosts[0]`. On
2026-09-10 the paired smoke ran on the Vast A100 and was refused for missing
`${VSKIPPER_DATA_ROOT}/...` paths: it had validated `hpc.json`, first alphabetically, on a box that
is not the HPC cluster, while the campaign contract named `deploy/hosts/a100.json`.

The false refusal was the harmless half. Had the first config's paths happened to exist, the
gate would have PASSED while certifying a host config the run does not use -- the same shape
as, where a value never reached the server and every gate went green.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GATE = ROOT / "test/vp/gates/campaign_preflight.py"


def _tree(tmp_path: Path, host_files: dict[str, dict]) -> Path:
    tree = tmp_path / "tree"
    (tree / "deploy" / "hosts").mkdir(parents=True)
    (tree / "python").mkdir(parents=True)
    (tree / "deploy" / "active_arm").write_text("integrated_it4\n")
    for name, body in host_files.items():
        (tree / "deploy" / "hosts" / f"{name}.json").write_text(json.dumps(body))
    return tree


def _run(tree: Path, workdir: Path, host_config: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GATE), "--tree", str(tree), "--upstream", str(tree),
         "--workdir", str(workdir), "--host-config", host_config],
        capture_output=True, text=True,
    )


def test_the_host_config_is_REQUIRED_not_defaulted():
    """A default would reintroduce the gate choosing its own subject."""
    done = subprocess.run(
        [sys.executable, str(GATE), "--tree", ".", "--upstream", ".", "--workdir", "."],
        capture_output=True, text=True,
    )
    assert done.returncode != 0
    assert "--host-config" in done.stderr


def test_it_checks_the_NAMED_config_not_the_alphabetically_first(tmp_path):
    """hpc sorts before a100 and points at paths that do not exist here. The gate must
    report on a100 -- the one the run names -- and never mention the the HPC cluster paths."""
    real = tmp_path / "staged"
    real.mkdir()
    tree = _tree(tmp_path, {
        "hpc": {"flexidepth_weights": "/data/nonexistent/nope/router.pt",
                 "conditional_graph_helper": "/data/nonexistent/nope/helper.so",
                 "moe_config_dir": "/data/nonexistent/nope/moe"},
        "a100": {"flexidepth_weights": str(real), "conditional_graph_helper": str(real),
                      "moe_config_dir": str(real)},
    })
    done = _run(tree, tmp_path, "deploy/hosts/a100.json")
    assert "/data/nonexistent" not in done.stdout, (
        "the gate validated the HPC cluster's config on a host that is not the HPC cluster:\n" + done.stdout
    )
    assert "a100.json" in done.stdout


def test_a_named_config_whose_paths_are_missing_is_REFUSED(tmp_path):
    tree = _tree(tmp_path, {
        "a100": {"flexidepth_weights": "/definitely/not/here/router.pt",
                      "conditional_graph_helper": "/definitely/not/here/helper.so",
                      "moe_config_dir": "/definitely/not/here/moe"},
    })
    done = _run(tree, tmp_path, "deploy/hosts/a100.json")
    assert done.returncode != 0
    assert "host path flexidepth_weights" in done.stdout


def test_a_named_config_that_does_not_exist_is_REFUSED(tmp_path):
    tree = _tree(tmp_path, {"hpc": {"flexidepth_weights": "/x", "conditional_graph_helper": "/x",
                                     "moe_config_dir": "/x"}})
    done = _run(tree, tmp_path, "deploy/hosts/a100.json")
    assert done.returncode != 0
    assert "MISSING" in done.stdout
