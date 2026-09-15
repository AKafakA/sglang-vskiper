#!/usr/bin/env python3
"""The served-arm quality driver, pinned on the three things it exists to prevent.

`run_quality_2x2.py` is the caller that makes arm C of the 2x2 a GENUINE upstream server.
Three properties carry that, and each has already failed once in this project:

  1. **The upstream arm gets `--upstream-baseline`.** Without it the quality driver keeps
     the fork attestation, so a genuine upstream server -- which has no `vp_runtime` --
     would be refused, and the natural "fix" is to drop back to `ARMS["stock"]`, which is
     how arm C came to be the fork with the skipper off for months (/).
  2. **Suite names come from the spec, never from this file.** BBH's protocol is the
     difference between +10.83 pp and -6.54 pp, and it is decided by which frozen
     suite is named.
  3. **A refused workload does not abort its arm.** One refused rate killed a whole arm's
     GPU time on 2026-09-11  and the refusal was itself the finding.

Plus the boot path is IMPORTED, not re-expressed: a second implementation of "boot the
upstream arm" is a second thing that can silently boot the fork.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DRIVER_PATH = ROOT / "test/vp/run_quality_2x2.py"
DRIVER = DRIVER_PATH.read_text()


def _code_only(text: str) -> str:
    """Source with the module docstring and comment-only lines removed.

    The docstring shows an example spec, so it legitimately names suites; a name there is
    documentation, a name in the code is a decision.
    """
    module = ast.parse(text)
    body = module.body[1:] if ast.get_docstring(module) else module.body
    first = body[0].lineno - 1 if body else len(text.splitlines())
    lines = text.splitlines()[first:]
    return "\n".join(l for l in lines if not l.strip().startswith("#"))


CODE = _code_only(DRIVER)


def test_upstream_arm_is_given_the_upstream_baseline_flag():
    assert "--upstream-baseline" in CODE
    assert "arm_is_upstream(spec, arm)" in CODE


def test_no_suite_name_is_hardcoded_in_the_driver():
    """The suites live in the spec. A literal here would decide BBH's protocol in code."""
    for literal in ("gsm8k.d179", "coqa.d179", "bbh_cot.3shot.raw", ".d179"):
        assert literal not in CODE, f"{literal!r} is hard-coded in the driver"
    assert 'spec["quality_suites"][workload]' in CODE


def test_the_boot_path_is_imported_not_reimplemented():
    assert "from run_paired_campaign import" in CODE
    assert "Server," in CODE and "campaign_preflight_gate," in CODE
    # No second launch_server invocation.
    assert "launch_server" not in CODE


def test_g1a_runs_for_every_arm_before_anything_boots():
    gate = CODE.index("campaign_preflight_gate(spec, arm)")
    boot = CODE.index("with Server(spec, arm, args.port")
    assert gate < boot, "G1a must run before the first boot"
    assert "return 2" in CODE[gate:boot]


def test_a_failed_workload_is_recorded_and_the_arm_continues():
    """`failures += ...` inside the workload loop, never a break or a raise."""
    loop = CODE[CODE.index("for workload in args.workload:\n                    row"):]
    body = loop[: loop.index("except Exception")]
    assert "failures += row[\"returncode\"] != 0" in body
    assert "break" not in body and "raise" not in body


def test_the_card_is_checked_from_the_device_before_each_boot():
    assert "nvidia-smi" in CODE and "memory.used" in CODE
    assert "gpu_is_free" in CODE


def test_preflight_refuses_a_missing_suite_without_booting(tmp_path: Path):
    spec = {
        "tree": str(ROOT), "upstream_tree": str(tmp_path / "upstream"),
        "python": sys.executable, "model_path": str(tmp_path / "model"),
        "suites_dir": str(tmp_path / "suites"), "staging_root": str(tmp_path),
        "host_config": "deploy/hosts/a100.json",
        "quality_suites": {"gsm8k": "gsm8k.NOPE"},
    }
    (tmp_path / "suites").mkdir()
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    done = subprocess.run(
        [sys.executable, str(DRIVER_PATH), "--spec", str(spec_path),
         "--out-dir", str(tmp_path / "out"), "--arm", "upstream",
         "--workload", "gsm8k", "--dry-run"],
        capture_output=True, text=True)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "no frozen suite" in done.stderr
    assert not (tmp_path / "out").exists()


def test_preflight_refuses_a_spec_missing_required_keys(tmp_path: Path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"tree": str(ROOT)}))
    done = subprocess.run(
        [sys.executable, str(DRIVER_PATH), "--spec", str(spec_path),
         "--out-dir", str(tmp_path / "out"), "--arm", "upstream",
         "--workload", "gsm8k", "--dry-run"],
        capture_output=True, text=True)
    assert done.returncode == 2
    assert "missing" in done.stderr


def test_extracted_g1a_helper_still_binds_the_serving_pythonpath():
    """The perf lane's G1a must not have lost anything in the extraction."""
    paired = (ROOT / "test/vp/run_paired_campaign.py").read_text()
    assert "--serving-pythonpath" in paired
    assert "--serving-is-upstream" in paired
    assert '"--workdir", spec["staging_root"]' in paired
    assert '"--host-config", spec["host_config"]' in paired
    assert "campaign_preflight_gate(spec, arm)" in paired
