#!/usr/bin/env python3
"""The cross-arm gate must REFUSE an undeclared configuration difference.

This gate exists because the baseline became a genuinely different source tree:
upstream SGLang at 602c8615a1, 193 commits behind the fork. Identical CLI flags no longer
imply identical resolved configuration, and a config difference the paper does not declare is
an uncontrolled variable in all 54 paired cells.

The tests assert the REFUSAL, not the pass. Every gate this project has lost time to --
 -- was passing at the time.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VP = Path(__file__).resolve().parents[1]
GATE = VP / "gates/verify_cross_arm_config.py"


def manifest(tmp_path: Path, name: str, identity: dict) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps({"system_id": name, "server_identity": identity}))
    return p


def run(tmp_path, a_identity, b_identity, *extra):
    a = manifest(tmp_path, "upstream", a_identity)
    b = manifest(tmp_path, "integrated_it4", b_identity)
    return subprocess.run(
        [sys.executable, str(GATE), "--arm", f"upstream={a}",
         "--arm", f"integrated_it4={b}", *extra],
        capture_output=True, text=True,
    )


BASE = {"server": {"chunked_prefill_size": 8192, "attention_backend": "triton",
                   "max_total_num_tokens": 393319},
        "effective_max_running_requests_per_dp": [4096]}


def test_identical_arms_pass(tmp_path):
    done = run(tmp_path, BASE, json.loads(json.dumps(BASE)))
    assert done.returncode == 0, done.stdout
    assert "differ only in declared fields" in done.stdout


def test_undeclared_difference_is_REFUSED(tmp_path):
    """The case the gate exists for: upstream defaults differently and nobody noticed."""
    other = json.loads(json.dumps(BASE))
    other["server"]["chunked_prefill_size"] = 2048
    done = run(tmp_path, BASE, other)
    assert done.returncode == 1, done.stdout
    assert "REFUSED" in done.stdout
    assert "server.chunked_prefill_size" in done.stdout, done.stdout


def test_the_treatment_is_declarable_as_a_whole_subtree(tmp_path):
    """`vp_runtime` is present on the routed arm and absent upstream -- that IS the treatment,
    and allowing the prefix must cover every field beneath it."""
    treated = json.loads(json.dumps(BASE))
    treated["vp_runtime"] = {"served_design": {"arm": "integrated_it4", "skipper": "flexidepth"}}
    done = run(tmp_path, BASE, treated, "--allow", "vp_runtime")
    assert done.returncode == 0, done.stdout


def test_a_declared_field_does_not_hide_an_undeclared_one(tmp_path):
    """Allowing the treatment must not wave through a real config divergence alongside it."""
    treated = json.loads(json.dumps(BASE))
    treated["vp_runtime"] = {"served_design": {"arm": "integrated_it4"}}
    treated["server"]["attention_backend"] = "flashinfer"
    done = run(tmp_path, BASE, treated, "--allow", "vp_runtime")
    assert done.returncode == 1, done.stdout
    assert "server.attention_backend" in done.stdout


def test_report_mode_never_judges(tmp_path):
    """--report builds the allowlist from evidence; it must exit 0 even when fields differ,
    and must be unusable as a pass (it says so in its own output)."""
    other = json.loads(json.dumps(BASE))
    other["server"]["chunked_prefill_size"] = 2048
    done = run(tmp_path, BASE, other, "--report")
    assert done.returncode == 0
    assert "exiting 0 without judging" in done.stdout
    assert "server.chunked_prefill_size" in done.stdout


def test_list_valued_config_is_compared_whole(tmp_path):
    """A CUDA-graph batch-size ladder is ordered; comparing per-index would be noise, but a
    changed ladder must still refuse."""
    other = json.loads(json.dumps(BASE))
    other["effective_max_running_requests_per_dp"] = [2048]
    done = run(tmp_path, BASE, other)
    assert done.returncode == 1
    assert "effective_max_running_requests_per_dp" in done.stdout


def test_a_manifest_without_server_identity_cannot_be_compared(tmp_path):
    a = tmp_path / "a.json"; a.write_text(json.dumps({"system_id": "upstream"}))
    b = manifest(tmp_path, "integrated_it4", BASE)
    done = subprocess.run(
        [sys.executable, str(GATE), "--arm", f"upstream={a}", "--arm", f"integrated_it4={b}"],
        capture_output=True, text=True)
    assert done.returncode != 0
    assert "no server_identity" in (done.stdout + done.stderr)
