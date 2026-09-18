#!/usr/bin/env python3
"""Every arm in ARMS has an expectation, derived from the design rather than hand-written.

`run_paired_campaign.py` refuses any arm with no `expect_<arm>.json`. Those files had only ever
been written by hand, so only four existed -- for the four arms that had been run. The instant
the RandomSkip sweep added twelve arms, all twelve would have been REFUSED at preflight, AFTER
the headline had spent seven hours of GPU, with a message about a missing file rather than
about the design. Caught by dry-running one sweep spec before the sweep could run.

Deriving the content closes the class: a new arm gets its expectation for free, and an
expectation cannot disagree with the design it exists to check.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "vskipper/src/vskipper/experiments/make_expected_runtime.py"


def _tool():
    spec = importlib.util.spec_from_file_location("_vp_expect_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _tool()


def test_every_arm_in_the_design_gets_an_expectation():
    for arm in tool.ARMS:
        e = tool.expectation(arm)
        assert e["runtime"]["served_design"]["arm"] == arm


def test_upstream_asserts_the_ABSENCE_of_an_attestation():
    """Genuine upstream SGLang has no vpipe package, so there is no attestation block at all.
    Its absence IS the assertion (D-587/D-646) -- an upstream arm that attests anything is a
    mislaunched fork."""
    assert tool.expectation("upstream") == {"attestation": "absent"}


@pytest.mark.parametrize("arm,skipper,phases", [
    ("stock", None, []),
    ("vdec_fd", "flexidepth", ["decode"]),
    ("vpre_binarycohort", "flexidepth", ["prefill"]),
    ("integrated_it4", "flexidepth", ["decode", "prefill"]),
    ("integrated_randomskip_r25_d75", "deterministic_mock", ["decode", "prefill"]),
])
def test_the_expectation_matches_the_arm_it_describes(arm, skipper, phases):
    sd = tool.expectation(arm)["runtime"]["served_design"]
    assert sd["skipper"] == skipper
    assert sd["active_phases"] == phases


def test_it_reproduces_the_four_expectations_that_already_existed():
    """The load-bearing check: these four were hand-written and the headline depends on them,
    so a generator that did not reproduce them byte-for-byte could not be trusted to run
    against the live campaign directory. Verified `same` on the box for all four before the
    15 missing ones were written."""
    known = {
        "upstream": {"attestation": "absent"},
        "stock": {"attestation": "present", "runtime": {"served_design": {
            "arm": "stock", "skipper": None, "active_phases": []}}},
        "integrated_it4": {"attestation": "present", "runtime": {"served_design": {
            "arm": "integrated_it4", "skipper": "flexidepth",
            "active_phases": ["decode", "prefill"]}}},
        "integrated_alwaysskip": {"attestation": "present", "runtime": {"served_design": {
            "arm": "integrated_alwaysskip", "skipper": "flexidepth",
            "active_phases": ["decode", "prefill"]}}},
    }
    for arm, want in known.items():
        assert tool.expectation(arm) == want, arm


def test_all_twelve_sweep_arms_are_covered():
    sweep = [a for a in tool.ARMS if a.startswith("integrated_randomskip_")]
    assert len(sweep) == 12
    for arm in sweep:
        assert tool.expectation(arm)["runtime"]["served_design"]["skipper"] == "deterministic_mock"


def test_the_output_is_json_serialisable_as_written():
    for arm in list(tool.ARMS) + ["upstream"]:
        json.loads(json.dumps(tool.expectation(arm)))
