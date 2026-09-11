#!/usr/bin/env python3
"""The cross-arm allowlist is frozen in code, and it contains exactly two classified claims.

Measured 2026-09-11 by running `verify_cross_arm_config.py --report` on two REAL manifests
from this campaign — an upstream ladder boot and an integrated_it4 bank harvest. Of 33
compared fields exactly two differ, and the gate's own exit codes were checked: rc=1 with
only `vp_runtime` declared, rc=0 with both.

Every spec in the campaign hard-codes `"cross_arm_allow": ["vp_runtime"]`, so without the
frozen set G1c would have refused **100 % of cells** — the headline and all twelve sweep
points — on a field that had already been classified as conservative.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "test/vp/run_paired_campaign.py"


def _driver():
    spec = importlib.util.spec_from_file_location("_vp_paired_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _driver()


def test_the_frozen_allowlist_is_exactly_the_two_classified_fields():
    assert driver.CROSS_ARM_DECLARED == frozenset({"vp_runtime", "server.max_total_num_tokens"})


def test_the_treatment_field_is_declared():
    """vp_runtime present on the fork, [] on upstream. This difference IS the experiment."""
    assert "vp_runtime" in driver.CROSS_ARM_DECLARED


def test_the_kv_capacity_field_is_declared_and_its_direction_is_recorded():
    """upstream 393319 vs integrated_it4 389479 = -0.976 %. The router/projector weights take
    HBM, so the TREATMENT arm gets the smaller KV pool — it can only make our result look
    worse. That direction is the whole reason this is declarable rather than a defect."""
    assert "server.max_total_num_tokens" in driver.CROSS_ARM_DECLARED
    upstream, treatment = 393319, 389479
    assert treatment < upstream, "declaring this only holds while the TREATMENT has less"
    assert abs((treatment - upstream) / upstream) < 0.02, "still ~1 %, not a new regime"


def test_a_spec_can_ADD_to_the_allowlist_but_not_replace_it():
    """The failure mode being closed: a spec saying ["vp_runtime"] must not silently narrow
    the frozen set back down to one field."""
    spec_value = ["vp_runtime"]
    effective = driver.CROSS_ARM_DECLARED | set(spec_value)
    assert "server.max_total_num_tokens" in effective

    extra = driver.CROSS_ARM_DECLARED | {"server.something_new"}
    assert driver.CROSS_ARM_DECLARED <= extra and "server.something_new" in extra


def test_an_empty_spec_allowlist_still_gets_the_frozen_set():
    assert driver.CROSS_ARM_DECLARED | set([]) == driver.CROSS_ARM_DECLARED


def test_the_list_is_short_on_purpose():
    """An allowlist that grows is a gate being talked out of its job. Two entries, each with a
    recorded measurement and direction."""
    assert len(driver.CROSS_ARM_DECLARED) == 2
