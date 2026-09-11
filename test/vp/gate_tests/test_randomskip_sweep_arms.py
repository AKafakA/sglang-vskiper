#!/usr/bin/env python3
"""The skip-rate x depth sweep is twelve NAMED arms, and the original is not one of them.

Plan item 3, owner scope 2026-09-10: gsm8k only, all 12 points, one rate, 3 reps.

There is no sweep loop, no CLI and no value-taking arm field in this repo, and D-609 forbids
expressing experiment configuration as environment variables. Twelve named entries in ARMS is
the only supported parameterisation, so these tests hold the grid's shape: a missing or
transposed point would silently become a hole in the published curve.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "python/sglang/srt/vpipe/design.py"


def _design():
    spec = importlib.util.spec_from_file_location("_vp_design_under_test", DESIGN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


design = _design()
SWEEP = {k: v for k, v in design.ARMS.items() if k.startswith("integrated_randomskip_")}


def test_the_grid_is_complete_and_has_exactly_twelve_points():
    expected = {
        f"integrated_randomskip_r{r}_d{d}"
        for r in (10, 25, 50, 75)
        for d in (25, 50, 75)
    }
    assert set(SWEEP) == expected
    assert len(SWEEP) == 12


@pytest.mark.parametrize("name,arm", sorted(SWEEP.items()))
def test_every_point_matches_the_rate_and_depth_in_its_own_NAME(name, arm):
    """The name is the label that reaches the plot. If it disagreed with the config, the curve
    would be drawn against coordinates the system never ran."""
    _, _, coords = name.partition("integrated_randomskip_")
    rate_text, depth_text = coords.split("_")
    assert arm["mock_token_skip_rate"] == pytest.approx(int(rate_text[1:]) / 100)
    assert arm["mock_skipped_depth_ratio"] == pytest.approx(int(depth_text[1:]) / 100)


def test_one_seed_across_the_whole_sweep():
    """The curve must vary in rate and depth only."""
    assert {arm["mock_seed"] for arm in SWEEP.values()} == {1234}


def test_every_point_routes_BOTH_phases_like_the_served_system():
    """D-619: a decode-only mock would confound skip magnitude with phase coverage, because
    integrated_it4 -- the system this surface is meant to contextualise -- routes both."""
    assert all(arm["phases"] == "both" for arm in SWEEP.values())
    assert design.ARMS["integrated_it4"]["phases"] == "both"


def test_the_ORIGINAL_randomskip_arm_is_untouched():
    """Its values are the deleted arm_env_vdec_randomskip.sh's, verbatim. Changing them while
    'porting' the arm would be a silent design change that makes it incomparable to every
    earlier RandomSkip result."""
    base = design.ARMS["integrated_randomskip"]
    assert base["mock_token_skip_rate"] == 0.5
    assert base["mock_skipped_depth_ratio"] == 0.5
    assert base["mock_seed"] == 1234
    assert "integrated_randomskip" not in SWEEP


def test_r50_d50_is_configuration_identical_to_the_original():
    """Not a duplicate -- a free internal consistency check: two names, one configuration, so
    the sweep's own machinery can be checked against the arm with prior evidence behind it."""
    base = design.ARMS["integrated_randomskip"]
    twin = SWEEP["integrated_randomskip_r50_d50"]
    assert {k: twin[k] for k in base} == base


def test_depth_is_a_ratio_not_a_layer_range():
    """Depth is a fraction of the FIXED routed set. SERVED_ROUTED_LAYERS is a design constant
    shared by every arm; expressing depth by changing it would make verify_served_design
    correctly refuse the diff."""
    assert list(design.SERVED_ROUTED_LAYERS) == list(range(16, 32))
    assert all(0.0 < arm["mock_skipped_depth_ratio"] <= 1.0 for arm in SWEEP.values())


def test_the_sweep_does_not_disturb_the_canonical_arms():
    for name in ("stock", "vdec_fd", "vpre_binarycohort", "integrated_it4",
                 "integrated_alwaysskip", "integrated_randomskip"):
        assert name in design.ARMS
