#!/usr/bin/env python3
"""The derived decode band: the rule must REPRODUCE the shipped values, and fail closed.

`enter_rows=176` / `exit_rows=144` were defaults in a test fixture in the commit that created
the regime switch (D-705). The defence is to make the runtime compute them from device
properties. That defence is worth nothing unless the computation actually lands on the values
we shipped and measured -- so that is the load-bearing test here.

Owner ruling (D-705): this is a principled DEFAULT, never a claimed optimum. These tests pin
the rule, not the optimality of its output.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "python/sglang/srt/vpipe/roofline.py"
ARTIFACT = ROOT / "python/sglang/srt/vpipe/device_roofline.json"

# The live decode capture ladder, read from the running campaign's server_info on 2026-09-11:
# rungs of 8 from 16 to 256, then 16 to the decode cap.
LADDER = tuple([1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 1025, 16)))


def _roofline():
    spec = importlib.util.spec_from_file_location("vp_roofline", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # pure arithmetic: no torch, no GPU
    return module


def test_the_rule_reproduces_the_SHIPPED_a100_band():
    """The whole claim in one assertion."""
    assert _roofline().derived_decode_band("NVIDIA_A100", LADDER) == (144, 176)


def test_the_a100_ridge_is_the_published_machine_balance():
    assert abs(_roofline().ridge_rows("NVIDIA_A100") - 312e12 / 1935e9) < 1e-9


def test_the_band_is_not_knife_edge_on_the_jitter_percentile():
    """p90=13 and p95=15 round to the same rung; only p99=27 widens it."""
    rl = _roofline()
    assert rl.derived_decode_band("NVIDIA_A100", LADDER, 13) == (144, 176)
    assert rl.derived_decode_band("NVIDIA_A100", LADDER, 15) == (144, 176)
    assert rl.derived_decode_band("NVIDIA_A100", LADDER, 27) == (128, 192)


def test_an_unknown_device_FAILS_CLOSED_and_names_what_is_known():
    rl = _roofline()
    with pytest.raises(RuntimeError) as excinfo:
        rl.device_peaks("NVIDIA_H100")
    assert "no published roofline" in str(excinfo.value)
    assert "must not fall back to a heuristic" in str(excinfo.value)


def test_the_turing_box_is_absent_on_purpose():
    """It has no bf16 tensor cores, and it is feasibility-only by owner rule."""
    table = json.loads(ARTIFACT.read_text())
    assert "Quadro_RTX_8000" not in table
    assert "_omitted_on_purpose" in table


def test_the_artifact_holds_only_spec_sheet_values():
    """A number measured on our own box, presented as a device property, is the defect."""
    table = json.loads(ARTIFACT.read_text())
    assert "SPEC-SHEET CONSTANTS" in table["_source"]
    for key, entry in table.items():
        if key.startswith("_"):
            continue
        assert set(entry) == {"peak_tflops_bf16", "peak_bw_gbs"}


def test_both_endpoints_land_on_capture_rungs():
    """An off-rung threshold switches bodies at a PADDED batch."""
    exit_rows, enter_rows = _roofline().derived_decode_band("NVIDIA_A100", LADDER)
    assert exit_rows in LADDER and enter_rows in LADDER


def test_the_ladder_is_PASSED_IN_never_held_here():
    """A second copy of the runtime's ladder would drift from it."""
    code = MODULE.read_text()
    body = code[code.index("def _roofline_table"):]
    for literal in ("range(16, 257", "capture_bs", "176", "144"):
        assert literal not in body, f"{literal!r} is hardcoded below the docstring"


def test_a_device_whose_ridge_is_below_the_band_width_is_refused():
    """H20-class: ridge ~37 with a wide band would give a non-positive exit."""
    rl = _roofline()
    rl._roofline_table()["TINY"] = {"peak_tflops_bf16": 10.0, "peak_bw_gbs": 4000.0}
    with pytest.raises(ValueError, match="not positive"):
        rl.derived_decode_band("TINY", LADDER, 200)


def test_the_shipped_a100_band_PASSES_the_boot_assertion():
    _roofline().assert_band_follows_rule(
        served_exit_rows=144, served_enter_rows=176,
        device_key="NVIDIA_A100", capture_rungs=LADDER)


def test_a_drifted_band_is_REFUSED_and_the_message_names_both(tmp_path):
    rl = _roofline()
    with pytest.raises(RuntimeError) as excinfo:
        rl.assert_band_follows_rule(
            served_exit_rows=128, served_enter_rows=192,
            device_key="NVIDIA_A100", capture_rungs=LADDER)
    message = str(excinfo.value)
    assert "declared (exit=128, enter=192)" in message
    assert "rule gives (exit=144, enter=176)" in message
    assert "ridge 161.2" in message


def test_the_assertion_is_a_GATE_not_a_computation():
    """It must never return a value the caller could use to overwrite the served design."""
    import inspect
    rl = _roofline()
    # `from __future__ import annotations` makes annotations strings, so accept both forms.
    annotation = inspect.signature(rl.assert_band_follows_rule).return_annotation
    assert annotation in (None, "None"), annotation
    src = inspect.getsource(rl.assert_band_follows_rule)
    assert "return " not in src, "a gate that returns a band invites silent per-host design"
