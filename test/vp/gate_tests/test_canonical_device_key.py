#!/usr/bin/env python3
"""One artifact per architecture: the interconnect suffix is not a tuning axis.

`torch.cuda.get_device_name()` returns "NVIDIA A100-SXM4-80GB" on one A100 80GB and
"NVIDIA A100 80GB PCIe" on another. Same GA100 die, and this artifact holds Triton TILE
parameters, which are architectural. Keying on the raw name meant a tree built from tracked
files could not boot on the PCIe card at all -- exactly how the P0 smoke failed -- and was
"fixed" on the box by hand-copying the SXM4 file under a PCIe name, leaving the deployment
dependent on an untracked artifact claiming a measurement nobody had made.

`kernel.py` imports torch at module level, but `canonical_device_key` is pure string logic,
so the module is loaded BY PATH with a stub -- no GPU, no sglang package import.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "python/sglang/srt/vpipe/kernel.py"
CONFIGS = ROOT / "python/sglang/srt/vpipe/binary_cohort_configs"


def _load_kernel():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    spec = importlib.util.spec_from_file_location("_vp_kernel_under_test", KERNEL)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # triton/torch internals we do not need for a string function
        pytest.skip(f"kernel.py needs a real GPU stack here: {exc}")
    return module


kernel = _load_kernel()


@pytest.mark.parametrize(
    "raw", ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA_A100_80GB"]
)
def test_every_a100_80gb_spelling_maps_to_one_key(raw):
    assert kernel.canonical_device_key(raw) == "NVIDIA_A100_80GB"


def test_the_committed_artifact_exists_under_that_key():
    key = kernel.canonical_device_key("NVIDIA A100 80GB PCIe")
    assert (CONFIGS / f"{key}.json").is_file(), f"no tuned artifact for key {key}"


def test_memory_technology_is_NOT_stripped():
    """H100 80GB HBM3 and HBM2e are genuinely different parts; only form factor is noise."""
    assert kernel.canonical_device_key("NVIDIA H100 80GB HBM3") == "NVIDIA_H100_80GB_HBM3"


def test_no_form_factor_suffix_survives_in_any_committed_artifact():
    for path in CONFIGS.glob("*.json"):
        assert kernel.canonical_device_key(path.stem) == path.stem, (
            f"{path.name} is not already canonical -- it would never be found"
        )


def test_a_card_with_no_artifact_fails_closed_and_says_what_exists():
    with pytest.raises(RuntimeError) as exc:
        kernel.load_tuned_configs("NVIDIA GeForce RTX 4090")
    assert "tuned-config artifact missing" in str(exc.value)
    assert "NVIDIA_A100_80GB" in str(exc.value)  # the refusal lists what IS available
