#!/usr/bin/env python3
"""ONE artifact per architecture: neither interconnect nor capacity is a tuning axis.

Owner ruling 2026-09-10: *"all a100 has to be run the same kernel, smx or pcie suffix,
removing it"* -- so every A100 spelling resolves to one file, and "both A100s ran the same
kernel" becomes a property of the lookup rather than a claim someone has to check.

`torch.cuda.get_device_name()` returns "NVIDIA A100-SXM4-80GB" on one A100 80GB,
"NVIDIA A100 80GB PCIe" on another, and "NVIDIA A100-SXM4-40GB" on the 40 GiB node. Same
GA100 die -- 108 SMs, 192 KB shared memory per SM -- and this artifact holds Triton TILE
parameters, which are PER-SM resources. Keying on the raw name meant a tree built from tracked
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
    "raw",
    [
        "NVIDIA A100-SXM4-80GB",
        "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-40GB",
        "NVIDIA A100-PCIE-40GB",
        "NVIDIA_A100",
    ],
)
def test_every_a100_spelling_maps_to_one_key(raw):
    assert kernel.canonical_device_key(raw) == "NVIDIA_A100"


def test_the_two_campaign_nodes_resolve_to_the_SAME_artifact():
    """The headline row runs on the 80 GiB PCIe box and the generality row on the 40 GiB SXM4
    node. If these two ever diverged, the two rows would be measured on different tiles while
    every log still said 'A100'."""
    headline = kernel.canonical_device_key("NVIDIA A100 80GB PCIe")
    generality = kernel.canonical_device_key("NVIDIA A100-SXM4-40GB")
    assert headline == generality
    assert (CONFIGS / f"{headline}.json").is_file(), f"no tuned artifact for key {headline}"


def test_capacity_is_stripped_but_memory_TECHNOLOGY_is_not():
    """Capacity changes HBM size, not SM count or shared memory per SM, so it cannot change
    which tile shape fits. HBM3 vs HBM2e is a genuinely different part and stays in the key."""
    assert kernel.canonical_device_key("NVIDIA H100 80GB HBM3") == "NVIDIA_H100_HBM3"
    assert kernel.canonical_device_key("NVIDIA H100 PCIe HBM2e") == "NVIDIA_H100_HBM2e"


def test_no_form_factor_suffix_survives_in_any_committed_artifact():
    for path in CONFIGS.glob("*.json"):
        assert kernel.canonical_device_key(path.stem) == path.stem, (
            f"{path.name} is not already canonical -- it would never be found"
        )


def test_a_card_with_no_artifact_fails_closed_and_says_what_exists():
    with pytest.raises(RuntimeError) as exc:
        kernel.load_tuned_configs("NVIDIA GeForce RTX 4090")
    assert "tuned-config artifact missing" in str(exc.value)
    assert "NVIDIA_A100" in str(exc.value)  # the refusal lists what IS available


# --- H100 (plan step 4 prerequisite, 2026-09-12) ---------------------------------------------

def test_h100_form_factors_do_not_merge_and_have_roofline_entries():
    """SXM (80 GB HBM3, 3.35 TB/s) and NVL (94 GB, 3.9 TB/s) must derive DIFFERENT bands, so their
    CUDA names must not collapse to one key; PCIe (2 TB/s class) is deliberately absent until read."""
    import json
    from sglang.srt.vpipe.kernel import canonical_device_key
    from sglang.srt.vpipe.roofline import derived_decode_band, ridge_rows
    keys = {name: canonical_device_key(name) for name in
            ("NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe")}
    assert len(set(keys.values())) == 3, keys
    table = json.loads((ROOT / "python/sglang/srt/vpipe/device_roofline.json").read_text())
    for name in ("NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"):
        key = keys[name]
        assert key in table, (name, key)
        ridge = ridge_rows(key)
        ladder = tuple(range(8, 257, 8)) + tuple(range(272, 513, 16))  # the A100 capture ladder shape
        band = derived_decode_band(key, ladder)
        assert 200 <= ridge <= 320 and band[0] < ridge < band[1] + 32, (name, ridge, band)
    assert keys["NVIDIA H100 PCIe"] not in table, "PCIe must fail closed until its datasheet is read"
