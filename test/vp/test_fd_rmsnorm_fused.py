"""Exactness gate for F2 (lane-2 tax-removal track): FDRMSNorm.forward (3 launches) must be bit-identical to
FDRMSNorm.reference_forward (the released 8-launch body) — routes cross a 0.5 threshold, so one ulp matters.
Covers the router shapes (hidden 256 = 4096/16), rows 1..4096, fp16/bf16/fp32 inputs, fp16/fp32 weights,
random and adversarial (tiny / huge / zero-row) inputs. GPU only."""

import pytest
import torch

from sglang.srt.vpipe.routing import FDRMSNorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 3, 64, 257, 1024, 4096])
@pytest.mark.parametrize("hidden", [256, 128, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("wdtype", ["same", torch.float32])
def test_fused_matches_reference(rows, hidden, dtype, wdtype):
    gen = torch.Generator(device="cuda"); gen.manual_seed(rows * 31 + hidden)
    norm = FDRMSNorm(hidden, eps=1e-5).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden, generator=gen, device="cuda") * 0.5 + 1.0)
        norm.weight.data = norm.weight.data.to(dtype if wdtype == "same" else torch.float32)
    x = (torch.randn(rows, hidden, generator=gen, device="cuda") * 3.0).to(dtype)
    if rows >= 3:
        x[0] = 0  # zero row -> rsqrt(eps)
        x[1] = x[1] * 1e-3  # tiny
        x[2] = x[2] * 1e3   # large
    ref = norm.reference_forward(x)
    out = norm(x)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert torch.equal(out, ref), f"max|diff| {(out.float() - ref.float()).abs().max().item():.3e}"


def test_non_contiguous_falls_back():
    norm = FDRMSNorm(256).cuda()
    x = torch.randn(64, 512, device="cuda", dtype=torch.float16)[:, ::2]  # strided view
    assert torch.equal(norm(x), norm.reference_forward(x))


def test_special_values_match_reference():
    # Codex review (MINOR): rsqrt boundary corpus. The argument of rsqrt is var + eps >= eps, so it is never
    # subnormal; inf / nan / all-subnormal / huge rows must still agree bit for bit (inf*0 -> nan on both sides).
    norm = FDRMSNorm(256, eps=1e-5).cuda()
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    x[0] = 1e-30            # fp32 squares underflow to subnormal / zero -> variance ~0 -> rsqrt(eps)
    x[1] = float("inf")
    x[2, 0] = float("nan")
    x[3] = -float("inf")
    x[4] = 3e19            # squares overflow to inf in fp32
    x[5] = 0.0
    x[6, ::2] = float("inf")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        xd = x.to(dtype)
        ref = norm.reference_forward(xd)
        out = norm(xd)
        assert torch.equal(out.isnan(), ref.isnan())
        assert torch.equal(out.nan_to_num(nan=0.0), ref.nan_to_num(nan=0.0))


@pytest.mark.parametrize("case", ["cpu", "3d", "float64_input", "float64_weight", "strided_weight"])
def test_unsupported_shapes_and_dtypes_fall_back(case):
    # Codex review (MAJOR + MINOR): every guard takes the released body, so the result equals reference_forward
    norm = FDRMSNorm(256).cuda()
    x = torch.randn(16, 256, device="cuda", dtype=torch.float16)
    if case == "cpu":
        norm = FDRMSNorm(256); x = x.cpu()
    elif case == "3d":
        x = x.view(2, 8, 256)
    elif case == "float64_input":
        x = x.double()
    elif case == "float64_weight":
        norm.weight.data = norm.weight.data.double()
    elif case == "strided_weight":
        norm.weight = torch.nn.Parameter(torch.randn(512, device="cuda", dtype=torch.float16)[::2])
    ref = norm.reference_forward(x)
    out = norm(x)
    assert out.dtype == ref.dtype and torch.equal(out, ref)
