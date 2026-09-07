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
