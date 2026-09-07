"""Exactness gate for F4 (lane-2 tax-removal track): count_matmul_gridexit with the gather folded into the A-load and the
route-weighted scatter folded into the epilogue must be bit-identical, row for row, to pack_rows -> GEMM and
GEMM -> weighted_scatter. Same tile config on both sides (the arithmetic is untouched by construction). GPU only."""

import pytest
import torch

from sglang.srt.vpipe.kernel import count_matmul_gridexit, pack_rows, weighted_scatter
from sglang.srt.vpipe.routing import build_route_maps_from_mask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

CFG = dict(block_m=64, block_n=128, block_k=32, group_m=8, num_warps=8, num_stages=2)


def _maps(rows, gen, frac=0.5):
    mask = torch.rand(rows, generator=gen, device="cuda") < frac
    stats = torch.zeros(3, dtype=torch.int64, device="cuda")
    run_map, proj_map, counts = build_route_maps_from_mask(mask, None, stats)
    return run_map, proj_map, counts


@pytest.mark.parametrize("rows", [1, 33, 64, 200, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kn", [(4096, 512), (256, 4096), (1792, 4096)])
def test_gather_matches_pack(rows, dtype, kn):
    K, N = kn
    gen = torch.Generator(device="cuda"); gen.manual_seed(rows * 7 + K)
    x = (torch.randn(rows, K, generator=gen, device="cuda") * 0.5).to(dtype)
    w = (torch.randn(N, K, generator=gen, device="cuda") * 0.02).to(dtype)
    run_map, _, counts = _maps(rows, gen)
    count = counts[0:1]
    compact = torch.empty(rows, K, dtype=dtype, device="cuda")
    pack_rows(x, run_map, count, compact)
    ref = torch.zeros(rows, N, dtype=dtype, device="cuda")
    count_matmul_gridexit(compact, w, count, ref, **CFG)
    out = torch.zeros(rows, N, dtype=dtype, device="cuda")
    count_matmul_gridexit(x, w, count, out, gather_index=run_map, **CFG)
    n = int(count.item())
    assert torch.equal(out[:n], ref[:n])


@pytest.mark.parametrize("rows", [1, 33, 64, 200, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("invert", [False, True])
def test_scatter_matches_weighted_scatter(rows, dtype, invert):
    K, N = 512, 4096
    gen = torch.Generator(device="cuda"); gen.manual_seed(rows * 11 + int(invert))
    a = (torch.randn(rows, K, generator=gen, device="cuda") * 0.5).to(dtype)
    w = (torch.randn(N, K, generator=gen, device="cuda") * 0.02).to(dtype)
    weights = torch.rand(rows, generator=gen, device="cuda").to(dtype)
    run_map, _, counts = _maps(rows, gen)
    count = counts[0:1]
    final = torch.zeros(rows, N, dtype=dtype, device="cuda")
    count_matmul_gridexit(a, w, count, final, **CFG)
    ref = torch.zeros(rows, N, dtype=dtype, device="cuda")
    weighted_scatter(final, run_map, weights, count, ref, invert_weight=invert)
    out = torch.zeros(rows, N, dtype=dtype, device="cuda")
    count_matmul_gridexit(a, w, count, out, scatter_index=run_map, scatter_weights=weights, scatter_invert=invert, **CFG)
    assert torch.equal(out, ref)


def test_strided_output_view_scatter():
    # the PROJECT branch writes through a column view of the shared scratch; the fused scatter must honour stride_cm
    rows, K, N = 96, 256, 1024
    gen = torch.Generator(device="cuda"); gen.manual_seed(3)
    a = torch.randn(rows, K, generator=gen, device="cuda").half()
    w = (torch.randn(N, K, generator=gen, device="cuda") * 0.02).half()
    weights = torch.rand(rows, generator=gen, device="cuda").half()
    _, proj_map, counts = _maps(rows, gen)
    count = counts[1:2]
    ref = torch.zeros(rows, 2 * N, dtype=torch.float16, device="cuda")
    final = torch.zeros(rows, N, dtype=torch.float16, device="cuda")
    count_matmul_gridexit(a, w, count, final, **CFG)
    weighted_scatter(final, proj_map, weights, count, ref[:, :N], invert_weight=True)
    out = torch.zeros(rows, 2 * N, dtype=torch.float16, device="cuda")
    count_matmul_gridexit(a, w, count, out[:, :N], scatter_index=proj_map, scatter_weights=weights, scatter_invert=True, **CFG)
    assert torch.equal(out, ref)


def test_fails_closed_on_bad_indices():
    a = torch.randn(64, 256, device="cuda").half(); w = torch.randn(128, 256, device="cuda").half()
    count = torch.tensor([64], dtype=torch.int32, device="cuda"); out = torch.zeros(64, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError):
        count_matmul_gridexit(a, w, count, out, gather_index=torch.arange(64, device="cuda"), **CFG)  # int64
    with pytest.raises(ValueError):
        count_matmul_gridexit(a, w, count, out, scatter_index=torch.arange(64, dtype=torch.int32, device="cuda"), **CFG)  # no weights
    idx = torch.arange(64, dtype=torch.int32, device="cuda"); wts = torch.ones(64, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError):  # one index pointer in the kernel: gather + scatter in one launch is refused
        count_matmul_gridexit(a, w, count, out, gather_index=idx, scatter_index=idx, scatter_weights=wts, **CFG)
