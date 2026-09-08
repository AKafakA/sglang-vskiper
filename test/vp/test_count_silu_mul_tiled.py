"""Exactness + capture-safety of the tiled ``count_silu_mul`` (lane-2 adapter-lane
track, 2026-09-08) against the row-walking launch it replaced.

The served bodies (``_binary_cohort_mlp`` RUN + PROJECT branches, the P3
eager lane) call ``count_silu_mul`` on a scratch of fixed capacity with a
device-resident count; the tiled kernel must produce the SAME bytes for every
count in [0, capacity], every width the bodies use (RUN intermediate 14336,
PROJECT bottleneck 896, plus odd widths), both half dtypes, and under CUDA
graph capture/replay (the grid is capacity-static; only the count changes).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="count_silu_mul needs a CUDA device"
)


def _run(kernel, capacity, width, count, dtype, seed):
    from sglang.srt.vpipe import kernel as vp_kernel

    g = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = torch.randn((capacity, 2 * width), device="cuda", dtype=torch.float32, generator=g)
    gate_up = (gate_up * 3.0).to(dtype)
    out = torch.full((capacity, width), float("nan"), device="cuda", dtype=dtype)
    count_t = torch.tensor([count], device="cuda", dtype=torch.int64)
    kernel(gate_up, count_t, out)
    torch.cuda.synchronize()
    return gate_up, out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "capacity,width,count",
    [
        (64, 14336, 64),
        (64, 14336, 0),
        (64, 14336, 1),
        (1040, 14336, 777),
        (1040, 896, 1040),
        (1040, 896, 3),
        (2208, 14336, 2208),
        (2208, 14336, 1300),
        (300, 1000, 299),   # width not a power of two, count = capacity - 1
        (7, 40, 5),         # tiny shapes: single partial tile in both dims
    ],
)
def test_tiled_matches_rowloop_bitwise(dtype, capacity, width, count):
    from sglang.srt.vpipe import kernel as vp_kernel

    gate_up, out_tiled = _run(vp_kernel.count_silu_mul, capacity, width, count, dtype, 7)
    gate_up_ref, out_ref = _run(vp_kernel.count_silu_mul_rowloop, capacity, width, count, dtype, 7)
    assert torch.equal(gate_up, gate_up_ref)
    # rows below the count: bit-identical
    assert torch.equal(out_tiled[:count], out_ref[:count])
    # rows at/above the count are never written by either launch
    assert torch.isnan(out_tiled[count:].float()).all()
    assert torch.isnan(out_ref[count:].float()).all()
    # and the value is the fp32 formula, not an approximation
    g = gate_up[:count, :width].float()
    u = gate_up[:count, width:].float()
    expect = (g * torch.sigmoid(g) * u).to(dtype)
    assert torch.equal(out_tiled[:count], expect)


def test_count_larger_than_capacity_is_clamped_by_the_grid():
    # The tiled kernel never reads past the scratch: a count above the
    # capacity writes exactly the capacity rows (same as the row-walking
    # launch, whose programs stop at `row < count` but only have `capacity`
    # rows of scratch to read).
    from sglang.srt.vpipe import kernel as vp_kernel

    capacity, width = 96, 512
    gate_up = torch.randn((capacity, 2 * width), device="cuda", dtype=torch.float16)
    out = torch.full((capacity, width), float("nan"), device="cuda", dtype=torch.float16)
    count_t = torch.tensor([capacity], device="cuda", dtype=torch.int64)
    vp_kernel.count_silu_mul(gate_up, count_t, out)
    torch.cuda.synchronize()
    assert not torch.isnan(out.float()).any()


def test_rejects_capacity_mismatch():
    from sglang.srt.vpipe import kernel as vp_kernel

    gate_up = torch.zeros((8, 64), device="cuda", dtype=torch.float16)
    out = torch.zeros((9, 32), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        vp_kernel.count_silu_mul(gate_up, torch.tensor([8], device="cuda"), out)


def test_capture_replay_tracks_the_device_count():
    from sglang.srt.vpipe import kernel as vp_kernel

    capacity, width = 512, 14336
    gate_up = (torch.randn((capacity, 2 * width), device="cuda") * 3).to(torch.float16)
    out = torch.zeros((capacity, width), device="cuda", dtype=torch.float16)
    count_t = torch.tensor([capacity], device="cuda", dtype=torch.int64)
    # warm-up outside capture (JIT compile), then capture one launch
    vp_kernel.count_silu_mul(gate_up, count_t, out)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            vp_kernel.count_silu_mul(gate_up, count_t, out)
    torch.cuda.current_stream().wait_stream(stream)
    for count in (capacity, 200, 1, 0, 333):
        out.fill_(float("nan"))
        count_t.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        ref = torch.full_like(out, float("nan"))
        vp_kernel.count_silu_mul_rowloop(gate_up, count_t, ref)
        torch.cuda.synchronize()
        assert torch.equal(out[:count], ref[:count]), count
        assert torch.isnan(out[count:].float()).all(), count
