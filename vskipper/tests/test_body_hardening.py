"""Hardening invariants of the routed-MLP bodies (D-450 era, P2 of the 2026-09-05 plan).

1. ``weighted_scatter`` fails closed on strided route weights: the kernel reads
   ``weights_ptr + dst`` with no stride argument, so a strided 1-D view would be
   consumed silently wrong.
2. (removed 2026-09-08, D-574: `_mapped_run_base_mlp` and the other per-layer
   capacity bodies were deleted with the layer-policy machinery.)
   the compact PROJECT branch is stubbed so the test runs without Triton/CUDA.
"""

import types

import pytest
import torch

from vskipper.kernels import kernel as vp_kernel
from vskipper.kernels import mlp as vp_mlp


def test_weighted_scatter_rejects_strided_route_weights():
    rows, hidden = 8, 16
    compact = torch.zeros(rows, hidden)
    output = torch.zeros(rows, hidden)
    index = torch.arange(rows, dtype=torch.int32)
    count = torch.tensor([rows], dtype=torch.int32)
    strided = torch.rand(rows, 2)[:, 0]  # stride 2, contiguous() would hide it
    assert strided.stride(0) == 2
    with pytest.raises(ValueError, match="stride 1"):
        vp_kernel.weighted_scatter(
            compact, index, strided, count, output, invert_weight=False
        )
    # a broadcast (stride-0) weight is equally wrong for dst > 0
    expanded = torch.tensor([0.5]).expand(rows)
    assert expanded.stride(0) == 0
    with pytest.raises(ValueError, match="stride 1"):
        vp_kernel.weighted_scatter(
            compact, index, expanded, count, output, invert_weight=True
        )
