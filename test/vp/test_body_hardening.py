"""Hardening invariants of the routed-MLP bodies (D-450 era, P2 of the 2026-09-05 plan).

1. ``weighted_scatter`` fails closed on strided route weights: the kernel reads
   ``weights_ptr + dst`` with no stride argument, so a strided 1-D view would be
   consumed silently wrong.
2. ``_mapped_run_base_mlp`` zeroes invalid (padded) rows like every other body;
   the compact PROJECT branch is stubbed so the test runs without Triton/CUDA.
"""

import types

import pytest
import torch

from sglang.srt.vpipe import kernel as vp_kernel
from sglang.srt.vpipe import mlp as vp_mlp


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


def test_run_base_body_zeroes_padded_rows(monkeypatch):
    rows, hidden = 6, 4
    hidden_states = torch.arange(rows * hidden, dtype=torch.float32).view(rows, hidden) + 1
    route_weights = torch.full((rows, 1), 0.5)
    run_mask = torch.tensor([[True], [True], [False], [True], [False], [True]])
    valid_rows = torch.tensor([True, True, True, True, False, False])
    layer = types.SimpleNamespace(mlp=lambda h: h * 2.0)
    proj = types.SimpleNamespace(
        _fused_gate_down_weight=lambda: torch.zeros(2, hidden),
        up_proj=types.SimpleNamespace(weight=torch.zeros(hidden, 1)),
    )
    seen = {}

    def fake_compact_branch(proj_, h, weights, project_active, output, **kwargs):
        seen["project_active"] = project_active.clone()
        return torch.zeros((), dtype=torch.int64)

    monkeypatch.setattr(vp_mlp, "_mapped_single_compact_branch", fake_compact_branch)
    output, _ = vp_mlp._mapped_run_base_mlp(
        layer, proj, hidden_states, route_weights, run_mask,
        capacity=rows, valid_rows=valid_rows,
    )
    expected = hidden_states * 2.0 * 0.5
    assert torch.equal(output[:4], expected[:4])
    assert torch.all(output[4:] == 0), "padded rows must be zero like every other body"
    # padded rows never enter the PROJECT branch
    assert seen["project_active"].tolist() == [False, False, True, False, False, False]

    # valid_rows=None keeps the historical all-rows behaviour (nothing padded)
    output_all, _ = vp_mlp._mapped_run_base_mlp(
        layer, proj, hidden_states, route_weights, run_mask, capacity=rows,
    )
    assert torch.equal(output_all, expected)
