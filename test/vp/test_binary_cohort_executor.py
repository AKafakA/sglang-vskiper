#!/usr/bin/env python3
"""Eager executor A/B: _binary_cohort_mlp vs the semantic reference,
with REPEATED alternating-mask calls over the shared scratch (the
state-contamination hazard the T0 ladder later re-tests under capture).
"""

import sys
import types

import torch

from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)

# The executor sizes its shared scratch from the deployment's prefill
# ceiling, so tests must establish that context (same pattern as the
# kernel tuner).
set_global_server_args_for_scheduler(
    ServerArgs(model_path="dummy", chunked_prefill_size=4096)
)

from sglang.srt.vp.flexidepth_full_graph import _binary_cohort_mlp

sys.path.insert(0, "test/vp")
from binary_cohort_reference import reference_binary_cohort_mlp


def linear_mock(weight):
    mock = types.SimpleNamespace()
    mock.weight = weight
    return mock


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 1
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    rows, hidden_size, inter, bottleneck = 192, 256, 512, 64
    dtype = torch.float16
    run_gate_up = (
        torch.randn(2 * inter, hidden_size, device=device) * 0.08
    ).to(dtype)
    run_down = (
        torch.randn(hidden_size, inter, device=device) * 0.08
    ).to(dtype)
    proj_gate_down = (
        torch.randn(2 * bottleneck, hidden_size, device=device) * 0.08
    ).to(dtype)
    proj_up = (
        torch.randn(hidden_size, bottleneck, device=device) * 0.08
    ).to(dtype)

    layer = types.SimpleNamespace(
        layer_id=25,
        mlp=types.SimpleNamespace(
            gate_up_proj=linear_mock(run_gate_up),
            down_proj=linear_mock(run_down),
        ),
    )
    proj = types.SimpleNamespace(
        up_proj=linear_mock(proj_up),
        _fused_gate_down_weight=lambda: proj_gate_down,
    )

    failures = 0
    for step, engagement in enumerate((0.0, 1.0, 0.16, 0.84, 0.0, 0.46)):
        hidden = (torch.randn(rows, hidden_size, device=device) * 0.5).to(
            dtype
        )
        run_mask = (torch.rand(rows, device=device) >= engagement).view(
            -1, 1
        )
        weights = torch.rand(rows, device=device).to(dtype).view(-1, 1)
        valid = torch.rand(rows, device=device) >= 0.1
        got = _binary_cohort_mlp(
            layer, proj, hidden, weights, run_mask, valid
        )
        want = reference_binary_cohort_mlp(
            hidden.float(),
            run_mask.view(-1),
            weights.view(-1).float(),
            valid,
            run_gate_up.float(),
            run_down.float(),
            proj_gate_down.float(),
            proj_up.float(),
        )
        max_abs = (got.float() - want).abs().max().item()
        ok = max_abs < 2e-3  # fp16 2-GEMM envelope at these scales
        print(
            f"step={step} engagement={engagement} max_abs={max_abs:.2e} "
            + ("PASS" if ok else "FAIL")
        )
        failures += 0 if ok else 1
    print("ALL_OK" if failures == 0 else f"FAILURES={failures}")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
