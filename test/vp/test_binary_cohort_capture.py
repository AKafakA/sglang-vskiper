#!/usr/bin/env python3
"""T0 capture ladder for binary_cohort (the three designated tests).

1. ALTERNATING MASKS under ONE captured graph: replay with zero /
   full / random / clustered cohorts and match the eager reference
   every time (exposes stale counts, un-zeroed atomics, frozen
   cohort sizes).
2. BRANCH-OUTPUT POISONING: NaN-fill the output buffer before each
   replay; every valid row must be rewritten correct and every pad
   row must be zero (exposes rows the captured bodies never touch).
3. ROUTE SEPARATION: RUN rows must carry w*MLP and PROJECT rows
   (1-w)*FDProj — checked per row against the reference, so a body
   writing the wrong cohort cannot hide in an aggregate norm.
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

TOL = 3e-3


def linear_mock(weight):
    return types.SimpleNamespace(weight=weight)


def mask_pattern(name, rows, device):
    if name == "all_run":
        return torch.ones(rows, dtype=torch.bool, device=device)
    if name == "all_project":
        return torch.zeros(rows, dtype=torch.bool, device=device)
    if name == "clustered":
        mask = torch.zeros(rows, dtype=torch.bool, device=device)
        mask[: rows // 4] = True
        return mask
    if name == "alternating":
        return (
            torch.arange(rows, device=device) % 2 == 0
        )
    return torch.rand(rows, device=device) >= 0.5


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 1
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    dtype = torch.float16
    rows, hidden_size, inter, bottleneck = 192, 256, 512, 64

    run_gate_up = (
        torch.randn(2 * inter, hidden_size, device=device) * 0.08
    ).to(dtype)
    run_down = (torch.randn(hidden_size, inter, device=device) * 0.08).to(dtype)
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

    static_hidden = (torch.randn(rows, hidden_size, device=device) * 0.5).to(dtype)
    static_mask = torch.ones(rows, dtype=torch.bool, device=device)
    static_weights = torch.rand(rows, device=device).to(dtype)
    static_valid = torch.ones(rows, dtype=torch.bool, device=device)

    # Warmup on a side stream (capture prerequisite), then capture once.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            _binary_cohort_mlp(
                layer, proj, static_hidden, static_weights, static_mask,
                static_valid,
            )
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = _binary_cohort_mlp(
            layer, proj, static_hidden, static_weights, static_mask,
            static_valid,
        )
    print("captured OK")

    failures = 0
    for pattern in (
        "all_run", "all_project", "random", "clustered", "alternating",
        "all_project", "all_run", "random",
    ):
        static_hidden.copy_(
            (torch.randn(rows, hidden_size, device=device) * 0.5).to(dtype)
        )
        static_mask.copy_(mask_pattern(pattern, rows, device))
        static_weights.copy_(torch.rand(rows, device=device).to(dtype))
        static_valid.copy_(torch.rand(rows, device=device) >= 0.1)
        # test 2: poison every output row before replay
        captured_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        got = captured_out.clone().float()
        want = reference_binary_cohort_mlp(
            static_hidden.float(), static_mask, static_weights.float(),
            static_valid, run_gate_up.float(), run_down.float(),
            proj_gate_down.float(), proj_up.float(),
        )
        finite = bool(torch.isfinite(got).all())
        pads_zero = bool((got[~static_valid] == 0).all())
        run_rows = static_mask & static_valid
        proj_rows = (~static_mask) & static_valid
        run_err = (
            (got[run_rows] - want[run_rows]).abs().max().item()
            if run_rows.any() else 0.0
        )
        proj_err = (
            (got[proj_rows] - want[proj_rows]).abs().max().item()
            if proj_rows.any() else 0.0
        )
        ok = finite and pads_zero and run_err < TOL and proj_err < TOL
        print(
            f"{pattern:12s} run={int(run_rows.sum()):3d} "
            f"proj={int(proj_rows.sum()):3d} run_err={run_err:.2e} "
            f"proj_err={proj_err:.2e} finite={finite} pads0={pads_zero} "
            + ("PASS" if ok else "FAIL")
        )
        failures += 0 if ok else 1

    print("ALL_OK" if failures == 0 else f"FAILURES={failures}")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
