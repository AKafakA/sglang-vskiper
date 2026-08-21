#!/usr/bin/env python3
"""End-to-end composition test: pack -> count_matmul -> activation ->
count_matmul -> weighted_scatter, both branches, vs the v2 reference.

Index vectors come from torch nonzero here (the capture-time index
build is a separate, review-gated slice); activation runs on the
host-known prefix (the count-bounded activation kernel is the next
primitive). This proves the primitives COMPOSE to the exact weighted
semantics of reference_binary_cohort_mlp.
"""

import sys

import torch
import torch.nn.functional as F

from sglang.srt.vp.binary_cohort_kernels import (
    count_matmul,
    count_matmul_fused_silu,
    count_silu_mul,
    pack_rows,
    weighted_scatter,
)

sys.path.insert(0, "test/vp")
from binary_cohort_reference import reference_binary_cohort_mlp


def branch(
    hidden, idx, count_value, gate_up_w, down_w, weights, out, invert,
    fused=False,
):
    device = hidden.device
    cap = hidden.shape[0]
    hidden_size = hidden.shape[1]
    inter2 = gate_up_w.shape[0]
    count = torch.tensor(count_value, dtype=torch.int32, device=device)
    compact = torch.zeros(cap, hidden_size, device=device)
    pack_rows(hidden, idx, count, compact)
    gate_up = torch.zeros(cap, inter2, device=device)
    count_matmul(compact, gate_up_w, count, gate_up)
    final = torch.zeros(cap, hidden_size, device=device)
    if fused:
        count_matmul_fused_silu(gate_up, down_w, count, final)
    else:
        activated = torch.zeros(cap, inter2 // 2, device=device)
        count_silu_mul(gate_up, count, activated)
        count_matmul(activated, down_w, count, final)
    weighted_scatter(final, idx, weights, count, out, invert_weight=invert)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 1
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    cap, hidden_size, inter, bottleneck = 192, 256, 384, 64
    run_gate_up = torch.randn(2 * inter, hidden_size, device=device) * 0.02
    run_down = torch.randn(hidden_size, inter, device=device) * 0.02
    proj_gate_down = (
        torch.randn(2 * bottleneck, hidden_size, device=device) * 0.02
    )
    proj_up = torch.randn(hidden_size, bottleneck, device=device) * 0.02

    failures = 0
    for engagement in (0.0, 0.16, 0.46, 0.84, 1.0):
        hidden = torch.randn(cap, hidden_size, device=device)
        run_mask = torch.rand(cap, device=device) >= engagement
        weights = torch.rand(cap, device=device)
        valid = torch.rand(cap, device=device) >= 0.1
        want = reference_binary_cohort_mlp(
            hidden, run_mask, weights, valid,
            run_gate_up, run_down, proj_gate_down, proj_up,
        )
        run_idx = (run_mask & valid).nonzero(as_tuple=True)[0].to(torch.int64)
        proj_idx = (
            (~run_mask & valid).nonzero(as_tuple=True)[0].to(torch.int64)
        )
        for fused in (False, True):
            got = torch.zeros_like(hidden)
            branch(
                hidden, run_idx, run_idx.numel(), run_gate_up, run_down,
                weights, got, invert=False, fused=fused,
            )
            branch(
                hidden, proj_idx, proj_idx.numel(), proj_gate_down, proj_up,
                weights, got, invert=True, fused=fused,
            )
            max_abs = (got - want).abs().max().item()
            ok = max_abs < 1e-3
            print(
                f"engagement={engagement} fused={fused} "
                f"run={run_idx.numel()} proj={proj_idx.numel()} "
                f"max_abs={max_abs:.2e} " + ("PASS" if ok else "FAIL")
            )
            failures += 0 if ok else 1
    print("ALL_OK" if failures == 0 else f"FAILURES={failures}")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
