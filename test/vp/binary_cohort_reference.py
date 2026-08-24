#!/usr/bin/env python3
"""Semantic reference for the binary-cohort kernel (D-302, ladder rung 1).

The contract the triton kernel must match (within fp tolerance):
given hidden [N, H] and a boolean RUN mask, RUN rows go through the
dense SwiGLU MLP (gate_up [2I, H] fused + down [H, I]) and PROJECT
rows through the FDProj bottleneck (fused gate_down [2m, H] + up
[H, m]) — both grounded in vp/flexidepth.py:FDProj.forward and the
sglang Llama MLP. Pure torch gather/matmul/scatter; also a per-row
loop implementation, so the reference validates itself before it
judges any kernel.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def swiglu_mlp(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor):
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down)


def fdproj(x: torch.Tensor, gate_down: torch.Tensor, up: torch.Tensor):
    gate, down = F.linear(x, gate_down).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * down, up)


def reference_binary_cohort_mlp(
    hidden: torch.Tensor,
    run_mask: torch.Tensor,
    route_weights: torch.Tensor,
    valid_rows: torch.Tensor,
    run_gate_up: torch.Tensor,
    run_down: torch.Tensor,
    proj_gate_down: torch.Tensor,
    proj_up: torch.Tensor,
) -> torch.Tensor:
    """Gather/compute/scatter reference — the kernel's semantic contract.

    The executor's epilogue weights each branch (virtual_cohort.py:229,
    INVERT_WEIGHT): RUN rows emit ``w * MLP(x)``, PROJECT rows emit
    ``(1 - w) * FDProj(x)``. Invalid (capture-pad) rows emit ZERO —
    exactly one write per valid row, none for invalid (kernel-design
    review finding P1-4).
    """

    if run_mask.dtype != torch.bool or run_mask.shape != hidden.shape[:1]:
        raise ValueError("run_mask must be bool [N]")
    if valid_rows.dtype != torch.bool or valid_rows.shape != run_mask.shape:
        raise ValueError("valid_rows must be bool [N]")
    if route_weights.shape != run_mask.shape:
        raise ValueError("route_weights must be [N]")
    out = torch.zeros_like(hidden)
    run_idx = (run_mask & valid_rows).nonzero(as_tuple=True)[0]
    proj_idx = (~run_mask & valid_rows).nonzero(as_tuple=True)[0]
    if run_idx.numel():
        out[run_idx] = swiglu_mlp(
            hidden[run_idx], run_gate_up, run_down
        ) * route_weights[run_idx, None]
    if proj_idx.numel():
        out[proj_idx] = fdproj(
            hidden[proj_idx], proj_gate_down, proj_up
        ) * (1.0 - route_weights[proj_idx, None])
    return out


def rowloop_binary_cohort_mlp(
    hidden, run_mask, route_weights, valid_rows,
    run_gate_up, run_down, proj_gate_down, proj_up
) -> torch.Tensor:
    """Row-at-a-time implementation used ONLY to validate the reference."""

    rows = []
    for index in range(hidden.shape[0]):
        row = hidden[index : index + 1]
        if not bool(valid_rows[index]):
            rows.append(torch.zeros_like(row))
        elif bool(run_mask[index]):
            rows.append(
                swiglu_mlp(row, run_gate_up, run_down)
                * float(route_weights[index])
            )
        else:
            rows.append(
                fdproj(row, proj_gate_down, proj_up)
                * (1.0 - float(route_weights[index]))
            )
    return torch.cat(rows, dim=0)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 1
    torch.manual_seed(20260820)
    device = torch.device("cuda")
    hidden_size, inter, bottleneck = 256, 512, 64  # scaled-down shapes
    run_gate_up = torch.randn(
        2 * inter, hidden_size, device=device, dtype=torch.float32
    ) * 0.02
    run_down = torch.randn(
        hidden_size, inter, device=device, dtype=torch.float32
    ) * 0.02
    proj_gate_down = torch.randn(
        2 * bottleneck, hidden_size, device=device, dtype=torch.float32
    ) * 0.02
    proj_up = torch.randn(
        hidden_size, bottleneck, device=device, dtype=torch.float32
    ) * 0.02

    failures = 0
    for rows, engagement in (
        (1, 0.0),
        (1, 1.0),
        (7, 0.3),
        (64, 0.16),
        (64, 0.46),
        (200, 0.84),
    ):
        hidden = torch.randn(
            rows, hidden_size, device=device, dtype=torch.float32
        )
        run_mask = torch.rand(rows, device=device) >= engagement
        route_weights = torch.rand(rows, device=device)
        valid_rows = torch.rand(rows, device=device) >= 0.1  # ~10% pad
        got = reference_binary_cohort_mlp(
            hidden, run_mask, route_weights, valid_rows,
            run_gate_up, run_down, proj_gate_down, proj_up,
        )
        want = rowloop_binary_cohort_mlp(
            hidden, run_mask, route_weights, valid_rows,
            run_gate_up, run_down, proj_gate_down, proj_up,
        )
        max_abs = (got - want).abs().max().item()
        ok = max_abs < 1e-4
        print(
            f"rows={rows} engagement={engagement} max_abs={max_abs:.2e} "
            + ("PASS" if ok else "FAIL")
        )
        failures += 0 if ok else 1
    print("ALL_OK" if failures == 0 else f"FAILURES={failures}")
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
