"""Binary-cohort GEMM primitives — the count-adaptive routed-MLP kernel (D-302/D-303).

Replaces the grouped ``fused_moe`` dispatch on routed layers. At low router
engagement that path taxed the ~84% of rows that still RUN, consuming ~19% of
GPU time where dense sits in peak cuBLAS tiles; this kernel removes that tax at
the source rather than gating around it.

Pipeline per routed layer, per cohort:
``pack -> count-bounded tuned GEMM -> silu*mul -> GEMM -> route-weighted scatter``

Parameter-free by construction: exact cohort sizes, **no capacity fraction and
no semantic knobs** — only offline per-hardware autotuning, the same class as
cuBLAS heuristics. The grid is STATIC (sized to bucket capacity) and CTAs
early-exit against the DEVICE count, so one captured graph serves any cohort
size with no host sync.

Measured on A100: 1.18-1.23x the exact-count ideal at working counts, and
dominant over every DEPLOYABLE alternative at partial cohorts (0.11x / 0.35x /
0.65x dense at counts 205 / 1024 / 2048).

Numerics are standard triton tile accumulation — NOT bit-identical to the
grouped path or cuBLAS. The quality lane is the gate; the bit-exact gates bind
the K/V commit path, not the MLP.

Two variants present in the frozen tree are deliberately ABSENT here:
  * ``count_matmul``/``_kernel`` — the non-grid-exit variant, superseded by
    ``count_matmul_gridexit`` (persistent-while loses 2.4-2.7x on A100).
  * ``count_matmul_fused_silu``/``_fused_kernel`` — mainloop activation fusion,
    REJECTED by D-303 (breaks TC pipelining: 1.68-2.31x vs 1.18-1.41x unfused).
Neither is imported by the executor. Kernel bodies below are byte-identical to
the frozen tree so route digests and numerics are unchanged.
"""

from __future__ import annotations

import functools
import torch


@functools.cache
def _row_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def pack_rows_kernel(
        source_ptr,
        index_ptr,
        compact_ptr,
        count_ptr,
        H: tl.constexpr,
        stride_source: tl.constexpr,
        stride_compact: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NUM_PROGS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        count = tl.load(count_ptr)
        row = pid
        while row < count:
            src = tl.load(index_ptr + row)
            for h0 in range(0, H, BLOCK_H):
                offs = h0 + tl.arange(0, BLOCK_H)
                values = tl.load(
                    source_ptr + src * stride_source + offs,
                    mask=offs < H,
                )
                tl.store(
                    compact_ptr + row * stride_compact + offs,
                    values,
                    mask=offs < H,
                )
            row += NUM_PROGS

    @triton.jit
    def weighted_scatter_kernel(
        compact_ptr,
        index_ptr,
        weights_ptr,
        out_ptr,
        count_ptr,
        H: tl.constexpr,
        stride_compact: tl.constexpr,
        stride_out: tl.constexpr,
        INVERT_WEIGHT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NUM_PROGS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        count = tl.load(count_ptr)
        row = pid
        while row < count:
            dst = tl.load(index_ptr + row)
            weight = tl.load(weights_ptr + dst).to(tl.float32)
            if INVERT_WEIGHT:
                weight = 1.0 - weight
            for h0 in range(0, H, BLOCK_H):
                offs = h0 + tl.arange(0, BLOCK_H)
                values = tl.load(
                    compact_ptr + row * stride_compact + offs,
                    mask=offs < H,
                ).to(tl.float32)
                tl.store(
                    out_ptr + dst * stride_out + offs,
                    (values * weight).to(out_ptr.dtype.element_ty),
                    mask=offs < H,
                )
            row += NUM_PROGS

    @triton.jit
    def silu_mul_kernel(
        gate_up_ptr,
        out_ptr,
        count_ptr,
        I: tl.constexpr,
        stride_in: tl.constexpr,
        stride_out: tl.constexpr,
        BLOCK_I: tl.constexpr,
        NUM_PROGS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        count = tl.load(count_ptr)
        row = pid
        while row < count:
            for i0 in range(0, I, BLOCK_I):
                offs = i0 + tl.arange(0, BLOCK_I)
                live = offs < I
                gate = tl.load(
                    gate_up_ptr + row * stride_in + offs, mask=live
                ).to(tl.float32)
                up = tl.load(
                    gate_up_ptr + row * stride_in + I + offs, mask=live
                ).to(tl.float32)
                silu = gate * tl.sigmoid(gate)
                tl.store(
                    out_ptr + row * stride_out + offs,
                    (silu * up).to(out_ptr.dtype.element_ty),
                    mask=live,
                )
            row += NUM_PROGS

    return pack_rows_kernel, weighted_scatter_kernel, silu_mul_kernel
def _num_programs(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
def pack_rows(
    source: torch.Tensor,
    index: torch.Tensor,
    count: torch.Tensor,
    compact: torch.Tensor,
    *,
    block_h: int = 256,
) -> None:
    """compact[i] = source[index[i]] for i < count (device-resident)."""

    if source.dim() != 2 or compact.dim() != 2:
        raise ValueError("pack_rows operands must be 2-D")
    if source.shape[1] != compact.shape[1]:
        raise ValueError("pack_rows hidden widths differ")
    if source.stride(1) != 1 or compact.stride(1) != 1:
        raise ValueError("pack_rows rows must be contiguous")
    num_programs = _num_programs(source.device)
    _row_kernels()[0][(num_programs,)](
        source,
        index,
        compact,
        count,
        source.shape[1],
        source.stride(0),
        compact.stride(0),
        block_h,
        num_programs,
    )
def weighted_scatter(
    compact: torch.Tensor,
    index: torch.Tensor,
    route_weights: torch.Tensor,
    count: torch.Tensor,
    output: torch.Tensor,
    *,
    invert_weight: bool,
    block_h: int = 256,
) -> None:
    """out[index[i]] = compact[i] * (w or 1-w) for i < count.

    The route-weight epilogue (reference v2 / review P1-4): RUN uses
    the weight directly, PROJECT passes invert_weight=True for (1-w).
    Weights are indexed by the DESTINATION row (per-token weights).
    """

    if compact.dim() != 2 or output.dim() != 2:
        raise ValueError("weighted_scatter operands must be 2-D")
    if compact.shape[1] != output.shape[1]:
        raise ValueError("weighted_scatter hidden widths differ")
    if compact.stride(1) != 1 or output.stride(1) != 1:
        raise ValueError("weighted_scatter rows must be contiguous")
    if route_weights.dim() != 1:
        raise ValueError("route_weights must be 1-D per-token")
    num_programs = _num_programs(compact.device)
    _row_kernels()[1][(num_programs,)](
        compact,
        index,
        route_weights,
        output,
        count,
        compact.shape[1],
        compact.stride(0),
        output.stride(0),
        invert_weight,
        block_h,
        num_programs,
    )
@functools.cache
def _grid_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def gridexit_count_matmul(
        a_ptr,
        w_ptr,
        c_ptr,
        count_ptr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_am: tl.constexpr,
        stride_cm: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP_M: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        # Diagnostic variant: capacity-sized 2-D grid with early exit —
        # isolates whether the persistent while-loop defeats pipelining.
        pid = tl.program_id(0)
        count = tl.load(count_ptr)
        num_m_cap = tl.num_programs(0) // tl.cdiv(N, BN)
        num_n = tl.cdiv(N, BN)
        group_size = GROUP_M * num_n
        group_id = pid // group_size
        first_m = group_id * GROUP_M
        group_rows = tl.minimum(num_m_cap - first_m, GROUP_M)
        pid_m = first_m + (pid % group_size) % group_rows
        pid_n = (pid % group_size) // group_rows
        if pid_m * BM >= count:
            return
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        m_live = offs_m[:, None] < count
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BK)[None, :]
        w_ptrs = w_ptr + offs_n[None, :] * K + tl.arange(0, BK)[:, None]
        for k0 in range(0, K, BK):
            if EVEN_K:
                a_tile = tl.load(a_ptrs, mask=m_live, other=0.0)
                w_tile = tl.load(w_ptrs)
            else:
                k_live = k0 + tl.arange(0, BK) < K
                a_tile = tl.load(
                    a_ptrs, mask=m_live & k_live[None, :], other=0.0
                )
                w_tile = tl.load(
                    w_ptrs,
                    mask=k_live[:, None] & (offs_n[None, :] < N),
                    other=0.0,
                )
            acc += tl.dot(a_tile, w_tile)
            a_ptrs += BK
            w_ptrs += BK
        store_mask = m_live & (offs_n[None, :] < N)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
            acc.to(c_ptr.dtype.element_ty),
            mask=store_mask,
        )

    return gridexit_count_matmul
def count_matmul_gridexit(
    compact_a: torch.Tensor,
    weight: torch.Tensor,
    count: torch.Tensor,
    output: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 128,
    block_k: int = 32,
    group_m: int = 8,
    num_warps: int = 8,
    num_stages: int = 2,
) -> None:
    """Production launcher: capacity grid + per-tile early exit.

    Promoted by D-302/D-303 over the persistent-while ``count_matmul`` (which
    lost 2.4-2.7x on A100); it is what the binary-cohort MLP path calls.
    """

    cap, k_dim = compact_a.shape
    n_dim = weight.shape[0]
    import triton as _triton

    grid = (_triton.cdiv(cap, block_m) * _triton.cdiv(n_dim, block_n),)
    _grid_kernel()[grid](
        compact_a,
        weight,
        output,
        count,
        k_dim,
        n_dim,
        compact_a.stride(0),
        output.stride(0),
        block_m,
        block_n,
        block_k,
        group_m,
        k_dim % block_k == 0,
        num_warps=num_warps,
        num_stages=num_stages,
    )
def count_silu_mul(
    gate_up: torch.Tensor,
    count: torch.Tensor,
    output: torch.Tensor,
    *,
    block_i: int = 512,
) -> None:
    """out[:count] = silu(gate) * up over the [gate | up] compact rows."""

    if gate_up.dim() != 2 or output.dim() != 2:
        raise ValueError("count_silu_mul operands must be 2-D")
    if gate_up.shape[1] != 2 * output.shape[1]:
        raise ValueError("gate_up width must be 2x the output width")
    if gate_up.stride(1) != 1 or output.stride(1) != 1:
        raise ValueError("count_silu_mul rows must be contiguous")
    num_programs = _num_programs(gate_up.device)
    _row_kernels()[2][(num_programs,)](
        gate_up,
        output,
        count,
        output.shape[1],
        gate_up.stride(0),
        output.stride(0),
        block_i,
        num_programs,
    )
@functools.cache
def load_tuned_configs(device_name: str) -> dict:
    """Load the committed per-device config artifact; fail closed.

    Artifacts live beside this module (binary_cohort_configs/<device>.json,
    the moe-configs pattern). A missing artifact for an active
    binary_cohort deployment is a configuration error, never a silent
    heuristic fallback (design v2.1 attestation contract).
    """

    import json
    from pathlib import Path

    path = (
        Path(__file__).parent
        / "binary_cohort_configs"
        / f"{device_name.replace(' ', '_')}.json"
    )
    if not path.exists():
        raise RuntimeError(
            f"binary_cohort tuned-config artifact missing for device "
            f"{device_name!r} (expected {path})"
        )
    return json.loads(path.read_text())
def select_config(
    tuned: dict, op: str, expected_count: int
) -> dict[str, int]:
    """Pick the nearest tuned count-band config for (op, count).

    ``tuned`` is the banked sweep artifact loaded by
    :func:`load_tuned_configs` from ``binary_cohort_configs/<device>.json``
    beside this module. Capture-time selection only (one config per
    bucket/op — design v2.1 item 6).
    """

    candidates = []
    for key, record in tuned.items():
        record_op, _, count_text = key.partition("@")
        if record_op == op:
            candidates.append((int(count_text), record["config"]))
    if not candidates:
        raise ValueError(f"no tuned configs for op {op!r}")
    grid_max = max(count for count, _ in candidates)
    if expected_count > grid_max:
        raise ValueError(
            f"expected_count {expected_count} exceeds the tuned grid "
            f"(max {grid_max}) for op {op!r} — extend the artifact "
            "(nearest-grid selection applies only within the grid)"
        )
    _, config = min(
        candidates, key=lambda item: abs(item[0] - expected_count)
    )
    block_m, block_n, block_k, num_warps, num_stages = config
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
