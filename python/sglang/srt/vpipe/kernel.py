"""Binary-cohort GEMM primitives — the count-adaptive routed-MLP kernel (/).

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
    REJECTED by (breaks TC pipelining: 1.68-2.31x vs 1.18-1.41x unfused).
Neither is imported by the executor. Kernel bodies below are byte-identical to
the frozen tree so route digests and numerics are unchanged.
"""

from __future__ import annotations

import functools
import re
from typing import Optional

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
        SCALE_WEIGHT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NUM_PROGS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        count = tl.load(count_ptr)
        row = pid
        while row < count:
            dst = tl.load(index_ptr + row)
            if SCALE_WEIGHT:
                weight = tl.load(weights_ptr + dst).to(tl.float32)
                if INVERT_WEIGHT:
                    weight = 1.0 - weight
            else:
                # hard_mask gate (v1.5): hard selection, no w scaling -- the multiply by
                # 1.0 keeps the rounding sequence identical to the scaled path.
                weight = 1.0
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

    @triton.jit
    def silu_mul_tile_kernel(
        gate_up_ptr,
        out_ptr,
        count_ptr,
        capacity,
        I: tl.constexpr,
        stride_in: tl.constexpr,
        stride_out: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        # Lane-2 adapter-lane track (2026-09-08): the same per-element fp32
        # silu(gate) * up as silu_mul_kernel, but tiled over a 2-D grid
        # (row tiles x column tiles) instead of 108 programs walking rows one
        # at a time -- enough bytes in flight to reach HBM bandwidth. The grid
        # is fixed by the scratch capacity (capture-safe); rows >= count are
        # masked, so the served count decides the work exactly as before.
        pid_r = tl.program_id(0)
        pid_i = tl.program_id(1)
        count = tl.load(count_ptr)
        row0 = pid_r * BLOCK_R
        # No early `return` inside the runtime condition (Codex review
        # 2026-09-08: rejected by some Triton versions); the whole body sits
        # under the condition instead. `capacity` bounds every row so an
        # out-of-contract count can never read/write past the scratch.
        if row0 < count:
            rows = (row0 + tl.arange(0, BLOCK_R)).to(tl.int64)
            cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
            live = (
                (rows[:, None] < count)
                & (rows[:, None] < capacity)
                & (cols[None, :] < I)
            )
            in_off = rows[:, None] * stride_in + cols[None, :]
            gate = tl.load(gate_up_ptr + in_off, mask=live).to(tl.float32)
            up = tl.load(gate_up_ptr + in_off + I, mask=live).to(tl.float32)
            silu = gate * tl.sigmoid(gate)
            tl.store(
                out_ptr + rows[:, None] * stride_out + cols[None, :],
                (silu * up).to(out_ptr.dtype.element_ty),
                mask=live,
            )

    return (
        pack_rows_kernel,
        weighted_scatter_kernel,
        silu_mul_kernel,
        silu_mul_tile_kernel,
    )
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
    scale_weight: bool = True,
    block_h: int = 256,
) -> None:
    """out[index[i]] = compact[i] * (w or 1-w) for i < count; ``scale_weight=False`` = hard mask.

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
    if route_weights.numel() > 1 and route_weights.stride(0) != 1:
        # The kernel indexes weights_ptr + dst with no stride argument; a
        # strided view (e.g. one column of a (rows, k) tensor reshaped to 1-D)
        # would be read silently wrong. Assert the invariant, don't absorb it.
        raise ValueError(
            "weighted_scatter route_weights must be contiguous (stride 1); got "
            f"stride {route_weights.stride(0)}"
        )
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
        scale_weight,
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
        idx_ptr,
        weights_ptr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_am: tl.constexpr,
        stride_cm: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP_M: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_N: tl.constexpr,
        GATHER_A: tl.constexpr,
        SCATTER_C: tl.constexpr,
        INVERT_W: tl.constexpr,
        SCALE_W: tl.constexpr,
    ):
        # Capacity-sized 2-D grid with early exit (/). Lane-2
        # tax-removal track / F4 (2026-09-07): the optional GATHER_A reads
        # row ``idx[m]`` of the FULL activation instead of a pre-packed row
        # (folds ``pack_rows``), and the optional SCATTER_C writes tile row m
        # to row ``idx[m]`` of the FULL output scaled by ``w`` or ``1-w``
        # (folds ``weighted_scatter``). The tile arithmetic is untouched, so
        # every row's result is bit-identical to the packed path; the scatter
        # epilogue repeats weighted_scatter's exact rounding sequence
        # (narrow to the output dtype, widen, multiply in fp32, narrow).
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
        if GATHER_A:
            src_rows = tl.load(idx_ptr + offs_m, mask=offs_m < count, other=0)
            a_ptrs = a_ptr + src_rows[:, None] * stride_am + tl.arange(0, BK)[None, :]
        else:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BK)[None, :]
        w_ptrs = w_ptr + offs_n[None, :] * K + tl.arange(0, BK)[:, None]
        for k0 in range(0, K, BK):
            if EVEN_K:
                a_tile = tl.load(a_ptrs, mask=m_live, other=0.0)
                if EVEN_N:
                    w_tile = tl.load(w_ptrs)
                else:
                    # v1.5 : N is not a multiple of BN -- Qwen3-4B's fused
                    # projector width 1216 at BN=128 -- so the last n-tile must
                    # not read past the weight's end (every Llama-3-8B N is a
                    # multiple of every tuned BN, which is why the unmasked load
                    # never faulted there). The masked columns are discarded by
                    # the store mask; EVEN_N shapes compile to the same kernel.
                    w_tile = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
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
        if SCATTER_C:
            dst_rows = tl.load(idx_ptr + offs_m, mask=offs_m < count, other=0)
            narrowed = acc.to(c_ptr.dtype.element_ty).to(tl.float32)
            if SCALE_W:
                wgt = tl.load(weights_ptr + dst_rows, mask=offs_m < count, other=0.0).to(tl.float32)
                if INVERT_W:
                    wgt = 1.0 - wgt
                scaled = narrowed * wgt[:, None]
            else:
                # hard_mask gate (v1.5): no w multiply; the narrow-and-widen keeps the
                # stored value bit-identical to the unweighted scatter.
                scaled = narrowed
            tl.store(
                c_ptr + dst_rows[:, None] * stride_cm + offs_n[None, :],
                scaled.to(c_ptr.dtype.element_ty),
                mask=store_mask,
            )
        else:
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
    gather_index: Optional[torch.Tensor] = None,
    scatter_index: Optional[torch.Tensor] = None,
    scatter_weights: Optional[torch.Tensor] = None,
    scatter_invert: bool = False,
    scatter_scale: bool = True,
) -> None:
    """Production launcher: capacity grid + per-tile early exit.

    ``scatter_scale=False`` (hard_mask gate, v1.5) writes the unweighted result: no ``w``
    multiply in the epilogue, hard selection by the route map alone.

    Promoted by/ over the persistent-while ``count_matmul`` (which
    lost 2.4-2.7x on A100); it is what the binary-cohort MLP path calls.

    F4 (lane-2 tax-removal track): ``gather_index`` makes the kernel read row
    ``gather_index[m]`` of ``compact_a`` (then the FULL activation) for compact
    row ``m`` -- ``pack_rows`` folded in; ``scatter_index`` + ``scatter_weights``
    make it write compact row ``m`` to row ``scatter_index[m]`` of ``output``
    (then the FULL output) scaled by ``w`` or ``1-w`` -- ``weighted_scatter``
    folded in. Indices are the device-resident route maps; only rows below
    ``count`` are touched, exactly as before.
    """

    cap, k_dim = compact_a.shape
    n_dim = weight.shape[0]
    import triton as _triton

    gather = gather_index is not None
    scatter = scatter_index is not None
    if scatter != (scatter_weights is not None):
        raise ValueError("scatter_index and scatter_weights must be given together")
    if gather and scatter:
        # Codex review (2026-09-07, MINOR): the kernel takes ONE index pointer,
        # so gather and scatter in the same launch would silently share it.
        # The binary-cohort body never needs both; fail closed instead.
        raise ValueError("count_matmul_gridexit: gather_index and scatter_index cannot be combined in one launch")
    for name, index in (("gather_index", gather_index), ("scatter_index", scatter_index)):
        if index is not None and (
            index.dtype != torch.int32 or index.dim() != 1 or not index.is_contiguous()
            or index.device != compact_a.device
        ):
            raise ValueError(f"{name} must be a contiguous 1-D int32 device index")
    if scatter and (
        scatter_weights.dim() != 1 or not scatter_weights.is_contiguous()
        or scatter_weights.device != compact_a.device
    ):
        raise ValueError("scatter_weights must be a contiguous 1-D device tensor")
    if compact_a.stride(1) != 1 or output.stride(1) != 1:
        raise ValueError("count_matmul_gridexit rows must be contiguous")
    grid = (_triton.cdiv(cap, block_m) * _triton.cdiv(n_dim, block_n),)
    _grid_kernel()[grid](
        compact_a,
        weight,
        output,
        count,
        gather_index if gather else (scatter_index if scatter else count),
        scatter_weights if scatter else count,
        k_dim,
        n_dim,
        compact_a.stride(0),
        output.stride(0),
        block_m,
        block_n,
        block_k,
        group_m,
        k_dim % block_k == 0,
        n_dim % block_n == 0,
        gather,
        scatter,
        bool(scatter_invert),
        bool(scatter_scale),
        num_warps=num_warps,
        num_stages=num_stages,
    )
def count_silu_mul(
    gate_up: torch.Tensor,
    count: torch.Tensor,
    output: torch.Tensor,
    *,
    block_i: int = 1024,
    block_r: int = 4,
) -> None:
    """out[:count] = silu(gate) * up over the [gate | up] compact rows.

    Tiled launch (2-D grid over the scratch capacity x width; rows >= count
    masked). Per element it is the same fp32 ``gate * sigmoid(gate) * up``
    cast to the output dtype as the row-walking kernel it replaced, so the
    outputs are bit-identical (asserted by test_count_silu_mul_tiled.py).
    """

    if gate_up.dim() != 2 or output.dim() != 2:
        raise ValueError("count_silu_mul operands must be 2-D")
    if gate_up.shape[1] != 2 * output.shape[1]:
        raise ValueError("gate_up width must be 2x the output width")
    if gate_up.stride(1) != 1 or output.stride(1) != 1:
        raise ValueError("count_silu_mul rows must be contiguous")
    if gate_up.shape[0] != output.shape[0]:
        raise ValueError("count_silu_mul capacity mismatch between gate_up and output")
    capacity = int(output.shape[0])
    if capacity == 0:
        return
    import triton

    width = int(output.shape[1])
    block_i = min(block_i, triton.next_power_of_2(width))
    for name, value in (("block_i", block_i), ("block_r", block_r)):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"count_silu_mul {name} must be a power of two, got {value}")
    _row_kernels()[3][
        (triton.cdiv(capacity, block_r), triton.cdiv(width, block_i))
    ](
        gate_up,
        output,
        count,
        capacity,
        width,
        gate_up.stride(0),
        output.stride(0),
        block_r,
        block_i,
    )


def count_silu_mul_rowloop(
    gate_up: torch.Tensor,
    count: torch.Tensor,
    output: torch.Tensor,
    *,
    block_i: int = 512,
) -> None:
    """The pre-2026-09-08 108-program row-walking launch, kept as the
    exactness reference for the tiled kernel (test_count_silu_mul_tiled.py);
    not called by any served body."""

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
# EVERY A100 RUNS ONE KERNEL (owner, 2026-09-10: "all a100 has to be run the same kernel,
# smx or pcie suffix, removing it"). `torch.cuda.get_device_name()` spells the same GA100
# part four ways -- "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-40GB",
# "NVIDIA A100-PCIE-40GB" -- and this artifact holds Triton TILE PARAMETERS (block_m/n/k,
# num_warps, num_stages). Those are PER-SM resources, and all four are 108 SMs with 192 KB of
# shared memory per SM. Interconnect and capacity move memory bandwidth and power cap, which
# change absolute throughput but not which tile shape fits.
#
# Two token classes are therefore stripped, and only two:
#   form factor   SXM*/PCIE
#   capacity      a bare <n>GB token
# Memory TECHNOLOGY is NOT stripped: H100 HBM3 and HBM2e are genuinely different parts.
#
# The campaign reason this is a correctness rule, not tidiness: the headline row and the
# hardware-generality row are measured on two different A100s, and a per-node artifact would
# make "same kernel" an assumption instead of a property of the lookup.
_FORM_FACTOR_TOKENS = ("SXM5", "SXM4", "SXM3", "SXM2", "SXM", "PCIE")
_CAPACITY_TOKEN = re.compile(r"^\d+GB$", re.IGNORECASE)


def canonical_device_key(device_name: str) -> str:
    """Normalise a CUDA device name to its tuned-artifact key.

    >>> canonical_device_key("NVIDIA A100-SXM4-80GB")
    'NVIDIA_A100'
    >>> canonical_device_key("NVIDIA A100 80GB PCIe")
    'NVIDIA_A100'
    >>> canonical_device_key("NVIDIA A100-SXM4-40GB")
    'NVIDIA_A100'
    """
    parts = [
        token
        for token in re.split(r"[\s_-]+", device_name.strip())
        if token
        and token.upper() not in _FORM_FACTOR_TOKENS
        and not _CAPACITY_TOKEN.match(token)
    ]
    return "_".join(parts)


@functools.cache
def load_tuned_configs(device_name: str) -> dict:
    """Load the committed per-device config artifact; fail closed.

    Artifacts live beside this module (binary_cohort_configs/<device>.json,
    the moe-configs pattern). A missing artifact for an active
    binary_cohort deployment is a configuration error, never a silent
    heuristic fallback (design v2.1 attestation contract).

    The lookup key is the CANONICAL device name -- see `canonical_device_key`.
    """

    import json
    from pathlib import Path

    key = canonical_device_key(device_name)
    path = Path(__file__).parent / "binary_cohort_configs" / f"{key}.json"
    if not path.exists():
        available = sorted(
            p.stem for p in (Path(__file__).parent / "binary_cohort_configs").glob("*.json")
        )
        raise RuntimeError(
            f"binary_cohort tuned-config artifact missing for device "
            f"{device_name!r} (canonical key {key!r}, expected {path}). "
            f"Available: {available}"
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
