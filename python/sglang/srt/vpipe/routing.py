"""Routing — decide RUN vs PROJECT_ONLY per token, per routed layer.

This is the DECIDE half of the per-layer mechanism; `executor.py` is the
EXECUTE half:

    prepare_layer_route(...)  ->  PreparedRoute      (this module)
    execute_prepared_route(..., prepared)            (executor.py)

Contents:

* The trained FlexiDepth modules as published -- ``FDRouter`` (router_enc ->
  router_norm -> router_dec -> router_head), ``FDProj`` (the PROJECT_ONLY
  gate/up/down projector) and ``FDRMSNorm``. Weights are loaded per routed
  layer (16-31 on Llama-3-8B) from ``SGLANG_FD_WEIGHTS`` at layer construction.
  The router is trained over the FULL sequence (prompt + generation), so it is
  meaningful in prefill as well as decode.

* ``fd_prepare_layer_route_full_graph`` -- runs the router, turns logits into a
  per-token RUN/PROJECT mask, and builds the DEVICE-RESIDENT index maps for the
  routed cohort. Index construction stays on device: a host sync here would
  serialise the decode step, which is what made the eager body collapse.

Bodies are byte-identical to the frozen tree so route decisions -- and
therefore route digests -- are unchanged by construction.

EXCEPTION (2026-09-04, lane-2 stage 1), OPT-IN AND DEFAULT OFF: with
``SGLANG_FD_FUSED_ROUTER_NORM=1`` the ``router_norm`` becomes SGLang's fused
``RMSNorm`` instead of the local ``FDRMSNorm``, collapsing ~6-8 elementwise
launches per routed layer per step into one kernel. Enabling it forfeits the
"byte-identical bodies" provenance argument for that one module, so route
identity is no longer unchanged *by construction* and must be demonstrated
EMPIRICALLY: ``test/vp/gates/route_digest_compare.py`` (route counters + output
text sha256) plus a bit-exact ``route_weight``/``route_mask`` capture through
``test/vp/compare_fd_parity_traces.py``.

MEASURED 2026-09-04, and it is NOT bit-exact: one row of 735,888 flips
RUN -> PROJECT (``run_rows`` 389192 -> 389191). ``cast_x_before_out_mul=True``
does reproduce ``FDRMSNorm``'s elementwise order exactly, so the *formula*
matches -- but the fused kernel reduces the 256-wide variance with a parallel
tree while PyTorch's ``.pow(2).mean(-1)`` does not, and the two differ in the
last bit. The router feeds a 0.5 threshold, so one epsilon-close row is enough.
The effect is DETERMINISTIC (two boots of one tree give bit-identical counters).

The narrow lesson, which bounds any future fusion here: ELEMENTWISE ops are
order-independent and can be fused bit-exactly; REDUCTIONS cannot. Of
``FDRMSNorm``'s ~8 kernels only ``.mean(-1)`` is a reduction, so fusing the
other seven around PyTorch's own mean would keep routes identical.
"""

from __future__ import annotations

import triton.language as tl
from typing import Any, Mapping, Optional
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.vpipe.common import (
    _fdvp_router_graph_enabled,
    full_graph_fused_router_norm_enabled,
    _fdvp_timing_enabled,
    fdvp_fused_project_input_enabled,
    fdvp_fused_project_input_shared_storage_enabled,
    full_graph_compact_routed_qkv_enabled,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_trace_enabled,
)
import triton
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_capturing_breakable_cuda_graph,
)
from sglang.srt.vpipe.common import (
    forced_route_action,
    resolve_full_graph_skipper,
)
from sglang.srt.vpipe.env import (
    FD_FORCE_ROUTE_ENV,
    _FD_PARITY_TRACE_ENABLED,
    _VALID_FORCED_ROUTES,
)
from sglang.srt.vpipe.types import (
    FullGraphActionBatch,
    RUN_PROJECT_EXECUTION,
)
from sglang.srt.vpipe.common import (
    _FD_PARITY_EPOCHS,
    fd_parity_trace_target,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_record_cache_counter,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_record_cache_value,
)


@triton.jit
def _build_route_maps_kernel(
    run_active_ptr,
    project_active_ptr,
    run_rows_ptr,
    project_rows_ptr,
    counts_ptr,
    row_count: tl.constexpr,
    block_rows: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    valid = rows < row_count
    run_active = tl.load(run_active_ptr + rows, mask=valid, other=0).to(tl.int1)
    project_active = tl.load(
        project_active_ptr + rows, mask=valid, other=0
    ).to(tl.int1)
    counter_lanes = tl.zeros((block_rows,), dtype=tl.int32)
    run_slots = tl.atomic_add(
        counts_ptr + counter_lanes, 1, mask=valid & run_active
    )
    project_slots = tl.atomic_add(
        counts_ptr + 1 + counter_lanes, 1, mask=valid & project_active
    )
    tl.store(run_rows_ptr + run_slots, rows, mask=valid & run_active)
    tl.store(
        project_rows_ptr + project_slots,
        rows,
        mask=valid & project_active,
    )


@triton.jit
def _build_route_maps_masked_kernel(
    run_mask_ptr,
    valid_ptr,
    run_rows_ptr,
    project_rows_ptr,
    counts_ptr,
    stats_ptr,
    row_count: tl.constexpr,
    block_rows: tl.constexpr,
    has_valid: tl.constexpr,
):
    """Lane-2 Block 1B-1: the binary-cohort route maps from the RAW route mask.

    Folds the three per-layer mask kernels (``run & valid``, ``~run``,
    ``~run & valid``) and the three per-layer evidence kernels
    (``stats[0] += 1``, ``counts.to(int64)``, ``stats[1:3] += counts``) into
    the one launch that already builds the maps. Boolean logic and integer
    atomics only, so maps, counts and stats are bit-identical to the
    unfused path; slot order under atomics was never deterministic in
    either version and every downstream consumer is row-wise independent.
    """
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    in_range = rows < row_count
    run_mask = tl.load(run_mask_ptr + rows, mask=in_range, other=0).to(tl.int32)
    if has_valid:
        valid = tl.load(valid_ptr + rows, mask=in_range, other=0).to(tl.int32)
    else:
        valid = in_range.to(tl.int32)
    run_active = in_range & (valid != 0) & (run_mask != 0)
    project_active = in_range & (valid != 0) & (run_mask == 0)
    counter_lanes = tl.zeros((block_rows,), dtype=tl.int32)
    run_slots = tl.atomic_add(counts_ptr + counter_lanes, 1, mask=run_active)
    project_slots = tl.atomic_add(
        counts_ptr + 1 + counter_lanes, 1, mask=project_active
    )
    tl.store(run_rows_ptr + run_slots, rows, mask=run_active)
    tl.store(project_rows_ptr + project_slots, rows, mask=project_active)
    # Evidence accumulator [calls, run_rows, project_rows] (int64): one
    # program-level partial per launch, and exactly one call increment.
    n_run = tl.sum(run_active.to(tl.int64), axis=0)
    n_project = tl.sum(project_active.to(tl.int64), axis=0)
    tl.atomic_add(stats_ptr + 1, n_run)
    tl.atomic_add(stats_ptr + 2, n_project)
    tl.atomic_add(stats_ptr, (tl.program_id(0) == 0).to(tl.int64))
def fd_parity_trace_tensor(
    phase: str,
    tensor,
    *,
    layer_id: int,
    token_epoch: int,
) -> None:
    """Persist one target row for deterministic direct-vs-full-graph parity diagnostics."""

    directory = os.environ.get("SGLANG_FD_PARITY_TRACE_DIR", "")
    target_epoch = int(os.environ.get("SGLANG_FD_PARITY_TRACE_EPOCH", "0") or "0")
    if not directory or tensor is None or token_epoch != target_epoch:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        tensor.detach().to(device="cpu"),
        path / f"e{token_epoch:04d}_l{layer_id:02d}_{phase}.pt",
    )
def fd_parity_trace_context(layer_id: int, forward_batch):
    """Return the selected token row for a trace-only direct/VP comparison."""

    del layer_id
    target = fd_parity_trace_target()
    rids = getattr(forward_batch, "rids", None) or ()
    mode = getattr(forward_batch, "forward_mode", None)
    if not target or target not in rids or mode is None:
        return None
    request_index = rids.index(target)
    if mode.is_decode():
        epoch = int(_FD_PARITY_EPOCHS.get(target, 0))
        return request_index, epoch
    if not mode.is_extend():
        return None

    lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if lengths is None:
        lengths_tensor = getattr(forward_batch, "extend_seq_lens", None)
        if lengths_tensor is None:
            raise RuntimeError(
                "FlexiDepth prefill parity trace has no sequence lengths"
            )
        lengths = lengths_tensor.detach().to(device="cpu").tolist()
    lengths = tuple(int(value) for value in lengths)
    if len(lengths) != len(rids) or any(value <= 0 for value in lengths):
        raise RuntimeError(
            "FlexiDepth prefill parity trace sequence lengths do not match requests"
        )
    prefix_lengths = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if prefix_lengths is None:
        prefix_tensor = getattr(forward_batch, "extend_prefix_lens", None)
        if prefix_tensor is None:
            raise RuntimeError(
                "FlexiDepth prefill parity trace has no prefix lengths"
            )
        prefix_lengths = prefix_tensor.detach().to(device="cpu").tolist()
    prefix_lengths = tuple(int(value) for value in prefix_lengths)
    if len(prefix_lengths) != len(rids) or any(
        value < 0 for value in prefix_lengths
    ):
        raise RuntimeError(
            "FlexiDepth prefill parity trace prefix lengths do not match requests"
        )
    offset = int(
        os.environ.get("SGLANG_FD_PARITY_TRACE_PREFILL_OFFSET", "-1") or "-1"
    )
    request_tokens = lengths[request_index]
    normalized_offset = offset if offset >= 0 else request_tokens + offset
    if normalized_offset < 0 or normalized_offset >= request_tokens:
        raise ValueError(
            "FlexiDepth prefill parity trace offset is outside the target request"
        )
    row = sum(lengths[:request_index]) + normalized_offset
    token_position = prefix_lengths[request_index] + normalized_offset
    return row, token_position
def _fdvp_fused_router_dec_head():
    return os.environ.get(
        "SGLANG_FD_VP_FUSED_ROUTER_DEC_HEAD", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
def _fdvp_fused_project_input():
    return fdvp_fused_project_input_enabled()
def _fdvp_router_graph_max_rows():
    try:
        return max(
            0,
            int(os.environ.get("SGLANG_FD_VP_ROUTER_GRAPH_MAX_ROWS", "64") or "64"),
        )
    except ValueError:
        return 64
def _fdvp_router_graph_max_entries():
    try:
        return max(
            0,
            int(
                os.environ.get("SGLANG_FD_VP_ROUTER_GRAPH_MAX_ENTRIES", "4")
                or "4"
            ),
        )
    except ValueError:
        return 4
@triton.jit
def _fd_rmsnorm_square_kernel(x_ptr, sq_ptr, n_elements, BLOCK: tl.constexpr):
    """sq = x.to(fp32) * x.to(fp32) -- the fp32 tensor PyTorch's `.pow(2)` produces."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(sq_ptr + offs, x * x, mask=m)
@triton.jit
def _fd_rmsnorm_scale_kernel(
    x_ptr, var_ptr, w_ptr, out_ptr, eps, H: tl.constexpr, BLOCK_H: tl.constexpr
):
    """out[row] = weight * ((x32 * rsqrt(var[row] + eps)).to(x.dtype)) -- FDRMSNorm's order."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    m = offs < H
    x = tl.load(x_ptr + row * H + offs, mask=m, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + row)
    r = tl.rsqrt(var + eps)
    h = (x * r).to(x_ptr.dtype.element_ty)
    w = tl.load(w_ptr + offs, mask=m, other=0.0)
    y = (w.to(tl.float32) * h.to(tl.float32)).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * H + offs, y, mask=m)
def fd_rmsnorm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FDRMSNorm in 3 launches: square (Triton) -> mean (PyTorch, reference order) -> scale (Triton)."""

    rows, hidden = x.shape
    sq = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    n = rows * hidden
    if n:
        block = 1024
        _fd_rmsnorm_square_kernel[(triton.cdiv(n, block),)](x, sq, n, BLOCK=block)
    variance = sq.mean(-1, keepdim=True)  # the reference reduction, unchanged
    out = torch.empty(x.shape, dtype=torch.result_type(weight, x), device=x.device)
    if rows:
        _fd_rmsnorm_scale_kernel[(rows,)](
            x, variance, weight, out, eps, H=hidden, BLOCK_H=triton.next_power_of_2(hidden)
        )
    return out
class FDRMSNorm(nn.Module):
    """FlexiDepth/DDLlama RMSNorm: compute variance in fp32, return input dtype.

    RETAINED AS THE BIT-EXACTNESS REFERENCE, not as a live body. Since
    2026-09-04 ``FDRouter`` uses SGLang's fused ``RMSNorm`` instead (see the
    construction site), which collapses these ~6-8 elementwise launches into
    one. This class stays because it is what the fused path must reproduce
    exactly: note the weight multiply happens AFTER the narrowing cast
    (``self.weight * hidden_states.to(input_dtype)``), which is
    ``cast_x_before_out_mul=True`` and NOT the fused default. Keep it for the
    parity harness to diff against; do not reintroduce it as the live path
    without re-measuring the launch count.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def reference_forward(self, hidden_states):
        """The released FlexiDepth body, kernel for kernel (8 launches). Kept as
        the bit-exactness reference for `forward`."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def forward(self, hidden_states):
        # Lane-2 tax-removal track / F2 (2026-09-07): the SAME arithmetic in 3
        # launches instead of 8. The only reduction -- `.mean(-1)` -- stays
        # PyTorch's own kernel on an fp32 `x*x` tensor that the first Triton
        # kernel produces bit-identically (pow(2) IS x*x in fp32), so the
        # variance is reduced in exactly the reference order. The second Triton
        # kernel fuses the elementwise tail `(x32 * rsqrt(var + eps)).to(dtype)`
        # then `weight * h` (product formed in fp32 and rounded once, which is
        # PyTorch's opmath for a half/bfloat16 multiply). `rsqrt` is the CUDA
        # `rsqrtf` on both sides. Exactness is asserted by
        # test/vp/test_fd_rmsnorm_fused.py against `reference_forward`; any
        # non-CUDA / non-contiguous input takes the reference body.
        if (
            not hidden_states.is_cuda
            or hidden_states.dim() != 2
            or not hidden_states.is_contiguous()
            or hidden_states.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            # Codex review (2026-09-07, MAJOR): the fused tail forms the product
            # in fp32, which is only PyTorch's opmath when the weight is one of
            # these dtypes too; any other weight dtype (float64, complex, ...)
            # takes the reference body so result_type and rounding are untouched.
            or self.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not self.weight.is_contiguous()
        ):
            return self.reference_forward(hidden_states)
        return fd_rmsnorm_fused(hidden_states, self.weight, float(self.variance_epsilon))
class FDRouter(nn.Module):
    """DDLlamaRouter: bottleneck MLP -> scalar skip logit per token (no bias)."""

    def __init__(self, hidden_size=4096, reduction=16, eps=1e-5):
        super().__init__()
        r = hidden_size // reduction
        self.router_enc = nn.Linear(hidden_size, r, bias=False)
        # Router norm. DEFAULT = FDRMSNorm, which reproduces the released
        # FlexiDepth checkpoint's routing EXACTLY and is what the oracle and the
        # route-digest gate expect.
        #
        # Opting into the fused kernel (SGLANG_FD_FUSED_ROUTER_NORM=1) collapses
        # ~6-8 elementwise launches per routed layer per step into one and buys a
        # MEASURED 0.32 ms/step of flat decode tax, but it is NOT route-identical:
        # the fused kernel reduces the 256-wide variance with a parallel tree,
        # PyTorch's `.pow(2).mean(-1)` does not, and one row of 735,888 sitting
        # within an epsilon of the 0.5 threshold flips RUN -> PROJECT.
        # `cast_x_before_out_mul=True` is still load-bearing when enabled: it
        # selects `weight * x.to(orig_dtype)` (layers/layernorm.py:523-524),
        # FDRMSNorm's own order; the default (False) multiplies the weight in
        # fp32 via a different kernel and would perturb more. `eps` must stay
        # explicit -- RMSNorm defaults to 1e-6, FlexiDepth uses
        # config.rms_norm_eps (1e-5), threaded from seam.py.
        if full_graph_fused_router_norm_enabled():
            self.router_norm = RMSNorm(r, eps=eps, cast_x_before_out_mul=True)
        else:
            self.router_norm = FDRMSNorm(r, eps=eps)
        self.router_act = nn.Tanh()
        self.router_dec = nn.Linear(r, hidden_size, bias=False)
        self.router_head = nn.Linear(hidden_size, 1, bias=False)
        self.register_buffer(
            "_fdvp_fused_dec_head_weight", None, persistent=False
        )
        self._fdvp_fused_dec_head_signature = None
        self._fdvp_router_graphs = {}
        self._fdvp_router_graph_failed = set()

    def _fused_dec_head_weight(self):
        dec_weight = self.router_dec.weight
        head_weight = self.router_head.weight
        signature = (
            int(dec_weight.data_ptr()),
            int(head_weight.data_ptr()),
            int(dec_weight._version),
            int(head_weight._version),
            str(dec_weight.device),
            str(dec_weight.dtype),
        )
        cached = self._fdvp_fused_dec_head_weight
        if (
            cached is not None
            and self._fdvp_fused_dec_head_signature == signature
            and cached.device == dec_weight.device
            and cached.dtype == dec_weight.dtype
        ):
            _fdvp_record_cache_counter("fused_router_dec_head_cache_hits")
            return cached

        # There is no nonlinearity between router_dec and router_head. Compose
        # them once in fp32, then retain the model dtype for the serving GEMM.
        with torch.no_grad():
            fused = torch.matmul(
                head_weight.to(torch.float32),
                dec_weight.to(torch.float32),
            ).to(device=dec_weight.device, dtype=dec_weight.dtype)
        self._fdvp_fused_dec_head_weight = fused
        self._fdvp_fused_dec_head_signature = signature
        _fdvp_record_cache_counter("fused_router_dec_head_builds")
        return fused

    def _forward_eager(self, x):
        hidden = self.router_act(self.router_norm(self.router_enc(x)))
        if _fdvp_fused_router_dec_head():
            _fdvp_record_cache_counter("fused_router_dec_head_calls")
            return F.linear(hidden, self._fused_dec_head_weight())
        return self.router_head(self.router_dec(hidden))

    def _graph_key(self, x):
        return (
            str(x.device),
            str(x.dtype),
            tuple(int(dim) for dim in x.shape),
            bool(_fdvp_fused_router_dec_head()),
        )

    def _forward_graph(self, x):
        if not _fdvp_router_graph_enabled():
            return None
        if not torch.is_tensor(x) or not x.is_cuda or x.requires_grad:
            return None
        if _fdvp_cuda_stream_capture_active():
            return None
        if _fdvp_trace_enabled() and _fdvp_timing_enabled():
            return None
        max_rows = _fdvp_router_graph_max_rows()
        if max_rows > 0 and int(x.shape[0]) > max_rows:
            _fdvp_record_cache_counter("router_graph_row_skips")
            return None

        key = self._graph_key(x)
        if key in self._fdvp_router_graph_failed:
            _fdvp_record_cache_counter("router_graph_failed_skips")
            return None
        cached = self._fdvp_router_graphs.get(key)
        if cached is not None:
            static_input, static_output, graph = cached
            static_input.copy_(x)
            graph.replay()
            _fdvp_record_cache_counter("router_graph_hits")
            return static_output

        max_entries = _fdvp_router_graph_max_entries()
        if max_entries > 0 and len(self._fdvp_router_graphs) >= max_entries:
            _fdvp_record_cache_counter("router_graph_overflow")
            return None

        try:
            current_stream = torch.cuda.current_stream(x.device)
            capture_stream = torch.cuda.Stream(device=x.device)
            static_input = torch.empty_like(x)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(capture_stream):
                static_input.copy_(x)
                for _ in range(2):
                    self._forward_eager(static_input)
            capture_stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            # torch.cuda.graph(stream=None) selects PyTorch's internal capture
            # stream, even when an outer stream context is active. Bind the
            # graph to the stream warmed and synchronized above so the first
            # returned output cannot race capture execution.
            with torch.cuda.graph(graph, stream=capture_stream):
                static_output = self._forward_eager(static_input)
            capture_stream.synchronize()
            self._fdvp_router_graphs[key] = (static_input, static_output, graph)
            _fdvp_record_cache_counter("router_graph_misses")
            _fdvp_record_cache_value(
                "router_graph_entries", len(self._fdvp_router_graphs)
            )
            # Capture records the kernels but does not materialize a usable
            # output. Launch once for the input that caused this cache miss.
            graph.replay()
            return static_output
        except Exception as exc:
            del exc
            self._fdvp_router_graph_failed.add(key)
            _fdvp_record_cache_counter("router_graph_capture_failures")
            return None

    def forward(
        self, x, *, use_graph=False
    ):  # x = input_layernorm(hidden) ; returns logits [..., 1]
        if use_graph:
            graph_output = self._forward_graph(x)
            if graph_output is not None:
                return graph_output
        return self._forward_eager(x)
class FDProj(nn.Module):
    """DDLlamaProj: SwiGLU adapter with bottleneck intermediate (replaces MLP for skipped tokens)."""

    def __init__(self, hidden_size=4096, intermediate_size=14336, reduction=16, act="silu"):
        super().__init__()
        m = intermediate_size // reduction
        self.gate_proj = nn.Linear(hidden_size, m, bias=False)
        self.down_proj = nn.Linear(hidden_size, m, bias=False)
        self.up_proj = nn.Linear(m, hidden_size, bias=False)
        self.act_fn = F.silu if act == "silu" else getattr(F, act)
        self.register_buffer(
            "_fdvp_fused_gate_down_weight", None, persistent=False
        )
        self._fdvp_fused_gate_down_signature = None

    @staticmethod
    def _gate_down_signature(gate_weight, down_weight):
        return (
            int(gate_weight.data_ptr()),
            int(down_weight.data_ptr()),
            int(gate_weight._version),
            int(down_weight._version),
            str(gate_weight.device),
            str(gate_weight.dtype),
            fdvp_fused_project_input_shared_storage_enabled(),
        )

    def _fused_gate_down_weight(self):
        gate_weight = self.gate_proj.weight
        down_weight = self.down_proj.weight
        signature = self._gate_down_signature(gate_weight, down_weight)
        cached = self._fdvp_fused_gate_down_weight
        if (
            cached is not None
            and self._fdvp_fused_gate_down_signature == signature
            and cached.device == gate_weight.device
            and cached.dtype == gate_weight.dtype
        ):
            _fdvp_record_cache_counter("fused_project_input_cache_hits")
            return cached
        with torch.no_grad():
            fused = torch.cat((gate_weight, down_weight), dim=0).contiguous()
            if fdvp_fused_project_input_shared_storage_enabled():
                gate_rows = int(gate_weight.shape[0])
                gate_weight.set_(fused.narrow(0, 0, gate_rows))
                down_weight.set_(
                    fused.narrow(0, gate_rows, int(down_weight.shape[0]))
                )
        self._fdvp_fused_gate_down_weight = fused
        self._fdvp_fused_gate_down_signature = self._gate_down_signature(
            gate_weight, down_weight
        )
        _fdvp_record_cache_counter("fused_project_input_builds")
        return fused

    def forward(self, x):  # x = post_attention_layernorm(hidden)
        if _fdvp_fused_project_input():
            gate_down = F.linear(x, self._fused_gate_down_weight())
            gate, down = gate_down.chunk(2, dim=-1)
            _fdvp_record_cache_counter("fused_project_input_calls")
        else:
            gate = self.gate_proj(x)
            down = self.down_proj(x)
        return self.up_proj(self.act_fn(gate) * down)
def _fdvp_cuda_stream_capture_active():
    if is_capturing_breakable_cuda_graph():
        return True
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False
def build_route_maps(
    run_active: torch.Tensor, project_active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed-size RUN/PROJECT row maps and their device counts."""

    if not run_active.is_cuda or not project_active.is_cuda:
        raise ValueError("compact cohort route maps require CUDA tensors")
    if run_active.shape != project_active.shape or run_active.ndim != 1:
        raise ValueError("compact cohort route masks must be aligned 1D tensors")
    rows = int(run_active.numel())
    counts = torch.zeros(2, dtype=torch.int32, device=run_active.device)
    run_rows = torch.empty(rows, dtype=torch.int32, device=run_active.device)
    project_rows = torch.empty_like(run_rows)
    block_rows = 256
    _build_route_maps_kernel[(triton.cdiv(rows, block_rows),)](
        run_active,
        project_active,
        run_rows,
        project_rows,
        counts,
        row_count=rows,
        block_rows=block_rows,
    )
    return run_rows, project_rows, counts


def build_route_maps_from_mask(
    run_mask: torch.Tensor,
    valid_rows: Optional[torch.Tensor],
    stats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route maps + counts from the raw 1-D bool route mask (Block 1B-1).

    Equivalent to ``build_route_maps(run_mask & valid, ~run_mask & valid)``
    followed by ``stats[0] += 1; stats[1:3] += counts`` — in one launch.
    ``stats`` is the per-device int64 ``[calls, run_rows, project_rows]``
    accumulator the attestation reads.
    """

    if not run_mask.is_cuda or run_mask.ndim != 1 or run_mask.dtype != torch.bool:
        raise ValueError("route mask must be a 1-D CUDA bool tensor")
    if valid_rows is not None and (
        valid_rows.shape != run_mask.shape or valid_rows.dtype != torch.bool
    ):
        raise ValueError("valid_rows must be a bool tensor aligned with the route mask")
    if (
        stats.dtype != torch.int64
        or stats.numel() != 3
        or stats.device != run_mask.device
        or not stats.is_contiguous()
    ):
        raise ValueError("stats must be the contiguous int64 [3] accumulator on the mask device")
    # Fail closed on strided inputs instead of copying: a hidden `.contiguous()`
    # copy would add the launch this kernel exists to remove, and the unfused
    # kernel read its masks with unit-stride offsets too (Codex review, 2026-09-05).
    if not run_mask.is_contiguous() or (valid_rows is not None and not valid_rows.is_contiguous()):
        raise ValueError("route mask and valid_rows must be contiguous")
    rows = int(run_mask.numel())
    counts = torch.zeros(2, dtype=torch.int32, device=run_mask.device)
    run_rows = torch.empty(rows, dtype=torch.int32, device=run_mask.device)
    project_rows = torch.empty_like(run_rows)
    if rows == 0:
        # An empty batch launches no program, so the call counter would not
        # advance inside the kernel; keep the old semantics (one call, no rows).
        stats[0].add_(1)
        return run_rows, project_rows, counts
    block_rows = 256
    has_valid = valid_rows is not None
    _build_route_maps_masked_kernel[(triton.cdiv(rows, block_rows),)](
        run_mask,
        valid_rows if has_valid else run_mask,
        run_rows,
        project_rows,
        counts,
        stats,
        row_count=rows,
        block_rows=block_rows,
        has_valid=has_valid,
    )
    return run_rows, project_rows, counts
@triton.jit
def _route_decide_maps_kernel(
    weights_ptr,
    threshold,
    valid_ptr,
    run_mask_out_ptr,
    weight_tape_ptr,
    run_rows_ptr,
    project_rows_ptr,
    counts_ptr,
    decide_stats_ptr,
    row_count: tl.constexpr,
    block_rows: tl.constexpr,
    has_valid: tl.constexpr,
    has_weight_tape: tl.constexpr,
):
    """Lane-2 Track B / F1: route DECISION + tape writes + route maps, one launch.

    Fuses, per routed layer, ``torch.gt(branch_weights, threshold, out=tape_row)``
    (the RUN mask), the branch-weight tape copy, and the Block 1B-1 route
    maps/counts (``_build_route_maps_masked_kernel`` minus the stats accumulation,
    which stays in the executing body -- ``accumulate_route_counts`` -- so the
    ``binary_cohort.realized`` evidence keeps its meaning: bodies that ran).
    The compare is the same elementwise ``w > threshold`` on the same values
    (PyTorch compares fp16/bf16 in fp32 opmath; the threshold is exactly
    representable), the tape writes are exact copies, and the maps are the same
    boolean logic + integer atomics -- so mask, tape, maps and counts are
    bit-identical to the unfused sequence. Slot order under atomics is
    unspecified in both, as before.
    """
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    in_range = rows < row_count
    w = tl.load(weights_ptr + rows, mask=in_range, other=0.0)
    run = w.to(tl.float32) > threshold
    tl.store(run_mask_out_ptr + rows, run.to(tl.uint8), mask=in_range)
    if has_weight_tape:
        tl.store(
            weight_tape_ptr + rows,
            w.to(weight_tape_ptr.dtype.element_ty),
            mask=in_range,
        )
    run_mask = run.to(tl.int32)
    if has_valid:
        valid = tl.load(valid_ptr + rows, mask=in_range, other=0).to(tl.int32)
    else:
        valid = in_range.to(tl.int32)
    run_active = in_range & (valid != 0) & (run_mask != 0)
    project_active = in_range & (valid != 0) & (run_mask == 0)
    counter_lanes = tl.zeros((block_rows,), dtype=tl.int32)
    run_slots = tl.atomic_add(counts_ptr + counter_lanes, 1, mask=run_active)
    project_slots = tl.atomic_add(
        counts_ptr + 1 + counter_lanes, 1, mask=project_active
    )
    tl.store(run_rows_ptr + run_slots, rows, mask=run_active)
    tl.store(project_rows_ptr + project_slots, rows, mask=project_active)
    tl.atomic_add(decide_stats_ptr, (tl.program_id(0) == 0).to(tl.int64))
@triton.jit
def _accumulate_counts_kernel(counts_ptr, stats_ptr):
    """``stats[0] += 1; stats[1] += counts[0]; stats[2] += counts[1]`` (one program)."""
    tl.store(stats_ptr, tl.load(stats_ptr) + 1)
    tl.store(stats_ptr + 1, tl.load(stats_ptr + 1) + tl.load(counts_ptr).to(tl.int64))
    tl.store(stats_ptr + 2, tl.load(stats_ptr + 2) + tl.load(counts_ptr + 1).to(tl.int64))
@triton.jit
def _predicate_from_counts_kernel(counts_ptr, predicate_ptr):
    """all-valid-rows-RUN <=> the PROJECT count (valid & ~run) is zero."""
    tl.store(predicate_ptr, (tl.load(counts_ptr + 1) == 0).to(tl.int32))
def _route_decide_stats(device: torch.device) -> torch.Tensor:
    from sglang.srt.vpipe.env import _ROUTE_DECIDE_STATS

    stats = _ROUTE_DECIDE_STATS.get(device)
    if stats is None:
        stats = torch.zeros(1, dtype=torch.int64, device=device)
        _ROUTE_DECIDE_STATS[device] = stats
    return stats
def route_decide_and_maps(
    branch_weights: torch.Tensor,
    threshold: float,
    valid_rows: Optional[torch.Tensor],
    run_mask_out: torch.Tensor,
    weight_tape_out: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused route decision (F1): writes ``run_mask_out`` (= ``branch_weights >
    threshold``) and ``weight_tape_out`` (= a copy of ``branch_weights``), and
    returns the Block 1B-1 route maps ``(run_rows, project_rows, counts)`` for
    the executing body. Stats are NOT accumulated here (see
    ``accumulate_route_counts``). Every operand is 1-D, contiguous, on one CUDA
    device; anything else fails closed instead of copying.
    """

    if not branch_weights.is_cuda or branch_weights.ndim != 1:
        raise ValueError("route decision needs 1-D CUDA branch weights")
    if branch_weights.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("route decision needs float branch weights")
    rows = int(branch_weights.numel())
    if run_mask_out.dtype != torch.bool or run_mask_out.ndim != 1 or run_mask_out.numel() != rows:
        raise ValueError("route decision needs a 1-D bool run-mask target aligned with the weights")
    if run_mask_out.device != branch_weights.device:
        raise ValueError("route decision operands must share a device")
    if valid_rows is not None and (
        valid_rows.shape != branch_weights.shape or valid_rows.dtype != torch.bool
    ):
        raise ValueError("valid_rows must be a bool tensor aligned with the branch weights")
    if weight_tape_out is not None and (
        weight_tape_out.ndim != 1 or weight_tape_out.numel() != rows
    ):
        raise ValueError("weight tape target must be 1-D and aligned with the branch weights")
    # Codex review (2026-09-07, MAJOR): the optional operands are dereferenced by
    # the Triton launch, so a CPU or other-device tensor must fail closed here.
    for name, tensor in (("valid_rows", valid_rows), ("weight_tape_out", weight_tape_out)):
        if tensor is not None and tensor.device != branch_weights.device:
            raise ValueError(
                f"route decision operand {name} must live on {branch_weights.device}"
            )
    for name, tensor in (
        ("branch_weights", branch_weights),
        ("run_mask_out", run_mask_out),
        ("valid_rows", valid_rows),
        ("weight_tape_out", weight_tape_out),
    ):
        if tensor is not None and not tensor.is_contiguous():
            raise ValueError(f"route decision operand {name} must be contiguous")
    device = branch_weights.device
    counts = torch.zeros(2, dtype=torch.int32, device=device)
    run_rows = torch.empty(rows, dtype=torch.int32, device=device)
    project_rows = torch.empty_like(run_rows)
    if rows == 0:
        return run_rows, project_rows, counts
    block_rows = 256
    has_valid = valid_rows is not None
    has_weight_tape = weight_tape_out is not None
    _route_decide_maps_kernel[(triton.cdiv(rows, block_rows),)](
        branch_weights,
        float(threshold),
        valid_rows if has_valid else branch_weights,
        run_mask_out.view(torch.uint8),
        weight_tape_out if has_weight_tape else branch_weights,
        run_rows,
        project_rows,
        counts,
        _route_decide_stats(device),
        row_count=rows,
        block_rows=block_rows,
        has_valid=has_valid,
        has_weight_tape=has_weight_tape,
    )
    return run_rows, project_rows, counts
def accumulate_route_counts(counts: torch.Tensor, stats: torch.Tensor) -> None:
    """``stats += [1, counts[0], counts[1]]`` on device, one launch (the 1B-1
    stats semantics, kept in the executing body)."""

    if counts.dtype != torch.int32 or counts.numel() != 2 or not counts.is_contiguous():
        raise ValueError("route counts must be the contiguous int32 [2] tensor")
    if stats.dtype != torch.int64 or stats.numel() != 3 or not stats.is_contiguous():
        raise ValueError("stats must be the contiguous int64 [3] accumulator")
    if stats.device != counts.device:
        raise ValueError("route counts and stats must share a device")
    _accumulate_counts_kernel[(1,)](counts, stats)
def all_run_predicate_from_counts(
    counts: torch.Tensor, predicate_out: torch.Tensor
) -> None:
    """Write the conditional-graph predicate (int32 scalar, 1 = every valid
    row RUNs) from the fused route counts: ``predicate = counts[1] == 0``."""

    if counts.dtype != torch.int32 or counts.numel() != 2 or not counts.is_contiguous():
        raise ValueError("route counts must be the contiguous int32 [2] tensor")
    if predicate_out.dtype != torch.int32 or predicate_out.numel() != 1:
        raise ValueError("predicate target must be one int32 element")
    if predicate_out.device != counts.device:
        raise ValueError("route counts and predicate must share a device")
    _predicate_from_counts_kernel[(1,)](counts, predicate_out)
def full_graph_forced_route(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[bool]:
    """Return a sealed mechanism-control route, or ``None`` for the router."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_FORCE_ROUTE_ENV, "off") or "off").strip().lower()
    if value not in _VALID_FORCED_ROUTES:
        choices = ", ".join(sorted(_VALID_FORCED_ROUTES))
        raise ValueError(
            f"{FD_FORCE_ROUTE_ENV} must be one of {choices}; got {value!r}"
        )
    if value == "off":
        return None
    return value == "all_run"
@dataclass(slots=True)
class FullGraphPreparedLayerRoute:
    """Stable tensors shared by one routed layer's conditional bodies."""

    layer_id: int
    hidden_states: torch.Tensor
    residual: torch.Tensor
    route_weights: torch.Tensor
    run_mask: torch.Tensor
    parity_context: Optional[tuple[int, int]]
    inline_kv_index: Optional[int]
    action_batch: Optional[FullGraphActionBatch] = None
    run_row_map: Optional[torch.Tensor] = None
    project_row_map: Optional[torch.Tensor] = None
    route_counts: Optional[torch.Tensor] = None
def fd_prepare_layer_route_full_graph(
    layer: Any,
    hidden_states: torch.Tensor,
    forward_batch: Any,
    residual: Optional[torch.Tensor],
    router: Any,
) -> FullGraphPreparedLayerRoute:
    """Normalize and route one layer without executing either compute body."""

    layer_id = int(layer.layer_id)
    parity_context = None
    if _FD_PARITY_TRACE_ENABLED:

        parity_context = fd_parity_trace_context(layer_id, forward_batch)
    if parity_context is not None:

        parity_row, parity_epoch = parity_context
        fd_parity_trace_tensor(
            "input_hidden",
            hidden_states[parity_row : parity_row + 1],
            layer_id=layer_id,
            token_epoch=parity_epoch,
        )
        if residual is not None:
            fd_parity_trace_tensor(
                "input_residual",
                residual[parity_row : parity_row + 1],
                layer_id=layer_id,
                token_epoch=parity_epoch,
            )

    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)

    if parity_context is not None:
        fd_parity_trace_tensor(
            "prepared_hidden",
            hidden_states[parity_row : parity_row + 1],
            layer_id=layer_id,
            token_epoch=parity_epoch,
        )
        fd_parity_trace_tensor(
            "prepared_residual",
            residual[parity_row : parity_row + 1],
            layer_id=layer_id,
            token_epoch=parity_epoch,
        )

    forced_route = full_graph_forced_route()
    skipper_adapter = getattr(
        forward_batch, "fd_full_graph_skipper_adapter", None
    )
    if skipper_adapter is None:
        skipper_adapter = resolve_full_graph_skipper()
    action_batch = skipper_adapter.route(
        hidden_states,
        router=router,
        forced_action=forced_route_action(forced_route),
        layer_id=layer_id,
        batch_state=getattr(
            forward_batch, "fd_full_graph_skipper_state", None
        ),
    )
    device_tape = getattr(
        forward_batch, "fd_full_graph_device_route_tape", None
    )
    route_weights = action_batch.route_weights
    fused_maps = None
    if (
        device_tape is not None
        and action_batch.execution_kind == RUN_PROJECT_EXECUTION
        and action_batch.forced_action is None
        and action_batch.explicit_run_mask is None
        and forward_batch.forward_mode.is_decode()
    ):
        # Lane-2 tax-removal track / F1: the router-driven decision, the tape
        # writes and the route maps in ONE launch (bit-identical to the unfused
        # sequence; see `_route_decide_maps_kernel`). DECODE passes only: the
        # decode skip body is captured by the conditional-graph backend, where
        # the per-layer launch count is the tax. Prefill keeps the unfused
        # sequence — the 2026-09-07 box gate showed captured (BCG) prefill
        # passes drifting in first-token logprobs on 8/32 prompts when the maps
        # crossed a segment boundary, while eager prefill was identical; prefill
        # is compute-bound and gains nothing from the launch saving. Sealed/
        # forced or explicit-mask skippers keep the unfused path as well.
        run_mask, fused_maps = device_tape.action_mask_fused(
            int(layer.layer_id),
            action_batch,
            forward_batch.fd_full_graph_valid_rows,
        )
    elif device_tape is not None:
        run_mask = device_tape.action_mask(
            int(layer.layer_id),
            action_batch,
        )
    else:
        run_mask = action_batch.write_run_storage()
    inline_kv_index = (
        device_tape.reserve_inline_kv_layer(layer_id)
        if device_tape is not None
        else None
    )
    if parity_context is not None:

        fd_parity_trace_tensor(
            "route_weight",
            route_weights[parity_row : parity_row + 1],
            layer_id=layer_id,
            token_epoch=parity_epoch,
        )
        fd_parity_trace_tensor(
            "route_mask",
            run_mask[parity_row : parity_row + 1],
            layer_id=layer_id,
            token_epoch=parity_epoch,
        )
    run_row_map = None
    project_row_map = None
    route_counts = None
    if fused_maps is not None:
        run_row_map, project_row_map, route_counts = fused_maps
    if full_graph_compact_routed_qkv_enabled():
        valid_rows = forward_batch.fd_full_graph_valid_rows
        run_rows = run_mask.squeeze(-1)
        if valid_rows is None or valid_rows.shape != run_rows.shape:
            raise RuntimeError(
                "compact routed QKV requires aligned graph-valid rows"
            )
        if fused_maps is None:
            # Unfused route (forced / explicit-mask skipper): build the maps
            # the compact-QKV path needs. The fused route already carries the
            # same maps (build_route_maps_from_mask is equivalent to
            # build_route_maps(run & valid, ~run & valid), Block 1B-1).
            run_row_map, project_row_map, route_counts = build_route_maps(
                run_rows & valid_rows,
                (~run_rows) & valid_rows,
            )
    return FullGraphPreparedLayerRoute(
        layer_id=layer_id,
        hidden_states=hidden_states,
        residual=residual,
        route_weights=route_weights,
        run_mask=run_mask,
        parity_context=parity_context,
        inline_kv_index=inline_kv_index,
        action_batch=action_batch,
        run_row_map=run_row_map,
        project_row_map=project_row_map,
        route_counts=route_counts,
    )
def fd_parity_trace_attention(
    *,
    layer_id: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    forward_batch,
    cache_written: bool,
) -> None:
    """Trace projected Q/K/V and, after attention, the physical K/V cache row."""

    context = fd_parity_trace_context(layer_id, forward_batch)
    if context is None:
        return
    row, epoch = context
    if not cache_written:
        for phase, tensor in (
            ("attention_q", q),
            ("attention_k", k),
            ("attention_v", v),
            ("token_position", positions),
            ("cache_position", getattr(forward_batch, "out_cache_loc", None)),
        ):
            if tensor is None:
                raise RuntimeError(f"FlexiDepth parity trace is missing {phase}")
            fd_parity_trace_tensor(
                phase,
                tensor[row : row + 1],
                layer_id=layer_id,
                token_epoch=epoch,
            )
        return

    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    pool = getattr(backend, "token_to_kv_pool", None)
    if pool is None or not hasattr(pool, "get_kv_buffer"):
        raise RuntimeError("FlexiDepth parity trace cannot access the K/V pool")
    cache_k, cache_v = pool.get_kv_buffer(layer_id)
    cache_locations = getattr(forward_batch, "out_cache_loc", None)
    if cache_locations is None:
        raise RuntimeError("FlexiDepth parity trace is missing cache locations")
    selected_location = cache_locations[row : row + 1].to(dtype=torch.long)
    fd_parity_trace_tensor(
        "cache_k",
        cache_k.index_select(0, selected_location),
        layer_id=layer_id,
        token_epoch=epoch,
    )
    fd_parity_trace_tensor(
        "cache_v",
        cache_v.index_select(0, selected_location),
        layer_id=layer_id,
        token_epoch=epoch,
    )
