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
"""

from __future__ import annotations

import triton.language as tl
from typing import Any, Mapping, Optional
from sglang.srt.vpipe.common import (
    full_graph_compact_routed_qkv_enabled,
)
from sglang.srt.vpipe.common import (
    _fdvp_router_graph_enabled,
    _fdvp_timing_enabled,
    fdvp_fused_project_input_enabled,
    fdvp_fused_project_input_shared_storage_enabled,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_trace_signal_flush_enabled,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_trace_enabled,
)
import json
import signal
import triton
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_capturing_breakable_cuda_graph,
)
from sglang.srt.vpipe.common import (
    forced_route_action,
    resolve_full_graph_skipper,
)
from sglang.srt.vpipe.env import (
    FD_COMPACT_ROUTED_QKV_ENV,
    FD_FORCE_ROUTE_ENV,
    _FD_PARITY_TRACE_ENABLED,
    _VALID_FORCED_ROUTES,
)
from sglang.srt.vpipe.env import (
    SUBLAYER_EXECUTION,
)
from sglang.srt.vpipe.types import (
    FullGraphActionBatch,
)
from sglang.srt.vpipe.common import (
    _FD_PARITY_EPOCHS,
    fd_parity_trace_target,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_state,
)
from sglang.srt.vpipe.kv_commit import (
    _fdvp_record_cache_counter,
)
from sglang.srt.vpipe.kv_commit import (
    _FDVP_TRACE_STATE,
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
def fd_parity_trace_tensor(
    phase: str,
    tensor,
    *,
    layer_id: int,
    token_epoch: int,
) -> None:
    """Persist one target row for deterministic direct/V2 parity diagnostics."""

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
class FDRMSNorm(nn.Module):
    """FlexiDepth/DDLlama RMSNorm: compute variance in fp32, return input dtype."""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
class FDRouter(nn.Module):
    """DDLlamaRouter: bottleneck MLP -> scalar skip logit per token (no bias)."""

    def __init__(self, hidden_size=4096, reduction=16, eps=1e-5):
        super().__init__()
        r = hidden_size // reduction
        self.router_enc = nn.Linear(hidden_size, r, bias=False)
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
    run_row_map: Optional[torch.Tensor] = None
    project_row_map: Optional[torch.Tensor] = None
    route_counts: Optional[torch.Tensor] = None
    action_batch: Optional[FullGraphActionBatch] = None
    attention_run_mask: Optional[torch.Tensor] = None
    mlp_run_mask: Optional[torch.Tensor] = None
    attention_skip_scale: Optional[torch.Tensor] = None
    mlp_skip_scale: Optional[torch.Tensor] = None
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
    if action_batch.execution_kind == SUBLAYER_EXECUTION:
        route_weights = action_batch.branch_weights
        if device_tape is not None:
            attention_run_mask, mlp_run_mask = (
                device_tape.sublayer_action_masks(
                    int(layer.layer_id),
                    action_batch,
                )
            )
        else:
            attention_run_mask, mlp_run_mask = (
                action_batch.write_sublayer_storage()
            )
        run_mask = attention_run_mask & mlp_run_mask
    else:
        route_weights = action_batch.route_weights
        if device_tape is not None:
            run_mask = device_tape.action_mask(
                int(layer.layer_id),
                action_batch,
            )
        else:
            run_mask = action_batch.write_run_storage()
        attention_run_mask = run_mask
        mlp_run_mask = run_mask
    inline_kv_index = (
        device_tape.reserve_inline_kv_layer(layer_id)
        if device_tape is not None
        else None
    )
    run_row_map = None
    project_row_map = None
    route_counts = None
    if full_graph_compact_routed_qkv_enabled():
        valid_rows = forward_batch.fd_full_graph_valid_rows
        run_rows = attention_run_mask.squeeze(-1)
        if valid_rows is None or valid_rows.shape != run_rows.shape:
            raise RuntimeError(
                "compact routed QKV requires aligned graph-valid rows"
            )

        run_row_map, project_row_map, route_counts = build_route_maps(
            run_rows & valid_rows,
            (~run_rows) & valid_rows,
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
    return FullGraphPreparedLayerRoute(
        layer_id=layer_id,
        hidden_states=hidden_states,
        residual=residual,
        route_weights=route_weights,
        run_mask=run_mask,
        parity_context=parity_context,
        inline_kv_index=inline_kv_index,
        run_row_map=run_row_map,
        project_row_map=project_row_map,
        route_counts=route_counts,
        action_batch=action_batch,
        attention_run_mask=attention_run_mask,
        mlp_run_mask=mlp_run_mask,
        attention_skip_scale=action_batch.attention_skip_scale,
        mlp_skip_scale=action_batch.mlp_skip_scale,
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
