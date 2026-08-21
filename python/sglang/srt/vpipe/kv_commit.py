"""Deferred PROJECT_ONLY K/V production, taken off the decode critical path.

Ordering is fenced by KVWorkIdentity(request_slot, request_id, token_epoch,
kv_position) plus PendingKVWork.layer_id: the fence guards slot reuse and drains
on finish. Every generated token still gets K/V from THAT layer's own projection
weights -- deferral changes WHEN the K/V is produced, never WHETHER."""

from __future__ import annotations

import json
import signal
from typing import Iterable, Sequence
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch


_BATCHED_KV_GRAPHS: dict[tuple, BatchedKVGraph] = {}
_BATCHED_KV_QUEUES: dict[tuple[int, int], list[BatchedKVWork]] = {}
_KV_READINESS_TRACKERS: dict[int, KVReadinessTracker] = {}
_BATCHED_KV_GRAPH_FAILED: set[tuple] = set()
def drain_request_kv_work(request_slot: int, request_id: str) -> int:
    count = 0
    for tracker in _KV_READINESS_TRACKERS.values():
        count += tracker.drain_request(
            request_slot, request_id, synchronize=True
        )
    return count
def reset_kv_readiness_trackers() -> None:
    _KV_READINESS_TRACKERS.clear()
def flush_request(request_slot: int, request_id: str, synchronize: bool = True) -> int:
    owner = int(request_slot), str(request_id)
    keys = [
        key
        for key, queue in _BATCHED_KV_QUEUES.items()
        if any(identity.owner == owner for work in queue for identity in work.identities)
    ]
    count = _flush_keys(keys, synchronize=synchronize)
    if count:
        _trace_counter("batched_kv_request_flushes")
    return count
def reset_batched_kv_queues() -> None:
    _BATCHED_KV_QUEUES.clear()
    _BATCHED_KV_GRAPHS.clear()
    _BATCHED_KV_GRAPH_FAILED.clear()
def fdvp_drain_async_kv(forward_batch):
    events = getattr(forward_batch, "_fdvp_async_kv_events", None)
    if not events:
        return 0
    if not torch.cuda.is_available():
        events.clear()
        return 0
    current_stream = torch.cuda.current_stream()
    for event in events:
        current_stream.wait_event(event)
    count = len(events)
    events.clear()
    drained = int(getattr(forward_batch, "_fdvp_async_kv_drains", 0))
    setattr(forward_batch, "_fdvp_async_kv_drains", drained + count)
    return count
def batched_kv_repair_graph_enabled() -> bool:
    """Replay exact projection, RoPE, and K/V scatter as one CUDA graph."""

    return (
        batched_kv_enabled()
        and os.environ.get("SGLANG_VP_ASYNC_KV_REPAIR_GRAPH", "0") == "1"
    )
class KVReadinessTracker:
    """Tracks side-stream K/V writes until their first dependent read."""

    def __init__(self) -> None:
        self._pending: list[PendingKVWork] = []

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def register(
        self,
        layer_id: int,
        event: object,
        identities: Sequence[KVWorkIdentity],
    ) -> bool:
        identities = tuple(identities)
        if not identities:
            return False
        self._pending.append(
            PendingKVWork(int(layer_id), event, identities)
        )
        return True

    @staticmethod
    def _wait_events(events: Iterable[object], stream=None, synchronize=False) -> int:
        events = tuple(events)
        for event in events:
            if synchronize and hasattr(event, "synchronize"):
                event.synchronize()
            elif stream is not None and hasattr(stream, "wait_event"):
                stream.wait_event(event)
            elif hasattr(event, "synchronize"):
                event.synchronize()
        return len(events)

    def _take(self, predicate) -> list[PendingKVWork]:
        selected = []
        keep = []
        for work in self._pending:
            (selected if predicate(work) else keep).append(work)
        self._pending = keep
        return selected

    def wait_for_reads(
        self,
        layer_id: int,
        readers: Sequence[KVWorkIdentity],
        stream=None,
    ) -> int:
        """Order reads after older writes for the same request and layer."""

        count, _ready = self.wait_for_reads_with_status(
            layer_id, readers, stream=stream, inspect_status=False
        )
        return count

    def wait_for_reads_with_status(
        self,
        layer_id: int,
        readers: Sequence[KVWorkIdentity],
        stream=None,
        inspect_status: bool = False,
    ) -> tuple[int, int]:
        """Order reads and optionally count events already complete at the fence."""

        reader_by_owner = {identity.owner: identity for identity in readers}

        def is_dependency(work: PendingKVWork) -> bool:
            if work.layer_id != int(layer_id):
                return False
            for writer in work.identities:
                reader = reader_by_owner.get(writer.owner)
                if reader is None:
                    continue
                if (
                    writer.token_epoch < reader.token_epoch
                    or writer.kv_position < reader.kv_position
                ):
                    return True
            return False

        selected = self._take(is_dependency)
        ready = 0
        if inspect_status:
            for work in selected:
                query = getattr(work.event, "query", None)
                if query is None:
                    continue
                try:
                    ready += int(bool(query()))
                except Exception:
                    pass
        count = self._wait_events(
            (work.event for work in selected), stream=stream
        )
        return count, ready

    def wait_for_slot_conflicts(
        self,
        current: Sequence[KVWorkIdentity],
        stream=None,
    ) -> int:
        """Fence stale work before a request-pool slot is reused by another RID."""

        owners_by_slot = {identity.request_slot: identity.request_id for identity in current}

        def conflicts(work: PendingKVWork) -> bool:
            return any(
                writer.request_slot in owners_by_slot
                and owners_by_slot[writer.request_slot] != writer.request_id
                for writer in work.identities
            )

        selected = self._take(conflicts)
        return self._wait_events((work.event for work in selected), stream=stream)

    def drain_request(
        self,
        request_slot: int,
        request_id: str,
        stream=None,
        synchronize: bool = False,
    ) -> int:
        owner = int(request_slot), str(request_id)
        selected = self._take(
            lambda work: any(identity.owner == owner for identity in work.identities)
        )
        return self._wait_events(
            (work.event for work in selected),
            stream=stream,
            synchronize=synchronize,
        )

    def drain_until_layer(self, max_layer_exclusive: Optional[int], stream=None) -> int:
        selected = self._take(
            lambda work: max_layer_exclusive is None
            or work.layer_id < 0
            or work.layer_id < int(max_layer_exclusive)
        )
        return self._wait_events((work.event for work in selected), stream=stream)

    def drain_all(self, stream=None, synchronize: bool = False) -> int:
        selected = self._take(lambda _work: True)
        return self._wait_events(
            (work.event for work in selected),
            stream=stream,
            synchronize=synchronize,
        )
def batched_kv_enabled() -> bool:
    return os.environ.get("SGLANG_VP_ASYNC_KV_BATCHED", "0") == "1"
@dataclass
class PendingKVWork:
    layer_id: int
    event: object
    identities: tuple[KVWorkIdentity, ...]
@dataclass
class BatchedKVGraph:
    hidden: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    graph: torch.cuda.CUDAGraph
@dataclass(frozen=True)
class KVWorkIdentity:
    request_slot: int
    request_id: str
    token_epoch: int
    kv_position: int

    @property
    def owner(self) -> tuple[int, str]:
        return self.request_slot, self.request_id
@dataclass
class BatchedKVWork:
    attn: object
    positions: torch.Tensor
    hidden: torch.Tensor
    out_cache_loc: torch.Tensor
    identities: tuple[KVWorkIdentity, ...]
    kv_pool: object
    layer_id: int
def _device_key(device) -> int:
    if device.type != "cuda":
        return -1
    return int(device.index if device.index is not None else torch.cuda.current_device())
def _trace_counter(name: str, delta: int = 1) -> None:
    try:

        _fdvp_record_cache_counter(name, delta)
    except Exception:
        pass
def _flush_keys(keys: Iterable[tuple[int, int]], synchronize: bool = False) -> int:
    works_flushed = 0
    rows_flushed = 0
    devices = set()
    for key in tuple(dict.fromkeys(keys)):
        works = _BATCHED_KV_QUEUES.pop(key, None)
        if not works:
            continue
        works_flushed += len(works)
        rows_flushed += _execute_batch(works)
        devices.add(works[0].hidden.device)
    if synchronize:
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    if works_flushed:
        _trace_counter("batched_kv_flushes")
        _trace_counter("batched_kv_flushed_work", works_flushed)
        _trace_counter("batched_kv_flushed_rows", rows_flushed)
    return works_flushed
def _execute_batch(
    works: Sequence[BatchedKVWork], *, use_repair_graph: bool = False
) -> int:
    if not works:
        return 0
    first = works[0]
    if any(work.attn is not first.attn or work.kv_pool is not first.kv_pool for work in works):
        raise RuntimeError("batched K/V queue mixed different layer or pool owners")

    if len(works) == 1:
        hidden = first.hidden
        positions = first.positions
        out_cache_loc = first.out_cache_loc
    else:
        hidden = torch.cat([work.hidden for work in works], dim=0)
        positions = torch.cat([work.positions for work in works], dim=0)
        out_cache_loc = torch.cat([work.out_cache_loc for work in works], dim=0)
    if not (
        use_repair_graph
        and _try_execute_repair_graph(first, hidden, positions, out_cache_loc)
    ):
        _execute_projection(first, hidden, positions, out_cache_loc)
    return int(hidden.shape[0])
def _fdvp_record_cache_counter(name, delta=1):
    if not _fdvp_trace_enabled():
        return
    state = _fdvp_state()
    state[name] = int(state.get(name, 0)) + int(delta)
def _execute_projection(first, hidden, positions, out_cache_loc) -> None:
    from sglang.srt.vpipe.cohort import (
        _fd_prepare_kv_only,
        _fd_prepare_qkv,
    )

    with torch.no_grad():
        qkv = _fd_prepare_kv_only(first.attn, positions, hidden)
        if qkv is None:
            _q, k, v = _fd_prepare_qkv(first.attn, positions, hidden)
        else:
            _q, k, v = qkv
            _trace_counter("batched_kv_kv_only_batches")
            _trace_counter("batched_kv_kv_only_rows", int(hidden.shape[0]))
        radix_attn = first.attn.attn
        k = k.view(-1, radix_attn.tp_k_head_num, radix_attn.qk_head_dim)
        v = v.view(-1, radix_attn.tp_v_head_num, radix_attn.v_head_dim)
        first.kv_pool.set_kv_buffer(
            radix_attn,
            out_cache_loc,
            k,
            v,
            radix_attn.k_scale,
            radix_attn.v_scale,
        )
def _try_execute_repair_graph(first, hidden, positions, out_cache_loc) -> bool:
    if not batched_kv_repair_graph_enabled():
        return False
    if (
        not hidden.is_cuda
        or not positions.is_cuda
        or not out_cache_loc.is_cuda
        or hidden.requires_grad
    ):
        return False
    rows = int(hidden.shape[0])
    max_rows = _repair_graph_max_rows()
    if max_rows > 0 and rows > max_rows:
        _trace_counter("batched_kv_repair_graph_row_skips")
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        return False

    key = _repair_graph_key(first, hidden, positions, out_cache_loc)
    if key in _BATCHED_KV_GRAPH_FAILED:
        _trace_counter("batched_kv_repair_graph_failed_skips")
        return False
    cached = _BATCHED_KV_GRAPHS.get(key)
    if cached is not None:
        cached.hidden.copy_(hidden)
        cached.positions.copy_(positions)
        cached.out_cache_loc.copy_(out_cache_loc)
        cached.graph.replay()
        _trace_counter("batched_kv_repair_graph_hits")
        _trace_counter("batched_kv_repair_graph_hit_rows", rows)
        return True

    max_entries = _repair_graph_max_entries()
    if max_entries > 0 and len(_BATCHED_KV_GRAPHS) >= max_entries:
        _trace_counter("batched_kv_repair_graph_overflow")
        return False

    try:
        stream = torch.cuda.current_stream(hidden.device)
        static_hidden = torch.empty_like(hidden)
        static_positions = torch.empty_like(positions)
        static_out_cache_loc = torch.empty_like(out_cache_loc)
        static_hidden.copy_(hidden)
        static_positions.copy_(positions)
        static_out_cache_loc.copy_(out_cache_loc)
        for _ in range(2):
            _execute_projection(
                first, static_hidden, static_positions, static_out_cache_loc
            )
        stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _execute_projection(
                first, static_hidden, static_positions, static_out_cache_loc
            )
        stream.synchronize()
        cached = BatchedKVGraph(
            hidden=static_hidden,
            positions=static_positions,
            out_cache_loc=static_out_cache_loc,
            graph=graph,
        )
        _BATCHED_KV_GRAPHS[key] = cached
        _trace_counter("batched_kv_repair_graph_misses")
        _trace_counter("batched_kv_repair_graph_miss_rows", rows)
        _trace_value("batched_kv_repair_graph_entries", len(_BATCHED_KV_GRAPHS))
        # Capture records the side effects but does not execute them.
        graph.replay()
        return True
    except Exception:
        _BATCHED_KV_GRAPH_FAILED.add(key)
        _trace_counter("batched_kv_repair_graph_capture_failures")
        return False
def _fdvp_trace_enabled():
    return os.environ.get("SGLANG_FD_VP_TRACE") == "1"
def _fdvp_state():
    _fdvp_install_trace_signal_handlers()
    if _FDVP_TRACE_STATE:
        return _FDVP_TRACE_STATE
    _FDVP_TRACE_STATE.update(
        {
            "calls": 0,
            "rows": 0,
            "kept_rows": 0,
            "skipped_rows": 0,
            "all_run_calls": 0,
            "all_skip_calls": 0,
            "mixed_calls": 0,
            "fallback_calls": 0,
            "timing_ms": {},
            "branch_timing_ms": {},
            "row_count_hist": {},
            "kept_count_hist": {},
            "mask_hist": {},
            "mask_hist_overflow": 0,
            "shape_hist": {},
            "shape_hist_overflow": 0,
            "cohort_transition_hist": {},
            "cohort_transition_overflow": 0,
            "phases": {},
            "layers": {},
        }
    )
    return _FDVP_TRACE_STATE
_FDVP_TRACE_STATE = {}
def _repair_graph_max_rows() -> int:
    try:
        return max(
            0,
            int(os.environ.get("SGLANG_VP_ASYNC_KV_REPAIR_GRAPH_MAX_ROWS", "64") or "64"),
        )
    except ValueError:
        return 64
def _repair_graph_max_entries() -> int:
    try:
        return max(
            0,
            int(
                os.environ.get("SGLANG_VP_ASYNC_KV_REPAIR_GRAPH_MAX_ENTRIES", "64")
                or "64"
            ),
        )
    except ValueError:
        return 64
def _trace_value(name: str, value: int) -> None:
    try:
        
        _fdvp_record_cache_value(name, value)
    except Exception:
        pass
def _repair_graph_key(first, hidden, positions, out_cache_loc) -> tuple:
    return (
        _device_key(hidden.device),
        id(first.attn),
        id(first.kv_pool),
        tuple(int(dim) for dim in hidden.shape),
        str(hidden.dtype),
        tuple(int(dim) for dim in positions.shape),
        str(positions.dtype),
        tuple(int(dim) for dim in out_cache_loc.shape),
        str(out_cache_loc.dtype),
        os.environ.get("SGLANG_VP_KV_ONLY_QKV", "0") == "1"
        or os.environ.get("SGLANG_FD_VP_ALL_SKIP_KV_ONLY_QKV", "0") == "1",
    )
def _fdvp_install_trace_signal_handlers():
    global _FDVP_TRACE_SIGNAL_HANDLERS_INSTALLED
    if _FDVP_TRACE_SIGNAL_HANDLERS_INSTALLED:
        return
    if not _fdvp_trace_enabled() or not _fdvp_trace_signal_flush_enabled():
        return
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, _fdvp_signal_emit_and_exit)
        except Exception:
            pass
    _FDVP_TRACE_SIGNAL_HANDLERS_INSTALLED = True
_FDVP_TRACE_SIGNAL_HANDLERS_INSTALLED = False
def _fdvp_trace_signal_flush_enabled():
    return os.environ.get("SGLANG_FD_VP_TRACE_SIGNAL_FLUSH", "1") != "0"
def _fdvp_signal_emit_and_exit(signum, _frame):
    _fdvp_emit(f"signal_{int(signum)}")
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
def _fdvp_record_cache_value(name, value):
    if not _fdvp_trace_enabled():
        return
    state = _fdvp_state()
    state[name] = int(value)
def _fdvp_emit(reason):
    if not _fdvp_trace_enabled() or not _FDVP_TRACE_STATE:
        return
    payload = {"event": "FDVP_TRACE", "reason": reason, **_FDVP_TRACE_STATE}
    line = json.dumps(payload, sort_keys=True)
    path = os.environ.get("SGLANG_FD_VP_TRACE_FILE", "")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    else:
        print(line, flush=True)
