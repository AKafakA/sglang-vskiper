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


def _device_key(device) -> int:
    if device.type != "cuda":
        return -1
    return int(device.index if device.index is not None else torch.cuda.current_device())
def _trace_counter(name: str, delta: int = 1) -> None:
    try:

        _fdvp_record_cache_counter(name, delta)
    except Exception:
        pass
def _fdvp_record_cache_counter(name, delta=1):
    if not _fdvp_trace_enabled():
        return
    state = _fdvp_state()
    state[name] = int(state.get(name, 0)) + int(delta)
def _fdvp_trace_enabled():
    return False  # [D-609] pinned; trace knob, not configurable
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
    return True  # [D-609] pinned; trace knob, not configurable
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
    path = ""  # [D-609] pinned; trace knob, not configurable
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    else:
        print(line, flush=True)
