"""Device-resident route tape — record what each token did, without stalling.

Route decisions are accumulated ON DEVICE and only summarised to the host at
batch end. A per-layer host read would put ~32 device-to-host syncs per step on
the decode critical path, which is the shape of the original eager collapse.

The tape feeds two consumers: the runtime attestation surface (so a deployment
can prove which policy actually executed, not merely which was configured) and
the route digest used to establish faithfulness against the reference
implementation.
"""

from __future__ import annotations

import triton.language as tl
from types import SimpleNamespace
from typing import Any, Mapping, Optional
import triton
import msgspec
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch
from sglang.srt.vpipe.common import (
    full_graph_decode_enabled,
)
from sglang.srt.vpipe.env import (
    FD_DEVICE_ROUTE_DIGEST_ENV,
    FD_DEVICE_ROUTE_TAPE_ENV,
    FD_FUSED_EVIDENCE_ENV,
    FD_LAYER_COUNTERS_ENV,
    FD_ROUTE_ACCOUNTING_ENV,
)
from sglang.srt.vpipe.types import (
    FullGraphActionBatch,
)
from sglang.srt.vpipe.skipper import (
    _positive_int,
)
from sglang.srt.vpipe.common import (
    _strict_bool,
)
from sglang.srt.vpipe.env import (
    REBATCH_SHADOW_ENV,
    REBATCH_SHADOW_MAX_DISPATCH_ROWS_ENV,
    REBATCH_SHADOW_MAX_ROWS_ENV,
    REBATCH_SHADOW_MIN_ROWS_ENV,
    REBATCH_SHADOW_QUEUE_CAPACITY_ENV,
    REBATCH_SHADOW_WAIT_ROUNDS_ENV,
)


def _env_positive_int(
    values: Mapping[str, str], name: str, default: int
) -> int:
    raw = str(values.get(name, default)).strip()
    try:
        result = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result
COVERAGE_REASON_ROWS = "rows_gt_max_bs"
COVERAGE_REASON_INELIGIBLE = "not_graph_eligible"
COVERAGE_REASON_NO_RUNNER = "no_graph_runner"
COVERAGE_REASONS = (
    COVERAGE_REASON_ROWS,
    COVERAGE_REASON_INELIGIBLE,
    COVERAGE_REASON_NO_RUNNER,
)
_graph_lifecycle_depth = 0
@triton.jit
def _build_first_stage_admission_kernel(
    actions_ptr,
    tape_valid_rows_ptr,
    request_slots_ptr,
    logical_request_ids_ptr,
    token_epochs_ptr,
    cache_positions_ptr,
    branch_weights_ptr,
    round_id_ptr,
    queue_ids_ptr,
    output_valid_rows_ptr,
    work_ids_ptr,
    epoch_slots_ptr,
    output_request_slots_ptr,
    output_logical_request_ids_ptr,
    output_token_epochs_ptr,
    output_kv_positions_ptr,
    output_actions_ptr,
    output_branch_weights_ptr,
    ready_us_ptr,
    deadline_us_ptr,
    row_count: tl.constexpr,
    capacity: tl.constexpr,
    wait_rounds: tl.constexpr,
    block_rows: tl.constexpr,
):
    rows = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    in_capacity = rows < capacity
    in_tape = rows < row_count
    valid = in_tape & tl.load(
        tape_valid_rows_ptr + rows,
        mask=in_tape,
        other=0,
    ).to(tl.int1)
    run = tl.load(
        actions_ptr + rows,
        mask=in_tape,
        other=1,
    ).to(tl.int1)
    request_slot = tl.load(
        request_slots_ptr + rows, mask=in_tape, other=-1
    ).to(tl.int32)
    logical_request_id = tl.load(
        logical_request_ids_ptr + rows, mask=in_tape, other=-1
    ).to(tl.int64)
    token_epoch = tl.load(
        token_epochs_ptr + rows, mask=in_tape, other=-1
    ).to(tl.int64)
    cache_position = tl.load(
        cache_positions_ptr + rows, mask=in_tape, other=-1
    ).to(tl.int64)
    branch_weight = tl.load(
        branch_weights_ptr + rows, mask=in_tape, other=0.0
    ).to(tl.float32)
    action = tl.where(run, 0, 1).to(tl.int32)
    queue_id = action
    round_id = tl.load(round_id_ptr).to(tl.int64)
    work_id = (
        (logical_request_id + 1) * 1_000_003
        + (token_epoch + 1) * 1_000_033
    )

    tl.store(queue_ids_ptr + rows, queue_id, mask=in_capacity)
    tl.store(output_valid_rows_ptr + rows, valid, mask=in_capacity)
    tl.store(work_ids_ptr + rows, work_id, mask=in_capacity)
    tl.store(epoch_slots_ptr + rows, request_slot, mask=in_capacity)
    tl.store(
        output_request_slots_ptr + rows,
        request_slot,
        mask=in_capacity,
    )
    tl.store(
        output_logical_request_ids_ptr + rows,
        logical_request_id,
        mask=in_capacity,
    )
    tl.store(
        output_token_epochs_ptr + rows, token_epoch, mask=in_capacity
    )
    tl.store(
        output_kv_positions_ptr + rows,
        cache_position,
        mask=in_capacity,
    )
    tl.store(output_actions_ptr + rows, action, mask=in_capacity)
    tl.store(
        output_branch_weights_ptr + rows,
        branch_weight,
        mask=in_capacity,
    )
    tl.store(ready_us_ptr + rows, round_id, mask=in_capacity)
    tl.store(
        deadline_us_ptr + rows,
        round_id + wait_rounds,
        mask=in_capacity,
    )
class CoverageDenseCounters(msgspec.Struct):
    """Process-wide (c3) evidence counters, served via ``/server_info``.

    Two-site parity (F2/F18): site A (dispatch-seam decision) writes
    ``dense_overflow_*``; site B (dense-body entry at the first ROUTED layer,
    deduped per pass via ``vp_fd_coverage_counted``) writes ``dense_body_*``.
    The gate is ``decisions == passes`` and ``rows == rows`` — removing either
    site fails the cross-check.
    """

    # Site A — dispatch-seam decision witnesses.
    dense_overflow_decisions: int = 0
    dense_overflow_rows: int = 0
    overflow_reason: dict[str, int] = {}
    # Site B — dense-body positive witnesses (the Candidate-A lesson: the body
    # that ran is attested at the body, not at the decision).
    dense_body_passes: int = 0
    dense_body_rows: int = 0
    # Negative witnesses (invariant: exactly 0 in serving; >0 => cell VOID).
    eager_skip_decode_layer_calls: int = 0
    # V4/V2 scheduler-owned route body (F3 tripwire; pinned 0 while the V4
    # decode scheduler is disabled — nonzero forces the C-E obligation).
    v4_route_execute_decode_calls: int = 0
    # Body-mix accounting per SS5 (token == decode row here).
    fd_tokens_skip_body: int = 0
    fd_tokens_prod_allrun_band: int = 0
    fd_tokens_dense_overflow: int = 0
    # Composition observability.
    dispatch_graph_steps: int = 0
    regime_observe_eager: int = 0
    coverage_stamp_w1_composed: int = 0
    # Graph-lifecycle mark evidence (F1): guarded sections entered, and each
    # mid-serving recapture with its forward-pass index (visible evidence, not
    # a silent counter hole).
    graph_lifecycle_sections: int = 0
    recapture_events: list[int] = []

    @classmethod
    def create(cls) -> "CoverageDenseCounters":
        return cls(
            overflow_reason={reason: 0 for reason in COVERAGE_REASONS},
            recapture_events=[],
        )
def vp_graph_lifecycle_active() -> bool:
    """True inside capture()/warmup()/recapture in the decode graph runner.

    ``torch.cuda.is_current_stream_capturing()`` excludes only the capture
    invocation itself; the backend runs every body twice EAGERLY before
    capture, and warmup/recapture also execute bodies eagerly on decode
    dummies — the sentinel must not count any of them.
    """

    return _graph_lifecycle_depth > 0
def _serving_decode_layer_call(forward_batch: Any) -> bool:
    mode = forward_batch.forward_mode
    if mode is None or not mode.is_decode():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if vp_graph_lifecycle_active():
        return False
    return True
def record_eager_skip_decode_layer_call(forward_batch: Any) -> None:
    """Negative witness at the entry of every FD skip decode body.

    Invariant: exactly 0 in serving.  Excluded: stream capture, and the
    graph-lifecycle sections (capture warmups, boot warmup, recapture).
    """

    if _serving_decode_layer_call(forward_batch):
        _c3_counters.eager_skip_decode_layer_calls += 1
def resolve_model_route_tape_shadow(model: Any) -> Optional[Any]:
    """Find the optional Llama route-tape shadow without model-family coupling."""

    language_model = getattr(model, "model", None)
    return getattr(
        language_model,
        "_fd_full_graph_rebatching_shadow",
        None,
    )
@dataclass(frozen=True, slots=True)
class RouteTapeShadowConfig:
    enabled: bool
    max_rows: int
    queue_capacity: int
    min_dispatch_rows: int
    max_dispatch_rows: int
    wait_rounds: int
def route_tape_shadow_config(
    environ: Optional[Mapping[str, str]] = None,
) -> RouteTapeShadowConfig:
    values = os.environ if environ is None else environ
    enabled = _strict_bool(values, REBATCH_SHADOW_ENV)
    max_rows = _env_positive_int(values, REBATCH_SHADOW_MAX_ROWS_ENV, 4096)
    queue_capacity = _env_positive_int(
        values, REBATCH_SHADOW_QUEUE_CAPACITY_ENV, 4096
    )
    min_dispatch_rows = _env_positive_int(
        values, REBATCH_SHADOW_MIN_ROWS_ENV, 16
    )
    max_dispatch_rows = _env_positive_int(
        values, REBATCH_SHADOW_MAX_DISPATCH_ROWS_ENV, 64
    )
    wait_rounds = _env_positive_int(
        values, REBATCH_SHADOW_WAIT_ROUNDS_ENV, 4
    )
    if max_rows > 4096 or queue_capacity > 4096:
        raise ValueError("route-tape shadow rows and capacity cannot exceed 4096")
    if max_dispatch_rows > queue_capacity:
        raise ValueError(
            "route-tape shadow max dispatch exceeds queue capacity"
        )
    if min_dispatch_rows > max_dispatch_rows:
        raise ValueError(
            "route-tape shadow min dispatch exceeds max dispatch"
        )
    return RouteTapeShadowConfig(
        enabled=enabled,
        max_rows=max_rows,
        queue_capacity=queue_capacity,
        min_dispatch_rows=min_dispatch_rows,
        max_dispatch_rows=max_dispatch_rows,
        wait_rounds=wait_rounds,
    )
class FullGraphRouteTapeShadowBridge:
    """Observe first-stage route work without changing model execution."""

    def __init__(
        self,
        *,
        stage_id: int,
        max_rows: int,
        queue_capacity: int,
        min_dispatch_rows: int,
        max_dispatch_rows: int,
        wait_rounds: int,
        device: torch.device | str,
    ) -> None:
        # The V4 device-rebatching backend this bridge drives
        # (DeviceRebatchingQueues / DeviceDispatchResult) is NOT part of this
        # build -- it was removed as unreachable from every deployed arm.
        # Refuse clearly instead of dying on a NameError further down.
        raise RuntimeError(
            "SGLANG_FD_FULL_GRAPH_REBATCH_SHADOW is not supported by this "
            "build: the V4 device-rebatching backend it depends on is absent"
        )
        if stage_id < 0:
            raise ValueError("route-tape bridge stage_id must be non-negative")
        if max_rows <= 0:
            raise ValueError("route-tape bridge max_rows must be positive")
        if wait_rounds < 0:
            raise ValueError("route-tape bridge wait_rounds must be non-negative")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("route-tape bridge requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        buckets = []
        bucket = 1
        while bucket < max_dispatch_rows:
            buckets.append(bucket)
            bucket *= 2
        buckets.append(max_dispatch_rows)

        self.stage_id = stage_id
        self.max_rows = max_rows
        self.wait_rounds = wait_rounds
        self.device = device
        self.queues = DeviceRebatchingQueues(
            queue_count=2,
            queue_capacity=queue_capacity,
            max_insert_rows=max_rows,
            device=device,
            queue_stage_ids=(stage_id, stage_id),
            graph_buckets=tuple(buckets),
            min_dispatch_rows=min_dispatch_rows,
            max_dispatch_rows=max_dispatch_rows,
            max_wait_us=wait_rounds,
        )
        self.round_id = torch.zeros(1, dtype=torch.int64, device=device)
        self.queue_ids = torch.zeros(
            max_rows, dtype=torch.int32, device=device
        )
        self.valid_rows = torch.zeros(
            max_rows, dtype=torch.bool, device=device
        )
        self.metadata = {
            "work_ids": torch.full(
                (max_rows,), -1, dtype=torch.int64, device=device
            ),
            "epoch_slots": torch.full(
                (max_rows,), -1, dtype=torch.int32, device=device
            ),
            "request_slots": torch.full(
                (max_rows,), -1, dtype=torch.int32, device=device
            ),
            "logical_request_ids": torch.full(
                (max_rows,), -1, dtype=torch.int64, device=device
            ),
            "token_epochs": torch.full(
                (max_rows,), -1, dtype=torch.int64, device=device
            ),
            "kv_positions": torch.full(
                (max_rows,), -1, dtype=torch.int64, device=device
            ),
            "actions": torch.full(
                (max_rows,), -1, dtype=torch.int32, device=device
            ),
            "branch_weights": torch.full(
                (max_rows,), -1.0, dtype=torch.float32, device=device
            ),
            "ready_us": torch.full(
                (max_rows,), -1, dtype=torch.int64, device=device
            ),
            "deadline_us": torch.full(
                (max_rows,), -1, dtype=torch.int64, device=device
            ),
        }

    def _validate_tape(self, tape: Any) -> int:
        required = (
            "actions",
            "valid_rows",
            "request_slots",
            "logical_request_ids",
            "token_epochs",
            "cache_positions",
            "branch_weights",
        )
        values = {name: getattr(tape, name, None) for name in required}
        if any(value is None for value in values.values()):
            raise RuntimeError(
                "route-tape bridge requires complete logical identity metadata"
            )
        actions = values["actions"]
        rows = int(actions.shape[1]) if actions.ndim == 2 else -1
        if (
            actions.dtype != torch.bool
            or actions.shape[0] <= 0
            or rows <= 0
            or rows > self.max_rows
        ):
            raise RuntimeError("route-tape bridge action shape is unsupported")
        branch_weights = values["branch_weights"]
        if (
            branch_weights.shape != actions.shape
            or branch_weights.dtype != torch.float32
        ):
            raise RuntimeError(
                "route-tape bridge branch weights changed shape or dtype"
            )
        if values["valid_rows"].shape != (rows,):
            raise RuntimeError("route-tape bridge valid rows changed shape")
        if any(
            values[name].shape != (rows,)
            for name in (
                "request_slots",
                "logical_request_ids",
                "token_epochs",
                "cache_positions",
            )
        ):
            raise RuntimeError("route-tape bridge identity rows changed shape")
        tensors = tuple(values.values())
        if any(not tensor.is_cuda for tensor in tensors):
            raise RuntimeError("route-tape bridge requires CUDA route metadata")
        if any(tensor.device != self.device for tensor in tensors):
            raise RuntimeError(
                "route-tape bridge metadata must share the bridge device"
            )
        return rows

    def observe(self, tape: Any) -> DeviceDispatchResult:
        """Insert first-stage actions and run one output-neutral device round."""

        rows = self._validate_tape(tape)
        self.round_id.add_(1)
        block_rows = 256
        _build_first_stage_admission_kernel[
            (triton.cdiv(self.max_rows, block_rows),)
        ](
            tape.actions,
            tape.valid_rows,
            tape.request_slots,
            tape.logical_request_ids,
            tape.token_epochs,
            tape.cache_positions,
            tape.branch_weights[0],
            self.round_id,
            self.queue_ids,
            self.valid_rows,
            self.metadata["work_ids"],
            self.metadata["epoch_slots"],
            self.metadata["request_slots"],
            self.metadata["logical_request_ids"],
            self.metadata["token_epochs"],
            self.metadata["kv_positions"],
            self.metadata["actions"],
            self.metadata["branch_weights"],
            self.metadata["ready_us"],
            self.metadata["deadline_us"],
            row_count=rows,
            capacity=self.max_rows,
            wait_rounds=self.wait_rounds,
            block_rows=block_rows,
        )
        self.queues.insert(
            queue_ids=self.queue_ids,
            valid_rows=self.valid_rows,
            metadata=self.metadata,
        )
        return self.queues.select_and_drain(now_us=self.round_id)

    def warmup(self) -> None:
        """Compile all bridge kernels before production CUDA-graph capture."""

        rows = self.max_rows
        valid_rows = torch.zeros(rows, dtype=torch.bool, device=self.device)
        valid_rows[0] = True
        row_ids = torch.arange(rows, dtype=torch.int64, device=self.device)
        tape = SimpleNamespace(
            actions=torch.ones(
                (1, rows), dtype=torch.bool, device=self.device
            ),
            valid_rows=valid_rows,
            request_slots=row_ids.to(torch.int32),
            logical_request_ids=row_ids,
            token_epochs=torch.zeros_like(row_ids),
            cache_positions=row_ids,
            branch_weights=torch.ones(
                (1, rows), dtype=torch.float32, device=self.device
            ),
        )
        self.observe(tape)
        torch.cuda.synchronize(self.device)
        self.reset()

    def reset(self) -> None:
        self.round_id.zero_()
        self.queue_ids.zero_()
        self.valid_rows.zero_()
        for tensor in self.metadata.values():
            tensor.fill_(-1)
        self.queues.reset()

    def debug_snapshot(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "max_rows": self.max_rows,
            "wait_rounds": self.wait_rounds,
            "round_id": int(self.round_id.item()),
            "queues": self.queues.debug_snapshot(),
            "dispatch": self.queues.debug_dispatch(),
        }
_c3_counters = CoverageDenseCounters.create()
