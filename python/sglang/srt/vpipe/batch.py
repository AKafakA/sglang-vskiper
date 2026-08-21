"""Batch lifecycle and the device route tape.

`prepare_full_graph_batch` runs once per forward batch before any routed layer:
it sizes the per-bucket buffers and arms the route tape. `finalize_full_graph_batch`
closes the batch out, publishing route counters and digests.

The route tape is DEVICE-RESIDENT. Route decisions are recorded on-GPU and only
summarised to the host at batch end, because a per-layer host read would put ~32
device-to-host syncs on the decode critical path -- the shape of the original
eager collapse.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch
from sglang.srt.vpipe.common import (
    full_graph_decode_enabled,
)
from sglang.srt.vpipe.env import (
    FD_EXECUTION_FULL_GRAPH,
)
from sglang.srt.vpipe.common import (
    FullGraphDeviceRouteTape,
    _accumulate_device_route_digest,
    _route_derived_compact_stats,
    _route_tape_request_slots,
)
from sglang.srt.vpipe.common import (
    resolve_full_graph_skipper,
)
from sglang.srt.vpipe.skipper import (
    route_digest_uses_logical_request_ids,
)
from sglang.srt.vpipe.env import (
    SUBLAYER_EXECUTION,
)
from sglang.srt.vpipe.config import (
    full_graph_compact_config,
    full_graph_device_route_digest_enabled,
    full_graph_device_route_tape_enabled,
    full_graph_fused_evidence_enabled,
    full_graph_layer_counters_enabled,
    full_graph_route_accounting_enabled,
    full_graph_scheduler_convergence_enabled,
)
from sglang.srt.vpipe.common import (
    flexidepth_execution_mode,
)
from sglang.srt.vpipe.common import (
    flexidepth_forward_phase,
    flexidepth_phase_enabled,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_phases,
    full_graph_low_row_policy,
)


def full_graph_compact_phase_enabled(forward_batch: Any) -> bool:
    phase = flexidepth_forward_phase(forward_batch)
    return phase is not None and phase in full_graph_compact_phases()
def full_graph_prefill_enabled(forward_batch: Any) -> bool:
    if flexidepth_execution_mode() != FD_EXECUTION_FULL_GRAPH:
        return False
    return (
        flexidepth_forward_phase(forward_batch) == "prefill"
        and flexidepth_phase_enabled(forward_batch)
    )
def full_graph_routed_enabled(forward_batch: Any) -> bool:
    return full_graph_decode_enabled(forward_batch) or full_graph_prefill_enabled(
        forward_batch
    )
def _route_policy_request_ids(
    forward_batch: Any, row_ids: torch.Tensor
) -> torch.Tensor:
    request_ids = getattr(forward_batch, "rids_int", None)
    if request_ids is None:
        raise RuntimeError(
            "full-graph skipper requires stable logical request IDs"
        )
    if request_ids.dtype != torch.int64:
        raise RuntimeError("full-graph logical request IDs must be int64")
    if full_graph_decode_enabled(forward_batch):
        if request_ids.shape != row_ids.shape:
            raise RuntimeError(
                "full-graph decode request IDs do not match routed rows"
            )
        return request_ids

    extend_starts = getattr(forward_batch, "extend_start_loc", None)
    if extend_starts is None:
        raise RuntimeError(
            "full-graph prefill request IDs require extend_start_loc"
        )
    request_rows = torch.bucketize(
        row_ids.to(dtype=extend_starts.dtype), extend_starts[1:], right=True
    )
    if request_ids.shape != (int(extend_starts.shape[0]),):
        raise RuntimeError(
            "full-graph prefill request IDs do not match request rows"
        )
    return request_ids.index_select(0, request_rows)
def _accumulate_inline_kv_readiness(
    tape: FullGraphDeviceRouteTape, counters: torch.Tensor
) -> None:
    """Attest inline own-layer K/V completion without a hot-path readback."""

    if counters.shape != (7,):
        raise RuntimeError("FlexiDepth inline K/V readiness counter shape changed")
    tape.require_inline_kv_complete()
    ready = tape.inline_kv_ready
    metadata = (
        tape.request_slots,
        tape.token_epochs,
        tape.cache_positions,
    )
    if ready is None or any(value is None for value in metadata):
        raise RuntimeError("FlexiDepth inline K/V readiness metadata is incomplete")

    request_slots, token_epochs, cache_positions = metadata
    assert request_slots is not None
    assert token_epochs is not None
    assert cache_positions is not None
    layer_ids = tape.layer_ids.to(dtype=torch.int64).view(-1, 1)
    request_slots = request_slots.to(dtype=torch.int64).view(1, -1)
    token_epochs = token_epochs.to(dtype=torch.int64).view(1, -1)
    cache_positions = cache_positions.to(dtype=torch.int64).view(1, -1)
    valid = tape.valid_rows.view(1, -1)
    expected = valid.expand_as(ready)
    ready_valid = ready & expected

    identity_codes = (
        (request_slots + 1) * 1_000_003
        + (token_epochs + 1) * 1_000_033
        + (layer_ids + 1) * 1_000_037
        + (cache_positions + 1) * 1_000_081
    )
    payload = torch.where(
        ready_valid,
        identity_codes,
        torch.zeros((), dtype=torch.int64, device=ready.device),
    )
    row_ids = torch.arange(
        ready.shape[1], dtype=torch.int64, device=ready.device
    ).view(1, -1)
    cell_weights = (layer_ids + 1) * 65_537 + (row_ids + 1) * 65_539
    expected_cells = expected.sum(dtype=torch.int64)
    ready_cells = ready_valid.sum(dtype=torch.int64)
    digest_sum = payload.sum(dtype=torch.int64)
    digest_weighted = (payload * cell_weights).sum(dtype=torch.int64)
    dispatch_ordinal = counters[0] + 1

    counters[0].add_(1)
    counters[1].add_(expected_cells)
    counters[2].add_(ready_cells)
    counters[3].add_(expected_cells - ready_cells)
    counters[4].add_(valid.sum(dtype=torch.int64))
    counters[5].add_(digest_sum)
    counters[6].add_(digest_weighted + digest_sum * dispatch_ordinal)
def prepare_full_graph_batch(
    forward_batch: Any,
    hidden_states: torch.Tensor,
    *,
    positions: Optional[torch.Tensor] = None,
    route_layer_ids: Optional[torch.Tensor] = None,
    route_layer_order: tuple[int, ...] = (),
    compact_evidence_specs: Optional[torch.Tensor] = None,
) -> None:
    """Create replay-updated valid rows and the per-forward route state."""

    if not full_graph_routed_enabled(forward_batch):
        return
    forward_batch.fd_full_graph_skipper_adapter = resolve_full_graph_skipper()
    num_valid = (
        getattr(forward_batch, "num_token_non_padded", None)
        if full_graph_decode_enabled(forward_batch)
        else None
    )
    if num_valid is None:
        valid_rows = torch.ones(
            hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device
        )
    else:
        row_ids = torch.arange(
            hidden_states.shape[0], dtype=num_valid.dtype, device=hidden_states.device
        )
        valid_rows = row_ids < num_valid.reshape(())
    forward_batch.fd_full_graph_valid_rows = valid_rows
    policy_request_ids = None
    policy_request_slots = None
    digest_enabled = full_graph_device_route_digest_enabled()
    logical_route_digest = (
        digest_enabled
        and route_digest_uses_logical_request_ids(
            forward_batch.fd_full_graph_skipper_adapter
        )
    )
    policy_requires_rows = (
        logical_route_digest
        or forward_batch.fd_full_graph_skipper_adapter.requires_stable_request_ids
        or forward_batch.fd_full_graph_skipper_adapter.requires_request_slots
    )
    if policy_requires_rows:
        if positions is None or positions.shape != (hidden_states.shape[0],):
            raise RuntimeError(
                "full-graph skipper token epochs do not match routed rows"
            )
        policy_row_ids = torch.arange(
            hidden_states.shape[0],
            dtype=torch.int64,
            device=hidden_states.device,
        )
        if (
            logical_route_digest
            or forward_batch.fd_full_graph_skipper_adapter
            .requires_stable_request_ids
        ):
            policy_request_ids = _route_policy_request_ids(
                forward_batch, policy_row_ids
            )
        if forward_batch.fd_full_graph_skipper_adapter.requires_request_slots:
            policy_request_slots = _route_tape_request_slots(
                forward_batch, policy_row_ids
            )
    forward_batch.fd_full_graph_skipper_state = (
        forward_batch.fd_full_graph_skipper_adapter.prepare_batch(
            hidden_states=hidden_states,
            request_ids=policy_request_ids,
            request_slots=policy_request_slots,
            token_epochs=positions,
            valid_rows=valid_rows,
            route_layer_order=route_layer_order,
            phase=flexidepth_forward_phase(forward_batch),
            batch_request_ids=getattr(forward_batch, "rids_int", None),
            batch_request_slots=getattr(
                forward_batch, "req_pool_indices", None
            ),
        )
    )
    accounting_enabled = full_graph_route_accounting_enabled()
    tape_enabled = full_graph_device_route_tape_enabled()
    layer_counters_enabled = full_graph_layer_counters_enabled()
    scheduler_convergence = full_graph_scheduler_convergence_enabled()
    fused_evidence = full_graph_fused_evidence_enabled()
    if fused_evidence:
        enabled = (
            accounting_enabled,
            tape_enabled,
            digest_enabled,
            layer_counters_enabled,
            scheduler_convergence,
        )
        if not all(enabled):
            raise RuntimeError(
                "fused route evidence runtime dependencies are incomplete"
            )
        expected_specs_shape = (len(route_layer_order), 3)
        if (
            compact_evidence_specs is None
            or compact_evidence_specs.shape != expected_specs_shape
            or compact_evidence_specs.dtype != torch.float32
            or compact_evidence_specs.device != hidden_states.device
            or not compact_evidence_specs.is_contiguous()
        ):
            raise RuntimeError(
                "fused route evidence compact metadata is incompatible"
            )
        if hidden_states.shape[0] > 1024:
            raise RuntimeError(
                "fused route evidence currently supports at most 1024 rows"
            )
    if tape_enabled:
        if route_layer_ids is None or not route_layer_order:
            raise RuntimeError(
                "FlexiDepth device route tape requires the loaded layer order"
            )
        if route_layer_ids.shape != (len(route_layer_order),):
            raise RuntimeError(
                "FlexiDepth device route tape layer metadata shape changed"
            )
        row_count = hidden_states.shape[0]
        request_slots = token_epochs = cache_positions = row_ids = None
        logical_request_ids = None
        if digest_enabled:
            row_ids = torch.arange(
                row_count, dtype=torch.int64, device=hidden_states.device
            )
            request_slots = _route_tape_request_slots(forward_batch, row_ids)
            token_epochs = positions
            cache_positions = getattr(forward_batch, "out_cache_loc", None)
            for name, value in (
                ("token epochs", token_epochs),
                ("cache positions", cache_positions),
            ):
                if value is None or value.shape != (row_count,):
                    raise RuntimeError(
                        f"FlexiDepth device route digest {name} do not match rows"
                    )
            if logical_route_digest:
                logical_request_ids = policy_request_ids
                if logical_request_ids is None:
                    logical_request_ids = _route_policy_request_ids(
                        forward_batch, row_ids
                    )
        forward_batch.fd_full_graph_device_route_tape = FullGraphDeviceRouteTape(
            actions=torch.empty(
                (len(route_layer_order), row_count),
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            valid_rows=valid_rows,
            layer_ids=route_layer_ids,
            layer_order=route_layer_order,
            request_slots=request_slots,
            token_epochs=token_epochs,
            cache_positions=cache_positions,
            row_ids=row_ids,
            adapter_name=forward_batch.fd_full_graph_skipper_adapter.name,
            branch_weights=torch.empty(
                (len(route_layer_order), row_count),
                dtype=torch.float32,
                device=hidden_states.device,
            ),
            logical_request_ids=logical_request_ids,
            mlp_actions=(
                torch.empty(
                    (len(route_layer_order), row_count),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                if forward_batch.fd_full_graph_skipper_adapter.execution_kind
                == SUBLAYER_EXECUTION
                else None
            ),
            compact_evidence_specs=compact_evidence_specs,
            inline_kv_ready=(
                torch.empty(
                    (len(route_layer_order), row_count),
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                if scheduler_convergence and not fused_evidence
                else None
            ),
            structural_inline_kv_ready=scheduler_convergence and fused_evidence,
        )
    else:
        forward_batch.fd_full_graph_device_route_tape = None
    forward_batch.fd_full_graph_route_masks = (
        [] if accounting_enabled and not tape_enabled else None
    )
    forward_batch.fd_full_graph_compact_stats = (
        []
        if accounting_enabled and not fused_evidence and not tape_enabled
        else None
    )
    forward_batch.fd_full_graph_compact_phase_enabled = (
        full_graph_compact_phase_enabled(forward_batch)
    )
def finalize_full_graph_batch(
    forward_batch: Any,
    route_counters: Optional[torch.Tensor],
    layer_route_counters: Optional[torch.Tensor] = None,
    route_digest_counters: Optional[torch.Tensor] = None,
    inline_kv_readiness_counters: Optional[torch.Tensor] = None,
    phase_route_counters: Optional[torch.Tensor] = None,
    low_row_counters: Optional[torch.Tensor] = None,
) -> None:
    """Accumulate logical route rows once per replay, excluding padded rows."""

    skipper_adapter = getattr(
        forward_batch, "fd_full_graph_skipper_adapter", None
    )
    if skipper_adapter is not None:
        skipper_adapter.finalize_batch(
            batch_state=getattr(
                forward_batch, "fd_full_graph_skipper_state", None
            )
        )
    device_tape = getattr(
        forward_batch, "fd_full_graph_device_route_tape", None
    )
    if device_tape is not None:
        device_tape.require_complete()
        attention_routes = device_tape.actions
        routes = (
            torch.cat((attention_routes, device_tape.mlp_actions), dim=0)
            if device_tape.mlp_actions is not None
            else attention_routes
        )
    else:
        route_masks = getattr(forward_batch, "fd_full_graph_route_masks", None)
        if not route_masks:
            return
        routes = torch.stack(route_masks, dim=0)
    if low_row_counters is not None:
        if low_row_counters.shape != (3,):
            raise RuntimeError(
                "FlexiDepth low-row counter shape must be dispatch/graph/logical"
            )
        low_row_policy, low_row_max_rows = full_graph_low_row_policy()
        if low_row_policy == "off":
            raise RuntimeError("FlexiDepth low-row counters require an active policy")
        if (
            forward_batch.fd_full_graph_compact_phase_enabled
            and routes.shape[1] <= low_row_max_rows
        ):
            logical_rows = (
                forward_batch.fd_full_graph_valid_rows.sum(dtype=torch.int64)
                * routes.shape[0]
            )
            low_row_counters.add_(
                torch.stack(
                    (
                        logical_rows.new_ones(()),
                        logical_rows.new_ones(()).mul_(routes.shape[1]),
                        logical_rows,
                    )
                )
            )
    if full_graph_fused_evidence_enabled():
        if not routes.is_cuda:
            raise RuntimeError("fused route evidence requires CUDA routes")
        if device_tape is None:
            raise RuntimeError("fused route evidence requires the device tape")
        device_tape.require_inline_kv_complete()
        metadata = (
            device_tape.request_slots,
            device_tape.token_epochs,
            device_tape.cache_positions,
            device_tape.compact_evidence_specs,
            route_counters,
            layer_route_counters,
            route_digest_counters,
            inline_kv_readiness_counters,
        )
        if any(value is None for value in metadata):
            raise RuntimeError("fused route evidence metadata is incomplete")
        from sglang.srt.vpipe.cohort import (
            accumulate_fused_route_evidence,
        )

        _, compact_min_rows, _, compact_multiple = full_graph_compact_config()
        accumulate_fused_route_evidence(
            actions=routes,
            valid_rows=device_tape.valid_rows,
            layer_ids=device_tape.layer_ids,
            request_slots=device_tape.request_slots,
            logical_request_ids=device_tape.logical_request_ids,
            token_epochs=device_tape.token_epochs,
            cache_positions=device_tape.cache_positions,
            compact_specs=device_tape.compact_evidence_specs,
            route_counters=route_counters,
            layer_counters=layer_route_counters,
            route_digest_counters=route_digest_counters,
            readiness_counters=inline_kv_readiness_counters,
            compact_active=(
                forward_batch.fd_full_graph_compact_phase_enabled
                and routes.shape[1] >= compact_min_rows
            ),
            capacity_multiple=compact_multiple,
        )
        return
    if route_digest_counters is not None:
        if device_tape is None:
            raise RuntimeError(
                "FlexiDepth device route digest requires the device tape"
            )
        _accumulate_device_route_digest(device_tape, route_digest_counters)
    if inline_kv_readiness_counters is not None:
        if device_tape is None:
            raise RuntimeError(
                "FlexiDepth inline K/V readiness requires the device tape"
            )
        _accumulate_inline_kv_readiness(
            device_tape, inline_kv_readiness_counters
        )
    if route_counters is None:
        return
    valid_rows = forward_batch.fd_full_graph_valid_rows
    logical_rows = valid_rows.sum(dtype=torch.int64) * routes.shape[0]
    run_rows = (routes & valid_rows.unsqueeze(0)).sum(dtype=torch.int64)
    counts = torch.stack((logical_rows, run_rows, logical_rows - run_rows))
    route_counters[:3].add_(counts)
    if phase_route_counters is not None:
        if phase_route_counters.shape != (2, 3):
            raise RuntimeError(
                "FlexiDepth phase route counter shape must be decode/prefill by 3"
            )
        phase = flexidepth_forward_phase(forward_batch)
        if phase not in {"decode", "prefill"}:
            raise RuntimeError(
                "FlexiDepth phase route accounting has no active phase"
            )
        phase_route_counters[0 if phase == "decode" else 1].add_(counts)
    if route_counters.numel() >= 6:
        if device_tape is not None:
            compact_specs = device_tape.compact_evidence_specs
            compact_enabled, compact_min_rows, _, compact_multiple = (
                full_graph_compact_config()
            )
            compact_active = (
                compact_enabled
                and forward_batch.fd_full_graph_compact_phase_enabled
                and routes.shape[1] >= compact_min_rows
            )
            if compact_specs is None and compact_active:
                raise RuntimeError(
                    "FlexiDepth route-tape compact accounting metadata is missing"
                )
            if compact_specs is not None:
                route_counters[3:6].add_(
                    _route_derived_compact_stats(
                        routes,
                        valid_rows,
                        compact_specs,
                        compact_active=compact_active,
                        capacity_multiple=compact_multiple,
                    )
                )
        else:
            compact_stats = getattr(
                forward_batch, "fd_full_graph_compact_stats", None
            )
            if compact_stats:
                route_counters[3:6].add_(
                    torch.stack(compact_stats).sum(dim=0)
                )
    if layer_route_counters is not None:
        sublayer_tape = (
            device_tape is not None and device_tape.mlp_actions is not None
        )
        expected_layers = (
            device_tape.actions.shape[0] if sublayer_tape else routes.shape[0]
        )
        if layer_route_counters.shape != (expected_layers, 3):
            raise RuntimeError(
                "full-graph per-layer counter shape does not match the route tape"
            )
        if sublayer_tape:
            valid_per_layer = (
                valid_rows.sum(dtype=torch.int64) * 2
            ).expand(expected_layers)
            run_per_layer = (
                (device_tape.actions & valid_rows.unsqueeze(0)).sum(
                    dim=1, dtype=torch.int64
                )
                + (device_tape.mlp_actions & valid_rows.unsqueeze(0)).sum(
                    dim=1, dtype=torch.int64
                )
            )
        else:
            valid_per_layer = valid_rows.sum(dtype=torch.int64).expand(
                expected_layers
            )
            run_per_layer = (routes & valid_rows.unsqueeze(0)).sum(
                dim=1, dtype=torch.int64
            )
        layer_route_counters.add_(
            torch.stack(
                (
                    valid_per_layer,
                    run_per_layer,
                    valid_per_layer - run_per_layer,
                ),
                dim=1,
            )
        )
