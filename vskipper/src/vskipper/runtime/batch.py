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
from vskipper.runtime.common import (
    full_graph_decode_enabled,
)
from vskipper.runtime.env import (
    COMPACT_CAPACITY_MULTIPLE,
    FD_EXECUTION_FULL_GRAPH,
)
from vskipper.runtime.common import (
    FullGraphDeviceRouteTape,
    _accumulate_device_route_digest,
    _route_derived_compact_stats,
    _route_tape_request_slots,
)
from vskipper.runtime.common import (
    resolve_full_graph_skipper,
)
from vskipper.runtime.skipper import (
    route_digest_uses_logical_request_ids,
)
from vskipper.runtime.config import (
    full_graph_compact_config,
    full_graph_device_route_digest_enabled,
    full_graph_device_route_tape_enabled,
    full_graph_fused_evidence_enabled,
    full_graph_layer_counters_enabled,
    full_graph_route_accounting_enabled,
    full_graph_scheduler_convergence_enabled,
)
from vskipper.runtime.common import (
    flexidepth_execution_mode,
)
from vskipper.runtime.common import (
    flexidepth_forward_phase,
    flexidepth_phase_enabled,
)
from vskipper.runtime.common import (
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
    # Lane-2 cut3 item 4 (D-359): the fused evidence kernel is decode-layout and
    # capped at 1024 rows, which is why `validation.py` refused it unless the
    # deployment was decode-only. The shipping arm serves BOTH phases, so decide
    # PER PASS instead of per deployment: fused on a decode pass within the row
    # cap, the existing per-pass accumulators otherwise. The decision is stamped
    # on the batch so `finalize_full_graph_batch` reads the same fact the
    # allocation used, rather than re-reading the environment.
    fused_evidence = full_graph_fused_evidence_enabled()
    if fused_evidence:
        rows_now = int(hidden_states.shape[0])
        fused_evidence = (
            forward_batch.forward_mode.is_decode() and rows_now <= 1024
        )
    forward_batch.fd_full_graph_fused_evidence_active = fused_evidence
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
            # lane-2 cut3: the branch-weight TAPE is write-only. `FullGraphDeviceRouteTape`
            # copies the per-layer route weights into it (common.py:380, :416) and NOTHING
            # reads them back -- not the executor, not the compaction kernels (whose own
            # `branch_weights` parameters are the per-pass route weights, a different
            # object), and not the attestation, which never reports a branch-weight tape.
            # `None` is a first-class state, guarded at common.py:374, :410 and :428, so
            # passing it skips the [layers, rows] float32 allocation and 16 device copies
            # per pass without introducing a new code path.
            branch_weights=None,
            logical_request_ids=logical_request_ids,
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
        routes = device_tape.actions
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
        low_row_policy = full_graph_low_row_policy()
        if low_row_policy == "off":
            raise RuntimeError("FlexiDepth low-row counters require an active policy")
        # The counter condition must mirror the DISPATCH condition in `mlp.py`
        # exactly, or the attested dispatch count silently disagrees with the
        # body that ran. [D-582] With the row bound deleted, native_dense
        # dispatches at EVERY occupancy, so the counter is unconditional here.
        if True:
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
    # Lane-2 cut3 item 4: read the PER-PASS stamp, not the environment, so the
    # accumulation path always matches the allocation decision made in
    # `prepare_full_graph_batch` (a prefill pass, or a decode pass above the
    # kernel's row cap, keeps the unfused accumulators).
    if getattr(forward_batch, "fd_full_graph_fused_evidence_active", False):
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
        from vskipper.runtime.cohort import (
            accumulate_fused_route_evidence,
        )

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
            # [D-578] compact_active no longer carries a min-row condition:
            # after D-574 the routed body compacts whenever compaction is on,
            # so gating the counter on a row threshold reported a pass as
            # uncompacted that in fact compacted.
            compact_active=forward_batch.fd_full_graph_compact_phase_enabled,
            capacity_multiple=COMPACT_CAPACITY_MULTIPLE,
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
            compact_active = (
                full_graph_compact_config()
                and forward_batch.fd_full_graph_compact_phase_enabled
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
                        capacity_multiple=COMPACT_CAPACITY_MULTIPLE,
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
        expected_layers = routes.shape[0]
        if layer_route_counters.shape != (expected_layers, 3):
            raise RuntimeError(
                "full-graph per-layer counter shape does not match the route tape"
            )
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


# ---------------------------------------------------------------------------
# [D-849, 2026-09-21] Per-request body pinning: partition a mixed-pin decode
# batch into uniform sub-batches and merge their outputs back in row order.
# CPU-testable (plain tensor indexing; no runner state).
# ---------------------------------------------------------------------------

_VP_SPLIT_ROW_TENSORS = (
    "input_ids",
    "req_pool_indices",
    "seq_lens",
    "out_cache_loc",
    "positions",
    "seq_lens_cpu",
    "orig_seq_lens",
    "rids_int",
    "bootstrap_room_ids_int",
)
_VP_SPLIT_ROW_LISTS = (
    "lora_ids",
    "rids",
    "top_logprobs_nums",
    "token_ids_logprobs",
    "vp_body_rows",
    # one entry per request in this fork; None for text-only requests
    "mm_inputs",
)
_VP_SPLIT_REFUSED_FIELDS = (
    "spec_info",
    "input_embeds",
    "replace_embeds",
    "global_num_tokens_cpu",
    "mamba_track_indices",
    "mamba_cow_src_indices",
    "mamba_clear_indices",
    "encoder_lens",
    "attn_cp_metadata",
    "attn_dcp_metadata",
    "tbo_children",
)


def vp_force_run_rows(forward_batch: Any) -> Optional[torch.Tensor]:
    """[D-849 forced-RUN] The bool-per-row mask of a MIXED decode batch's
    stock-pinned rows (forced to RUN inside the routed body), or ``None`` for a
    uniform / unpinned batch."""

    from vskipper.runtime.common import REQUEST_BODY_STOCK, VP_BODY_MIXED

    if forward_batch.vp_body != VP_BODY_MIXED:
        return None
    pins = forward_batch.vp_body_rows
    if pins is None or len(pins) != int(forward_batch.batch_size):
        raise RuntimeError("[D-849] vp_body_rows missing or not one per row")
    return torch.tensor(
        [p == REQUEST_BODY_STOCK for p in pins],
        dtype=torch.bool,
        device=forward_batch.input_ids.device,
    )


def vp_split_decode_forward_batch(
    forward_batch: Any, max_rows: Optional[int] = None, by_pin: bool = True
) -> list[tuple[str, torch.Tensor, Any]]:
    """Split a pinned DECODE forward batch into pin-uniform sub-batches.

    With ``max_rows`` (the captured decode ladder's top), every pin group is
    further cut into consecutive chunks of at most ``max_rows`` rows, so an
    fd-pinned batch larger than the ladder is served by its captured graph in
    several replays instead of falling to the dense body (a pin violation;
    2026-09-21 bbh overload: 22 % of steps above 1,024 rows).

    Returns ``[(pin, row_index_tensor, sub_forward_batch), ...]`` with the
    stock sub-batch first. Each sub-batch is a shallow copy whose per-row
    tensors/lists are ``index_select``-ed/sliced by the pin's rows, with
    ``batch_size``, ``seq_lens_sum`` and ``vp_body`` recomputed and every
    per-pass plan/stamp reset so the runner plans and stamps the sub-batch
    itself. Refuses anything that is not a plain decode batch (speculative,
    embeddings, DP/MLP-sync padding, mamba, encoder, context-parallel, TBO):
    those never run under the served design, and a silent partial split would
    be worse than a loud refusal.
    """

    import copy

    from vskipper.runtime.common import REQUEST_BODY_FD, REQUEST_BODY_STOCK

    if not forward_batch.forward_mode.is_decode():
        raise RuntimeError("[D-849] only DECODE batches are partitioned by pin")
    pins = forward_batch.vp_body_rows
    if pins is None or forward_batch.batch_size != len(pins):
        raise RuntimeError("[D-849] vp_body_rows missing or not one per row")
    for name in _VP_SPLIT_REFUSED_FIELDS:
        if getattr(forward_batch, name) is not None:
            raise RuntimeError(
                f"[D-849] cannot partition a decode batch with {name} set"
            )
    mm = forward_batch.mm_inputs
    if mm is not None and any(item is not None for item in mm):
        raise RuntimeError("[D-849] cannot partition a decode batch with multimodal inputs")
    device = forward_batch.input_ids.device
    parts: list[tuple[str, torch.Tensor, Any]] = []
    if max_rows is not None and int(max_rows) < 1:
        raise RuntimeError(f"[D-849] max_rows must be >= 1, got {max_rows!r}")
    from vskipper.runtime.regime import batch_pin_of

    groups: list[tuple[str, list[int]]] = []
    if by_pin:
        for pin in (REQUEST_BODY_STOCK, REQUEST_BODY_FD):
            rows = [i for i, p in enumerate(pins) if p == pin]
            if not rows:
                continue
            if max_rows is None:
                groups.append((pin, rows))
            else:
                step = int(max_rows)
                groups.extend((pin, rows[i : i + step]) for i in range(0, len(rows), step))
    else:
        # size-only chunks in row order; a chunk keeps whatever pins it holds
        # (a mixed chunk is served by ONE routed replay with forced-RUN rows)
        if max_rows is None:
            raise RuntimeError("[D-849] size-only chunking needs max_rows")
        step = int(max_rows)
        all_rows = list(range(len(pins)))
        groups = [(batch_pin_of([pins[i] for i in all_rows[i0 : i0 + step]]), all_rows[i0 : i0 + step]) for i0 in range(0, len(pins), step)]
    for pin, rows in groups:
        index = torch.tensor(rows, dtype=torch.int64, device=device)
        sub = copy.copy(forward_batch)
        for name in _VP_SPLIT_ROW_TENSORS:
            value = getattr(forward_batch, name)
            if value is None:
                continue
            if value.shape[0] != len(pins):
                raise RuntimeError(
                    f"[D-849] {name} has {value.shape[0]} rows for {len(pins)} pins"
                )
            setattr(sub, name, value.index_select(0, index.to(value.device)))
        for name in _VP_SPLIT_ROW_LISTS:
            value = getattr(forward_batch, name)
            if value is None:
                continue
            if len(value) != len(pins):
                raise RuntimeError(
                    f"[D-849] {name} has {len(value)} entries for {len(pins)} pins"
                )
            setattr(sub, name, [value[i] for i in rows])
        sub.batch_size = len(rows)
        sub.seq_lens_sum = int(sub.seq_lens_cpu.sum()) if sub.seq_lens_cpu is not None else int(sub.seq_lens.sum().item())
        sub.vp_body = pin
        # Per-pass plan/stamp state is NOT inherited: the runner plans and stamps
        # each sub-batch itself (a pre-plan of the union would be stale).
        sub.forward_metadata_ready = False
        sub.forward_metadata_planned_bs = None
        sub.forward_metadata_planned_num_tokens = None
        sub.forward_metadata_replan_equivalent = False
        sub.vp_seam_batch_routed = None
        sub.vp_seam_batch_eager = None
        sub.vp_fd_decode_dense = False
        sub.vp_fd_decode_coverage_dense = False
        sub.vp_fd_coverage_counted = False
        sub.fd_full_graph_force_production_attention = False
        sub.fd_full_graph_device_route_tape = None
        sub.fd_full_graph_skipper_adapter = None
        sub.fd_full_graph_skipper_state = None
        sub.fd_full_graph_route_masks = None
        sub.fd_full_graph_compact_stats = None
        sub.fd_full_graph_valid_rows = None
        sub.fd_full_graph_force_run_rows = None
        sub.fd_full_graph_attention_run_mask = None
        sub.fd_full_graph_kv_write_mask = None
        parts.append((pin, index, sub))
    if len(parts) < 2:
        raise RuntimeError(
            "[D-849] partition called on a batch that is neither mixed nor above the ladder"
        )
    return parts


def vp_detach_logits_output(out: Any) -> Any:
    """Copy a sub-pass's forward-produced rows out of the graph runner's static buffers.

    A captured-graph replay hands back views of the backend's per-shape output
    tensors (``FullCudaGraphBackend.replay`` returns ``self._outputs[shape_key]``),
    which the NEXT replay overwrites. A partitioned step runs two replays before
    it merges, so the first sub-pass's logits must be materialised before the
    second sub-pass runs (2026-09-21 tiny-band gate: every stock-pinned row
    sampled the fd rows' tokens, 16/16 for 12 consecutive steps).
    """

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    if not isinstance(out, LogitsProcessorOutput):
        raise RuntimeError("[D-849] pinned sub-pass returned a non-logits output")
    return LogitsProcessorOutput(
        next_token_logits=(
            out.next_token_logits.clone() if out.next_token_logits is not None else None
        ),
        full_logits=out.full_logits.clone() if out.full_logits is not None else None,
        hidden_states=out.hidden_states.clone() if out.hidden_states is not None else None,
        customized_info=out.customized_info,
    )


def vp_merge_logits_outputs(
    parts: list[tuple[str, torch.Tensor, Any]], batch_size: int
) -> Any:
    """Merge the sub-passes' LogitsProcessorOutput rows back into batch order.

    Only the forward-produced fields are merged (``next_token_logits``,
    ``hidden_states``); the sampler fills the rest on the merged output.
    """

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    def merge(name: str):
        first = getattr(parts[0][2], name)
        if first is None:
            for _, _, out in parts[1:]:
                if getattr(out, name) is not None:
                    raise RuntimeError(f"[D-849] sub-pass outputs disagree on {name}")
            return None
        merged = torch.empty(
            (batch_size,) + tuple(first.shape[1:]), dtype=first.dtype, device=first.device
        )
        for _, index, out in parts:
            value = getattr(out, name)
            if value is None or value.shape[0] != index.numel():
                raise RuntimeError(f"[D-849] sub-pass output {name} rows do not match its index")
            merged.index_copy_(0, index.to(merged.device), value)
        return merged

    for _, _, out in parts:
        if not isinstance(out, LogitsProcessorOutput):
            raise RuntimeError("[D-849] pinned sub-pass returned a non-logits output")
        if out.full_logits is not None:
            raise RuntimeError("[D-849] full_logits (dLLM) is not partitionable")
    return LogitsProcessorOutput(
        next_token_logits=merge("next_token_logits"),
        hidden_states=merge("hidden_states"),
    )
