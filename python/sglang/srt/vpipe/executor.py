"""Executor — run the layer under the prepared route.

The EXECUTE half of the per-layer mechanism (`routing.py` is DECIDE):

    prepared = prepare_layer_route(...)              (routing.py)
    execute_prepared_route(..., prepared)            (this module)

Attention runs for every row with run-only cache writes; the MLP body is then
chosen by the layer's policy:

  run_base       dense body on the low-engagement band (layers 16-21 engage
                 5-16% on EVERY workload measured, so routing them costs more
                 than it saves)
  compact        packed cohort through the grouped path
  dual_compact   both cohorts packed, per-bucket minimum rows
  binary_cohort  the count-adaptive indexed GEMM in `kernel.py` -- the shipped
                 prefill body (D-302/D-303)

Policy dispatch is precedence-ordered so binary_cohort is chosen BEFORE the
grouped/compact gates; getting that order wrong silently runs the old body
while still attesting the new policy name.

PROJECT_ONLY rows never skip K/V: they take their K/V from that layer's own
projection weights, which is what keeps the cache complete for later attention.

Bodies are byte-identical to the frozen tree.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch
from sglang.srt.vpipe.common import (
    full_graph_decode_enabled,
)
from sglang.srt.vpipe.attention import (
    _apply_attention_run_mask,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_o_proj_min_rows,
)
from sglang.srt.vpipe.batch import (
    full_graph_prefill_enabled,
)
from sglang.srt.vpipe.env import (
    FD_DEFER_PROJECT_KV_ENV,
    FD_EAGER_SEMANTIC_DEBUG_ENV,
    FD_FORCED_ALL_RUN_FASTPATH_ENV,
    FD_LAYER_POLICIES_ENV,
    FD_WEIGHTED_SCATTER_ENV,
    _VALID_LAYER_POLICIES,
)
from sglang.srt.vpipe.mlp import (
    fd_conditional_mlp_full_graph,
)
from sglang.srt.vpipe.mlp_compact import (
    _compact_capacity,
)
from sglang.srt.vpipe.coverage import (
    record_eager_skip_decode_layer_call,
)
from sglang.srt.vpipe.routing import (
    FullGraphPreparedLayerRoute,
    fd_prepare_layer_route_full_graph,
)
from sglang.srt.vpipe.env import (
    SUBLAYER_EXECUTION,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_routed_qkv_enabled,
)
from sglang.srt.vpipe.config import (
    full_graph_compact_config,
    full_graph_compact_o_proj_config,
    full_graph_defer_project_kv_enabled,
    full_graph_forced_all_run_fastpath_enabled,
    full_graph_forced_all_run_production_attention_enabled,
    full_graph_mapped_decode_attention_enabled,
    full_graph_masked_decode_attention_enabled,
    full_graph_prefill_grouped_mlp_enabled,
)
from sglang.srt.vpipe.mlp_compact import (
    _one_expert_mlp,
)


def _execute_sublayer_route_full_graph(
    layer: Any,
    positions: torch.Tensor,
    forward_batch: Any,
    prepared: FullGraphPreparedLayerRoute,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute independent AdaSkip attention/MLP actions with complete K/V."""

    action_batch = prepared.action_batch
    if action_batch is None or action_batch.execution_kind != SUBLAYER_EXECUTION:
        raise RuntimeError("sublayer execution requires a sublayer action batch")
    attention_run_mask = prepared.attention_run_mask
    mlp_run_mask = prepared.mlp_run_mask
    attention_scale = prepared.attention_skip_scale
    mlp_scale = prepared.mlp_skip_scale
    shape = (prepared.hidden_states.shape[0], 1)
    for name, value in (
        ("attention actions", attention_run_mask),
        ("MLP actions", mlp_run_mask),
        ("attention compensation", attention_scale),
        ("MLP compensation", mlp_scale),
    ):
        if value is None or value.shape != shape:
            raise RuntimeError(f"AdaSkip {name} do not match routed rows")

    valid_rows = forward_batch.fd_full_graph_valid_rows
    if valid_rows is None or valid_rows.shape != (shape[0],):
        raise RuntimeError("AdaSkip graph-valid rows are missing")
    attention_active = attention_run_mask.squeeze(-1) & valid_rows
    forward_batch.fd_full_graph_attention_run_mask = attention_active
    forward_batch.fd_full_graph_attention_static_run = (
        action_batch.static_attention_run
    )
    forward_batch.fd_full_graph_kv_write_mask = None
    try:
        attention = layer.self_attn(
            positions=positions,
            hidden_states=prepared.hidden_states,
            forward_batch=forward_batch,
        )
    finally:
        forward_batch.fd_full_graph_attention_run_mask = None
        forward_batch.fd_full_graph_attention_static_run = None
        forward_batch.fd_full_graph_kv_write_mask = None

    device_tape = getattr(
        forward_batch, "fd_full_graph_device_route_tape", None
    )
    if device_tape is not None:
        device_tape.record_inline_kv_ready(
            int(layer.layer_id),
            index=prepared.inline_kv_index,
        )
    if action_batch.static_attention_run is True:
        attention_delta = attention
    elif action_batch.static_attention_run is False:
        attention_delta = prepared.residual * (attention_scale - 1.0)
    else:
        attention_delta = torch.where(
            attention_run_mask,
            attention,
            prepared.residual * (attention_scale - 1.0),
        )
    hidden_states, residual = layer.post_attention_layernorm(
        attention_delta,
        prepared.residual,
    )

    if action_batch.static_mlp_run is True:
        output = layer.mlp(hidden_states)
    elif action_batch.static_mlp_run is False:
        output = residual * (mlp_scale - 1.0)
    else:
        mlp_active = mlp_run_mask & valid_rows.view(-1, 1)
        if action_batch.dense_reference_mlp:
            run_output = layer.mlp(hidden_states)
        else:
            run_output = _one_expert_mlp(
                hidden_states,
                layer.mlp.gate_up_proj.weight.unsqueeze(0),
                layer.mlp.down_proj.weight.unsqueeze(0),
                torch.ones_like(mlp_scale),
                mlp_active,
            )
        output = torch.where(
            mlp_run_mask,
            run_output,
            residual * (mlp_scale - 1.0),
        )

    skipper_adapter = getattr(
        forward_batch, "fd_full_graph_skipper_adapter", None
    )
    if skipper_adapter is not None and skipper_adapter.observes_mlp:
        skipper_adapter.observe_mlp(
            layer_id=prepared.layer_id,
            pre_mlp_hidden=residual,
            post_mlp_hidden=residual + output,
            batch_state=getattr(
                forward_batch, "fd_full_graph_skipper_state", None
            ),
        )

    route_masks = getattr(forward_batch, "fd_full_graph_route_masks", None)
    if route_masks is not None:
        route_masks.extend(
            (attention_run_mask.squeeze(-1), mlp_run_mask.squeeze(-1))
        )
    parity_context = prepared.parity_context
    if parity_context is not None:
        from sglang.srt.vpipe.cohort import (
            fd_parity_trace_advance,
        )
        from sglang.srt.vpipe.routing import (
            fd_parity_trace_tensor,
        )

        parity_row, parity_epoch = parity_context
        fd_parity_trace_tensor(
            "output_hidden",
            output[parity_row : parity_row + 1],
            layer_id=prepared.layer_id,
            token_epoch=parity_epoch,
        )
        fd_parity_trace_tensor(
            "output_residual",
            residual[parity_row : parity_row + 1],
            layer_id=prepared.layer_id,
            token_epoch=parity_epoch,
        )
        fd_parity_trace_advance(prepared.layer_id, forward_batch)
    return output, residual
def fd_execute_prepared_layer_route_full_graph(
    layer: Any,
    positions: torch.Tensor,
    forward_batch: Any,
    proj: Any,
    prepared: FullGraphPreparedLayerRoute,
    *,
    force_dense_all_run: Optional[bool] = None,
    force_filtered_all_run: bool = False,
    force_production_attention: Optional[bool] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute one exact RUN or mixed body from prepared device routes."""

    if (
        prepared.action_batch is not None
        and prepared.action_batch.execution_kind == SUBLAYER_EXECUTION
    ):
        if proj is not None:
            raise RuntimeError("sublayer execution must not load a projector")
        if any(
            value
            for value in (
                force_dense_all_run,
                force_filtered_all_run,
                force_production_attention,
            )
            if value is not None
        ):
            raise ValueError("sublayer execution does not support forced bodies")
        return _execute_sublayer_route_full_graph(
            layer,
            positions,
            forward_batch,
            prepared,
        )

    layer_id = prepared.layer_id
    hidden_states = prepared.hidden_states
    residual = prepared.residual
    route_weights = prepared.route_weights
    run_mask = prepared.run_mask
    parity_context = prepared.parity_context
    device_tape = getattr(
        forward_batch, "fd_full_graph_device_route_tape", None
    )
    if force_dense_all_run is None:
        force_dense_all_run = full_graph_forced_all_run_fastpath_enabled()
    if force_production_attention is None:
        force_production_attention = (
            full_graph_forced_all_run_production_attention_enabled()
        )
    if force_production_attention and not force_dense_all_run:
        raise ValueError(
            "production attention requires the exact dense all-RUN body"
        )
    if force_dense_all_run and force_filtered_all_run:
        raise ValueError("dense and filtered all-RUN bodies are mutually exclusive")

    masked_decode_attention = (
        full_graph_masked_decode_attention_enabled()
        and full_graph_decode_enabled(forward_batch)
        and not force_dense_all_run
    )
    prebound_kv_write_mask = getattr(
        forward_batch, "fd_full_graph_kv_write_mask", None
    )
    if masked_decode_attention:
        attention_run_mask = (
            run_mask.squeeze(-1) & forward_batch.fd_full_graph_valid_rows
        )
        forward_batch.fd_full_graph_attention_run_mask = attention_run_mask
        forward_batch.fd_full_graph_kv_write_mask = (
            attention_run_mask
            if full_graph_defer_project_kv_enabled()
            else prebound_kv_write_mask
        )
        forward_batch.fd_full_graph_attention_row_map = None
        forward_batch.fd_full_graph_attention_row_count = None
        forward_batch.fd_full_graph_attention_row_map_layer = None
        forward_batch.fd_full_graph_attention_worker_rows = None
        if full_graph_compact_routed_qkv_enabled():
            if prepared.run_row_map is None or prepared.route_counts is None:
                raise RuntimeError(
                    "compact routed QKV requires prepared RUN row metadata"
                )
            forward_batch.fd_full_graph_qkv_run_row_map = prepared.run_row_map
            forward_batch.fd_full_graph_qkv_run_row_count = prepared.route_counts[
                0:1
            ]

        if full_graph_mapped_decode_attention_enabled():
            compact_enabled, _, _, multiple = full_graph_compact_config()
            compact_o_enabled, layer_fractions = (
                full_graph_compact_o_proj_config()
            )
            layer_id = int(getattr(layer.self_attn.attn, "layer_id", -1))
            fraction = layer_fractions.get(layer_id)
            rows = int(hidden_states.shape[0])
            min_rows = full_graph_compact_o_proj_min_rows()
            worker_rows = (
                _compact_capacity(rows, fraction, multiple)
                if fraction is not None
                else rows
            )
            if (
                compact_enabled
                and compact_o_enabled
                and fraction is not None
                and rows >= min_rows
                and worker_rows < rows
            ):
                from sglang.srt.vpipe.cohort import (
                    build_row_map,
                )

                row_map, count = build_row_map(attention_run_mask)
                forward_batch.fd_full_graph_attention_row_map = row_map
                forward_batch.fd_full_graph_attention_row_count = count
                forward_batch.fd_full_graph_attention_row_map_layer = layer_id
                forward_batch.fd_full_graph_attention_worker_rows = worker_rows
    if force_production_attention:
        forward_batch.fd_full_graph_force_production_attention = True
    try:
        attention = layer.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
    finally:
        if force_production_attention:
            forward_batch.fd_full_graph_force_production_attention = False
        if masked_decode_attention:
            forward_batch.fd_full_graph_attention_run_mask = None
            forward_batch.fd_full_graph_kv_write_mask = prebound_kv_write_mask
            forward_batch.fd_full_graph_attention_row_map = None
            forward_batch.fd_full_graph_attention_row_count = None
            forward_batch.fd_full_graph_attention_row_map_layer = None
            forward_batch.fd_full_graph_attention_worker_rows = None
            forward_batch.fd_full_graph_qkv_run_row_map = None
            forward_batch.fd_full_graph_qkv_run_row_count = None
    if device_tape is not None and not full_graph_defer_project_kv_enabled():
        device_tape.record_inline_kv_ready(
            int(layer.layer_id),
            index=prepared.inline_kv_index,
        )
    attention = _apply_attention_run_mask(
        attention,
        run_mask,
        masked_decode_attention=masked_decode_attention,
    )
    hidden_states, residual = layer.post_attention_layernorm(attention, residual)

    route_masks = getattr(forward_batch, "fd_full_graph_route_masks", None)
    if route_masks is not None:
        route_masks.append(run_mask.squeeze(-1))
    output = fd_conditional_mlp_full_graph(
        layer,
        proj,
        hidden_states,
        route_weights,
        run_mask,
        valid_rows=forward_batch.fd_full_graph_valid_rows,
        compact_stats=forward_batch.fd_full_graph_compact_stats,
        compact_phase_enabled=forward_batch.fd_full_graph_compact_phase_enabled,
        prefill_grouped_enabled=(
            full_graph_prefill_grouped_mlp_enabled()
            and full_graph_prefill_enabled(forward_batch)
        ),
        force_dense_all_run=force_dense_all_run,
        force_filtered_all_run=force_filtered_all_run,
    )
    if parity_context is not None:
        from sglang.srt.vpipe.cohort import (
            fd_parity_trace_advance,
        )
        from sglang.srt.vpipe.routing import (
            fd_parity_trace_tensor,
        )

        parity_row, parity_epoch = parity_context
        fd_parity_trace_tensor(
            "output_hidden",
            output[parity_row : parity_row + 1],
            layer_id=layer_id,
            token_epoch=parity_epoch,
        )
        fd_parity_trace_tensor(
            "output_residual",
            residual[parity_row : parity_row + 1],
            layer_id=layer_id,
            token_epoch=parity_epoch,
        )
        fd_parity_trace_advance(layer_id, forward_batch)
    return output, residual
def fd_layer_forward_full_graph(
    layer: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: Any,
    residual: Optional[torch.Tensor],
    router: Any,
    proj: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fixed-shape decode or explicitly grouped prefill routing."""

    # (c3) negative witness: a SERVING-phase eager decode execution of this
    # skip body is the collapse regime and must be structurally unreachable
    # (invariant: exactly 0; capture/warmup/recapture excluded via the
    # stream-capture check + graph-lifecycle mark, F1).
    record_eager_skip_decode_layer_call(forward_batch)
    prepared = fd_prepare_layer_route_full_graph(
        layer,
        hidden_states,
        forward_batch,
        residual,
        router,
    )
    return fd_execute_prepared_layer_route_full_graph(
        layer,
        positions,
        forward_batch,
        proj,
        prepared,
    )
