"""Attention path for routed layers.

Runs for every row, with cache writes restricted to RUN rows. The masked and
mapped decode variants exist because the routed decode batch is sparse in a way
the stock attention kernels do not expect."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch
import torch.nn.functional as F
from sglang.srt.vpipe.env import (
    FD_COMPACT_O_PROJ_ENV,
    FD_COMPACT_O_PROJ_LAYERS_ENV,
    FD_COMPACT_O_PROJ_MIN_ROWS_ENV,
    FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV,
    FD_MAPPED_DECODE_ATTN_ENV,
    FD_MASKED_DECODE_ATTN_ENV,
)
from sglang.srt.vpipe.common import (
    _fixed_capacity_mapped_linear,
    full_graph_compact_q_proj_enabled,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_routed_qkv_enabled,
)
from sglang.srt.vpipe.config import (
    full_graph_compact_config,
    full_graph_compact_o_proj_config,
)
from sglang.srt.vpipe.env import (
    FD_CONTIGUOUS_ROUTED_QKV_ENV,
    FD_ROUTED_QKV_CAPACITIES_ENV,
    FD_ROUTED_QKV_CAPACITY_MULTIPLE_ENV,
    FD_ROUTED_QKV_MIN_ROWS_ENV,
)
from sglang.srt.vpipe.mlp_compact import (
    _compact_capacity,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_o_proj_min_rows,
    full_graph_contiguous_routed_qkv_config,
)


def _apply_attention_run_mask(
    attention: torch.Tensor,
    run_mask: torch.Tensor,
    *,
    masked_decode_attention: bool,
) -> torch.Tensor:
    """Zero non-RUN rows of the attention output. Always.

    653c21ec7f elided this multiply when `masked_decode_attention` is set, on the premise
    in the old docstring -- "masked decode plus bias-free o_proj emits zeros". That premise
    holds ONLY under the triton decode kernel, which is the sole reader of
    fd_full_graph_attention_run_mask; flashinfer has no such reader. On any other backend
    the elision left jump rows carrying a real non-zero attention output that
    post_attention_layernorm folded into the residual, on every routed layer without a
    compact-o_proj backstop -- 799 of 1184 (layer, bucket) cells in the arm where it shipped.

    The multiply is restored unconditionally as defence in depth. Under triton it is a
    NO-OP (the kernel already writes exact 0.0 for jump rows, decode_attention.py:1028-1034)
    and it is idempotent against the compact-o_proj path, which zeroes via new_zeros +
    scatter_cohort. Cost is one elementwise multiply per routed layer per decode step.
    ModelRunner._get_attention_backend now fails closed on a non-triton backend, so this is
    a second line of defence, not the primary one -- but the primary one did not exist when
    the elision shipped, and this is what would have caught it.
    """

    del masked_decode_attention  # retained for call-site compatibility; no longer gates
    return attention * run_mask.to(attention.dtype)
def fd_attention_o_proj_full_graph(
    attention: Any,
    attention_output: torch.Tensor,
    forward_batch: Any,
) -> torch.Tensor:
    """Project exact RUN rows through compact cuBLAS plus mapped overflow."""

    enabled, layer_fractions = full_graph_compact_o_proj_config()
    layer_id = int(getattr(attention.attn, "layer_id", -1))
    run_mask = getattr(forward_batch, "fd_full_graph_attention_run_mask", None)
    static_run = getattr(
        forward_batch, "fd_full_graph_attention_static_run", None
    )
    if (
        run_mask is not None
        and getattr(attention.o_proj, "bias", None) is not None
    ):
        raise RuntimeError(
            "FlexiDepth masked decode attention requires a bias-free o_proj"
        )
    if static_run is False:
        return attention_output.new_zeros(
            (attention_output.shape[0], int(attention.o_proj.weight.shape[0]))
        )
    if not enabled or layer_id not in layer_fractions or run_mask is None:
        output, _ = attention.o_proj(attention_output)
        return output

    compact_enabled, _, _, multiple = full_graph_compact_config()
    min_rows = full_graph_compact_o_proj_min_rows()
    rows = int(attention_output.shape[0])
    capacity = _compact_capacity(rows, layer_fractions[layer_id], multiple)
    if not compact_enabled or rows < min_rows or capacity >= rows:
        output, _ = attention.o_proj(attention_output)
        return output
    if attention_output.ndim != 2 or run_mask.shape != (rows,):
        raise RuntimeError(
            "FlexiDepth compact o_proj requires aligned 2D attention rows"
        )
    if getattr(attention.o_proj, "bias", None) is not None:
        raise RuntimeError("FlexiDepth compact o_proj requires a bias-free layer")

    from sglang.srt.vpipe.cohort import (
        build_row_map,
        scatter_cohort,
    )
    from sglang.srt.vpipe.kernel import (
        pack_rows,
    )
    from sglang.srt.vpipe.cohort import (
        mapped_linear,
    )

    row_map = getattr(forward_batch, "fd_full_graph_attention_row_map", None)
    count = getattr(forward_batch, "fd_full_graph_attention_row_count", None)
    row_map_layer = getattr(
        forward_batch, "fd_full_graph_attention_row_map_layer", None
    )
    if row_map is None or count is None or row_map_layer != layer_id:
        row_map, count = build_row_map(run_mask)
    packed_attention = pack_rows(
        attention_output,
        row_map,
        count,
        capacity=capacity,
    )
    packed_output, _ = attention.o_proj(packed_attention)
    output = attention_output.new_zeros(
        (rows, int(attention.o_proj.weight.shape[0]))
    )
    scatter_cohort(output, packed_output, row_map, count)
    mapped_linear(
        attention_output,
        attention.o_proj.weight,
        row_map,
        count,
        output,
        row_offset=capacity,
    )
    return output
def fd_attention_qkv_full_graph(
    attention: Any,
    hidden_states: torch.Tensor,
    forward_batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project own-weight QKV for the exact foreground RUN cohort."""

    enabled = full_graph_compact_q_proj_enabled()
    compact_routed_qkv = full_graph_compact_routed_qkv_enabled()
    (
        contiguous_routed_qkv,
        routed_qkv_min_rows,
        routed_qkv_multiple,
        routed_qkv_capacities,
    ) = full_graph_contiguous_routed_qkv_config()
    compact_o_enabled, layer_fractions = full_graph_compact_o_proj_config()
    layer_id = int(getattr(attention.attn, "layer_id", -1))
    run_mask = getattr(forward_batch, "fd_full_graph_attention_run_mask", None)
    compact_enabled, _, _, multiple = full_graph_compact_config()
    rows = int(hidden_states.shape[0])
    expected_width = attention.q_size + 2 * attention.kv_size
    qkv_weight = attention.qkv_proj.weight
    static_run = getattr(
        forward_batch, "fd_full_graph_attention_static_run", None
    )
    if static_run is False:
        if getattr(attention.qkv_proj, "bias", None) is not None:
            raise RuntimeError("AdaSkip K/V-only projection requires no QKV bias")
        if qkv_weight.ndim != 2 or int(qkv_weight.shape[0]) != expected_width:
            raise RuntimeError("AdaSkip K/V-only projection has incompatible weights")
        q = hidden_states.new_zeros((rows, attention.q_size))
        kv = F.linear(hidden_states, qkv_weight[attention.q_size : expected_width])
        k, v = kv.split([attention.kv_size, attention.kv_size], dim=-1)
        return q, k, v
    if compact_routed_qkv:
        row_map = getattr(
            forward_batch, "fd_full_graph_qkv_run_row_map", None
        )
        count = getattr(
            forward_batch, "fd_full_graph_qkv_run_row_count", None
        )
        if (
            hidden_states.ndim != 2
            or run_mask is None
            or run_mask.shape != (rows,)
            or row_map is None
            or row_map.shape != (rows,)
            or count is None
            or count.numel() != 1
        ):
            raise RuntimeError(
                "compact routed QKV requires aligned RUN row metadata"
            )
        if getattr(attention.qkv_proj, "bias", None) is not None:
            raise RuntimeError("compact routed QKV requires a bias-free layer")
        if qkv_weight.ndim != 2 or int(qkv_weight.shape[0]) != expected_width:
            raise RuntimeError("compact routed QKV has incompatible weights")

        qkv = hidden_states.new_empty((rows, expected_width))
        capacity_fractions = routed_qkv_capacities.get(layer_id)
        if contiguous_routed_qkv and rows >= routed_qkv_min_rows:
            if capacity_fractions is None:
                raise RuntimeError(
                    "contiguous routed QKV is missing the active layer"
                )
            run_capacity = _compact_capacity(
                rows, capacity_fractions[0], routed_qkv_multiple
            )
            _fixed_capacity_mapped_linear(
                hidden_states,
                qkv_weight,
                row_map,
                count,
                qkv,
                capacity=run_capacity,
            )
        else:
            from sglang.srt.vpipe.cohort import (
                mapped_linear,
            )

            mapped_linear(hidden_states, qkv_weight, row_map, count, qkv)
        return qkv.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
        )

    min_rows = full_graph_compact_o_proj_min_rows()
    fraction = layer_fractions.get(layer_id)
    capacity = (
        _compact_capacity(rows, fraction, multiple)
        if fraction is not None
        else rows
    )
    if (
        not enabled
        or not compact_o_enabled
        or run_mask is None
        or fraction is None
        or not compact_enabled
        or rows < min_rows
        or capacity >= rows
    ):
        qkv, _ = attention.qkv_proj(hidden_states)
        return qkv.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
        )

    if hidden_states.ndim != 2 or run_mask.shape != (rows,):
        raise RuntimeError(
            "FlexiDepth compact Q projection requires aligned 2D attention rows"
        )
    if getattr(attention.qkv_proj, "bias", None) is not None:
        raise RuntimeError(
            "FlexiDepth compact Q projection requires a bias-free layer"
        )
    if qkv_weight.ndim != 2 or int(qkv_weight.shape[0]) != expected_width:
        raise RuntimeError(
            "FlexiDepth compact Q projection has incompatible weights"
        )

    from sglang.srt.vpipe.cohort import (
        build_row_map,
        scatter_cohort,
    )
    from sglang.srt.vpipe.kernel import (
        pack_rows,
    )
    from sglang.srt.vpipe.cohort import (
        mapped_linear,
    )

    row_map = getattr(forward_batch, "fd_full_graph_attention_row_map", None)
    count = getattr(forward_batch, "fd_full_graph_attention_row_count", None)
    row_map_layer = getattr(
        forward_batch, "fd_full_graph_attention_row_map_layer", None
    )
    if row_map is None or count is None or row_map_layer != layer_id:
        row_map, count = build_row_map(run_mask)
    packed_hidden = pack_rows(
        hidden_states,
        row_map,
        count,
        capacity=capacity,
    )
    q_weight = qkv_weight[: attention.q_size]
    packed_q = F.linear(packed_hidden, q_weight)
    q = hidden_states.new_zeros((rows, attention.q_size))
    scatter_cohort(q, packed_q, row_map, count)
    mapped_linear(
        hidden_states,
        q_weight,
        row_map,
        count,
        q,
        row_offset=capacity,
    )

    kv = F.linear(hidden_states, qkv_weight[attention.q_size : expected_width])
    k, v = kv.split([attention.kv_size, attention.kv_size], dim=-1)
    forward_batch.fd_full_graph_attention_row_map = row_map
    forward_batch.fd_full_graph_attention_row_count = count
    forward_batch.fd_full_graph_attention_row_map_layer = layer_id
    return q, k, v
