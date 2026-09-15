"""Attention path for routed layers.

Runs for every row, with cache writes restricted to RUN rows. The masked and
mapped decode variants exist because the routed decode batch is sparse in a way
the stock attention kernels do not expect."""

from __future__ import annotations

from typing import Any

import torch


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
    """Project the attention rows through the layer's own ``o_proj``.

    [lane-2 knob cleanup] The compact-o_proj cohort branch that used to
    live here is deleted with its three env vars
    (``SGLANG_FD_FULL_GRAPH_COMPACT_O_PROJ{,_LAYERS,_MIN_ROWS}``). It was never
    enabled in any served arm — the gates default off — and before 2026-09-03 it
    could not have run at all: it passed ``capacity=`` to ``kernel.pack_rows``,
    which takes a preallocated destination, so its first use raised TypeError.
    What remains is the plain projection plus the bias guard that the masked
    decode path genuinely needs.
    """

    run_mask = getattr(forward_batch, "fd_full_graph_attention_run_mask", None)
    if (
        run_mask is not None
        and getattr(attention.o_proj, "bias", None) is not None
    ):
        raise RuntimeError(
            "FlexiDepth masked decode attention requires a bias-free o_proj"
        )
    output, _ = attention.o_proj(attention_output)
    return output
def fd_attention_qkv_full_graph(
    attention: Any,
    hidden_states: torch.Tensor,
    forward_batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project QKV for the whole batch through the layer's own weights.

    [lane-2 knob cleanup] The compact-Q cohort branch is deleted along
    with ``SGLANG_FD_FULL_GRAPH_COMPACT_Q_PROJ`` and the o_proj layer-fraction
    gates it shared. Like the o_proj branch above, it was never enabled in any
    served arm and carried the same latent defect: it called
    ``kernel.pack_rows`` with a ``capacity=`` keyword that function does not
    accept, so its first execution would have raised TypeError.
    """

    qkv, _ = attention.qkv_proj(hidden_states)
    return qkv.split(
        [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
    )

