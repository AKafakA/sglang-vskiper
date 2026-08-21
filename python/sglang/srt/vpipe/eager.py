"""Eager FlexiDepth reference body — the quality-attribution instrument.

This is NOT a performance path. It exists so a quality result can be attributed:
if a task score drops, running the same checkpoint through this body answers
whether the loss comes from the FlexiDepth checkpoint itself or from vPipe's
graph-captured execution of it. Without it, a quality regression is unfalsifiable.

It is deliberately the *simple* implementation: run the router, gate attention by
the mask, then gather the routed rows and run either the MLP or the projector.
No compaction, no device-resident route tape, no deferred K/V. Those are the
mechanisms the full-graph path exists to provide, and keeping them out of here is
the point — this body is the thing they are compared against.

Written against the current package rather than lifted from the pre-refactor
tree: the frozen version dragged in parity-trace scaffolding, a
Candidate-A diagnostic branch, and the async-K/V wait, none of which belong in a
reference body.

CUDA graphs are incompatible with this body by construction. The row gather is
data-dependent (`nonzero`), which is a device->host sync and illegal under
stream capture -- the pre-refactor implementation failed the same way
(`cudaErrorStreamCaptureUnsupported`). Run it with `--disable-cuda-graph`;
`validation.py` refuses the combination at startup rather than letting capture
fail later.

K/V-completeness holds here exactly as in the full-graph path: `self_attn` runs
for EVERY row, so every token gets K/V from this layer's own projection weights.
Only the attention *output* is masked. K/V is never copied across layers.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.srt.vpipe.coverage import (
    record_eager_skip_decode_layer_call,
)

ROUTE_THRESHOLD = 0.5


def fd_layer_forward_eager(
    layer: Any,
    positions: torch.Tensor,
    hidden: torch.Tensor,
    forward_batch: Any,
    residual: Any,
    router: Any,
    proj: Any,
):
    """One FlexiDepth decoder layer, eagerly.

    Reproduces the released FlexiDepth semantics in SGLang's fused-residual
    convention:

        w    = sigmoid(router(norm(h)))          per row, in [0, 1]
        mask = w > 0.5                           1 = RUN, 0 = PROJECT_ONLY
        attention runs for every row; its OUTPUT is multiplied by mask
        RUN rows          -> mlp(h)  * w
        PROJECT_ONLY rows -> proj(h) * (1 - w)

    Returns ``(out, residual)``; the residual add is deferred to the next
    layer's fused norm, matching the surrounding SGLang convention.
    """

    # (c3) negative witness. In serving this body must be structurally
    # unreachable, so this counter is a gate invariant pinned at 0; the
    # graph-lifecycle sections (capture warmup, recapture) are excluded.
    record_eager_skip_decode_layer_call(forward_batch)

    if residual is None:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
    else:
        hidden, residual = layer.input_layernorm(hidden, residual)

    weight = torch.sigmoid(router(hidden))                      # [tokens, 1]
    mask = (weight > ROUTE_THRESHOLD).to(hidden.dtype)

    # Attention runs unconditionally: that is what makes the layer K/V-complete.
    # Masking its output is what makes a PROJECT_ONLY row "skipped".
    attn = layer.self_attn(
        positions=positions,
        hidden_states=hidden,
        forward_batch=forward_batch,
    )
    attn = attn * mask

    hidden, residual = layer.post_attention_layernorm(attn, residual)

    # Gather so the expensive MLP is genuinely skipped for PROJECT_ONLY rows and
    # the cheap projector runs instead. This gather is the data-dependent step
    # that makes the body uncapturable.
    run_rows = mask.squeeze(-1).bool()
    out = torch.empty_like(hidden)
    run_index = run_rows.nonzero(as_tuple=True)[0]
    project_index = (~run_rows).nonzero(as_tuple=True)[0]
    if run_index.numel() > 0:
        out.index_copy_(
            0,
            run_index,
            layer.mlp(hidden.index_select(0, run_index))
            * weight.index_select(0, run_index),
        )
    if project_index.numel() > 0:
        out.index_copy_(
            0,
            project_index,
            proj(hidden.index_select(0, project_index))
            * (1.0 - weight.index_select(0, project_index)),
        )
    return out, residual
