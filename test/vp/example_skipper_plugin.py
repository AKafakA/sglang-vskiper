#!/usr/bin/env python3
"""A minimal third skipper plugin — the worked example the contract promises.

This file is documentation that has to compile. It is what the paper's appendix
quotes, and it is deliberately the SMALLEST thing that satisfies the contract:
a policy that routes a fixed fraction of the routed layers for every token,
reusing the released checkpoint's projector.

Nothing in the shipped system imports it. It is not an arm, it measures nothing,
and it is not a reimplementation of any published system -- a static, input-
independent layer subset is the shape of Unified Layer Skipping (Liu et al.,
arXiv:2404.06954), and if it were ever measured it would have to be described as
"in the style of", never as that system.

To make it servable you would add, in `vpipe/design.py`:

    ARMS["integrated_staticdepth"] = {
        "skipper": "static_depth", "phases": "both", "regime_switch": True,
        "static_depth_ratio": 0.5,
    }

and call `register_skipper("static_depth", build_static_depth)` once at import.
Selection stays in the tree; only the implementation is pluggable (D-609).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from sglang.srt.vpipe.skipper import FLEXIDEPTH_PROJECTOR, register_skipper
from sglang.srt.vpipe.types import (
    FULL_GRAPH_ACTION_CONTRACT,
    FullGraphActionBatch,
    FullGraphSkipperAdapter,
    LogicalAction,
    _LOGICAL_ACTION_CODES,
)


class StaticDepthSkipper(FullGraphSkipperAdapter):
    """Project the deepest `ratio` of the routed layers, for every token.

    Three declarations carry the whole contract:

      * `supported_actions` — the executor refuses anything outside the binary
        pair rather than lowering it (so a sublayer policy fails closed);
      * `requires_router_weights = False` — this policy reads no trained gate;
      * `projector_kind` — but it still needs a projector, because a projected
        row must leave this layer's output and its own K/V behind.
    """

    name = "static_depth"
    supported_actions = frozenset((LogicalAction.RUN, LogicalAction.PROJECT_ONLY))
    requires_router_weights = False
    projector_kind = FLEXIDEPTH_PROJECTOR
    payload_semantics = "run=mlp(hidden)*1;project_only=proj(hidden)*(1-0)"

    def __init__(self, ratio: float) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"static_depth_ratio must be in [0, 1]; got {ratio}")
        self.ratio = float(ratio)
        self._projected_layers: frozenset[int] = frozenset()

    def prepare_batch(self, *, route_layer_order: tuple[int, ...], **_: Any) -> None:
        """Decide the layer subset once per routed batch, not per row.

        The subset is taken from the TAIL of the routed order: depth is expressed
        as a ratio *within* the design's routed set, never by editing
        SERVED_ROUTED_LAYERS -- that constant is shared by every arm and
        `verify_served_design` diffs against it.
        """
        take = int(len(route_layer_order) * self.ratio)
        self._projected_layers = frozenset(route_layer_order[len(route_layer_order) - take:])
        return None

    def route(
        self,
        hidden_states: Any,
        *,
        router: Any,
        forced_action: Optional[LogicalAction] = None,
        layer_id: Optional[int] = None,
        batch_state: Any = None,
    ) -> FullGraphActionBatch:
        """Return DEVICE decisions and execute neither branch.

        `router` is ignored: this policy brings its own rule. Returning a
        constant mask is legal under CUDA-graph capture precisely because it is
        data-independent -- no `nonzero()`, no device->host sync.
        """
        del router, batch_state
        rows = hidden_states.shape[0]
        project = layer_id is not None and layer_id in self._projected_layers
        run_mask = torch.full(
            (rows,), not project, dtype=torch.bool, device=hidden_states.device
        )
        return FullGraphActionBatch(
            supported_actions=self.supported_actions,
            branch_weights=run_mask.to(hidden_states.dtype),
            explicit_run_mask=run_mask,
            forced_action=forced_action,
            threshold=0.5,
        )

    def attestation(self) -> dict[str, Any]:
        """Publish the coordinates, so a mislabelled cell is detectable."""
        return {
            "name": self.name,
            "contract": FULL_GRAPH_ACTION_CONTRACT,
            "logical_action_codes": dict(_LOGICAL_ACTION_CODES),
            "supported_actions": ["run", "project_only"],
            "decision_granularity": "layer",
            "decision": "static_tail_fraction_of_routed_layers",
            "static_depth_ratio": self.ratio,
            "requires_router_weights": self.requires_router_weights,
            "projector_kind": self.projector_kind,
            "payload_semantics": self.payload_semantics,
            "state_semantics": "own_layer_kv_materialized_for_every_action",
            "unsupported_actions": ["reuse", "skip_attn", "skip_mlp", "veto"],
        }


def build_static_depth(arm: Mapping[str, Any]) -> FullGraphSkipperAdapter:
    """Factory: a policy's parameters come from the ARM, never the environment."""
    ratio = arm.get("static_depth_ratio")
    if ratio is None:
        raise ValueError(
            f"arm {arm.get('name', '<unnamed>')!r} selects static_depth but omits "
            "static_depth_ratio (define it in vpipe/design.py ARMS)"
        )
    return StaticDepthSkipper(float(ratio))


def install() -> None:
    """Register the policy. Call once; the arm then names it as its `skipper`."""
    register_skipper("static_depth", build_static_depth)
