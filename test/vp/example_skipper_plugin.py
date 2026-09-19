#!/usr/bin/env python3
"""A minimal third skipper plugin -- the worked example the contract promises.

This file is documentation that has to run. It is exactly what the paper's
appendix quotes, and it is deliberately the SMALLEST thing that satisfies the
contract: a policy that projects a fixed fraction of the routed layers for
every valid row, reusing the released checkpoint's projector.

Nothing in the shipped system imports it. It is not an arm, it measures nothing,
and it is not a reimplementation of any published system -- a static, input-
independent layer subset is the shape of Unified Layer Skipping (Liu et al.,
arXiv:2404.06954), and if it were ever measured it would have to be described as
"in the style of", never as that system.

The batch returned to the executor follows the contract exactly: an adapter
name, [rows, 1] branch weights and mask, an explicit mask WITHOUT a threshold or
forced action, and a non-empty payload-semantics string. Padding rows always RUN.

To make it servable you would add, in the design module:

    ARMS["integrated_staticdepth"] = {
        "skipper": "static_depth", "phases": "both",
        "regime_switch": True, "static_depth_ratio": 0.5,
    }

Selection stays in the tree; only the implementation is pluggable.
"""
from __future__ import annotations

import torch

from sglang.srt.vpipe.skipper import FLEXIDEPTH_PROJECTOR, register_skipper
from sglang.srt.vpipe.types import (
    FullGraphActionBatch,
    FullGraphSkipperAdapter,
    LogicalAction,
)


class StaticDepthSkipper(FullGraphSkipperAdapter):
    name = "static_depth"
    supported_actions = frozenset((LogicalAction.RUN,
                                   LogicalAction.PROJECT_ONLY))
    requires_router_weights = False
    projector_kind = FLEXIDEPTH_PROJECTOR
    payload_semantics = "binary_static_depth"

    def __init__(self, ratio):
        self.ratio = float(ratio)
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")

    def prepare_batch(self, *, hidden_states, valid_rows,
                      route_layer_order, **_):
        # Choose a tail within the engine's existing routed set.
        take = int(len(route_layer_order) * self.ratio)
        layers = frozenset(
            route_layer_order[len(route_layer_order) - take:])
        project_mask = (~valid_rows).view(-1, 1)
        run_mask = torch.ones_like(project_mask)
        # Preallocate device payloads; padding rows always RUN.
        dtype = hidden_states.dtype
        return (layers, run_mask, run_mask.to(dtype),
                project_mask, project_mask.to(dtype))

    def route(self, hidden_states, *, router, forced_action=None,
              layer_id=None, batch_state=None):
        if forced_action is not None:
            raise ValueError("forced routes are unsupported")
        if layer_id is None or batch_state is None:
            raise ValueError("layer and prepared batch required")
        layers, run, run_w, project, project_w = batch_state
        mask, weights = ((project, project_w) if layer_id in layers
                         else (run, run_w))
        return FullGraphActionBatch(
            adapter_name=self.name,
            supported_actions=self.supported_actions,
            branch_weights=weights, explicit_run_mask=mask,
            payload_semantics=self.payload_semantics)

    def attestation(self):
        return {"name": self.name, "decision": "static_tail_fraction",
                "static_depth_ratio": self.ratio,
                "requires_router_weights": self.requires_router_weights,
                "projector_kind": self.projector_kind,
                "supported_actions": ["run", "project_only"]}

def build_static_depth(arm):
    return StaticDepthSkipper(arm["static_depth_ratio"])

register_skipper("static_depth", build_static_depth)

