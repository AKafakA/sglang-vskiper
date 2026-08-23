"""Pluggable skipper policies — the generality axis (v1.5).

One interface, multiple policies, so "does vPipe depend on FlexiDepth?" is
answered by evidence rather than assertion:

* ``FlexiDepthFullGraphAdapter``  — the trained router (the payload).
* ``DeterministicMockFullGraphAdapter`` — RandomSkip: independent per-token
  skips whose expected skipped-depth equals a preset K, hashed on
  request_id (x) token_epoch (x) seed so it is reproducible under replay and
  capture. Measured 0.25005 against an expected 0.25 over 4.4M layer-rows.

(The AdaSkip sublayer adapter was dropped 2026-08-23 by owner ruling — its
name and knobs are refused at validation; see the removed-feature register.)

Policies emit ``LogicalAction``; unsupported actions fail closed rather than
being lowered into the binary RUN/PROJECT executor.
"""

from __future__ import annotations

import os
from dataclasses import (
    dataclass,
)
from functools import lru_cache
from math import ceil
from typing import Any, Mapping, Optional
import torch
from sglang.srt.vpipe.types import (
    FULL_GRAPH_ACTION_CONTRACT,
    FullGraphActionBatch,
    FullGraphSkipperAdapter,
    LogicalAction,
)
from sglang.srt.vpipe.types import (
    _LOGICAL_ACTION_CODES,
)
from sglang.srt.vpipe.env import (
    FULL_GRAPH_MOCK_SEED_ENV,
    FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO_ENV,
    FULL_GRAPH_MOCK_TOKEN_SKIP_RATE_ENV,
)
from sglang.srt.vpipe.env import (
    DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER,
    FLEXIDEPTH_FULL_GRAPH_SKIPPER,
    FULL_GRAPH_SKIPPER_ENV,
)


def route_digest_uses_logical_request_ids(
    adapter: FullGraphSkipperAdapter,
) -> bool:
    """Whether route evidence is keyed by stable logical request identity."""

    return bool(
        adapter.requires_stable_request_ids
        or adapter.route_digest_requires_stable_request_ids
    )
class FlexiDepthFullGraphAdapter(FullGraphSkipperAdapter):
    """Exact released FlexiDepth router and continuous branch-weight equation."""

    name = FLEXIDEPTH_FULL_GRAPH_SKIPPER
    route_digest_requires_stable_request_ids = True
    supported_actions = frozenset(
        (LogicalAction.RUN, LogicalAction.PROJECT_ONLY)
    )
    threshold = 0.5
    payload_semantics = (
        "run=mlp(hidden)*weight;project_only=proj(hidden)*(1-weight)"
    )

    def route(
        self,
        hidden_states: Any,
        *,
        router: Any,
        forced_action: Optional[LogicalAction] = None,
        layer_id: Optional[int] = None,
        batch_state: Any = None,
    ) -> FullGraphActionBatch:
        del layer_id, batch_state
        if router is None:
            raise RuntimeError("FlexiDepth full-graph routing requires router weights")

        import torch

        branch_weights = torch.sigmoid(router(hidden_states, use_graph=False))
        return FullGraphActionBatch(
            adapter_name=self.name,
            branch_weights=branch_weights,
            supported_actions=self.supported_actions,
            threshold=self.threshold,
            forced_action=forced_action,
            payload_semantics=self.payload_semantics,
        )

    def attestation(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contract": FULL_GRAPH_ACTION_CONTRACT,
            "logical_action_codes": dict(_LOGICAL_ACTION_CODES),
            "supported_actions": [
                LogicalAction.RUN.name.lower(),
                LogicalAction.PROJECT_ONLY.name.lower(),
            ],
            "decision_granularity": "token_layer",
            "decision": "trained_router_sigmoid_threshold_per_row",
            "threshold": self.threshold,
            "continuous_payload": "sigmoid_router_weight",
            "payload_semantics": self.payload_semantics,
            "state_semantics": "own_layer_kv_materialized_for_every_action",
            "requires_stable_request_ids": self.requires_stable_request_ids,
            "execution_storage": "bool_run_mask",
            "unsupported_actions": [
                action.name.lower()
                for action in (
                    LogicalAction.REUSE,
                    LogicalAction.SKIP_ATTN,
                    LogicalAction.SKIP_MLP,
                    LogicalAction.VETO,
                )
            ],
        }
_MOCK_HASH_BUCKETS = 1_000_003
_MOCK_EPOCH_MULTIPLIER = 1_000_000_007
@lru_cache(maxsize=None)
def _deterministic_mock_adapter(
    token_skip_rate: float,
    skipped_depth_ratio: float,
    seed: int,
) -> DeterministicMockFullGraphAdapter:
    return DeterministicMockFullGraphAdapter(
        token_skip_rate=token_skip_rate,
        skipped_depth_ratio=skipped_depth_ratio,
        seed=seed,
    )
def _parse_mock_config(values: Mapping[str, str]) -> tuple[float, float, int]:
    try:
        token_skip_rate = float(values.get(FULL_GRAPH_MOCK_TOKEN_SKIP_RATE_ENV, ""))
        skipped_depth_ratio = float(
            values.get(FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO_ENV, "")
        )
        seed = int(values.get(FULL_GRAPH_MOCK_SEED_ENV, "0"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "invalid deterministic full-graph mock configuration"
        ) from error
    return token_skip_rate, skipped_depth_ratio, seed
class DeterministicMockFullGraphAdapter(FullGraphSkipperAdapter):
    """Order-independent token-rate x skipped-depth systems ablation."""

    name = DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER
    supported_actions = frozenset(
        (LogicalAction.RUN, LogicalAction.PROJECT_ONLY)
    )
    requires_stable_request_ids = True
    payload_semantics = (
        "run=mlp(hidden)*1;project_only=proj(hidden)*(1-0)"
    )

    def __init__(
        self,
        *,
        token_skip_rate: float,
        skipped_depth_ratio: float,
        seed: int,
    ) -> None:
        if not 0.0 <= token_skip_rate <= 1.0:
            raise ValueError(
                f"{FULL_GRAPH_MOCK_TOKEN_SKIP_RATE_ENV} must be in [0, 1]"
            )
        if not 0.0 <= skipped_depth_ratio <= 1.0:
            raise ValueError(
                f"{FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO_ENV} must be in [0, 1]"
            )
        if not -(1 << 63) <= seed < (1 << 63):
            raise ValueError(f"{FULL_GRAPH_MOCK_SEED_ENV} must fit signed int64")
        self.token_skip_rate = token_skip_rate
        self.skipped_depth_ratio = skipped_depth_ratio
        self.seed = seed

    def prepare_batch(
        self,
        *,
        hidden_states: Any,
        request_ids: Any,
        token_epochs: Any,
        valid_rows: Any,
        route_layer_order: tuple[int, ...],
        request_slots: Any = None,
        phase: Optional[str] = None,
        batch_request_ids: Any = None,
        batch_request_slots: Any = None,
    ) -> DeterministicMockBatchState:
        del (
            request_slots,
            phase,
            batch_request_ids,
            batch_request_slots,
        )
        import torch

        rows = int(hidden_states.shape[0])
        expected_shape = (rows,)
        for name, value in (
            ("stable request IDs", request_ids),
            ("token epochs", token_epochs),
            ("valid rows", valid_rows),
        ):
            if value is None or tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"deterministic mock {name} do not match routed rows"
                )
            if value.device != hidden_states.device:
                raise RuntimeError(
                    f"deterministic mock {name} must share the hidden-state device"
                )
        if request_ids.dtype != torch.int64 or token_epochs.dtype != torch.int64:
            raise RuntimeError(
                "deterministic mock request IDs and token epochs must be int64"
            )
        if valid_rows.dtype != torch.bool:
            raise RuntimeError("deterministic mock valid rows must be boolean")
        if not route_layer_order:
            raise RuntimeError("deterministic mock requires routed layers")

        mixed = torch.bitwise_xor(
            request_ids,
            token_epochs * _MOCK_EPOCH_MULTIPLIER,
        )
        if self.seed:
            mixed = torch.bitwise_xor(mixed, self.seed)
        cutoff = int(self.token_skip_rate * _MOCK_HASH_BUCKETS)
        selected = (torch.remainder(mixed, _MOCK_HASH_BUCKETS) < cutoff) & valid_rows
        mixed_run_mask = (~selected).view(-1, 1)
        mixed_branch_weights = mixed_run_mask.to(dtype=hidden_states.dtype)
        all_run_mask = torch.ones_like(mixed_run_mask)
        all_run_branch_weights = torch.ones_like(mixed_branch_weights)

        project_layer_count = min(
            len(route_layer_order),
            ceil(self.skipped_depth_ratio * len(route_layer_order)),
        )
        project_layers = frozenset(
            route_layer_order[len(route_layer_order) - project_layer_count :]
            if project_layer_count
            else ()
        )
        return DeterministicMockBatchState(
            mixed_run_mask=mixed_run_mask,
            mixed_branch_weights=mixed_branch_weights,
            all_run_mask=all_run_mask,
            all_run_branch_weights=all_run_branch_weights,
            project_layers=project_layers,
        )

    def route(
        self,
        hidden_states: Any,
        *,
        router: Any,
        forced_action: Optional[LogicalAction] = None,
        layer_id: Optional[int] = None,
        batch_state: Any = None,
    ) -> FullGraphActionBatch:
        del hidden_states, router
        if forced_action is not None:
            raise ValueError("deterministic mock does not support forced routes")
        if layer_id is None:
            raise RuntimeError("deterministic mock requires a physical layer ID")
        if not isinstance(batch_state, DeterministicMockBatchState):
            raise RuntimeError("deterministic mock batch state is missing")

        if layer_id in batch_state.project_layers:
            run_mask = batch_state.mixed_run_mask
            branch_weights = batch_state.mixed_branch_weights
        else:
            run_mask = batch_state.all_run_mask
            branch_weights = batch_state.all_run_branch_weights
        return FullGraphActionBatch(
            adapter_name=self.name,
            branch_weights=branch_weights,
            supported_actions=self.supported_actions,
            explicit_run_mask=run_mask,
            payload_semantics=self.payload_semantics,
        )

    def attestation(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contract": FULL_GRAPH_ACTION_CONTRACT,
            "logical_action_codes": dict(_LOGICAL_ACTION_CODES),
            "supported_actions": ["run", "project_only"],
            "decision_granularity": "request_token_then_suffix_layer",
            "decision": "stable_request_id_token_epoch_hash",
            "token_skip_rate": self.token_skip_rate,
            "skipped_depth_ratio": self.skipped_depth_ratio,
            "seed": self.seed,
            "request_id_hash": "blake2b_64_signed",
            "hash_buckets": _MOCK_HASH_BUCKETS,
            "epoch_multiplier": _MOCK_EPOCH_MULTIPLIER,
            "continuous_payload": "binary_branch_weight",
            "payload_semantics": self.payload_semantics,
            "state_semantics": "own_layer_kv_materialized_for_every_action",
            "requires_stable_request_ids": self.requires_stable_request_ids,
            "execution_storage": "bool_run_mask",
            "unsupported_actions": ["reuse", "skip_attn", "skip_mlp", "veto"],
        }
@dataclass(frozen=True, slots=True)
class DeterministicMockBatchState:
    """Device tensors shared by every layer in one deterministic mock batch."""

    mixed_run_mask: Any
    mixed_branch_weights: Any
    all_run_mask: Any
    all_run_branch_weights: Any
    project_layers: frozenset[int]
def configured_full_graph_skipper_name(
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the normalized configured name without importing policy code."""

    values = os.environ if environ is None else environ
    return str(
        values.get(FULL_GRAPH_SKIPPER_ENV, FLEXIDEPTH_FULL_GRAPH_SKIPPER)
        or FLEXIDEPTH_FULL_GRAPH_SKIPPER
    ).strip().lower()
_ADAPTERS: dict[str, FullGraphSkipperAdapter] = {
    FLEXIDEPTH_FULL_GRAPH_SKIPPER: FlexiDepthFullGraphAdapter(),
}
