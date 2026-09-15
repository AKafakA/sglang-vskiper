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
from typing import Any, Callable, Mapping, Optional
import torch
from sglang.srt.vpipe.types import (
    FULL_GRAPH_ACTION_CONTRACT,
    PROJECTOR_CONTRACT,
    FullGraphActionBatch,
    FullGraphSkipperAdapter,
    LogicalAction,
    ProjectorAdapter,
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
FLEXIDEPTH_PROJECTOR = "flexidepth"


class FlexiDepthProjector(ProjectorAdapter):
    """The released FlexiDepth ``router_proj``: a low-rank gate/up/down MLP.

    This is the ONLY projector, and it is the worked example. A PROJECT_ONLY row
    must still leave this layer's output and this layer's own K/V behind
    (K/V-completeness), so a projector is what makes "skip" cheap rather than
    absent. Adding another means: subclass ``ProjectorAdapter``, name the
    per-layer checkpoint tensors it loads, register it here, and point an arm's
    policy at it by name.
    """

    name = FLEXIDEPTH_PROJECTOR
    checkpoint_tensor_suffixes = (
        "router_proj.gate_proj.weight",
        "router_proj.down_proj.weight",
        "router_proj.up_proj.weight",
    )

    def attestation(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contract": PROJECTOR_CONTRACT,
            "payload": "low_rank_gate_up_down_mlp",
            "reduction_config_key": "proj_reduction_factor",
            "checkpoint_tensor_suffixes": list(self.checkpoint_tensor_suffixes),
            "state_semantics": "own_layer_kv_materialized_for_every_action",
        }


_PROJECTORS: dict[str, ProjectorAdapter] = {
    FLEXIDEPTH_PROJECTOR: FlexiDepthProjector(),
}


def register_projector(projector: ProjectorAdapter) -> None:
    """Add a projector implementation under its own ``name``."""

    existing = _PROJECTORS.get(projector.name)
    if existing is not None and type(existing) is not type(projector):
        raise ValueError(f"projector {projector.name!r} is already registered")
    _PROJECTORS[projector.name] = projector


def resolve_projector(name: str) -> ProjectorAdapter:
    """Return the named projector, or refuse with the names that do exist."""

    try:
        return _PROJECTORS[name]
    except KeyError:
        raise ValueError(
            f"unknown projector {name!r}; registered: "
            + ", ".join(sorted(_PROJECTORS))
        ) from None


class FlexiDepthFullGraphAdapter(FullGraphSkipperAdapter):
    """Exact released FlexiDepth router and continuous branch-weight equation."""

    name = FLEXIDEPTH_FULL_GRAPH_SKIPPER
    route_digest_requires_stable_request_ids = True
    # Reads the trained per-layer gate, and projects with the checkpoint's own
    # router_proj. Both halves come from the same weights file, which is why they
    # were one boolean until.
    requires_router_weights = True
    projector_kind = FLEXIDEPTH_PROJECTOR
    supported_actions = frozenset(
        (LogicalAction.RUN, LogicalAction.PROJECT_ONLY)
    )
    threshold = 0.5

    @property
    def payload_semantics(self) -> str:
        # [v1.5] Reported from the ARM's gate mode, so the attestation cannot describe the
        # released arithmetic while a hard_mask checkpoint is served.
        from sglang.srt.vpipe.common import full_graph_gate_mode

        if full_graph_gate_mode() == "hard_mask":
            return "run=mlp(hidden);project_only=proj(hidden)"
        return "run=mlp(hidden)*weight;project_only=proj(hidden)*(1-weight)"

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
class DeterministicMockFullGraphAdapter(FullGraphSkipperAdapter):
    """Order-independent token-rate x skipped-depth systems ablation."""

    name = DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER
    supported_actions = frozenset(
        (LogicalAction.RUN, LogicalAction.PROJECT_ONLY)
    )
    requires_stable_request_ids = True
    # The policy is a hash of (request_id, token_epoch, seed): NO trained gate is
    # read. It still needs a projector, because a skipped row must leave this
    # layer's output and K/V behind. That asymmetry is the whole point of the
    # split -- before it, this adapter was described as *requiring
    # FlexiDepth weights*, which was true of the projector and false of the router.
    requires_router_weights = False
    projector_kind = FLEXIDEPTH_PROJECTOR
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

    # The skipper comes from the ACTIVE ARM, never the environment. This read
    # defaulted to `flexidepth`, so `vdec_randomskip` -- whose arm declares
    # deterministic_mock -- served TRAINED FlexiDepth routes while attesting a random
    # mock, and passed the smoke gate doing it. The arm's declaration was decorative:
    # exactly the declared-vs-served split exists to prevent.
    from sglang.srt.vpipe.design import active_arm

    return str(active_arm().get("skipper") or FLEXIDEPTH_FULL_GRAPH_SKIPPER).strip().lower()
# A policy is built FROM THE ACTIVE ARM, so a parameterised policy (RandomSkip's
# rate x depth x seed) needs no special case at the resolver -- which is what it
# had, and what made "add a third skipper" a two-place edit in library code.
SkipperFactory = Callable[[Mapping[str, Any]], FullGraphSkipperAdapter]

_FLEXIDEPTH_ADAPTER = FlexiDepthFullGraphAdapter()


def _build_flexidepth(arm: Mapping[str, Any]) -> FullGraphSkipperAdapter:
    del arm
    return _FLEXIDEPTH_ADAPTER


def _build_deterministic_mock(arm: Mapping[str, Any]) -> FullGraphSkipperAdapter:
    # The mock's parameters come from the ARM definition, like its name.
    # These are the RandomSkip study's independent variables (skip rate x depth),
    # so they must be as durable and attestable as the design itself.
    missing = [
        key
        for key in ("mock_token_skip_rate", "mock_skipped_depth_ratio", "mock_seed")
        if arm.get(key) is None
    ]
    if missing:
        raise ValueError(
            f"arm {arm.get('name', '<unnamed>')!r} selects the deterministic mock "
            "but omits " + ", ".join(missing)
            + " (define them in vpipe/design.py ARMS, D-611)"
        )
    return _deterministic_mock_adapter(
        float(arm["mock_token_skip_rate"]),
        float(arm["mock_skipped_depth_ratio"]),
        int(arm["mock_seed"]),
    )


_SKIPPERS: dict[str, SkipperFactory] = {
    FLEXIDEPTH_FULL_GRAPH_SKIPPER: _build_flexidepth,
    DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER: _build_deterministic_mock,
}


def register_skipper(name: str, factory: SkipperFactory) -> None:
    """Add a policy under ``name``; an arm then selects it by that name.

    Selection stays IN THE TREE (`design.py` ARMS), never in the environment
     -- this only decides what a name may resolve to.
    """

    if name in _SKIPPERS and _SKIPPERS[name] is not factory:
        raise ValueError(f"skipper {name!r} is already registered")
    _SKIPPERS[name] = factory


def available_skippers() -> tuple[str, ...]:
    """Registered policy names, for error messages and attestation."""

    return tuple(sorted(_SKIPPERS))


def build_skipper(name: str, arm: Mapping[str, Any]) -> FullGraphSkipperAdapter:
    """Resolve a policy name against the registry, refusing an unknown name.

    The old `_ADAPTERS[name]` raised a bare KeyError, which is fail-closed but
    tells an operator nothing about what they could have written instead.
    """

    factory = _SKIPPERS.get(name)
    if factory is None:
        raise ValueError(
            f"unknown skipper {name!r}; registered: " + ", ".join(available_skippers())
        )
    return factory(arm)
