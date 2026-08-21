"""Pluggable skipper policies — the generality axis (v1.5).

One interface, three policies, so "does vPipe depend on FlexiDepth?" is answered
by evidence rather than assertion:

* ``FlexiDepthFullGraphAdapter``  — the trained router (the payload).
* ``DeterministicMockFullGraphAdapter`` — RandomSkip: independent per-token
  skips whose expected skipped-depth equals a preset K, hashed on
  request_id (x) token_epoch (x) seed so it is reproducible under replay and
  capture. Measured 0.25005 against an expected 0.25 over 4.4M layer-rows.
* AdaSkip — sublayer attention/MLP actions, in ``skipper_adaskip.py``.

AdaSkip lives in its own module because it is the largest adapter by far and
because its action-semantics declaration is still ⛔ owner-gated for PROMOTION;
keeping it separable means the gate is visible in the file layout.

Policies emit ``LogicalAction``; unsupported actions fail closed rather than
being lowered into the binary RUN/PROJECT executor.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from typing import Any, Mapping, Optional
import torch
from sglang.srt.vpipe.env import (
    ADASKIP_MAX_GRAPH_ROWS_ENV,
    ADASKIP_MAX_REQUEST_SLOTS_ENV,
)
from sglang.srt.vpipe.env import (
    RUN_PROJECT_EXECUTION,
    SUBLAYER_EXECUTION,
)
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
    ADASKIP_DENSE_REFERENCE_MLP_ENV,
    FULL_GRAPH_MOCK_SEED_ENV,
    FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO_ENV,
    FULL_GRAPH_MOCK_TOKEN_SKIP_RATE_ENV,
)
from sglang.srt.vpipe.env import (
    ADASKIP_FULL_GRAPH_SKIPPER,
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
ADASKIP_OFFICIAL_SOURCE_REVISION = (
    "4ef686d6fd7b94756e4959f3f80191708dcec9e3"
)
ADASKIP_OFFICIAL_WINDOW = 20
ADASKIP_PROFILE_SCHEMA = "vpipe-adaskip-fixed-sublayer-profile-v2"
ADASKIP_OFFICIAL_EXTRA_MLP_MINIMUM = 0
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
@lru_cache(maxsize=None)
def _adaskip_adapter(
    profile_path: str,
    max_request_slots: Optional[int],
    max_graph_rows: Optional[int],
    dense_reference_mlp: bool,
) -> AdaSkipFixedProfileFullGraphAdapter:
    return AdaSkipFixedProfileFullGraphAdapter(
        load_adaskip_profile(profile_path),
        max_request_slots=max_request_slots,
        max_graph_rows=max_graph_rows,
        dense_reference_mlp=dense_reference_mlp,
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
def _parse_adaskip_capacity(values: Mapping[str, str], name: str) -> int:
    try:
        result = int(values.get(name, ""))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result
@dataclass(slots=True)
class AdaSkipOnlineDeviceState:
    """Persistent device tables keyed by scheduler request slot."""

    request_ids: Any
    active: Any
    decode_counts: Any
    mlp_similarity: Any
    mlp_ratio: Any
    activity_counters: Any
@dataclass(frozen=True, slots=True)
class AdaSkipFixedBatchState:
    """Graph-resident fixed actions plus optional per-request online state."""

    layer_to_index: Mapping[int, int]
    attention_run_masks: Any
    mlp_run_masks: Any
    attention_skip_scales: Any
    mlp_skip_scales: Any
    valid_rows: Any
    phase: Optional[str]
    request_slots: Any = None
    request_ids: Any = None
    active_rows: Any = None
    decode_counts: Any = None
    online_mlp_similarity: Any = None
    online_mlp_ratio: Any = None
    online_device_state: Optional[AdaSkipOnlineDeviceState] = None
def adaskip_dense_reference_mlp_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether online AdaSkip uses a dense, same-action MLP reference."""

    values = os.environ if environ is None else environ
    value = str(
        values.get(ADASKIP_DENSE_REFERENCE_MLP_ENV, "0")
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"{ADASKIP_DENSE_REFERENCE_MLP_ENV} must be a boolean value"
    )
class AdaSkipFixedProfileFullGraphAdapter(FullGraphSkipperAdapter):
    """Paper-faithful AdaSkip sublayers with optional online MLP adaptation."""

    name = ADASKIP_FULL_GRAPH_SKIPPER
    supported_actions = frozenset(
        (
            LogicalAction.RUN,
            LogicalAction.SKIP_ATTN,
            LogicalAction.SKIP_MLP,
        )
    )
    requires_flexidepth_weights = False
    execution_kind = SUBLAYER_EXECUTION
    payload_semantics = (
        "skip_attn=stream*prefill_attention_norm_ratio;"
        "skip_mlp=stream*prefill_mlp_norm_ratio"
    )

    def __init__(
        self,
        profile: AdaSkipProfile,
        *,
        max_request_slots: Optional[int] = None,
        max_graph_rows: Optional[int] = None,
        dense_reference_mlp: bool = False,
    ) -> None:
        if not profile.routed_layer_ids:
            raise ValueError("AdaSkip profile does not select any sublayers")
        online = profile.online_decode_extra_mlp
        capacities = (max_request_slots, max_graph_rows)
        if online and any(value is None for value in capacities):
            raise ValueError(
                "AdaSkip online decode requires request-slot and graph-row "
                "capacities"
            )
        if not online and any(value is not None for value in capacities):
            raise ValueError(
                "fixed AdaSkip profiles cannot set online state capacities"
            )
        if online and any(int(value) <= 0 for value in capacities):
            raise ValueError("AdaSkip online state capacities must be positive")
        if dense_reference_mlp and not online:
            raise ValueError(
                "AdaSkip dense MLP reference requires online decode"
            )
        self.profile = profile
        self.requires_stable_request_ids = online
        self.requires_request_slots = online
        self.observes_mlp = online
        self.max_request_slots = int(max_request_slots) if online else None
        self.max_graph_rows = int(max_graph_rows) if online else None
        self.dense_reference_mlp = dense_reference_mlp
        ranked_similarity = sorted(
            (
                *(layer.attention_similarity for layer in profile.layers),
                *(layer.mlp_similarity for layer in profile.layers),
            ),
            reverse=True,
        )
        self.extra_skip_threshold = ranked_similarity[
            profile.skip_sublayer_count - 1
        ]
        self._online_states: dict[Any, AdaSkipOnlineDeviceState] = {}
        # Per-(device, dtype) fixed-profile constants. Captured graphs bake
        # these tensors' addresses, so the cache is never cleared while the
        # adapter lives (reset_runtime_state must NOT drop it).
        self._fixed_tensor_cache: dict[Any, tuple] = {}

    def routed_layer_ids(
        self,
        *,
        num_hidden_layers: int,
        flexidepth_layer_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        if flexidepth_layer_ids:
            raise ValueError(
                "AdaSkip must not load FlexiDepth router/projector weights"
            )
        if self.profile.num_hidden_layers != num_hidden_layers:
            raise ValueError(
                "AdaSkip profile layer count does not match the served model"
            )
        if self.profile.online_decode_extra_mlp:
            return tuple(range(num_hidden_layers))
        return self.profile.routed_layer_ids

    def attention_routed_layer_ids(
        self,
        *,
        num_hidden_layers: int,
        flexidepth_layer_ids: tuple[int, ...],
    ) -> tuple[int, ...]:
        self.routed_layer_ids(
            num_hidden_layers=num_hidden_layers,
            flexidepth_layer_ids=flexidepth_layer_ids,
        )
        return tuple(
            layer.layer_id
            for layer in self.profile.layers
            if layer.skip_attention
        )

    def _online_state(self, hidden_states: Any) -> AdaSkipOnlineDeviceState:
        import torch

        device = hidden_states.device
        state = self._online_states.get(device)
        if state is not None:
            return state
        capacity = self.max_request_slots + self.max_graph_rows
        layers = self.profile.num_hidden_layers
        state = AdaSkipOnlineDeviceState(
            request_ids=torch.zeros(capacity, dtype=torch.int64, device=device),
            active=torch.zeros(capacity, dtype=torch.bool, device=device),
            decode_counts=torch.zeros(
                capacity, dtype=torch.int64, device=device
            ),
            mlp_similarity=torch.zeros(
                (capacity, layers), dtype=torch.float64, device=device
            ),
            mlp_ratio=torch.zeros(
                (capacity, layers), dtype=torch.float64, device=device
            ),
            activity_counters=torch.zeros(5, dtype=torch.int64, device=device),
        )
        self._online_states[device] = state
        return state

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
    ) -> AdaSkipFixedBatchState:
        del token_epochs
        import torch

        expected_order = (
            tuple(range(self.profile.num_hidden_layers))
            if self.profile.online_decode_extra_mlp
            else self.profile.routed_layer_ids
        )
        if route_layer_order != expected_order:
            raise RuntimeError("AdaSkip routed layer order changed")
        rows = int(hidden_states.shape[0])
        if valid_rows is None or tuple(valid_rows.shape) != (rows,):
            raise RuntimeError("AdaSkip valid rows do not match the graph bucket")
        if (
            valid_rows.dtype != torch.bool
            or valid_rows.device != hidden_states.device
        ):
            raise RuntimeError(
                "AdaSkip valid rows must be boolean and share the graph device"
            )
        layer_profiles = tuple(
            self.profile.layer(layer_id) for layer_id in route_layer_order
        )
        # Profile constants must not be rebuilt host-side per batch: an
        # unpinned CPU->CUDA copy is illegal during CUDA graph capture
        # (torch>=2.11 enforces it). Build once per (device, dtype) from a
        # PINNED host tensor (capture-legal even on a first capture-time
        # call) and reuse the device copies; expand() is a view and safe.
        cache_key = (hidden_states.device, hidden_states.dtype)
        cached = self._fixed_tensor_cache.get(cache_key)
        if cached is None:
            attention_run_base = torch.tensor(
                [not layer.skip_attention for layer in layer_profiles],
                dtype=torch.bool,
                pin_memory=True,
            ).to(hidden_states.device)
            mlp_run_base = torch.tensor(
                [not layer.skip_mlp for layer in layer_profiles],
                dtype=torch.bool,
                pin_memory=True,
            ).to(hidden_states.device)
            attention_scale_base = torch.tensor(
                [layer.attention_scale for layer in layer_profiles],
                dtype=hidden_states.dtype,
                pin_memory=True,
            ).to(hidden_states.device)
            mlp_scale_base = torch.tensor(
                [layer.mlp_scale for layer in layer_profiles],
                dtype=hidden_states.dtype,
                pin_memory=True,
            ).to(hidden_states.device)
            cached = (
                attention_run_base,
                mlp_run_base,
                attention_scale_base,
                mlp_scale_base,
            )
            self._fixed_tensor_cache[cache_key] = cached
        attention_run = cached[0].view(-1, 1, 1).expand(-1, rows, -1)
        mlp_run = cached[1].view(-1, 1, 1).expand(-1, rows, -1)
        attention_scale = cached[2].view(-1, 1, 1).expand(-1, rows, -1)
        mlp_scale = cached[3].view(-1, 1, 1).expand(-1, rows, -1)
        fixed_state = dict(
            layer_to_index={
                layer_id: index
                for index, layer_id in enumerate(route_layer_order)
            },
            attention_run_masks=attention_run,
            mlp_run_masks=mlp_run,
            attention_skip_scales=attention_scale,
            mlp_skip_scales=mlp_scale,
            valid_rows=valid_rows,
            phase=phase,
        )
        if not self.profile.online_decode_extra_mlp:
            return AdaSkipFixedBatchState(**fixed_state)
        if phase == "prefill":
            for name, value in (
                ("request IDs", batch_request_ids),
                ("request slots", batch_request_slots),
            ):
                if value is None or value.ndim != 1:
                    raise RuntimeError(
                        f"AdaSkip prefill {name} must be one-dimensional"
                    )
                if value.device != hidden_states.device:
                    raise RuntimeError(
                        f"AdaSkip prefill {name} must share the device"
                    )
            if batch_request_ids.shape != batch_request_slots.shape:
                raise RuntimeError(
                    "AdaSkip prefill request IDs and slots must align"
                )
            if batch_request_ids.dtype != torch.int64:
                raise RuntimeError("AdaSkip prefill request IDs must be int64")
            if batch_request_slots.dtype not in (torch.int32, torch.int64):
                raise RuntimeError(
                    "AdaSkip prefill request slots must be integral"
                )
            state = self._online_state(hidden_states)
            slots = batch_request_slots.to(dtype=torch.int64)
            state.request_ids.index_copy_(0, slots, batch_request_ids)
            state.active.index_fill_(0, slots, True)
            state.decode_counts.index_fill_(0, slots, 0)
            state.mlp_similarity.index_fill_(0, slots, 0.0)
            state.mlp_ratio.index_fill_(0, slots, 0.0)
            return AdaSkipFixedBatchState(**fixed_state)
        if phase != "decode":
            return AdaSkipFixedBatchState(**fixed_state)

        for name, value in (
            ("request IDs", request_ids),
            ("request slots", request_slots),
        ):
            if value is None or tuple(value.shape) != (rows,):
                raise RuntimeError(f"AdaSkip online {name} do not match rows")
            if value.device != hidden_states.device:
                raise RuntimeError(f"AdaSkip online {name} must share the device")
        if request_ids.dtype != torch.int64:
            raise RuntimeError("AdaSkip online request IDs must be int64")
        if request_slots.dtype not in (torch.int32, torch.int64):
            raise RuntimeError("AdaSkip online request slots must be integral")
        if rows > self.max_graph_rows:
            raise RuntimeError(
                "AdaSkip decode graph rows exceed "
                f"{ADASKIP_MAX_GRAPH_ROWS_ENV}"
            )

        row_ids = torch.arange(rows, dtype=torch.int64, device=hidden_states.device)
        padding_slots = self.max_request_slots + row_ids
        safe_slots = torch.where(
            valid_rows,
            request_slots.to(dtype=torch.int64),
            padding_slots,
        )
        online_state = self._online_state(hidden_states)
        active_rows = online_state.active.index_select(0, safe_slots)
        stored_request_ids = online_state.request_ids.index_select(0, safe_slots)
        new_requests = valid_rows & (
            (~active_rows) | (stored_request_ids != request_ids)
        )
        decode_counts = online_state.decode_counts.index_select(0, safe_slots)
        decode_counts = torch.where(
            new_requests,
            torch.zeros_like(decode_counts),
            decode_counts,
        )
        online_similarity = online_state.mlp_similarity.index_select(
            0, safe_slots
        )
        online_ratio = online_state.mlp_ratio.index_select(0, safe_slots)
        online_similarity = torch.where(
            new_requests.view(-1, 1),
            torch.zeros_like(online_similarity),
            online_similarity,
        )
        online_ratio = torch.where(
            new_requests.view(-1, 1),
            torch.zeros_like(online_ratio),
            online_ratio,
        )
        return AdaSkipFixedBatchState(
            **fixed_state,
            request_slots=safe_slots,
            request_ids=request_ids,
            active_rows=active_rows,
            decode_counts=decode_counts,
            online_mlp_similarity=online_similarity,
            online_mlp_ratio=online_ratio,
            online_device_state=online_state,
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
        del hidden_states
        if router is not None:
            raise RuntimeError("AdaSkip does not consume FlexiDepth router weights")
        if forced_action is not None:
            raise ValueError("AdaSkip does not support forced whole-layer routes")
        if layer_id is None:
            raise RuntimeError("AdaSkip requires a physical layer ID")
        if not isinstance(batch_state, AdaSkipFixedBatchState):
            raise RuntimeError("AdaSkip graph batch state is missing")
        try:
            index = batch_state.layer_to_index[layer_id]
        except KeyError as error:
            raise RuntimeError(f"AdaSkip layer {layer_id} is not routed") from error
        layer_profile = self.profile.layer(layer_id)
        attention_run = batch_state.attention_run_masks[index]
        fixed_mlp_run = batch_state.mlp_run_masks[index]
        mlp_scale = batch_state.mlp_skip_scales[index]
        static_mlp_run: Optional[bool] = not layer_profile.skip_mlp
        if (
            self.profile.online_decode_extra_mlp
            and batch_state.phase == "decode"
        ):
            import torch

            mature = (
                batch_state.decode_counts
                >= self.profile.online_decode_window
            ).view(-1, 1)
            extra_skip = mature & (
                batch_state.online_mlp_similarity[:, layer_id : layer_id + 1]
                >= self.extra_skip_threshold
            )
            valid = batch_state.valid_rows.view(-1, 1)
            batch_state.online_device_state.activity_counters[1].add_(
                (mature & valid).sum(dtype=torch.int64)
            )
            batch_state.online_device_state.activity_counters[2].add_(
                (extra_skip & valid).sum(dtype=torch.int64)
            )
            mlp_run = fixed_mlp_run & ~extra_skip
            mlp_scale = torch.where(
                extra_skip,
                batch_state.online_mlp_ratio[:, layer_id : layer_id + 1],
                mlp_scale,
            ).to(dtype=batch_state.mlp_skip_scales.dtype)
            static_mlp_run = None
        else:
            mlp_run = fixed_mlp_run
        return FullGraphActionBatch(
            adapter_name=self.name,
            branch_weights=mlp_run.to(dtype=batch_state.mlp_skip_scales.dtype),
            supported_actions=self.supported_actions,
            payload_semantics=self.payload_semantics,
            execution_kind=self.execution_kind,
            attention_run_mask=attention_run,
            mlp_run_mask=mlp_run,
            attention_skip_scale=batch_state.attention_skip_scales[index],
            mlp_skip_scale=mlp_scale,
            static_attention_run=not layer_profile.skip_attention,
            static_mlp_run=static_mlp_run,
            dense_reference_mlp=self.dense_reference_mlp,
        )

    def observe_mlp(
        self,
        *,
        layer_id: int,
        pre_mlp_hidden: Any,
        post_mlp_hidden: Any,
        batch_state: Any,
    ) -> None:
        if (
            not self.profile.online_decode_extra_mlp
            or not isinstance(batch_state, AdaSkipFixedBatchState)
            or batch_state.phase != "decode"
            or self.profile.layer(layer_id).skip_mlp
        ):
            return
        import torch
        import torch.nn.functional as F

        collect = batch_state.valid_rows & (
            batch_state.decode_counts < self.profile.online_decode_window
        )
        batch_state.online_device_state.activity_counters[0].add_(
            collect.sum(dtype=torch.int64)
        )
        old_similarity = batch_state.online_mlp_similarity[:, layer_id]
        old_ratio = batch_state.online_mlp_ratio[:, layer_id]
        observation_similarity = F.cosine_similarity(
            pre_mlp_hidden.float(),
            post_mlp_hidden.float(),
            dim=-1,
            eps=1e-8,
        )
        observation_ratio = (
            torch.linalg.vector_norm(post_mlp_hidden, dim=-1).to(torch.float64)
            / torch.linalg.vector_norm(pre_mlp_hidden, dim=-1).to(torch.float64)
        )
        observation_similarity = observation_similarity.to(torch.float64)
        count = batch_state.decode_counts.to(dtype=torch.float64)
        denominator = count + 1.0
        next_similarity = (
            old_similarity * count + observation_similarity
        ) / denominator
        next_ratio = (old_ratio * count + observation_ratio) / denominator
        old_similarity.copy_(
            torch.where(collect, next_similarity, old_similarity)
        )
        old_ratio.copy_(torch.where(collect, next_ratio, old_ratio))

    def validate_runtime_capacities(
        self,
        *,
        request_pool_slots: int,
        decode_graph_rows: int,
    ) -> None:
        if not self.profile.online_decode_extra_mlp:
            return
        if request_pool_slots > self.max_request_slots:
            raise ValueError(
                f"{ADASKIP_MAX_REQUEST_SLOTS_ENV}={self.max_request_slots} "
                "cannot cover the resolved SGLang request pool with "
                f"{request_pool_slots} slots"
            )
        if decode_graph_rows > self.max_graph_rows:
            raise ValueError(
                f"{ADASKIP_MAX_GRAPH_ROWS_ENV}={self.max_graph_rows} "
                "cannot cover the largest resolved decode graph with "
                f"{decode_graph_rows} rows"
            )

    def finalize_batch(self, *, batch_state: Any) -> None:
        if (
            not self.profile.online_decode_extra_mlp
            or not isinstance(batch_state, AdaSkipFixedBatchState)
            or batch_state.phase != "decode"
        ):
            return
        import torch

        device_state = batch_state.online_device_state
        if device_state is None:
            raise RuntimeError("AdaSkip online device state is missing")
        next_counts = torch.where(
            batch_state.valid_rows
            & (batch_state.decode_counts < self.profile.online_decode_window),
            batch_state.decode_counts + 1,
            batch_state.decode_counts,
        )
        valid_rows = batch_state.valid_rows
        device_state.activity_counters[3].add_(
            valid_rows.sum(dtype=torch.int64)
        )
        device_state.activity_counters[4].add_(
            (
                valid_rows
                & (
                    batch_state.decode_counts
                    >= self.profile.online_decode_window
                )
            ).sum(dtype=torch.int64)
        )
        device_state.request_ids.index_copy_(
            0, batch_state.request_slots, batch_state.request_ids
        )
        device_state.active.index_copy_(
            0,
            batch_state.request_slots,
            batch_state.valid_rows | batch_state.active_rows,
        )
        device_state.decode_counts.index_copy_(
            0, batch_state.request_slots, next_counts
        )
        device_state.mlp_similarity.index_copy_(
            0, batch_state.request_slots, batch_state.online_mlp_similarity
        )
        device_state.mlp_ratio.index_copy_(
            0, batch_state.request_slots, batch_state.online_mlp_ratio
        )

    def reset_runtime_state(self) -> None:
        for state in self._online_states.values():
            state.request_ids.zero_()
            state.active.zero_()
            state.decode_counts.zero_()
            state.mlp_similarity.zero_()
            state.mlp_ratio.zero_()
            state.activity_counters.zero_()

    def _online_activity(self) -> dict[str, Any]:
        totals = [0] * 5
        for state in self._online_states.values():
            values = state.activity_counters.detach().cpu().tolist()
            totals = [left + int(right) for left, right in zip(totals, values)]
        mature_eligible = totals[1]
        return {
            "observed_mlp_rows": totals[0],
            "mature_eligible_mlp_rows": mature_eligible,
            "extra_skip_mlp_rows": totals[2],
            "extra_skip_ratio": (
                totals[2] / mature_eligible if mature_eligible else 0.0
            ),
            "decode_request_rows": totals[3],
            "mature_request_rows": totals[4],
        }

    def attestation(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "contract": FULL_GRAPH_ACTION_CONTRACT,
            "logical_action_codes": dict(_LOGICAL_ACTION_CODES),
            "supported_actions": ["run", "skip_attn", "skip_mlp"],
            "decision_granularity": (
                "fixed_model_attention_and_request_token_mlp"
                if self.profile.online_decode_extra_mlp
                else "fixed_model_sublayer"
            ),
            "decision": (
                "offline_topk_then_online_20_token_mlp_threshold"
                if self.profile.online_decode_extra_mlp
                else "offline_topk_concat_attention_then_mlp_similarity"
            ),
            "skip_sublayer_count": self.profile.skip_sublayer_count,
            "routed_layer_ids": list(
                range(self.profile.num_hidden_layers)
                if self.profile.online_decode_extra_mlp
                else self.profile.routed_layer_ids
            ),
            "fixed_routed_layer_ids": list(self.profile.routed_layer_ids),
            "attention_skip_layer_ids": [
                layer.layer_id
                for layer in self.profile.layers
                if layer.skip_attention
            ],
            "mlp_skip_layer_ids": [
                layer.layer_id for layer in self.profile.layers if layer.skip_mlp
            ],
            "calibration_request_count": self.profile.calibration_request_count,
            "calibration_dataset_id": self.profile.calibration_dataset_id,
            "calibration_dataset_revision": (
                self.profile.calibration_dataset_revision
            ),
            "calibration_sha256": self.profile.calibration_sha256,
            "profile_sha256": self.profile.input_sha256,
            "source_revision": self.profile.source_revision,
            "model_id": self.profile.model_id,
            "model_revision": self.profile.model_revision,
            "online_decode_window": self.profile.online_decode_window,
            "online_decode_extra_mlp": self.profile.online_decode_extra_mlp,
            "online_extra_mlp_minimum": self.profile.online_extra_mlp_minimum,
            "online_accumulator_dtype": (
                "float64_python_scalar_equivalent"
                if self.profile.online_decode_extra_mlp
                else None
            ),
            "online_activation": (
                "first_decode_forward_after_20_observed_decode_forwards"
                if self.profile.online_decode_extra_mlp
                else None
            ),
            "online_extra_skip_threshold": self.extra_skip_threshold,
            "online_state_key": (
                "scheduler_request_slot_stable_request_hash"
                if self.profile.online_decode_extra_mlp
                else None
            ),
            "online_max_request_slots": self.max_request_slots,
            "online_max_graph_rows": self.max_graph_rows,
            "dense_reference_mlp": {
                "enabled": self.dense_reference_mlp,
                "scope": (
                    "online_dynamic_mlp_all_rows_dense_then_same_action_select"
                    if self.dense_reference_mlp
                    else "disabled"
                ),
                "performance_claim_allowed": False,
            },
            "online_activity": (
                self._online_activity()
                if self.profile.online_decode_extra_mlp
                else None
            ),
            "continuous_payload": "per_sublayer_norm_ratio",
            "payload_semantics": self.payload_semantics,
            "native_state_semantics": "skip_attention_omits_layer_kv",
            "state_semantics": "own_layer_kv_materialized_for_every_action",
            "execution_storage": "independent_attention_mlp_run_masks",
            "requires_stable_request_ids": self.requires_stable_request_ids,
            "unsupported_actions": ["project_only", "reuse", "veto"],
        }
@dataclass(frozen=True, slots=True)
class AdaSkipProfile:
    model_id: str
    model_revision: str
    source_revision: str
    calibration_dataset_id: str
    calibration_dataset_revision: str
    calibration_request_count: int
    calibration_sha256: str
    skip_sublayer_count: int
    online_decode_window: int
    online_decode_extra_mlp: bool
    online_extra_mlp_minimum: int
    layers: tuple[AdaSkipLayerProfile, ...]
    input_sha256: str

    @property
    def num_hidden_layers(self) -> int:
        return len(self.layers)

    @property
    def routed_layer_ids(self) -> tuple[int, ...]:
        return tuple(layer.layer_id for layer in self.layers if layer.is_routed)

    def layer(self, layer_id: int) -> AdaSkipLayerProfile:
        if layer_id < 0 or layer_id >= len(self.layers):
            raise ValueError(f"AdaSkip layer {layer_id} is outside the profile")
        result = self.layers[layer_id]
        if result.layer_id != layer_id:
            raise RuntimeError("AdaSkip profile layer ordering is corrupt")
        return result

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        input_sha256: str,
    ) -> "AdaSkipProfile":
        _require_keys(
            value,
            required=frozenset(
                (
                    "schema",
                    "model",
                    "source",
                    "calibration",
                    "policy",
                    "layers",
                )
            ),
            context="AdaSkip profile",
        )
        if value["schema"] != ADASKIP_PROFILE_SCHEMA:
            raise ValueError(
                f"AdaSkip profile schema must be {ADASKIP_PROFILE_SCHEMA}"
            )
        model = value["model"]
        source = value["source"]
        calibration = value["calibration"]
        policy = value["policy"]
        layers = value["layers"]
        for name, item in (
            ("model", model),
            ("source", source),
            ("calibration", calibration),
            ("policy", policy),
        ):
            if not isinstance(item, dict):
                raise ValueError(f"AdaSkip profile {name} must be an object")
        if not isinstance(layers, list) or not layers:
            raise ValueError("AdaSkip profile layers must be a nonempty list")
        _require_keys(
            model,
            required=frozenset(("id", "revision", "num_hidden_layers")),
            context="AdaSkip profile model",
        )
        num_layers = _positive_int(
            model["num_hidden_layers"], name="AdaSkip profile num_hidden_layers"
        )
        if len(layers) != num_layers:
            raise ValueError(
                f"AdaSkip profile requires {num_layers} ordered layer records"
            )
        _require_keys(
            source,
            required=frozenset(("repository", "revision")),
            context="AdaSkip profile source",
        )
        if source["repository"] != "https://github.com/ASISys/AdaSkip":
            raise ValueError("AdaSkip profile source is not the official repository")
        source_revision = _nonempty_string(
            source["revision"], name="AdaSkip profile source revision"
        )
        if source_revision != ADASKIP_OFFICIAL_SOURCE_REVISION:
            raise ValueError(
                "AdaSkip profile source revision does not match the audited source"
            )
        _require_keys(
            calibration,
            required=frozenset(
                (
                    "dataset_id",
                    "dataset_revision",
                    "request_count",
                    "sha256",
                )
            ),
            context="AdaSkip profile calibration",
        )
        request_count = _positive_int(
            calibration["request_count"],
            name="AdaSkip profile calibration request_count",
        )
        if request_count != ADASKIP_OFFICIAL_WINDOW:
            raise ValueError(
                "AdaSkip profile requires the audited 20-request calibration"
            )
        calibration_sha256 = _nonempty_string(
            calibration["sha256"], name="AdaSkip calibration sha256"
        )
        if len(calibration_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in calibration_sha256
        ):
            raise ValueError("AdaSkip calibration sha256 must be lowercase hex")
        _require_keys(
            policy,
            required=frozenset(
                (
                    "skip_sublayer_count",
                    "selection",
                    "compensation",
                    "online_decode_window",
                    "online_decode_extra_mlp",
                    "online_extra_mlp_minimum",
                    "native_kv",
                    "vp_kv",
                )
            ),
            context="AdaSkip profile policy",
        )
        skip_count = _positive_int(
            policy["skip_sublayer_count"],
            name="AdaSkip skip_sublayer_count",
        )
        if skip_count > 2 * num_layers:
            raise ValueError("AdaSkip skip_sublayer_count exceeds model sublayers")
        if policy["selection"] != "topk_concat_attention_then_mlp_similarity":
            raise ValueError("AdaSkip profile selection semantics changed")
        if policy["compensation"] != "mean_output_norm_over_input_norm":
            raise ValueError("AdaSkip profile compensation semantics changed")
        online_window = _positive_int(
            policy["online_decode_window"],
            name="AdaSkip online_decode_window",
        )
        if online_window != ADASKIP_OFFICIAL_WINDOW:
            raise ValueError("AdaSkip online decode window must be 20 tokens")
        online_extra = policy["online_decode_extra_mlp"]
        if not isinstance(online_extra, bool):
            raise ValueError("AdaSkip online_decode_extra_mlp must be boolean")
        online_extra_minimum = policy["online_extra_mlp_minimum"]
        if (
            isinstance(online_extra_minimum, bool)
            or not isinstance(online_extra_minimum, int)
            or online_extra_minimum != ADASKIP_OFFICIAL_EXTRA_MLP_MINIMUM
        ):
            raise ValueError(
                "AdaSkip online extra MLP minimum must match released value 0"
            )
        if policy["native_kv"] != "omit_on_skip_attention":
            raise ValueError("AdaSkip native K/V semantics changed")
        if policy["vp_kv"] != "materialize_own_layer_projection":
            raise ValueError("AdaSkip VP K/V semantics changed")

        parsed_layers: list[AdaSkipLayerProfile] = []
        for expected_layer, item in enumerate(layers):
            if not isinstance(item, dict):
                raise ValueError("AdaSkip layer records must be objects")
            _require_keys(
                item,
                required=frozenset(
                    (
                        "layer_id",
                        "attention_similarity",
                        "mlp_similarity",
                        "attention_scale",
                        "mlp_scale",
                        "skip_attention",
                        "skip_mlp",
                    )
                ),
                context=f"AdaSkip layer {expected_layer}",
            )
            if item["layer_id"] != expected_layer:
                raise ValueError("AdaSkip layer records must be ordered and complete")
            if not isinstance(item["skip_attention"], bool) or not isinstance(
                item["skip_mlp"], bool
            ):
                raise ValueError("AdaSkip skip actions must be boolean")
            parsed_layers.append(
                AdaSkipLayerProfile(
                    layer_id=expected_layer,
                    attention_similarity=_cosine_float(
                        item["attention_similarity"],
                        name=f"AdaSkip attention_similarity[{expected_layer}]",
                    ),
                    mlp_similarity=_cosine_float(
                        item["mlp_similarity"],
                        name=f"AdaSkip mlp_similarity[{expected_layer}]",
                    ),
                    attention_scale=_positive_float(
                        item["attention_scale"],
                        name=f"AdaSkip attention_scale[{expected_layer}]",
                    ),
                    mlp_scale=_positive_float(
                        item["mlp_scale"],
                        name=f"AdaSkip mlp_scale[{expected_layer}]",
                    ),
                    skip_attention=item["skip_attention"],
                    skip_mlp=item["skip_mlp"],
                )
            )

        observed = sum(
            int(layer.skip_attention) + int(layer.skip_mlp)
            for layer in parsed_layers
        )
        if observed != skip_count:
            raise ValueError(
                "AdaSkip selected sublayer count does not match the policy"
            )
        ranked = sorted(
            range(2 * num_layers),
            key=lambda index: (
                -(
                    parsed_layers[index].attention_similarity
                    if index < num_layers
                    else parsed_layers[index - num_layers].mlp_similarity
                ),
                index,
            ),
        )
        expected = frozenset(ranked[:skip_count])
        selected = frozenset(
            (
                layer.layer_id
                if layer.skip_attention
                else -1
                for layer in parsed_layers
            )
        ) | frozenset(
            (
                num_layers + layer.layer_id
                if layer.skip_mlp
                else -1
                for layer in parsed_layers
            )
        )
        selected = selected - {-1}
        if selected != expected:
            raise ValueError(
                "AdaSkip selected sublayers do not match concatenated top-k similarity"
            )
        return cls(
            model_id=_nonempty_string(model["id"], name="AdaSkip profile model id"),
            model_revision=_nonempty_string(
                model["revision"], name="AdaSkip profile model revision"
            ),
            source_revision=source_revision,
            calibration_dataset_id=_nonempty_string(
                calibration["dataset_id"],
                name="AdaSkip profile calibration dataset id",
            ),
            calibration_dataset_revision=_nonempty_string(
                calibration["dataset_revision"],
                name="AdaSkip profile calibration dataset revision",
            ),
            calibration_request_count=request_count,
            calibration_sha256=calibration_sha256,
            skip_sublayer_count=skip_count,
            online_decode_window=online_window,
            online_decode_extra_mlp=online_extra,
            online_extra_mlp_minimum=online_extra_minimum,
            layers=tuple(parsed_layers),
            input_sha256=input_sha256,
        )
@lru_cache(maxsize=None)
def load_adaskip_profile(path: str | Path) -> AdaSkipProfile:
    resolved = Path(path).expanduser().resolve()
    value, payload = _load_json_bytes(resolved)
    return AdaSkipProfile.from_mapping(
        value,
        input_sha256=_sha256_bytes(payload),
    )
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
class AdaSkipLayerProfile:
    layer_id: int
    attention_similarity: float
    mlp_similarity: float
    attention_scale: float
    mlp_scale: float
    skip_attention: bool
    skip_mlp: bool

    @property
    def is_routed(self) -> bool:
        return self.skip_attention or self.skip_mlp
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
def _require_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    keys = frozenset(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{context} is missing: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")
def _nonempty_string(value: Any, *, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} must be a nonempty string")
    return result
def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
def _cosine_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
    return result
def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result
def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
def _load_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read AdaSkip JSON {path}: {error}") from error
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid AdaSkip JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("AdaSkip JSON root must be an object")
    return value, payload
_ADAPTERS: dict[str, FullGraphSkipperAdapter] = {
    FLEXIDEPTH_FULL_GRAPH_SKIPPER: FlexiDepthFullGraphAdapter(),
}
def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
