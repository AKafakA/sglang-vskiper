"""AdaSkip — sublayer attention/MLP skipping, as a pluggable policy.

Separate module because it is the largest adapter and because its
action-semantics declaration is ⛔ owner-gated for PROMOTION (F7 evidence is
banked: sublayer skip 0.5000 exact at 42.77M rows, both phases).

The contract that must not be violated: AdaSkip's independent attention and MLP
actions are NEVER lowered into the binary RUN/PROJECT executor. An unsupported
action fails closed. Same-layers-for-every-token would be pruning, not the
dynamic skipping this system is about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping, Optional
import torch
from pathlib import PurePosixPath
from sglang.srt.vpipe.env import (
    ADASKIP_DENSE_REFERENCE_MLP_ENV,
    ADASKIP_FULL_GRAPH_SKIPPER,
    ADASKIP_MAX_GRAPH_ROWS_ENV,
    ADASKIP_MAX_REQUEST_SLOTS_ENV,
)
from sglang.srt.vpipe.env import (
    SUBLAYER_EXECUTION,
)
from sglang.srt.vpipe.types import (
    FullGraphActionBatch,
    FullGraphSkipperAdapter,
    LogicalAction,
)
from sglang.srt.vpipe.types import (
    FULL_GRAPH_ACTION_CONTRACT,
    _LOGICAL_ACTION_CODES,
)
from sglang.srt.vpipe.adaskip_profile import (
    AdaSkipProfile,
    load_adaskip_profile,
)


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

    def assert_profile_matches_checkpoint(
        self, model_identity: Optional[Mapping[str, str]]
    ) -> None:
        """Bind the calibration profile to the CHECKPOINT it was measured on.

        Layer count alone is not identity. Every Llama-3-8B derivative has 32
        layers, so a profile calibrated on a different checkpoint used to load
        happily and apply ITS skip masks and compensation scales -- silently
        changing what the server generates, with no error and no attestation
        difference. Skip decisions are checkpoint-specific by construction.

        Fail closed: a profile with no declared provenance, or a deployment
        whose checkpoint identity cannot be determined, is refused rather than
        assumed compatible.
        """

        want_rev = (self.profile.model_revision or "").strip()
        want_id = (self.profile.model_id or "").strip()
        if not want_rev and not want_id:
            raise ValueError(
                "AdaSkip profile declares neither model_revision nor model_id, "
                "so it cannot be bound to the served checkpoint; rebuild it "
                "with provenance"
            )
        ident = dict(model_identity or {})
        got_rev = (ident.get("revision") or "").strip()
        got_id = (ident.get("model_id") or "").strip()
        if want_rev and got_rev:
            if got_rev != want_rev:
                raise ValueError(
                    f"AdaSkip profile was calibrated on revision {want_rev!r} "
                    f"but the served checkpoint is {got_rev!r}; its skip masks "
                    "and compensation scales do not describe this model"
                )
            return
        # No revision on one side: fall back to the checkpoint DIRECTORY NAME,
        # which pins the released snapshot even when the absolute path differs
        # between hosts. Comparing full paths would reject a correct profile
        # merely staged elsewhere.
        if want_id and got_id:
            if PurePosixPath(want_id).name != PurePosixPath(got_id).name:
                raise ValueError(
                    f"AdaSkip profile was calibrated on {PurePosixPath(want_id).name!r} "
                    f"but the served checkpoint is {PurePosixPath(got_id).name!r}"
                )
            return
        raise ValueError(
            "AdaSkip cannot verify the served checkpoint against the profile "
            f"(profile revision={want_rev or '<none>'} id={want_id or '<none>'}; "
            f"served revision={got_rev or '<none>'} id={got_id or '<none>'}). "
            "Refusing rather than applying another checkpoint's skip masks."
        )

    def routed_layer_ids(
        self,
        *,
        num_hidden_layers: int,
        flexidepth_layer_ids: tuple[int, ...],
        model_identity: Optional[Mapping[str, str]] = None,
    ) -> tuple[int, ...]:
        if flexidepth_layer_ids:
            raise ValueError(
                "AdaSkip must not load FlexiDepth router/projector weights"
            )
        if self.profile.num_hidden_layers != num_hidden_layers:
            raise ValueError(
                "AdaSkip profile layer count does not match the served model"
            )
        self.assert_profile_matches_checkpoint(model_identity)
        if self.profile.online_decode_extra_mlp:
            return tuple(range(num_hidden_layers))
        return self.profile.routed_layer_ids

    def attention_routed_layer_ids(
        self,
        *,
        num_hidden_layers: int,
        flexidepth_layer_ids: tuple[int, ...],
        model_identity: Optional[Mapping[str, str]] = None,
    ) -> tuple[int, ...]:
        self.routed_layer_ids(
            num_hidden_layers=num_hidden_layers,
            flexidepth_layer_ids=flexidepth_layer_ids,
            model_identity=model_identity,
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
