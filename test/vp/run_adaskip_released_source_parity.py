#!/usr/bin/env python3
"""Differential gate against the hash-pinned released AdaSkip E2E source.

This is a semantic and numerical correctness gate, not performance evidence.
It executes the released ``LlamaDecoderLayer.forward`` function directly from
its parsed source, without importing AdaSkip's version-pinned Transformers
module, then compares its actions and equations with the VP adapter.  The
online filtered MLP is checked separately against the dense Llama equation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.vpipe.adaskip_profile import (
    ADASKIP_CALIBRATION_SCHEMA,
    AdaSkipCalibration,
    profile_from_calibration,
    selected_sublayer_ids,
)
from sglang.srt.vpipe.skipper import (
    ADASKIP_OFFICIAL_SOURCE_REVISION,
)
from sglang.srt.vpipe.executor import (
    _execute_sublayer_route_full_graph,
)
from sglang.srt.vpipe.mlp_compact import (
    _one_expert_mlp,
)
from sglang.srt.vpipe.routing import (
    FullGraphPreparedLayerRoute,
)
from sglang.srt.vpipe.common import (
    resolve_full_graph_skipper,
)
from sglang.srt.vpipe.env import (
    SUBLAYER_EXECUTION,
)
from sglang.srt.vpipe.types import (
    FullGraphActionBatch,
)
from sglang.srt.vpipe.types import (
    LogicalAction,
)


OFFICIAL_E2E_MODEL_SHA256 = (
    "a59b20c898986964c8c1152f9236d8f25a2f3727dc4856225e122fbf858c14da"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _released_symbols(path: Path) -> tuple[Any, Any]:
    if _sha256(path) != OFFICIAL_E2E_MODEL_SHA256:
        raise RuntimeError("released AdaSkip E2E model hash changed")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    select_node = None
    forward_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "select_indices":
            select_node = node
        if isinstance(node, ast.ClassDef) and node.name == "LlamaDecoderLayer":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "forward":
                    forward_node = child
                    break
    if select_node is None or forward_node is None:
        raise RuntimeError("released AdaSkip symbols are missing")
    namespace = {
        "torch": torch,
        "Any": Any,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "Cache": object,
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[select_node, forward_node], type_ignores=[])
    )
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["select_indices"], namespace["forward"]


def _profile_payload(*, online: bool) -> tuple[Any, bytes]:
    calibration = AdaSkipCalibration.from_mapping(
        {
            "schema": ADASKIP_CALIBRATION_SCHEMA,
            "model": {
                "id": "NousResearch/Meta-Llama-3-8B-Instruct",
                "revision": "53346005fb0ef11d3b6a83b12c895cca40156b6c",
                "num_hidden_layers": 4,
            },
            "source": {
                "repository": "https://github.com/ASISys/AdaSkip",
                "revision": ADASKIP_OFFICIAL_SOURCE_REVISION,
            },
            "calibration": {
                "dataset_id": "released-source-differential-fixture",
                "dataset_revision": "fixture",
                "request_count": 20,
            },
            "measurements": {
                "attention_similarity": [0.1, 0.99, 0.2, 0.98],
                "mlp_similarity": [0.97, 0.3, 0.4, 0.5],
                "attention_scale": [1.01, 1.02, 1.03, 1.04],
                "mlp_scale": [1.05, 1.06, 1.07, 1.08],
            },
        },
        input_sha256="a" * 64,
    )
    return profile_from_calibration(
        calibration,
        skip_sublayer_count=3,
        online_decode_extra_mlp=online,
    )


class _Identity:
    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _RecordingIdentity:
    def __init__(self) -> None:
        self.last_input: Optional[torch.Tensor] = None

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.last_input = value
        return value


class _ReleasedAttention:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        past_key_value: Any = None,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, None, Any]:
        self.calls += 1
        return hidden_states * 0.125, None, past_key_value


class _VPAttention:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, hidden_states: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        self.calls += 1
        return hidden_states * 0.125


class _DeltaMLP:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return value * 0.375


class _VPPostAttentionNorm:
    def __call__(
        self, delta: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_hidden = residual + delta
        return full_hidden, full_hidden


def _released_layer(layer_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        layer_idx=layer_id,
        input_layernorm=_Identity(),
        self_attn=_ReleasedAttention(),
        post_attention_layernorm=_RecordingIdentity(),
        mlp=_DeltaMLP(),
        cos_sim=torch.nn.CosineSimilarity(dim=2, eps=1e-8),
    )


def _vp_layer(layer_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id,
        self_attn=_VPAttention(),
        post_attention_layernorm=_VPPostAttentionNorm(),
        mlp=_DeltaMLP(),
    )


def _released_forward_call(
    forward: Any,
    layer: SimpleNamespace,
    hidden_states: torch.Tensor,
    *,
    attention_scales: list[float],
    mlp_scales: list[float],
    fixed_attention: list[int],
    fixed_mlp: list[int],
    extra_mlp: list[int],
    online_similarity: list[float],
    online_ratio: list[float],
    decode_count: int,
) -> torch.Tensor:
    zeros = [0.0] * len(attention_scales)
    return forward(
        layer,
        hidden_states,
        use_cache=False,
        attn_prefill_sim_diffreq_avg=zeros.copy(),
        mlp_prefill_sim_diffreq_avg=zeros.copy(),
        attn_prefill_ratio_diffreq_avg=attention_scales,
        mlp_prefill_ratio_diffreq_avg=mlp_scales,
        mlp_decode_20_sim_thisreq_avg=online_similarity,
        mlp_decode_20_ratio_thisreq_avg=online_ratio,
        this_req_decode_count=decode_count,
        diff_req_count=20,
        FIX_SKIP_ATTN_LAYER_ID=fixed_attention,
        FIX_SKIP_MLP_LAYER_ID=fixed_mlp,
        EXTRA_SKIP_MLP_LAYER_ID=extra_mlp,
    )[0]


def _forward_batch(rows: int, adapter: Any = None, state: Any = None) -> Any:
    return SimpleNamespace(
        fd_full_graph_valid_rows=torch.ones(rows, dtype=torch.bool, device="cuda"),
        fd_full_graph_device_route_tape=None,
        fd_full_graph_route_masks=None,
        fd_full_graph_attention_run_mask=None,
        fd_full_graph_attention_static_run=None,
        fd_full_graph_kv_write_mask=None,
        fd_full_graph_skipper_adapter=adapter,
        fd_full_graph_skipper_state=state,
    )


def _prepared(
    *,
    layer_id: int,
    hidden_states: torch.Tensor,
    action: Any,
) -> FullGraphPreparedLayerRoute:
    residual = hidden_states
    return FullGraphPreparedLayerRoute(
        layer_id=layer_id,
        hidden_states=hidden_states,
        residual=residual,
        route_weights=action.branch_weights,
        run_mask=action.attention_run_mask & action.mlp_run_mask,
        parity_context=None,
        inline_kv_index=None,
        action_batch=action,
        attention_run_mask=action.attention_run_mask,
        mlp_run_mask=action.mlp_run_mask,
        attention_skip_scale=action.attention_skip_scale,
        mlp_skip_scale=action.mlp_skip_scale,
    )


def _fixed_source_parity(select_indices: Any, forward: Any) -> dict[str, Any]:
    profile, payload = _profile_payload(online=False)
    combined = [
        *(layer.attention_similarity for layer in profile.layers),
        *(layer.mlp_similarity for layer in profile.layers),
    ]
    official_selected = select_indices(combined, profile.skip_sublayer_count)
    vp_selected = []
    for action in selected_sublayer_ids(profile):
        name, value = action.split(":")
        layer_id = int(value)
        vp_selected.append(layer_id if name == "attention" else 4 + layer_id)
    vp_ranked = sorted(
        vp_selected, key=lambda index: (-combined[index], index)
    )
    if official_selected != vp_ranked:
        raise AssertionError(
            f"fixed action mismatch: released={official_selected} vp={vp_ranked}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        profile_path = Path(temp_dir) / "profile.json"
        profile_path.write_bytes(payload)
        adapter = resolve_full_graph_skipper(
            {
                "SGLANG_VP_FULL_GRAPH_SKIPPER": "adaskip",
                "SGLANG_VP_ADASKIP_PROFILE": str(profile_path),
            }
        )
        fixed_attention = [
            layer.layer_id for layer in profile.layers if layer.skip_attention
        ]
        fixed_mlp = [layer.layer_id for layer in profile.layers if layer.skip_mlp]
        attention_scales = [layer.attention_scale for layer in profile.layers]
        mlp_scales = [layer.mlp_scale for layer in profile.layers]
        cases = {}
        for dtype, atol in ((torch.float32, 1e-6), (torch.bfloat16, 0.015625)):
            hidden = torch.linspace(
                -0.75, 0.75, steps=24, device="cuda", dtype=dtype
            ).view(1, 3, 8)
            flat_hidden = hidden.view(-1, hidden.shape[-1])
            state = adapter.prepare_batch(
                hidden_states=flat_hidden,
                request_ids=None,
                token_epochs=None,
                valid_rows=torch.ones(3, dtype=torch.bool, device="cuda"),
                route_layer_order=profile.routed_layer_ids,
            )
            dtype_cases = {}
            for layer_id, expected_action in (
                (1, "skip_attention"),
                (0, "skip_mlp"),
            ):
                released_layer = _released_layer(layer_id)
                released = _released_forward_call(
                    forward,
                    released_layer,
                    hidden.clone(),
                    attention_scales=attention_scales,
                    mlp_scales=mlp_scales,
                    fixed_attention=fixed_attention,
                    fixed_mlp=fixed_mlp,
                    extra_mlp=[],
                    online_similarity=[0.0] * 4,
                    online_ratio=[0.0] * 4,
                    decode_count=0,
                )
                action = adapter.route(
                    flat_hidden,
                    router=None,
                    layer_id=layer_id,
                    batch_state=state,
                )
                vp_layer = _vp_layer(layer_id)
                output, residual = _execute_sublayer_route_full_graph(
                    vp_layer,
                    torch.arange(3, device="cuda"),
                    _forward_batch(3),
                    _prepared(
                        layer_id=layer_id,
                        hidden_states=flat_hidden,
                        action=action,
                    ),
                )
                vp_full = (output + residual).view_as(released)
                error = _error(vp_full, released)
                torch.testing.assert_close(
                    vp_full, released, rtol=0.0, atol=atol
                )
                attention_runs = bool(action.attention_run_mask[0].item())
                mlp_runs = bool(action.mlp_run_mask[0].item())
                if expected_action == "skip_attention":
                    assert not attention_runs and mlp_runs
                    assert released_layer.self_attn.calls == 0
                    assert vp_layer.self_attn.calls == 1
                else:
                    assert attention_runs and not mlp_runs
                    assert released_layer.mlp.calls == 0
                    assert vp_layer.mlp.calls == 0
                dtype_cases[str(layer_id)] = {
                    "action": expected_action,
                    "released_attention_calls": released_layer.self_attn.calls,
                    "vp_attention_projection_calls": vp_layer.self_attn.calls,
                    "exact_tensor_equal": torch.equal(vp_full, released),
                    **error,
                }
            cases[str(dtype).removeprefix("torch.")] = {
                "atol": atol,
                "layers": dtype_cases,
            }
    return {"selected_indices": official_selected, "cases": cases}


def _online_source_parity(forward: Any) -> dict[str, Any]:
    profile, payload = _profile_payload(online=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        profile_path = Path(temp_dir) / "profile.json"
        profile_path.write_bytes(payload)
        adapter = resolve_full_graph_skipper(
            {
                "SGLANG_VP_FULL_GRAPH_SKIPPER": "adaskip",
                "SGLANG_VP_ADASKIP_PROFILE": str(profile_path),
                "SGLANG_VP_ADASKIP_MAX_REQUEST_SLOTS": "8",
                "SGLANG_VP_ADASKIP_MAX_GRAPH_ROWS": "4",
            }
        )
        hidden = torch.linspace(
            0.25, 1.0, steps=8, device="cuda", dtype=torch.float32
        ).view(1, 1, 8)
        request_ids = torch.tensor([101], dtype=torch.int64, device="cuda")
        request_slots = torch.tensor([3], dtype=torch.int32, device="cuda")
        valid_rows = torch.ones(1, dtype=torch.bool, device="cuda")
        fixed_attention = [
            layer.layer_id for layer in profile.layers if layer.skip_attention
        ]
        fixed_mlp = [layer.layer_id for layer in profile.layers if layer.skip_mlp]
        attention_scales = [layer.attention_scale for layer in profile.layers]
        mlp_scales = [layer.mlp_scale for layer in profile.layers]
        released_similarity = [0.0] * 4
        released_ratio = [0.0] * 4
        max_similarity_delta = 0.0
        max_ratio_delta = 0.0
        for decode_count in range(profile.online_decode_window):
            released_layer = _released_layer(2)
            released_output = _released_forward_call(
                forward,
                released_layer,
                hidden.clone(),
                attention_scales=attention_scales,
                mlp_scales=mlp_scales,
                fixed_attention=fixed_attention,
                fixed_mlp=fixed_mlp,
                extra_mlp=[],
                online_similarity=released_similarity,
                online_ratio=released_ratio,
                decode_count=decode_count,
            )
            pre_mlp = released_layer.post_attention_layernorm.last_input
            assert pre_mlp is not None
            state = adapter.prepare_batch(
                hidden_states=hidden.view(1, -1),
                request_ids=request_ids,
                request_slots=request_slots,
                token_epochs=torch.tensor([decode_count], device="cuda"),
                valid_rows=valid_rows,
                route_layer_order=(0, 1, 2, 3),
                phase="decode",
            )
            action = adapter.route(
                hidden.view(1, -1),
                router=None,
                layer_id=2,
                batch_state=state,
            )
            assert action.mlp_run_mask.item() is True
            adapter.observe_mlp(
                layer_id=2,
                pre_mlp_hidden=pre_mlp.view(1, -1),
                post_mlp_hidden=released_output.view(1, -1),
                batch_state=state,
            )
            vp_similarity = float(state.online_mlp_similarity[0, 2].item())
            vp_ratio = float(state.online_mlp_ratio[0, 2].item())
            max_similarity_delta = max(
                max_similarity_delta,
                abs(vp_similarity - released_similarity[2]),
            )
            max_ratio_delta = max(
                max_ratio_delta,
                abs(vp_ratio - released_ratio[2]),
            )
            adapter.finalize_batch(batch_state=state)
        if max_similarity_delta > 1e-12 or max_ratio_delta > 1e-12:
            raise AssertionError(
                "online arithmetic mismatch: "
                f"similarity={max_similarity_delta} ratio={max_ratio_delta}"
            )

        mature_state = adapter.prepare_batch(
            hidden_states=hidden.view(1, -1),
            request_ids=request_ids,
            request_slots=request_slots,
            token_epochs=torch.tensor([profile.online_decode_window], device="cuda"),
            valid_rows=valid_rows,
            route_layer_order=(0, 1, 2, 3),
            phase="decode",
        )
        mature_action = adapter.route(
            hidden.view(1, -1),
            router=None,
            layer_id=2,
            batch_state=mature_state,
        )
        released_extra = [
            layer_id
            for layer_id, similarity in enumerate(released_similarity)
            if similarity >= adapter.extra_skip_threshold
        ]
        assert released_extra == [2]
        assert mature_action.mlp_run_mask.item() is False
        scale = float(mature_action.mlp_skip_scale.item())
        if not math.isclose(scale, released_ratio[2], rel_tol=1e-6, abs_tol=1e-6):
            raise AssertionError(
                f"online compensation mismatch: released={released_ratio[2]} vp={scale}"
            )
        released_layer = _released_layer(2)
        released_output = _released_forward_call(
            forward,
            released_layer,
            hidden.clone(),
            attention_scales=attention_scales,
            mlp_scales=mlp_scales,
            fixed_attention=fixed_attention,
            fixed_mlp=fixed_mlp,
            extra_mlp=released_extra,
            online_similarity=released_similarity,
            online_ratio=released_ratio,
            decode_count=profile.online_decode_window,
        )
        pre_mlp = released_layer.post_attention_layernorm.last_input
        assert pre_mlp is not None
        torch.testing.assert_close(
            released_output,
            pre_mlp * mature_action.mlp_skip_scale.view(1, 1, 1),
            rtol=0.0,
            atol=1e-6,
        )
        return {
            "window": profile.online_decode_window,
            "first_extra_skip_decode_forward": profile.online_decode_window + 1,
            "extra_skip_layers": released_extra,
            "max_similarity_delta": max_similarity_delta,
            "max_ratio_delta": max_ratio_delta,
            "compensation_scale": scale,
        }


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def _filtered_mlp_parity(
    *,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    dtype = torch.bfloat16
    hidden = torch.empty(
        (rows, hidden_size), device="cuda", dtype=dtype
    ).normal_(std=0.1)
    gate_up = torch.empty(
        (2 * intermediate_size, hidden_size), device="cuda", dtype=dtype
    ).normal_(std=0.01)
    down = torch.empty(
        (hidden_size, intermediate_size), device="cuda", dtype=dtype
    ).normal_(std=0.01)
    gate, up = F.linear(hidden, gate_up).chunk(2, dim=-1)
    dense = F.linear(F.silu(gate) * up, down)
    route_weights = torch.ones((rows, 1), device="cuda", dtype=dtype)

    all_active = torch.ones((rows, 1), device="cuda", dtype=torch.bool)
    filtered = _one_expert_mlp(
        hidden,
        gate_up.unsqueeze(0),
        down.unsqueeze(0),
        route_weights,
        all_active,
    )
    torch.cuda.synchronize()
    all_active_error = _error(filtered, dense)
    torch.testing.assert_close(filtered, dense, atol=atol, rtol=rtol)

    mixed_active = torch.arange(rows, device="cuda").remainder(2).eq(0).view(-1, 1)
    mixed = _one_expert_mlp(
        hidden,
        gate_up.unsqueeze(0),
        down.unsqueeze(0),
        route_weights,
        mixed_active,
    )
    torch.cuda.synchronize()
    expected_mixed = torch.where(mixed_active, dense, torch.zeros_like(dense))
    mixed_error = _error(mixed, expected_mixed)
    torch.testing.assert_close(mixed, expected_mixed, atol=atol, rtol=rtol)
    if mixed[~mixed_active.squeeze(-1)].count_nonzero().item() != 0:
        raise AssertionError("filtered MLP wrote an inactive row")
    return {
        "dtype": "bfloat16",
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "atol": atol,
        "rtol": rtol,
        "all_active": all_active_error,
        "mixed_active": mixed_error,
        "inactive_rows_exact_zero": True,
    }


class _DenseReferenceMLP:
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
    ) -> None:
        gate_up = torch.empty(
            (2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=dtype,
        ).normal_(std=0.01)
        down = torch.empty(
            (hidden_size, intermediate_size),
            device="cuda",
            dtype=dtype,
        ).normal_(std=0.01)
        self.gate_up_proj = SimpleNamespace(weight=gate_up)
        self.down_proj = SimpleNamespace(weight=down)
        self.calls = 0

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        gate, up = F.linear(
            hidden_states, self.gate_up_proj.weight
        ).chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, self.down_proj.weight)


def _dense_reference_executor_parity(
    *,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    dtype = torch.bfloat16
    hidden = torch.empty(
        (rows, hidden_size), device="cuda", dtype=dtype
    ).normal_(std=0.1)
    mlp = _DenseReferenceMLP(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
    )
    layer = SimpleNamespace(
        layer_id=2,
        self_attn=_VPAttention(),
        post_attention_layernorm=_VPPostAttentionNorm(),
        mlp=mlp,
    )
    attention_run = torch.ones((rows, 1), dtype=torch.bool, device="cuda")
    mlp_run = (
        torch.arange(rows, device="cuda").remainder(2).eq(0).view(-1, 1)
    )
    mlp_scale = torch.full(
        (rows, 1), 1.03125, dtype=dtype, device="cuda"
    )
    action = FullGraphActionBatch(
        adapter_name="adaskip",
        branch_weights=mlp_run.to(dtype=dtype),
        supported_actions=frozenset(
            (
                LogicalAction.RUN,
                LogicalAction.SKIP_ATTN,
                LogicalAction.SKIP_MLP,
            )
        ),
        execution_kind=SUBLAYER_EXECUTION,
        attention_run_mask=attention_run,
        mlp_run_mask=mlp_run,
        attention_skip_scale=torch.ones_like(mlp_scale),
        mlp_skip_scale=mlp_scale,
        static_attention_run=True,
        static_mlp_run=None,
        payload_semantics="released AdaSkip independent sublayer equations",
    )
    prepared = FullGraphPreparedLayerRoute(
        layer_id=2,
        hidden_states=hidden,
        residual=hidden,
        route_weights=action.branch_weights,
        run_mask=attention_run & mlp_run,
        parity_context=None,
        inline_kv_index=None,
        action_batch=action,
        attention_run_mask=attention_run,
        mlp_run_mask=mlp_run,
        attention_skip_scale=action.attention_skip_scale,
        mlp_skip_scale=action.mlp_skip_scale,
    )
    reference_action = replace(action, dense_reference_mlp=True)
    reference_prepared = replace(
        prepared,
        action_batch=reference_action,
    )
    forward_batch = _forward_batch(rows)
    positions = torch.arange(rows, dtype=torch.int64, device="cuda")
    reference, reference_residual = _execute_sublayer_route_full_graph(
        layer, positions, forward_batch, reference_prepared
    )
    reference_calls = mlp.calls
    optimized, optimized_residual = _execute_sublayer_route_full_graph(
        layer, positions, forward_batch, prepared
    )
    torch.cuda.synchronize()
    if reference_calls != 1 or mlp.calls != 1:
        raise AssertionError(
            "dense reference must execute one all-row MLP while the optimized "
            "path uses only the filtered expert"
        )
    torch.testing.assert_close(
        optimized_residual, reference_residual, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(optimized, reference, atol=atol, rtol=rtol)
    inactive = ~mlp_run.squeeze(-1)
    expected_skip = reference_residual[inactive] * (
        mlp_scale[inactive] - 1.0
    )
    torch.testing.assert_close(
        reference[inactive], expected_skip, atol=0.0, rtol=0.0
    )
    return {
        "dtype": "bfloat16",
        "rows": rows,
        "active_rows": int(mlp_run.sum().item()),
        "dense_reference_mlp_calls": reference_calls,
        "optimized_dense_mlp_calls": mlp.calls - reference_calls,
        "same_attention_residual": True,
        "optimized_vs_dense_reference": _error(optimized, reference),
        "atol": atol,
        "rtol": rtol,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-e2e-model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--atol", type=float, default=0.08)
    parser.add_argument("--rtol", type=float, default=0.02)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows < 2 or args.hidden_size <= 0 or args.intermediate_size <= 0:
        raise ValueError("invalid filtered-MLP shape")
    torch.manual_seed(20260716)
    torch.cuda.manual_seed_all(20260716)
    select_indices, released_forward = _released_symbols(args.official_e2e_model)
    report = {
        "status": "PASS",
        "evidence_class": "correctness-parity",
        "performance_claim_allowed": False,
        "official_revision": ADASKIP_OFFICIAL_SOURCE_REVISION,
        "official_e2e_model_sha256": OFFICIAL_E2E_MODEL_SHA256,
        "fixed_source_parity": _fixed_source_parity(
            select_indices, released_forward
        ),
        "online_source_parity": _online_source_parity(released_forward),
        "filtered_mlp_numerical_parity": _filtered_mlp_parity(
            rows=args.rows,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "dense_reference_executor_parity": _dense_reference_executor_parity(
            rows=args.rows,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            atol=args.atol,
            rtol=args.rtol,
        ),
        "claim_boundary": (
            "exact released-source action/arithmetic/residual equations plus "
            "bounded BF16 filtered-MLP and dense-reference executor parity; "
            "not full-model token, quality, latency, or throughput evidence"
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
