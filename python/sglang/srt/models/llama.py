# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/llama.py#L1
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_pp_indices,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.srt.utils import add_prefix, is_cuda, is_npu, is_xpu, make_layers
from sglang.srt.vpipe.coverage import (
    record_dense_body_pass,
)
from sglang.srt.vpipe.attention import (
    fd_attention_o_proj_full_graph,
    fd_attention_qkv_full_graph,
)
from sglang.srt.vpipe.attestation import (
    binary_cohort_attestation,
    full_graph_compact_evidence_specs,
)
from sglang.srt.vpipe.batch import (
    finalize_full_graph_batch,
    full_graph_routed_enabled,
    prepare_full_graph_batch,
)
from sglang.srt.vpipe.common import (
    full_graph_low_row_policy,
)
from sglang.srt.vpipe.common import (
    flexidepth_execution_mode,
    flexidepth_forward_phase,
    flexidepth_phase_enabled,
)
from sglang.srt.vpipe.common import (
    flexidepth_active_phases,
)
from sglang.srt.vpipe.config import (
    full_graph_device_route_digest_enabled,
    full_graph_device_route_tape_enabled,
    full_graph_fused_evidence_enabled,
    full_graph_layer_counters_enabled,
    full_graph_route_accounting_enabled,
    full_graph_scheduler_convergence_enabled,
)
from sglang.srt.vpipe.env import (
    FD_EXECUTION_FULL_GRAPH,
)
from sglang.srt.vpipe.executor import (
    fd_layer_forward_full_graph,
)
from sglang.srt.vpipe.routing import (
    full_graph_forced_route,
)
from sglang.srt.vpipe.validation import (
    validate_full_graph_model_configuration,
)
from sglang.srt.vpipe.common import (
    resolve_full_graph_skipper,
)
from sglang.srt.vpipe.skipper import (
    route_digest_uses_logical_request_ids,
)
from sglang.srt.vpipe.env import (
    RUN_PROJECT_EXECUTION,
)
from sglang.srt.vpipe.common import (
    regime_switch_config,
)
from sglang.srt.vpipe.regime import (
    PREFILL_BODY_DENSE,
    PREFILL_BODY_FD,
    prefill_regime_decision,
    regime_switch_zero_counters,
)
from sglang.utils import get_exception_traceback

_is_cuda = is_cuda()
_is_xpu = is_xpu()
_FD_PARITY_TRACE_ENABLED = bool(
    os.environ.get("SGLANG_FD_PARITY_TRACE_RID", "").strip()
)

logger = logging.getLogger(__name__)
_is_npu = is_npu()

if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkv_rmsnorm_rope


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
        use_dp_attention_reduce: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
            use_dp_attention_reduce=use_dp_attention_reduce,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        forward_batch=None,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x,
            skip_all_reduce=use_reduce_scatter,
        )
        return x


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.start_layer = start_layer
        tp_size = get_parallel().tp_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=rope_is_neox_style,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward_prepare_native(
        self, positions, hidden_states, forward_batch=None
    ):
        if forward_batch is None:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
        elif getattr(
            forward_batch, "fd_full_graph_attention_run_mask", None
        ) is not None:
            q, k, v = fd_attention_qkv_full_graph(
                self, hidden_states, forward_batch
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn.layer_id == self.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            is_neox_style=self.rotary_emb.is_neox_style,
        )
        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if (
            not _is_npu
            or not hasattr(self.rotary_emb, "get_cos_sin_with_position")
            or forward_batch.forward_mode.is_extend()
        ):
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if _FD_PARITY_TRACE_ENABLED:
            from sglang.srt.vpipe.routing import (
                fd_parity_trace_attention,
            )

            fd_parity_trace_attention(
                layer_id=int(self.attn.layer_id),
                q=q,
                k=k,
                v=v,
                positions=positions,
                forward_batch=forward_batch,
                cache_written=False,
            )
        attn_output = self.attn(q, k, v, forward_batch)
        if _FD_PARITY_TRACE_ENABLED:
            fd_parity_trace_attention(
                layer_id=int(self.attn.layer_id),
                q=q,
                k=k,
                v=v,
                positions=positions,
                forward_batch=forward_batch,
                cache_written=True,
            )
        if getattr(forward_batch, "fd_full_graph_attention_run_mask", None) is None:
            output, _ = self.o_proj(attn_output)
        else:
            output = fd_attention_o_proj_full_graph(
                self, attn_output, forward_batch
            )
        return output


@lru_cache(maxsize=1)
def _load_fd_weights(path: str):
    """Deserialize the FlexiDepth checkpoint ONCE for all routed layers.

    Every routed layer needs a different slice of the same ~400 MB file, and
    each LlamaDecoderLayer.__init__ used to torch.load it again -- 16 full
    deserializations per TP worker at boot. Cached on the path and released by
    _release_fd_weights_cache() as soon as the layers are built, so the payload
    does not outlive construction.
    """

    import torch as _t

    return _t.load(path, map_location="cpu")


def _release_fd_weights_cache() -> None:
    _load_fd_weights.cache_clear()


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.fd_execution_mode = flexidepth_execution_mode()
        self.vp_full_graph_routed = False
        self.vp_full_graph_attention_routed = False
        self.hidden_size = config.hidden_size
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", 10000)
            rope_scaling = rope_parameters
        else:
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        rope_is_neox_style = getattr(config, "rope_is_neox_style", True)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support llamafy/Qwen-Qwen2.5-7B-Instruct-llamafied with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=attention_bias,
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # FlexiDepth-in-vPipe (GOAL): attach the trained router + router_proj (EXTRACTED weights,
        # version-agnostic) to routing layers 16-31 on stock Llama-3. Mode-guarded by SGLANG_FD_WEIGHTS
        # (a .pt path); loaded here so it lives in the scheduler subprocess with the model. KV-complete
        # (self_attn still writes K/V); reproduces FlexiDepth's quality in the SGLang serving stack.
        self.fd_router = None
        self.fd_proj = None
        _fdw = os.environ.get("SGLANG_FD_WEIGHTS", "")
        if _fdw and 16 <= layer_id <= 31:
            from sglang.srt.vpipe.routing import (
                FDProj,
                FDRouter,
            )

            _sd = _load_fd_weights(_fdw)
            _fd_reduction = getattr(config, "router_reduction_factor", 16)
            self.fd_router = FDRouter(
                config.hidden_size,
                reduction=_fd_reduction,
                eps=config.rms_norm_eps,
            )
            self.fd_proj = FDProj(
                config.hidden_size,
                config.intermediate_size,
                reduction=getattr(config, "proj_reduction_factor", _fd_reduction),
                act=config.hidden_act,
            )
            self.fd_router.load_state_dict({
                "router_enc.weight": _sd[f"model.layers.{layer_id}.router.router_enc.weight"],
                "router_norm.weight": _sd[f"model.layers.{layer_id}.router.router_norm.weight"],
                "router_dec.weight": _sd[f"model.layers.{layer_id}.router.router_dec.weight"],
                "router_head.weight": _sd[f"model.layers.{layer_id}.router.router_head.weight"],
            })
            self.fd_proj.load_state_dict({
                "gate_proj.weight": _sd[f"model.layers.{layer_id}.router_proj.gate_proj.weight"],
                "down_proj.weight": _sd[f"model.layers.{layer_id}.router_proj.down_proj.weight"],
                "up_proj.weight": _sd[f"model.layers.{layer_id}.router_proj.up_proj.weight"],
            })
            # Inference-only: the router/projector are frozen published weights,
            # never trained here. Left as leaves that require grad they make the
            # capture path's in-place buffer copies illegal ("a leaf Variable
            # that requires grad is being used in an in-place operation"), which
            # fails CUDA-graph capture outright.
            self.fd_router.requires_grad_(False).eval()
            self.fd_proj.requires_grad_(False).eval()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and self.vp_full_graph_routed
            and full_graph_routed_enabled(forward_batch)
        ):
            return fd_layer_forward_full_graph(
                self,
                positions,
                hidden_states,
                forward_batch,
                residual,
                self.fd_router,
                self.fd_proj,
            )
        if self.fd_router is not None and flexidepth_phase_enabled(forward_batch):
            # direct_eager: the quality-attribution reference. It answers whether
            # a quality result is the checkpoint's or vPipe's execution of it, so
            # it is a gate instrument, never a performance path.
            #
            # Only the plain body is carried. The pre-refactor tree also had
            # fd_layer_forward_vp{,_coalesced} (gated on SGLANG_FD_VP_PROJECT,
            # the removed V1 path) and fd_layer_forward_eager_compact (gated on
            # the inert Candidate-A flag, which D-2026-07-25 showed never
            # executed in any measured cell). Neither is needed to attribute
            # quality, and both are omitted.
            from sglang.srt.vpipe.eager import fd_layer_forward_eager

            return fd_layer_forward_eager(
                self,
                positions,
                hidden_states,
                forward_batch,
                residual,
                self.fd_router,
                self.fd_proj,
            )
        # KV-complete dense fall-through — also the W1 regime-forced-dense
        # prefill body. For routed layers 16-31 as well, self_attn below writes
        # THIS layer's own K/V for every token, so any later attention that
        # reads this layer's K/V finds it. Forcing a prefill pass dense (the
        # grouped/eager FD gate returning False) therefore preserves KV-
        # completeness by construction: the full base-Llama layer runs and no
        # K/V is ever copied across layers.
        # (c3) site-B positive witness (F2/F18): on a coverage-stamped decode
        # pass, the FIRST routed layer reaching this dense fall-through counts
        # the pass exactly once (dedup via vp_fd_coverage_counted) — positive
        # proof the dense body EXECUTED, at the body, not the decision site.
        # Unstamped passes (production, W1 band-dense, prefill) return
        # immediately inside the helper.
        if self.fd_router is not None:
            record_dense_body_pass(forward_batch)
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _vp_regime_prefill_mixed_running_bs(forward_batch: ForwardBatch) -> int:
    """Best-effort count of appended running-decode rows in a MIXED pass.

    ``ScheduleBatch.mix_with_running`` appends exactly one extend token
    (length 1) per running decode request at the tail of ``extend_lens``, so the
    trailing run of length-1 entries in ``extend_seq_lens_cpu`` bounds
    ``running_bs`` from above. It over-counts only when the final *prefill* chunk
    is itself length 1, which biases the prefill-token estimate slightly
    downward — i.e. toward the dense body, the safe (never-loses) direction.

    RISK / GPU-verified refinement: the exact value is
    ``len(ScheduleBatch.mix_running_indices)``; carrying that onto ForwardBatch
    (or keying the threshold on the parent request's total prompt) is the
    correct integrated-endpoint refinement and must be validated on the A100.
    """

    lengths = forward_batch.extend_seq_lens_cpu
    if lengths is None:
        return 0
    running_bs = 0
    for value in reversed(lengths):
        if int(value) != 1:
            break
        running_bs += 1
    return running_bs


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: LlamaDecoderLayer(
                config=config,
                quant_config=quant_config,
                layer_id=idx,
                start_layer=pp_start_layer,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix="model.layers",
        )
        _release_fd_weights_cache()
        self.fd_execution_mode = flexidepth_execution_mode()
        full_graph_skipper = resolve_full_graph_skipper()
        loaded_fd_layers = [
            layer
            for layer in self.layers
            if getattr(layer, "fd_router", None) is not None
        ]
        loaded_fd_layer_ids = tuple(int(layer.layer_id) for layer in loaded_fd_layers)
        # Checkpoint identity, so a skipper carrying calibration data can verify
        # that data was measured on THIS model. Layer count alone is not
        # identity: every Llama-3-8B derivative has 32 layers.
        served_identity = {
            "revision": str(getattr(config, "_commit_hash", "") or ""),
            "model_id": str(getattr(config, "_name_or_path", "") or ""),
        }
        routed_layer_ids = full_graph_skipper.routed_layer_ids(
            num_hidden_layers=config.num_hidden_layers,
            flexidepth_layer_ids=loaded_fd_layer_ids,
            model_identity=served_identity,
        )
        attention_routed_layer_ids = (
            full_graph_skipper.attention_routed_layer_ids(
                num_hidden_layers=config.num_hidden_layers,
                flexidepth_layer_ids=loaded_fd_layer_ids,
                model_identity=served_identity,
            )
        )
        routed_layer_id_set = frozenset(routed_layer_ids)
        attention_routed_layer_id_set = frozenset(attention_routed_layer_ids)
        if not attention_routed_layer_id_set <= routed_layer_id_set:
            raise ValueError(
                "full-graph attention-routed layers must be skipper-routed"
            )
        for layer in self.layers:
            if hasattr(layer, "vp_full_graph_routed"):
                layer.vp_full_graph_routed = (
                    self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                    and int(layer.layer_id) in routed_layer_id_set
                )
                layer.vp_full_graph_attention_routed = (
                    self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                    and int(layer.layer_id) in attention_routed_layer_id_set
                )
        self._vp_full_graph_skipper_name = full_graph_skipper.name
        self._vp_full_graph_loaded_flexidepth_layer_ids = loaded_fd_layer_ids
        self._vp_full_graph_attention_route_layer_order = (
            attention_routed_layer_ids
        )
        self._fd_full_graph_route_layer_order = routed_layer_ids
        has_routed_layers = bool(routed_layer_ids)
        self.register_buffer(
            "_fd_full_graph_route_layer_ids",
            (
                torch.tensor(routed_layer_ids, dtype=torch.int64)
                if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and has_routed_layers
                and full_graph_device_route_tape_enabled()
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "_fd_full_graph_compact_evidence_specs",
            (
                torch.tensor(
                    full_graph_compact_evidence_specs(routed_layer_ids),
                    dtype=torch.float32,
                )
                if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and has_routed_layers
                and full_graph_skipper.execution_kind == RUN_PROJECT_EXECUTION
                and (
                    full_graph_fused_evidence_enabled()
                    or (
                        full_graph_device_route_tape_enabled()
                        and full_graph_route_accounting_enabled()
                    )
                )
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "_fd_full_graph_route_counters",
            (
                torch.zeros(6, dtype=torch.int64)
                if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and has_routed_layers
                and full_graph_route_accounting_enabled()
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "_fd_full_graph_phase_route_counters",
            (
                torch.zeros((2, 3), dtype=torch.int64)
                if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and has_routed_layers
                and full_graph_route_accounting_enabled()
                and flexidepth_active_phases() == {"decode", "prefill"}
                and not full_graph_fused_evidence_enabled()
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "_fd_full_graph_layer_route_counters",
            (
                torch.zeros((len(routed_layer_ids), 3), dtype=torch.int64)
                if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and has_routed_layers
                and full_graph_layer_counters_enabled()
                else None
            ),
            persistent=False,
        )
        low_row_policy, _ = full_graph_low_row_policy()
        self.register_buffer(
            "_fd_full_graph_low_row_counters",
            (
                torch.zeros(3, dtype=torch.int64)
                if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and has_routed_layers
                and full_graph_route_accounting_enabled()
                and low_row_policy != "off"
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "_fd_full_graph_route_digest_counters",
            (
                torch.zeros(6, dtype=torch.int64)
                if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and has_routed_layers
                and full_graph_device_route_digest_enabled()
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "_fd_full_graph_inline_kv_readiness_counters",
            (
                torch.zeros(7, dtype=torch.int64)
                if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and has_routed_layers
                and full_graph_scheduler_convergence_enabled()
                else None
            ),
            persistent=False,
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)
        self.layers_to_capture = []
        # [W1] Prefill regime-switch per-pass decision counters (grouped-FD vs
        # dense). Plain ints, always present so reads need no defensive getattr;
        # surfaced into the runtime-attestation regime_switch.counters block and
        # zeroed by vp_reset_runtime_counters between evaluation cells.
        self._vp_regime_prefill_fd_passes = 0
        self._vp_regime_prefill_dense_passes = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]], PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            # FIXME(@ying): reduce the number of proxy tensors by not fusing layer norms
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]
            deferred_norm = None

        # [W1] Unified regime switch — prefill leg (I5). Compute the per-pass
        # dense/FD prefill decision ONCE, before any FlexiDepth phase gate reads
        # it, and stamp it on forward_batch. When the switch is off (config
        # None) or its prefill leg is disabled, the batch is left unstamped and
        # flexidepth_phase_enabled never reads the stamp -> byte-identical FD.
        _regime_cfg = regime_switch_config()
        if _regime_cfg is not None and _regime_cfg.prefill.enabled:
            if flexidepth_forward_phase(forward_batch) == "prefill":
                _regime_is_mixed = forward_batch.forward_mode.is_mixed()
                if _regime_is_mixed and not _regime_cfg.prefill.include_mixed:
                    # MIXED pass with mixed switching disabled: leave FlexiDepth
                    # enabled (unchanged). The appended running-decode rows are
                    # governed by the decode leg (I6), not the prefill leg.
                    forward_batch.vp_fd_prefill_dense = False
                else:
                    _regime_running_bs = (
                        _vp_regime_prefill_mixed_running_bs(forward_batch)
                        if _regime_is_mixed
                        else 0
                    )
                    _regime_decision = prefill_regime_decision(
                        forward_batch.extend_num_tokens,
                        forward_batch.batch_size,
                        _regime_is_mixed,
                        _regime_running_bs,
                        _regime_cfg,
                    )
                    # A sub-threshold MIXED pass forces its WHOLE batch — the
                    # appended running-decode rows included — onto the dense
                    # body. This is an intentional consequence of stamping at
                    # pass granularity (documented for the integrated endpoint).
                    forward_batch.vp_fd_prefill_dense = (
                        _regime_decision == PREFILL_BODY_DENSE
                    )
                # Per-pass decision counter (dense vs grouped-FD), surfaced into
                # regime_switch.counters.prefill (stripped from deploy identity).
                if forward_batch.vp_fd_prefill_dense:
                    self._vp_regime_prefill_dense_passes += 1
                else:
                    self._vp_regime_prefill_fd_passes += 1

        if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH:
            prepare_full_graph_batch(
                forward_batch,
                hidden_states,
                positions=positions,
                route_layer_ids=self._fd_full_graph_route_layer_ids,
                route_layer_order=self._fd_full_graph_route_layer_order,
                compact_evidence_specs=self._fd_full_graph_compact_evidence_specs,
            )

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )

        if self.fd_execution_mode == FD_EXECUTION_FULL_GRAPH:
            finalize_full_graph_batch(
                forward_batch,
                self._fd_full_graph_route_counters,
                self._fd_full_graph_layer_route_counters,
                self._fd_full_graph_route_digest_counters,
                self._fd_full_graph_inline_kv_readiness_counters,
                self._fd_full_graph_phase_route_counters,
                self._fd_full_graph_low_row_counters,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn

            if hasattr(layer_self_attn.attn, "k_scale"):
                layer_self_attn.attn.k_scale = scaling_factor
                layer_self_attn.attn.v_scale = scaling_factor
            else:
                raise RuntimeError(
                    "Self attention has no KV cache scaling " "factor attribute!"
                )

    def get_input_embeddings(self) -> nn.Embedding:
        """Get input embeddings from the model."""
        return self.embed_tokens


class LlamaForCausalLM(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        ".q_proj": (".qkv_proj", 0),
        ".k_proj": (".qkv_proj", 1),
        ".v_proj": (".qkv_proj", 2),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = self._init_model(config, quant_config, add_prefix("model", prefix))
        loaded_fd_layers = [
            int(layer.layer_id)
            for layer in self.model.layers
            if getattr(layer, "fd_router", None) is not None
        ]
        routed_layers = list(self.model._fd_full_graph_route_layer_order)
        from sglang.srt.server_args import get_global_server_args

        validate_full_graph_model_configuration(
            loaded_layers=routed_layers,
            loaded_flexidepth_layers=loaded_fd_layers,
            tp_size=get_parallel().tp_size,
            pp_size=self.pp_group.world_size,
            quant_config=quant_config,
            cuda_graph_enabled=not get_global_server_args().disable_cuda_graph,
        )
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_flags().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        self.capture_aux_hidden_states = False

    def _init_model(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return LlamaModel(config, quant_config=quant_config, prefix=prefix)

    def vp_runtime_attestation(self) -> dict:
        loaded_flexidepth_layers = [
            int(layer.layer_id)
            for layer in self.model.layers
            if getattr(layer, "fd_router", None) is not None
        ]
        routed_layers = list(self.model._fd_full_graph_route_layer_order)
        attention_routed_layers = list(
            self.model._vp_full_graph_attention_route_layer_order
        )
        full_graph_adapter = resolve_full_graph_skipper()
        # Derive the attested execution from the LIVE dispatch, never from
        # legacy environment variables. The chain here previously reported
        # v4_split_stage_adapter / v2_scheduler_adapter / scheduler_stage_route
        # / inline_vp_project purely because an env var was set -- all four
        # implementations are deleted, so an experiment could be validated or
        # compared under a treatment that never ran. The retained
        # vast_a100_ladder.sh harness sets SGLANG_FD_VP_PROJECT=1, which made a
        # direct_eager run attest inline_vp_project.
        if not routed_layers and not loaded_flexidepth_layers:
            execution = "disabled"
        elif flexidepth_execution_mode() == FD_EXECUTION_FULL_GRAPH:
            execution = f"full_graph_{full_graph_adapter.name}"
        else:
            execution = "direct_eager"
        flexidepth_state = {
            "loaded": bool(routed_layers),
            "loaded_layers": routed_layers,
            "routed_layers": routed_layers,
            "attention_routed_layers": attention_routed_layers,
            "flexidepth_weights_loaded": bool(loaded_flexidepth_layers),
            "flexidepth_weight_layers": loaded_flexidepth_layers,
            "execution": execution,
            "active_phases": sorted(flexidepth_active_phases()),
        }
        if execution.startswith("full_graph_"):
            flexidepth_state["skipper_adapter"] = full_graph_adapter.attestation()
            # Realized binary_cohort dispatch evidence. Lives here (the
            # model-level surface every arm publishes) rather than only on
            # the decode conditional-graph backend: prefill-only V-pre arms
            # never instantiate that backend, so the counters were invisible
            # exactly where the executor runs.
            flexidepth_state["binary_cohort"] = binary_cohort_attestation()
            flexidepth_state["route_accounting_enabled"] = (
                full_graph_route_accounting_enabled()
            )
            forced_route = full_graph_forced_route()
            if forced_route is None:
                flexidepth_state["forced_route"] = "off"
            else:
                flexidepth_state["forced_route"] = (
                    "all_run" if forced_route else "all_project"
                )
        counters = getattr(self.model, "_fd_full_graph_route_counters", None)
        if counters is not None:
            values = [int(value) for value in counters.detach().cpu().tolist()]
            layer_rows, run_rows, project_rows = values[:3]
            if full_graph_adapter.execution_kind == RUN_PROJECT_EXECUTION:
                flexidepth_state["full_graph_routes"] = {
                    "layer_rows": layer_rows,
                    "run_rows": run_rows,
                    "project_rows": project_rows,
                    "skip_ratio": (
                        project_rows / layer_rows if layer_rows else 0.0
                    ),
                }
            else:
                flexidepth_state["full_graph_routes"] = {
                    "sublayer_rows": layer_rows,
                    "run_sublayer_rows": run_rows,
                    "skip_sublayer_rows": project_rows,
                    "skip_ratio": (
                        project_rows / layer_rows if layer_rows else 0.0
                    ),
                }
            if (
                len(values) >= 6
                and full_graph_adapter.execution_kind == RUN_PROJECT_EXECUTION
            ):
                compact_rows, run_overflow, project_overflow = values[3:6]
                hidden_element_bytes = self.model.layers[
                    routed_layers[0]
                ].mlp.gate_up_proj.weight.element_size()
                hidden_row_bytes = self.config.hidden_size * hidden_element_bytes
                flexidepth_state["full_graph_routes"].update(
                    {
                        "compact_layer_rows": compact_rows,
                        "compact_coverage": (
                            compact_rows / layer_rows if layer_rows else 0.0
                        ),
                        "compact_run_overflow_rows": run_overflow,
                        "compact_project_overflow_rows": project_overflow,
                        "compact_logical_gather_payload_bytes": (
                            compact_rows
                            * (2 * hidden_row_bytes + 2 * hidden_element_bytes)
                        ),
                        "compact_logical_scatter_payload_bytes": (
                            compact_rows * 2 * hidden_row_bytes
                        ),
                    }
                )
        phase_counters = getattr(
            self.model, "_fd_full_graph_phase_route_counters", None
        )
        if phase_counters is not None:
            phase_values = phase_counters.detach().cpu().tolist()
            by_phase = {}
            for phase, (layer_rows, run_rows, project_rows) in zip(
                ("decode", "prefill"), phase_values
            ):
                if full_graph_adapter.execution_kind == RUN_PROJECT_EXECUTION:
                    record = {
                        "layer_rows": int(layer_rows),
                        "run_rows": int(run_rows),
                        "project_rows": int(project_rows),
                    }
                else:
                    record = {
                        "sublayer_rows": int(layer_rows),
                        "run_sublayer_rows": int(run_rows),
                        "skip_sublayer_rows": int(project_rows),
                    }
                record["skip_ratio"] = (
                    project_rows / layer_rows if layer_rows else 0.0
                )
                by_phase[phase] = record
            flexidepth_state["full_graph_routes"]["by_phase"] = by_phase
        layer_counters = getattr(
            self.model, "_fd_full_graph_layer_route_counters", None
        )
        if layer_counters is not None:
            layer_values = layer_counters.detach().cpu().tolist()
            per_layer = []
            for layer_id, (layer_rows, run_rows, project_rows) in zip(
                routed_layers, layer_values
            ):
                if full_graph_adapter.execution_kind == RUN_PROJECT_EXECUTION:
                    record = {
                        "layer_id": layer_id,
                        "layer_rows": int(layer_rows),
                        "run_rows": int(run_rows),
                        "project_rows": int(project_rows),
                    }
                else:
                    record = {
                        "layer_id": layer_id,
                        "sublayer_rows": int(layer_rows),
                        "run_sublayer_rows": int(run_rows),
                        "skip_sublayer_rows": int(project_rows),
                    }
                record["run_ratio"] = (
                    run_rows / layer_rows if layer_rows else 0.0
                )
                per_layer.append(record)
            flexidepth_state["full_graph_routes"]["per_layer"] = per_layer
        low_row_counters = getattr(
            self.model, "_fd_full_graph_low_row_counters", None
        )
        if low_row_counters is not None:
            dispatches, graph_rows, logical_layer_rows = (
                int(value)
                for value in low_row_counters.detach().cpu().tolist()
            )
            flexidepth_state["full_graph_routes"]["low_row_execution"] = {
                "dispatches": dispatches,
                "graph_rows": graph_rows,
                "logical_layer_rows": logical_layer_rows,
                "body": "full_dual",
            }
        digest_counters = getattr(
            self.model, "_fd_full_graph_route_digest_counters", None
        )
        if digest_counters is not None:
            (
                dispatches,
                action_rows,
                run_rows,
                digest_sum,
                digest_ordered,
                metadata_rows,
            ) = [int(value) for value in digest_counters.detach().cpu().tolist()]
            mask = (1 << 64) - 1
            logical_digest = route_digest_uses_logical_request_ids(
                full_graph_adapter
            )
            sublayer_digest = (
                full_graph_adapter.execution_kind != RUN_PROJECT_EXECUTION
            )
            flexidepth_state["full_graph_routes"]["device_route_tape"] = {
                "dispatches": dispatches,
                "action_rows": action_rows,
                "run_rows": run_rows,
                "metadata_rows": metadata_rows,
                "digest_sum_u64": format(digest_sum & mask, "016x"),
                "digest_ordered_u64": format(digest_ordered & mask, "016x"),
                "action_digest": (
                    format(digest_sum & mask, "016x")
                    + format(digest_ordered & mask, "016x")
                ),
                "digest_input": (
                    "stable_request_hash_token_epoch_layer_component_action"
                    if logical_digest and sublayer_digest
                    else "stable_request_hash_token_epoch_layer_action"
                    if logical_digest
                    else "dispatch_order_layer_row_request_slot_token_epoch_"
                    "cache_position_action"
                ),
                "batching_invariant": logical_digest,
                "digest_algorithm": (
                    "batching_invariant_dual_sublayer_int64_weighted_"
                    "fingerprint"
                    if logical_digest and sublayer_digest
                    else "batching_invariant_logical_action_int64_weighted_"
                    "fingerprint"
                    if logical_digest
                    else "ordered_dual_int64_weighted_fingerprint"
                ),
            }
            if full_graph_adapter.execution_kind == RUN_PROJECT_EXECUTION:
                flexidepth_state["full_graph_routes"]["device_route_tape"][
                    "project_rows"
                ] = action_rows - run_rows
            else:
                flexidepth_state["full_graph_routes"]["device_route_tape"][
                    "skip_sublayer_rows"
                ] = action_rows - run_rows
        readiness_counters = getattr(
            self.model, "_fd_full_graph_inline_kv_readiness_counters", None
        )
        if readiness_counters is not None:
            (
                dispatches,
                expected_cells,
                ready_cells,
                missing_cells,
                metadata_rows,
                digest_sum,
                digest_ordered,
            ) = [
                int(value)
                for value in readiness_counters.detach().cpu().tolist()
            ]
            mask = (1 << 64) - 1
            flexidepth_state["full_graph_routes"]["inline_kv_readiness"] = {
                "dispatches": dispatches,
                "expected_cells": expected_cells,
                "ready_cells": ready_cells,
                "missing_cells": missing_cells,
                "metadata_rows": metadata_rows,
                "key_digest": (
                    format(digest_sum & mask, "016x")
                    + format(digest_ordered & mask, "016x")
                ),
                "key_fields": (
                    "request_slot_token_epoch_layer_cache_position"
                ),
                "completion": "inline_after_own_layer_attention",
            }
        return {
            "model_family": "llama",
            "flexidepth": flexidepth_state,
        }

    def vp_regime_switch_counters(self) -> dict:
        """Per-body regime-switch pass counters for the runtime attestation.

        Feeds regime_switch.counters (stripped from the deployment identity, so
        these live counts never move the pinned identity SHA). Prefill counts
        are real (I5); decode counts stay zero until the decode leg (I6) wires
        them.
        """

        counters = regime_switch_zero_counters()
        counters["prefill"][PREFILL_BODY_DENSE] = int(
            self.model._vp_regime_prefill_dense_passes
        )
        counters["prefill"][PREFILL_BODY_FD] = int(
            self.model._vp_regime_prefill_fd_passes
        )
        return counters

    def vp_reset_runtime_counters(self) -> None:
        resolve_full_graph_skipper().reset_runtime_state()
        counters = getattr(self.model, "_fd_full_graph_route_counters", None)
        if counters is not None:
            counters.zero_()
        phase_counters = getattr(
            self.model, "_fd_full_graph_phase_route_counters", None
        )
        if phase_counters is not None:
            phase_counters.zero_()
        layer_counters = getattr(
            self.model, "_fd_full_graph_layer_route_counters", None
        )
        if layer_counters is not None:
            layer_counters.zero_()
        low_row_counters = getattr(
            self.model, "_fd_full_graph_low_row_counters", None
        )
        if low_row_counters is not None:
            low_row_counters.zero_()
        digest_counters = getattr(
            self.model, "_fd_full_graph_route_digest_counters", None
        )
        if digest_counters is not None:
            digest_counters.zero_()
        readiness_counters = getattr(
            self.model, "_fd_full_graph_inline_kv_readiness_counters", None
        )
        if readiness_counters is not None:
            readiness_counters.zero_()
        # [W1] Reset the prefill regime-switch decision counters (always present).
        self.model._vp_regime_prefill_fd_passes = 0
        self.model._vp_regime_prefill_dense_passes = 0




    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
        # decoder layer
        for i in range(start, end):
            layer = self.model.layers[i]
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_module_name_from_weight_name(self, name):
        for param_name, weight_name, shard_id, num_shard in self.stacked_params_mapping:
            if weight_name in name:
                return (
                    name.replace(weight_name, param_name)[: -len(".weight")],
                    num_shard,
                )
        return name[: -len(".weight")], 1

    def get_num_params(self):
        params_dict = dict(self.named_parameters())
        return len(params_dict)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if name.endswith(".activation_scale"):
                name = name.replace(".activation_scale", ".input_scale")
            if name.endswith(".weight_scale_inv"):
                name = name.replace(".weight_scale_inv", ".weight_scale")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Handle FP8 kv-scale remapping
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading kv_scale from ckpts towards new design.
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        try:
            if name == "lm_head.weight" and self.config.tie_word_embeddings:
                logger.info(
                    "word embedding is tied for this model, return embed_tokens.weight as lm_head.weight."
                )
                return (
                    self.model.embed_tokens.weight.cpu()
                    .to(torch.float32)
                    .numpy()
                    .tolist()[:truncate_size]
                )

            mapped_name = name
            mapped_shard_id = None
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name in name:
                    mapped_name = name.replace(weight_name, param_name)
                    mapped_shard_id = shard_id
                    break
            params_dict = dict(self.named_parameters())
            param = params_dict[mapped_name]
            if mapped_shard_id is not None:
                if mapped_shard_id in ["q", "k", "v"]:
                    num_heads = self.config.num_attention_heads // tp_size
                    num_kv_heads = self.config.num_key_value_heads // tp_size
                    head_dim = (
                        self.config.hidden_size // self.config.num_attention_heads
                    )
                    if mapped_shard_id == "q":
                        offset = 0
                        size = num_heads * head_dim
                    elif mapped_shard_id == "k":
                        offset = num_heads * head_dim
                        size = num_kv_heads * head_dim
                    elif mapped_shard_id == "v":
                        offset = (num_heads + num_kv_heads) * head_dim
                        size = num_kv_heads * head_dim
                    weight = param.data.narrow(0, offset, size)
                elif mapped_shard_id in [0, 1]:
                    intermediate_size = self.config.intermediate_size
                    slice_size = intermediate_size // tp_size
                    if mapped_shard_id == 0:  # gate_proj
                        offset = 0
                        size = slice_size
                    elif mapped_shard_id == 1:  # up_proj
                        offset = slice_size
                        size = slice_size

                    weight = param.data.narrow(0, offset, size)
                else:
                    weight = param.data
            else:
                weight = param.data
            if tp_size > 1 and ("o_proj" in name or "down_proj" in name):
                gathered_weights = [torch.zeros_like(weight) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered_weights, weight)
                weight = torch.cat(gathered_weights, dim=1)
            return weight.cpu().to(torch.float32).numpy().tolist()[:truncate_size]

        except Exception:
            logger.error(
                f"Error getting weights by name {name} in LlamaForCausalLM: {get_exception_traceback()}"
            )
            return None

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        if _is_xpu:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        # NOTE: If draft hidden size != target hidden size, the embed weight cannot be shared for EAGLE3
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        if _is_xpu:
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            # we plus 1 here because in sglang, for the ith layer, it takes the output
            # of the (i-1)th layer as aux hidden state
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if not self.pp_group.is_last_rank:
            return

        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )

        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = [val + 1 for val in layer_ids]


class Phi3ForCausalLM(LlamaForCausalLM):
    pass


class InternLM3ForCausalLM(LlamaForCausalLM):
    pass


class IQuestCoderForCausalLM(LlamaForCausalLM):
    pass


EntryClass = [
    LlamaForCausalLM,
    Phi3ForCausalLM,
    InternLM3ForCausalLM,
    IQuestCoderForCausalLM,
]
