"""The model-side vpipe seam, shared across families.

Extracted MECHANICALLY from models/llama.py (D-307 Lane S commit 1;
transform script archived in vPipe-doc). Families call these helpers at
the exact sites llama.py did; behavior is byte-identical, proven by the
full box gate chain incl. route+output equality vs frozen. Layer-level
entry points take the decoder layer; model-level take the inner model;
lm-level take the ForCausalLM wrapper.
"""

import os
from functools import lru_cache

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.vpipe.env import SERVED_MODEL_REVISION_ENV
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
    full_graph_commit_overlap_enabled,
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


def init_fd_layer(layer, config, layer_id, routed_lo, routed_hi, qk_head_norms):
    """Decoder-layer seam: execution flags + routed FDRouter/FDProj attach.

    routed_lo..routed_hi (inclusive) is the family's trained routed-layer
    range (Llama-3-8B 16-31, Qwen3-8B 18-35, ...). qk_head_norms declares the
    per-head (q_norm, k_norm) the family applies between QKV split and RoPE
    (None for Llama) — the deferred PROJECT K/V repair replays it so repaired
    K matches the layer's own attention path exactly. Always set on the
    attention module so readers need no defensive access.
    """
    layer.fd_execution_mode = flexidepth_execution_mode()
    layer.vp_full_graph_routed = False
    layer.vp_full_graph_attention_routed = False
    layer.self_attn.fd_qk_head_norms = qk_head_norms
    # FlexiDepth-in-vPipe (GOAL): attach the trained router + router_proj (EXTRACTED weights,
    # version-agnostic) to routing layers 16-31 on stock Llama-3. Mode-guarded by SGLANG_FD_WEIGHTS
    # (a .pt path); loaded here so it lives in the scheduler subprocess with the model. KV-complete
    # (self_attn still writes K/V); reproduces FlexiDepth's quality in the SGLang serving stack.
    layer.fd_router = None
    layer.fd_proj = None
    _fdw = os.environ.get("SGLANG_FD_WEIGHTS", "")
    if _fdw and routed_lo <= layer_id <= routed_hi:
        from sglang.srt.vpipe.routing import (
            FDProj,
            FDRouter,
        )

        _sd = _load_fd_weights(_fdw)
        _fd_reduction = getattr(config, "router_reduction_factor", 16)
        layer.fd_router = FDRouter(
            config.hidden_size,
            reduction=_fd_reduction,
            eps=config.rms_norm_eps,
        )
        layer.fd_proj = FDProj(
            config.hidden_size,
            config.intermediate_size,
            reduction=getattr(config, "proj_reduction_factor", _fd_reduction),
            act=config.hidden_act,
        )
        layer.fd_router.load_state_dict({
            "router_enc.weight": _sd[f"model.layers.{layer_id}.router.router_enc.weight"],
            "router_norm.weight": _sd[f"model.layers.{layer_id}.router.router_norm.weight"],
            "router_dec.weight": _sd[f"model.layers.{layer_id}.router.router_dec.weight"],
            "router_head.weight": _sd[f"model.layers.{layer_id}.router.router_head.weight"],
        })
        layer.fd_proj.load_state_dict({
            "gate_proj.weight": _sd[f"model.layers.{layer_id}.router_proj.gate_proj.weight"],
            "down_proj.weight": _sd[f"model.layers.{layer_id}.router_proj.down_proj.weight"],
            "up_proj.weight": _sd[f"model.layers.{layer_id}.router_proj.up_proj.weight"],
        })
        # Inference-only: the router/projector are frozen published weights,
        # never trained here. Left as leaves that require grad they make the
        # capture path's in-place buffer copies illegal ("a leaf Variable
        # that requires grad is being used in an in-place operation"), which
        # fails CUDA-graph capture outright.
        layer.fd_router.requires_grad_(False).eval()
        layer.fd_proj.requires_grad_(False).eval()


def maybe_fd_layer_forward(layer, positions, hidden_states, forward_batch, residual):
    """Decoder-forward seam: FD dispatch; None = fall through to the stock body."""
    if (
        layer.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
        and layer.vp_full_graph_routed
        and full_graph_routed_enabled(forward_batch)
    ):
        return fd_layer_forward_full_graph(
            layer,
            positions,
            hidden_states,
            forward_batch,
            residual,
            layer.fd_router,
            layer.fd_proj,
        )
    if layer.fd_router is not None and flexidepth_phase_enabled(forward_batch):
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
            layer,
            positions,
            hidden_states,
            forward_batch,
            residual,
            layer.fd_router,
            layer.fd_proj,
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
    if layer.fd_router is not None:
        record_dense_body_pass(forward_batch)
    return None


def attach_vp_model(model, config):
    """Model-level seam: attach, revision guard, routed flags, evidence buffers."""
    _release_fd_weights_cache()
    model.fd_execution_mode = flexidepth_execution_mode()
    full_graph_skipper = resolve_full_graph_skipper()
    loaded_fd_layers = [
        layer
        for layer in model.layers
        if getattr(layer, "fd_router", None) is not None
    ]
    loaded_fd_layer_ids = tuple(int(layer.layer_id) for layer in loaded_fd_layers)
    # Checkpoint identity, so a skipper carrying calibration data can verify
    # that data was measured on THIS model. Layer count alone is not
    # identity: every Llama-3-8B derivative has 32 layers.
    # config._commit_hash is NOT the weights' identity. This runs during
    # __init__, BEFORE weights are loaded, and DefaultModelLoader resolves
    # weights from model_config.revision -- the requested branch/tag/None --
    # not from the commit the config happened to resolve to. A mutable Hub
    # ref can therefore yield commit A for the config and commit B for the
    # weights, and loaders like DummyModelLoader do not derive weights from
    # that snapshot at all.
    #
    # So the OPERATOR DECLARATION is authoritative, and the config commit is
    # only corroboration: if both are present they must AGREE, because a
    # disagreement means the deployment is not what it claims. A skipper
    # carrying checkpoint-specific calibration has to be told, explicitly,
    # which checkpoint is being served.
    _hub_commit = str(getattr(config, "_commit_hash", "") or "").strip()
    _declared = str(os.environ.get(SERVED_MODEL_REVISION_ENV, "") or "").strip()
    if _hub_commit and _declared and _hub_commit != _declared:
        raise ValueError(
            f"{SERVED_MODEL_REVISION_ENV}={_declared!r} disagrees with the "
            f"model config commit {_hub_commit!r}; refusing rather than "
            "guessing which describes the weights being loaded"
        )
    served_identity = {
        "revision": _declared,
        "revision_source": SERVED_MODEL_REVISION_ENV if _declared else "",
        "config_commit": _hub_commit,
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
    for layer in model.layers:
        if hasattr(layer, "vp_full_graph_routed"):
            layer.vp_full_graph_routed = (
                model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and int(layer.layer_id) in routed_layer_id_set
            )
            layer.vp_full_graph_attention_routed = (
                model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
                and int(layer.layer_id) in attention_routed_layer_id_set
            )
    model._vp_full_graph_skipper_name = full_graph_skipper.name
    model._vp_full_graph_loaded_flexidepth_layer_ids = loaded_fd_layer_ids
    model._vp_full_graph_attention_route_layer_order = (
        attention_routed_layer_ids
    )
    model._fd_full_graph_route_layer_order = routed_layer_ids
    has_routed_layers = bool(routed_layer_ids)
    model.register_buffer(
        "_fd_full_graph_route_layer_ids",
        (
            torch.tensor(routed_layer_ids, dtype=torch.int64)
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and has_routed_layers
            and full_graph_device_route_tape_enabled()
            else None
        ),
        persistent=False,
    )
    model.register_buffer(
        "_fd_full_graph_compact_evidence_specs",
        (
            torch.tensor(
                full_graph_compact_evidence_specs(routed_layer_ids),
                dtype=torch.float32,
            )
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
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
    model.register_buffer(
        "_fd_full_graph_route_counters",
        (
            torch.zeros(6, dtype=torch.int64)
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and has_routed_layers
            and full_graph_route_accounting_enabled()
            else None
        ),
        persistent=False,
    )
    model.register_buffer(
        "_fd_full_graph_phase_route_counters",
        (
            torch.zeros((2, 3), dtype=torch.int64)
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and has_routed_layers
            and full_graph_route_accounting_enabled()
            and flexidepth_active_phases() == {"decode", "prefill"}
            and not full_graph_fused_evidence_enabled()
            else None
        ),
        persistent=False,
    )
    model.register_buffer(
        "_fd_full_graph_layer_route_counters",
        (
            torch.zeros((len(routed_layer_ids), 3), dtype=torch.int64)
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and has_routed_layers
            and full_graph_layer_counters_enabled()
            else None
        ),
        persistent=False,
    )
    model.register_buffer(
        "_fd_full_graph_commit_overlap_counters",
        (
            torch.zeros(1, dtype=torch.int64)
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and has_routed_layers
            and full_graph_commit_overlap_enabled()
            else None
        ),
        persistent=False,
    )
    low_row_policy, _ = full_graph_low_row_policy()
    model.register_buffer(
        "_fd_full_graph_low_row_counters",
        (
            torch.zeros(3, dtype=torch.int64)
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and has_routed_layers
            and full_graph_route_accounting_enabled()
            and low_row_policy != "off"
            else None
        ),
        persistent=False,
    )
    model.register_buffer(
        "_fd_full_graph_route_digest_counters",
        (
            torch.zeros(6, dtype=torch.int64)
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and has_routed_layers
            and full_graph_device_route_digest_enabled()
            else None
        ),
        persistent=False,
    )
    model.register_buffer(
        "_fd_full_graph_inline_kv_readiness_counters",
        (
            torch.zeros(7, dtype=torch.int64)
            if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH
            and has_routed_layers
            and full_graph_scheduler_convergence_enabled()
            else None
        ),
        persistent=False,
    )
    # [W1] Prefill regime-switch per-pass decision counters (grouped-FD vs
    # dense). Plain ints, always present so reads need no defensive getattr;
    # surfaced into the runtime-attestation regime_switch.counters block and
    # zeroed by vp_reset_runtime_counters between evaluation cells.
    model._vp_regime_prefill_fd_passes = 0
    model._vp_regime_prefill_dense_passes = 0
    # Batch-composition telemetry: counted for every prefill-phase pass
    # independent of regime-switch config (mixed split via the trailing-run
    # estimator — see _vp_regime_prefill_mixed_running_bs for the bias note).
    model._vp_mixed_passes = 0
    model._vp_mixed_decode_rows_est = 0
    model._vp_mixed_prefill_tokens_est = 0
    model._vp_prefill_pass_tokens = 0


def stamp_prefill_regime(model, forward_batch):
    """Forward-side seam: the W1 prefill regime-switch leg (per-pass stamp)."""
    # [W1] Unified regime switch — prefill leg (I5). Compute the per-pass
    # dense/FD prefill decision ONCE, before any FlexiDepth phase gate reads
    # it, and stamp it on forward_batch. When the switch is off (config
    # None) or its prefill leg is disabled, the batch is left unstamped and
    # flexidepth_phase_enabled never reads the stamp -> byte-identical FD.
    if flexidepth_forward_phase(forward_batch) != "prefill":
        return
    _regime_is_mixed = forward_batch.forward_mode.is_mixed()
    _pass_tokens = forward_batch.extend_num_tokens
    _regime_running_bs = (
        _vp_regime_prefill_mixed_running_bs(forward_batch)
        if _regime_is_mixed
        else 0
    )
    # Batch-composition telemetry (regime-independent), surfaced into
    # regime_switch.counters.{mixed,volume}. Skipped (never crashed) on a
    # pass without extend accounting; the regime leg below keeps its own
    # original contract on such passes.
    if _pass_tokens is not None:
        model._vp_prefill_pass_tokens += int(_pass_tokens)
        if _regime_is_mixed:
            model._vp_mixed_passes += 1
            model._vp_mixed_decode_rows_est += _regime_running_bs
            model._vp_mixed_prefill_tokens_est += (
                int(_pass_tokens) - _regime_running_bs
            )
    _regime_cfg = regime_switch_config()
    if _regime_cfg is not None and _regime_cfg.prefill.enabled:
        if _regime_is_mixed and not _regime_cfg.prefill.include_mixed:
            # MIXED pass with mixed switching disabled: leave FlexiDepth
            # enabled (unchanged). The appended running-decode rows are
            # governed by the decode leg (I6), not the prefill leg.
            forward_batch.vp_fd_prefill_dense = False
        else:
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
            model._vp_regime_prefill_dense_passes += 1
        else:
            model._vp_regime_prefill_fd_passes += 1


def seam_prepare_batch(model, forward_batch, hidden_states, positions):
    if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH:
        prepare_full_graph_batch(
            forward_batch,
            hidden_states,
            positions=positions,
            route_layer_ids=model._fd_full_graph_route_layer_ids,
            route_layer_order=model._fd_full_graph_route_layer_order,
            compact_evidence_specs=model._fd_full_graph_compact_evidence_specs,
        )


def seam_finalize_batch(model, forward_batch):
    if model.fd_execution_mode == FD_EXECUTION_FULL_GRAPH:
        finalize_full_graph_batch(
            forward_batch,
            model._fd_full_graph_route_counters,
            model._fd_full_graph_layer_route_counters,
            model._fd_full_graph_route_digest_counters,
            model._fd_full_graph_inline_kv_readiness_counters,
            model._fd_full_graph_phase_route_counters,
            model._fd_full_graph_low_row_counters,
        )


def validate_vp_causal_lm(lm, quant_config):
    loaded_fd_layers = [
        int(layer.layer_id)
        for layer in lm.model.layers
        if getattr(layer, "fd_router", None) is not None
    ]
    routed_layers = list(lm.model._fd_full_graph_route_layer_order)
    from sglang.srt.server_args import get_global_server_args

    validate_full_graph_model_configuration(
        loaded_layers=routed_layers,
        loaded_flexidepth_layers=loaded_fd_layers,
        tp_size=get_parallel().tp_size,
        pp_size=lm.pp_group.world_size,
        quant_config=quant_config,
        cuda_graph_enabled=not get_global_server_args().disable_cuda_graph,
    )


def vp_runtime_attestation(lm) -> dict:
    loaded_flexidepth_layers = [
        int(layer.layer_id)
        for layer in lm.model.layers
        if getattr(layer, "fd_router", None) is not None
    ]
    routed_layers = list(lm.model._fd_full_graph_route_layer_order)
    attention_routed_layers = list(
        lm.model._vp_full_graph_attention_route_layer_order
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
    counters = getattr(lm.model, "_fd_full_graph_route_counters", None)
    if counters is not None:
        values = [int(value) for value in counters.detach().cpu().tolist()]
        layer_rows, run_rows, project_rows = values[:3]
        flexidepth_state["full_graph_routes"] = {
            "layer_rows": layer_rows,
            "run_rows": run_rows,
            "project_rows": project_rows,
            "skip_ratio": (
                project_rows / layer_rows if layer_rows else 0.0
            ),
        }
        if (
            len(values) >= 6
            and full_graph_adapter.execution_kind == RUN_PROJECT_EXECUTION
        ):
            compact_rows, run_overflow, project_overflow = values[3:6]
            hidden_element_bytes = lm.model.layers[
                routed_layers[0]
            ].mlp.gate_up_proj.weight.element_size()
            hidden_row_bytes = lm.config.hidden_size * hidden_element_bytes
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
        lm.model, "_fd_full_graph_phase_route_counters", None
    )
    if phase_counters is not None:
        phase_values = phase_counters.detach().cpu().tolist()
        by_phase = {}
        for phase, (layer_rows, run_rows, project_rows) in zip(
            ("decode", "prefill"), phase_values
        ):
            record = {
                "layer_rows": int(layer_rows),
                "run_rows": int(run_rows),
                "project_rows": int(project_rows),
            }
            record["skip_ratio"] = (
                project_rows / layer_rows if layer_rows else 0.0
            )
            by_phase[phase] = record
        flexidepth_state["full_graph_routes"]["by_phase"] = by_phase
    layer_counters = getattr(
        lm.model, "_fd_full_graph_layer_route_counters", None
    )
    if layer_counters is not None:
        layer_values = layer_counters.detach().cpu().tolist()
        per_layer = []
        for layer_id, (layer_rows, run_rows, project_rows) in zip(
            routed_layers, layer_values
        ):
            record = {
                "layer_id": layer_id,
                "layer_rows": int(layer_rows),
                "run_rows": int(run_rows),
                "project_rows": int(project_rows),
            }
            record["run_ratio"] = (
                run_rows / layer_rows if layer_rows else 0.0
            )
            per_layer.append(record)
        flexidepth_state["full_graph_routes"]["per_layer"] = per_layer
    low_row_counters = getattr(
        lm.model, "_fd_full_graph_low_row_counters", None
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
        lm.model, "_fd_full_graph_route_digest_counters", None
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
                "stable_request_hash_token_epoch_layer_action"
                if logical_digest
                else "dispatch_order_layer_row_request_slot_token_epoch_"
                "cache_position_action"
            ),
            "batching_invariant": logical_digest,
            "digest_algorithm": (
                "batching_invariant_logical_action_int64_weighted_"
                "fingerprint"
                if logical_digest
                else "ordered_dual_int64_weighted_fingerprint"
            ),
        }
        flexidepth_state["full_graph_routes"]["device_route_tape"][
            "project_rows"
        ] = action_rows - run_rows
    commit_overlap_counters = getattr(
        lm.model, "_fd_full_graph_commit_overlap_counters", None
    )
    if commit_overlap_counters is not None:
        flexidepth_state["full_graph_routes"]["commit_overlap"] = {
            "enabled": True,
            "commit_batches": int(commit_overlap_counters[0].item()),
            "fence": "graph_end_dependency_on_commit_and_suffix_leaves",
        }
    readiness_counters = getattr(
        lm.model, "_fd_full_graph_inline_kv_readiness_counters", None
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
        "model_family": lm.vp_model_family,
        "flexidepth": flexidepth_state,
    }


def vp_regime_switch_counters(lm) -> dict:
    """Per-body regime-switch pass counters for the runtime attestation.

    Feeds regime_switch.counters (stripped from the deployment identity, so
    these live counts never move the pinned identity SHA). Prefill counts
    are real (I5); decode counts stay zero until the decode leg (I6) wires
    them.
    """

    counters = regime_switch_zero_counters()
    counters["prefill"][PREFILL_BODY_DENSE] = int(
        lm.model._vp_regime_prefill_dense_passes
    )
    counters["prefill"][PREFILL_BODY_FD] = int(
        lm.model._vp_regime_prefill_fd_passes
    )
    counters["mixed"]["passes"] = int(lm.model._vp_mixed_passes)
    counters["mixed"]["decode_rows_est"] = int(lm.model._vp_mixed_decode_rows_est)
    counters["mixed"]["prefill_tokens_est"] = int(
        lm.model._vp_mixed_prefill_tokens_est
    )
    counters["volume"]["prefill_pass_tokens"] = int(
        lm.model._vp_prefill_pass_tokens
    )
    return counters


def vp_reset_runtime_counters(lm) -> None:
    resolve_full_graph_skipper().reset_runtime_state()
    counters = getattr(lm.model, "_fd_full_graph_route_counters", None)
    if counters is not None:
        counters.zero_()
    phase_counters = getattr(
        lm.model, "_fd_full_graph_phase_route_counters", None
    )
    if phase_counters is not None:
        phase_counters.zero_()
    layer_counters = getattr(
        lm.model, "_fd_full_graph_layer_route_counters", None
    )
    if layer_counters is not None:
        layer_counters.zero_()
    low_row_counters = getattr(
        lm.model, "_fd_full_graph_low_row_counters", None
    )
    if low_row_counters is not None:
        low_row_counters.zero_()
    digest_counters = getattr(
        lm.model, "_fd_full_graph_route_digest_counters", None
    )
    if digest_counters is not None:
        digest_counters.zero_()
    readiness_counters = getattr(
        lm.model, "_fd_full_graph_inline_kv_readiness_counters", None
    )
    if readiness_counters is not None:
        readiness_counters.zero_()
    commit_overlap_counters = getattr(
        lm.model, "_fd_full_graph_commit_overlap_counters", None
    )
    if commit_overlap_counters is not None:
        commit_overlap_counters.zero_()
    # [W1] Reset the prefill regime-switch decision counters (always present).
    lm.model._vp_regime_prefill_fd_passes = 0
    lm.model._vp_regime_prefill_dense_passes = 0
    lm.model._vp_mixed_passes = 0
    lm.model._vp_mixed_decode_rows_est = 0
    lm.model._vp_mixed_prefill_tokens_est = 0
    lm.model._vp_prefill_pass_tokens = 0
