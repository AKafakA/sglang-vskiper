"""Startup validation of the full-graph model configuration.

Runs once at boot and RAISES on any inconsistency: fail-closed checks, zero
tensor operations. A misconfigured routed deployment must refuse to start
rather than serve a posture nobody declared.
"""

from __future__ import annotations

from vskipper.runtime.design import (  # [D-609]
    conditional_graph_helper_path,
)

from typing import (
    Any,
    Mapping,
    Optional,
    Callable,
)
import os
from vskipper.runtime.attestation import (
    _conflicting_env_enabled,
    full_graph_conditional_branch_counters_enabled,
    full_graph_conditional_graph_enabled,
    full_graph_conditional_max_rows,
    full_graph_conditional_production_all_run_enabled,
    full_graph_defer_project_kv_diagnostic_stage,
)
from vskipper.runtime.common import (
    full_graph_compact_routed_qkv_enabled,
    full_graph_contiguous_routed_qkv_config,
    full_graph_compact_phases,
    full_graph_prefill_cublas_enabled,
    full_graph_prefill_fallback_min_project,
    full_graph_gate_mode,
    full_graph_low_row_policy,
    flexidepth_execution_mode,
    flexidepth_active_phases,
    regime_switch_config,
    resolve_full_graph_skipper,
)
from vskipper.runtime.config import (
    full_graph_compact_config,
    full_graph_batched_commit_enabled,
    full_graph_commit_overlap_enabled,
    full_graph_defer_project_kv_enabled,
    full_graph_device_route_digest_enabled,
    full_graph_device_route_tape_enabled,
    full_graph_eager_semantic_debug_enabled,
    full_graph_forced_all_run_fastpath_enabled,
    full_graph_forced_all_run_production_attention_enabled,
    full_graph_fused_evidence_enabled,
    full_graph_layer_counters_enabled,
    full_graph_layer_policies,
    full_graph_masked_decode_attention_enabled,
    full_graph_prefill_grouped_mlp_enabled,
    full_graph_route_accounting_enabled,
    full_graph_scheduler_convergence_enabled,
    full_graph_virtual_cohort_enabled,
    full_graph_weighted_scatter_enabled,
)
from vskipper.runtime.env import (
    FD_ACTIVE_PHASES_ENV,
    FD_COMPACT_ENABLED_ENV,
    FD_COMPACT_PHASES_ENV,
    FD_PREFILL_CUBLAS_ENV,
    FD_PREFILL_FALLBACK_ENV,
    FD_CONDITIONAL_BRANCH_COUNTERS_ENV,
    FD_CONDITIONAL_GRAPH_ENV,
    FD_CONDITIONAL_GRAPH_HELPER_ENV,
    FD_CONDITIONAL_MAX_ROWS_ENV,
    FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV,
    FD_BATCHED_COMMIT_ENV,
    FD_COMMIT_OVERLAP_ENV,
    FD_COMPACT_ROUTED_QKV_ENV,
    FD_CONTIGUOUS_ROUTED_QKV_ENV,
    FD_ROUTED_QKV_CAPACITIES_ENV,
    FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV,
    FD_DEFER_PROJECT_KV_ENV,
    FD_DEVICE_ROUTE_DIGEST_ENV,
    FD_DEVICE_ROUTE_TAPE_ENV,
    FD_EAGER_SEMANTIC_DEBUG_ENV,
    FD_EXECUTION_FULL_GRAPH,
    FD_EXECUTION_MODE_ENV,
    FD_FORCED_ALL_RUN_FASTPATH_ENV,
    FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV,
    FD_FORCE_ROUTE_ENV,
    FD_FUSED_EVIDENCE_ENV,
    FD_GATE_MODE_ENV,
    FD_LAYER_COUNTERS_ENV,
    FD_LAYER_POLICIES_ENV,
    FD_LOW_ROW_POLICY_ENV,
    FD_MASKED_DECODE_ATTN_ENV,
    FD_PREFILL_GROUPED_MLP_ENV,
    FD_ROUTE_ACCOUNTING_ENV,
    FD_SCHEDULER_CONVERGENCE_ENV,
    FD_VIRTUAL_COHORT_ENV,
    FD_WEIGHTED_SCATTER_ENV,
    _CONFLICTING_FULL_GRAPH_ENV,
    _REMOVED_FEATURE_ENVS,
    REGIME_SWITCH_ENV,
)
from vskipper.kernels.mlp_compact import (
    full_graph_dual_compact_min_rows,
)



def assert_no_removed_execution_flags(
    environ: Optional[Mapping[str, str]] = None,
) -> None:
    """Reject env flags that select an execution path removed from this build.

    MODEL-INDEPENDENT on purpose. This used to live inside
    validate_full_graph_model_configuration, whose only caller is
    LlamaForCausalLM.__init__ -- so a non-Llama server accepted
    SGLANG_VP_SCHED=1 unchallenged while scheduler.py still published a
    ``vp_runtime`` attestation block for it. A removed flag must be refused for
    every model, not just the one this package hooks.
    """

    values = os.environ if environ is None else environ

    # Two SHAPES here, and conflating them was a defect. The V2/V4 keys name a
    # config PATH: any non-empty value selects the removed runtime. The other
    # three are BOOLEANS: only a truthy value ever did. Rejecting everything
    # "not in ('', '0')" refused an explicit SGLANG_VP_SCHED=false -- a
    # deployment saying OFF -- which is the same over-broad refusal this
    # validator already got wrong once (it briefly rejected every vanilla
    # server). Truthy/falsy sets match the package convention in
    # common.py:215-220.
    truthy = frozenset({"1", "true", "yes", "on"})
    falsy = frozenset({"0", "false", "no", "off", ""})
    for removed_path in ("SGLANG_VP_V4_CONFIG", "SGLANG_VP_V2_CONFIG"):
        if str(values.get(removed_path, "")).strip():
            raise ValueError(
                f"{removed_path} names a config for the removed V2/V4 "
                "runtime, which is not part of this build. Unset it."
            )
    for removed_flag in (
        "SGLANG_VP_SCHED",
        "SGLANG_FD_VP_STAGE_ROUTE",
        "SGLANG_FD_VP_PROJECT",
    ):
        value = str(values.get(removed_flag, "")).strip().lower()
        if value in falsy:
            continue
        if value in truthy:
            raise ValueError(
                f"{removed_flag} selects an execution path that is not part "
                "of this build (V1/V2/V4 and the inline VP-project body were "
                "removed). Unset it, or set it to a falsy value."
            )
        raise ValueError(
            f"{removed_flag} must be a boolean value; got {value!r}"
        )


def assert_regime_switch_skipper_capability(adapter: Any) -> None:
    """Fail closed unless the resolved skipper is a binary RUN/PROJECT FlexiDepth.

    Ports the M3.57 prereq-B capability check
    (``v4/production_executor.py:188-206``) to the V3 full-graph activation
    site so the W1 regime switch never runs over a non-binary-action
    adapter.  The resolvers are imported lazily to keep the config/predicate
    surface dependency-light.
    """

    from vskipper.runtime.env import RUN_PROJECT_EXECUTION
    from vskipper.runtime.types import LogicalAction

    if adapter.execution_kind != RUN_PROJECT_EXECUTION:
        raise RuntimeError(
            "regime switch requires RUN/PROJECT execution; adapter "
            f"{adapter.name!r} declares {adapter.execution_kind!r}"
        )
    if adapter.supported_actions != frozenset(
        (LogicalAction.RUN, LogicalAction.PROJECT_ONLY)
    ):
        raise RuntimeError(
            "regime switch supports exactly RUN and PROJECT_ONLY; adapter "
            f"{adapter.name!r} declares "
            f"{sorted(action.name for action in adapter.supported_actions)}"
        )
    # What the regime switch actually needs is a PROJECTOR: below the row
    # threshold it runs the dense body, above it routes, and a routed row must
    # still leave this layer's output and its own K/V behind. It never reads the
    # policy's gate. Until D-701 this was written as "requires FlexiDepth
    # router/projector weights", which refused any policy that brings its own
    # gate -- including, on its own terms, the deterministic mock, which only
    # passed because it inherited the FlexiDepth default it does not satisfy.
    from vskipper.runtime.skipper import resolve_projector

    resolve_projector(adapter.projector_kind)
def validate_full_graph_model_configuration(
    *,
    loaded_layers: list[int],
    loaded_flexidepth_layers: Optional[list[int]] = None,
    tp_size: int,
    pp_size: int,
    quant_config: Any,
    environ: Optional[Mapping[str, str]] = None,
    cuda_graph_enabled: Optional[bool] = None,
) -> None:
    """Fail closed on configurations not covered by the first V3 kernel."""

    values = os.environ if environ is None else environ

    # D-250 (review finding 5): REJECT the removed layer-routed mechanism rather
    # than ignoring it. It reused the stock --prefill/--decode-attention-backend
    # flags as role selectors, giving the treated arm a different decode kernel
    # than the baseline (D-248), and it is gone. ~20 sealed system_configs still
    # set it to "1"; without this they would boot into a DIFFERENT backend
    # topology than their name and provenance imply, with no fail-closed signal.
    # A hard error makes the removal self-attesting.
    # Same fate for the forced-all-RUN PRODUCTION-ATTENTION variant (D-250
    # review finding 4): its ONLY reader was _backend_for_layer on the deleted
    # class, yet the attestation would still have reported
    # "production_flashinfer_full_exact" — a mechanism attested but not
    # running, the exact R6 class. Without per-layer dispatch the feature
    # cannot exist; reject rather than silently accept.
    _removed_forced = "SGLANG_FD_FULL_GRAPH_FORCED_ALL_RUN_PRODUCTION_ATTENTION"
    if str(values.get(_removed_forced, "")).strip():
        raise ValueError(
            f"{_removed_forced} is REMOVED (D-248/D-250): it forced routed "
            "layers onto the production backend via the deleted per-layer "
            "dispatch. One backend now serves every layer; delete this key."
        )

    _removed = "SGLANG_FD_FULL_GRAPH_LAYER_ROUTED_DECODE_ATTENTION"
    if str(values.get(_removed, "")).strip():
        raise ValueError(
            f"{_removed} is REMOVED (D-248/D-250): it built a hybrid attention "
            "backend out of the stock CLI flags, which made the treated arm's "
            "decode kernel differ from the baseline's. One backend now serves "
            "every layer of every arm, chosen by the official CLI. Delete this "
            "key from the config; do not set it to 0."
        )

    # Removed execution modes must be REJECTED, not silently ignored. Each of
    # these selected an implementation that no longer exists; leaving them
    # accepted let a run attest a treatment that never executed (the class this
    # cleanup removed the self-sized ladder for).
    assert_no_removed_execution_flags(values)

    skipper_adapter = resolve_full_graph_skipper(values)
    # W1 regime switch is config-compatible with every full-graph knob below,
    # grouped-prefill MLP included: the switch changes only the per-pass
    # dense/FD *decision* (the vp_fd_prefill_dense stamp that
    # flexidepth_phase_enabled consults), never the static routed topology.
    # No check in this validator may assume "grouped-MLP enabled => every
    # prefill pass runs FlexiDepth"; the predicate chooses the body per pass.
    if regime_switch_config(values) is not None:
        assert_regime_switch_skipper_capability(skipper_adapter)
    if loaded_flexidepth_layers is None:
        loaded_flexidepth_layers = list(loaded_layers)
    if skipper_adapter.requires_flexidepth_weights:
        if tuple(loaded_layers) != tuple(loaded_flexidepth_layers):
            raise ValueError(
                f"{skipper_adapter.name} routed layers must match loaded "
                "FlexiDepth layers"
            )
    elif loaded_flexidepth_layers:
        raise ValueError(
            f"{skipper_adapter.name} must not load FlexiDepth weights"
        )
    execution_mode = flexidepth_execution_mode(values)
    # The eager FlexiDepth bodies are not in this build (llama.py fails closed
    # if a routed layer reaches that dispatch). direct_eager is nonetheless the
    # DEFAULT of flexidepth_execution_mode(), so a routed deployment that does
    # not set the mode explicitly used to die later -- at CUDA-graph capture, or
    # on the first routed request when graphs are disabled. Refuse at STARTUP
    # instead: same outcome, but at the point where the operator can act on it,
    # and before any request is accepted.
    # direct_eager IS supported (it is the quality-attribution reference), but it
    # cannot coexist with CUDA graphs: its routed-row gather is data-dependent,
    # which is a device->host sync and illegal under stream capture. The
    # pre-refactor tree failed the same way, at capture, with
    # cudaErrorStreamCaptureUnsupported. Refuse the combination HERE so the
    # operator gets an actionable message at startup instead of a capture abort.
    #
    # Gate on ROUTED LAYERS ACTUALLY LOADED, not on the adapter's declared
    # requirement: resolve_full_graph_skipper() returns the FlexiDepth adapter by
    # default and requires_flexidepth_weights defaults to True, so keying off the
    # adapter would fire for a VANILLA Llama server with no vPipe configuration.
    # This validator runs unconditionally from LlamaForCausalLM.__init__ with no
    # early return, and no canonical arm covers that case.
    if (
        loaded_flexidepth_layers
        and skipper_adapter.requires_flexidepth_weights
        and execution_mode != FD_EXECUTION_FULL_GRAPH
    ):
        if cuda_graph_enabled:
            raise ValueError(
                f"{FD_EXECUTION_MODE_ENV}={execution_mode!r} cannot run with CUDA "
                "graphs: its routed-row gather is data-dependent and illegal "
                "under stream capture. Run the eager reference with "
                "--disable-cuda-graph, or set "
                f"{FD_EXECUTION_MODE_ENV}={FD_EXECUTION_FULL_GRAPH} to serve."
            )
    if (
        skipper_adapter.requires_stable_request_ids
        and execution_mode != FD_EXECUTION_FULL_GRAPH
    ):
        raise ValueError(
            f"{skipper_adapter.name} requires "
            f"{FD_EXECUTION_MODE_ENV}={FD_EXECUTION_FULL_GRAPH}"
        )
    if (
        not skipper_adapter.requires_flexidepth_weights
        and execution_mode != FD_EXECUTION_FULL_GRAPH
    ):
        raise ValueError(
            f"{skipper_adapter.name} requires "
            f"{FD_EXECUTION_MODE_ENV}={FD_EXECUTION_FULL_GRAPH}"
        )
    eager_semantic_debug = full_graph_eager_semantic_debug_enabled(values)
    prefill_grouped_mlp = full_graph_prefill_grouped_mlp_enabled(values)
    conditional_graph = full_graph_conditional_graph_enabled(values)
    conditional_production_all_run = (
        full_graph_conditional_production_all_run_enabled(values)
    )
    conditional_max_rows = full_graph_conditional_max_rows(values)
    conditional_branch_counters = (
        full_graph_conditional_branch_counters_enabled(values)
    )
    defer_project_kv = full_graph_defer_project_kv_enabled(values)
    defer_project_kv_stage = (
        full_graph_defer_project_kv_diagnostic_stage(values)
    )
    compact_routed_qkv = full_graph_compact_routed_qkv_enabled(values)
    (
        contiguous_routed_qkv,
        _,
        _,
        _capacities,
    ) = full_graph_contiguous_routed_qkv_config(values)
    if compact_routed_qkv and not defer_project_kv:
        raise ValueError(
            f"{FD_COMPACT_ROUTED_QKV_ENV}=1 requires "
            f"{FD_DEFER_PROJECT_KV_ENV}=1"
        )
    if compact_routed_qkv and defer_project_kv_stage != "full":
        raise ValueError(
            f"{FD_COMPACT_ROUTED_QKV_ENV}=1 requires "
            f"{FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV}=full"
        )
    if contiguous_routed_qkv and not compact_routed_qkv:
        raise ValueError(
            f"{FD_CONTIGUOUS_ROUTED_QKV_ENV}=1 requires "
            f"{FD_COMPACT_ROUTED_QKV_ENV}=1"
        )
    if contiguous_routed_qkv:
        # loaded_flexidepth_layers is normalized to a concrete list above.
        missing_capacity_layers = sorted(
            set(loaded_flexidepth_layers) - set(_capacities)
        )
        if missing_capacity_layers:
            raise ValueError(
                f"{FD_ROUTED_QKV_CAPACITIES_ENV} is missing routed layers "
                f"{missing_capacity_layers}; the contiguous lane fails at "
                "capture otherwise — cover every routed layer or disable it"
            )
    batched_commit = full_graph_batched_commit_enabled(values)
    if batched_commit and not defer_project_kv:
        raise ValueError(
            f"{FD_BATCHED_COMMIT_ENV}=1 requires {FD_DEFER_PROJECT_KV_ENV}=1"
        )
    if batched_commit and defer_project_kv_stage != "full":
        raise ValueError(
            f"{FD_BATCHED_COMMIT_ENV}=1 requires the full repair commit; "
            f"{FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV} must be full"
        )
    commit_overlap = full_graph_commit_overlap_enabled(values)
    if commit_overlap and not defer_project_kv:
        raise ValueError(
            f"{FD_COMMIT_OVERLAP_ENV}=1 requires {FD_DEFER_PROJECT_KV_ENV}=1"
        )
    if commit_overlap and defer_project_kv_stage != "full":
        raise ValueError(
            f"{FD_COMMIT_OVERLAP_ENV}=1 requires the full repair commit; "
            f"{FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV} must be full"
        )
    if conditional_branch_counters and not conditional_graph:
        raise ValueError(
            f"{FD_CONDITIONAL_BRANCH_COUNTERS_ENV}=1 requires "
            f"{FD_CONDITIONAL_GRAPH_ENV}=1"
        )
    if conditional_production_all_run and not conditional_graph:
        raise ValueError(
            f"{FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV}=1 requires "
            f"{FD_CONDITIONAL_GRAPH_ENV}=1"
        )
    if conditional_max_rows is not None and not conditional_graph:
        raise ValueError(
            f"{FD_CONDITIONAL_MAX_ROWS_ENV} requires "
            f"{FD_CONDITIONAL_GRAPH_ENV}=1"
        )
    if conditional_max_rows is not None and defer_project_kv:
        raise ValueError(
            f"{FD_CONDITIONAL_MAX_ROWS_ENV} requires foreground K/V; "
            f"{FD_DEFER_PROJECT_KV_ENV} must be disabled"
        )
    if defer_project_kv and not conditional_graph:
        raise ValueError(
            f"{FD_DEFER_PROJECT_KV_ENV}=1 requires "
            f"{FD_CONDITIONAL_GRAPH_ENV}=1"
        )
    if defer_project_kv_stage != "full" and not defer_project_kv:
        raise ValueError(
            f"{FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV}="
            f"{defer_project_kv_stage} requires {FD_DEFER_PROJECT_KV_ENV}=1"
        )
    if eager_semantic_debug:
        if execution_mode != FD_EXECUTION_FULL_GRAPH:
            raise ValueError(
                f"{FD_EAGER_SEMANTIC_DEBUG_ENV}=1 requires "
                f"{FD_EXECUTION_MODE_ENV}={FD_EXECUTION_FULL_GRAPH}"
            )
        if not str(values.get("SGLANG_FD_PARITY_TRACE_RID", "")).strip():
            raise ValueError(
                f"{FD_EAGER_SEMANTIC_DEBUG_ENV}=1 requires "
                "SGLANG_FD_PARITY_TRACE_RID"
            )
        if not str(values.get("SGLANG_FD_PARITY_TRACE_DIR", "")).strip():
            raise ValueError(
                f"{FD_EAGER_SEMANTIC_DEBUG_ENV}=1 requires "
                "SGLANG_FD_PARITY_TRACE_DIR"
            )
    if conditional_production_all_run and eager_semantic_debug:
        raise ValueError(
            f"{FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV}=1 conflicts with "
            f"{FD_EAGER_SEMANTIC_DEBUG_ENV}=1"
        )
    if conditional_production_all_run and defer_project_kv:
        raise ValueError(
            f"{FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV}=1 requires foreground "
            f"K/V; {FD_DEFER_PROJECT_KV_ENV} must be disabled"
        )
    active_phases = flexidepth_active_phases(values)
    # W1 regime switch (decode leg, I6b — fork option (b)): the "prod_allrun" low
    # band DISPATCHES THE STOCK production decode graph (the plain base-Llama
    # forward with production/flashinfer decode attention and no FlexiDepth
    # hooks), bypassing the FD conditional backend; the "skip" high band keeps
    # the M3.38 FlexiDepth conditional/whole-model body. The GPU parity gate
    # (2026-07-23) proved production-all-RUN *through* the conditional graph is
    # neither byte-identical (0/48) nor same-speed (+4%) vs stock, so the low
    # body must be the stock graph. Fail closed on the prerequisites the high
    # (skip) band still needs. Only enforce when decode is an active phase (a
    # prefill-only endpoint's decode leg is dormant).
    #
    # The old `conditional_max_rows >= decode.enter_rows` constraint is REMOVED:
    # with the low band on stock graphs, the conditional graph is never used
    # below the band, so band buckets need not be conditional buckets.
    _regime_cfg = regime_switch_config(values)
    if (
        _regime_cfg is not None
        and _regime_cfg.decode.enabled
        and "decode" in active_phases
    ):
        if not conditional_graph:
            raise ValueError(
                f"{REGIME_SWITCH_ENV} decode leg requires "
                f"{FD_CONDITIONAL_GRAPH_ENV}=1"
            )
        if defer_project_kv:
            raise ValueError(
                f"{REGIME_SWITCH_ENV} decode leg needs foreground K/V so a "
                "skip<->stock band crossing always reads complete own-layer "
                f"K/V; {FD_DEFER_PROJECT_KV_ENV} must be disabled"
            )
        if conditional_production_all_run:
            raise ValueError(
                f"{REGIME_SWITCH_ENV} decode leg drives the body per regime "
                f"(the low band is the stock decode graph); "
                f"{FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV} must stay disabled "
                "(the switch selects the body, not the env flag)"
            )
    compact_enabled = full_graph_compact_config(values)
    compact_phases = full_graph_compact_phases(values)
    low_row_policy = full_graph_low_row_policy(values)
    gate_mode = full_graph_gate_mode(values)  # validates the value; fails closed on an unknown one
    # [v1.5] `hard_mask` is honoured by EVERY routed-MLP body now, not only the dense ones:
    # the count-GEMM scatter epilogues, the weighted scatter, the mapped SwiGLU down-kernel and
    # the fused-MoE fallbacks all take the gate mode (SCALE_WEIGHT off = hard selection, no `w`
    # multiply), and the attestation publishes `gate_branch_scaling`. The old refusal
    # ("hard_mask requires low_row_policy=native_dense") is therefore gone; parity tests cover
    # the epilogues under both modes (test_count_gemm_fused_io.py, test_hard_mask_bodies.py).
    route_accounting = full_graph_route_accounting_enabled(values)
    layer_counters = full_graph_layer_counters_enabled(values)
    device_route_tape = full_graph_device_route_tape_enabled(values)
    device_route_digest = full_graph_device_route_digest_enabled(values)
    scheduler_convergence = full_graph_scheduler_convergence_enabled(values)
    forced_all_run_fastpath = full_graph_forced_all_run_fastpath_enabled(values)
    forced_all_run_production_attention = (
        full_graph_forced_all_run_production_attention_enabled(values)
    )
    virtual_cohort = full_graph_virtual_cohort_enabled(values)
    full_graph_dual_compact_min_rows(values)  # load-time validation
    weighted_scatter = full_graph_weighted_scatter_enabled(values)
    fused_evidence = full_graph_fused_evidence_enabled(values)
    masked_decode_attention = full_graph_masked_decode_attention_enabled(values)
    if full_graph_prefill_fallback_min_project(values) is not None:
        # D-508 fails closed: the per-layer fallback decides between a
        # compaction body and the dense full-dual body on prefill passes, so
        # it needs compaction on prefill and prefill among the active phases;
        # otherwise the value could never execute.
        if "prefill" not in full_graph_compact_phases(values):
            raise ValueError(
                f"{FD_PREFILL_FALLBACK_ENV} requires {FD_COMPACT_PHASES_ENV} to "
                "include prefill (both) — it can only execute on prefill passes"
            )
        if "prefill" not in flexidepth_active_phases(values):
            raise ValueError(
                f"{FD_PREFILL_FALLBACK_ENV} requires prefill among the active "
                "FlexiDepth phases"
            )
    if full_graph_prefill_cublas_enabled(values):
        # P3 fails closed: the cuBLAS branch lives inside the binary_cohort body
        # on prefill passes, so it needs compaction on prefill and at least one
        # binary_cohort layer policy; otherwise the flag could never execute.
        if "prefill" not in full_graph_compact_phases(values):
            raise ValueError(
                f"{FD_PREFILL_CUBLAS_ENV}=1 requires {FD_COMPACT_PHASES_ENV} to "
                "include prefill (both) — it can only execute on prefill passes"
            )
        if not any(
            policy[0] == "binary_cohort"
            for policy in full_graph_layer_policies(values).values()
        ):
            raise ValueError(
                f"{FD_PREFILL_CUBLAS_ENV}=1 requires at least one binary_cohort "
                f"layer policy in {FD_LAYER_POLICIES_ENV}"
            )
    if virtual_cohort and any(
        policy[0] == "project_filtered_run_compact"
        for policy in full_graph_layer_policies(values).values()
    ):
        raise ValueError(
            "project_filtered_run_compact requires virtual cohort OFF: the "
            "mapped_swiglu path ignores the bounded capacity, silently "
            "defeating the policy"
        )
    if low_row_policy != "off":
        if execution_mode != FD_EXECUTION_FULL_GRAPH:
            raise ValueError(
                f"{FD_LOW_ROW_POLICY_ENV} requires "
                f"{FD_EXECUTION_MODE_ENV}={FD_EXECUTION_FULL_GRAPH}"
            )
        if not compact_enabled:
            raise ValueError(
                f"{FD_LOW_ROW_POLICY_ENV} requires "
                f"{FD_COMPACT_ENABLED_ENV}=1"
            )
        # The posture resolves before every routed-MLP dispatch
        # (binary_cohort included), so it is well-defined for prefill passes
        # too. Decode must still be a compact phase; prefill may be as well
        # (the I3 vp_final posture is `both`).
        if "decode" not in compact_phases:
            raise ValueError(
                f"{FD_LOW_ROW_POLICY_ENV} requires decode in "
                f"{FD_COMPACT_PHASES_ENV}"
            )
        if "decode" not in active_phases:
            raise ValueError(
                f"{FD_LOW_ROW_POLICY_ENV} requires decode in "
                f"{FD_ACTIVE_PHASES_ENV}"
            )
        if eager_semantic_debug or forced_all_run_fastpath:
            raise ValueError(
                f"{FD_LOW_ROW_POLICY_ENV} conflicts with eager semantic "
                "debug or forced all-RUN execution"
            )
    if prefill_grouped_mlp:
        if execution_mode != FD_EXECUTION_FULL_GRAPH:
            raise ValueError(
                f"{FD_PREFILL_GROUPED_MLP_ENV}=1 requires "
                f"{FD_EXECUTION_MODE_ENV}={FD_EXECUTION_FULL_GRAPH}"
            )
        if "prefill" not in active_phases:
            raise ValueError(
                f"{FD_PREFILL_GROUPED_MLP_ENV}=1 requires prefill in "
                f"{FD_ACTIVE_PHASES_ENV}"
            )
        if skipper_adapter.name != "flexidepth":
            raise ValueError(
                f"{FD_PREFILL_GROUPED_MLP_ENV}=1 currently requires the "
                "FlexiDepth whole-layer adapter"
            )
        if eager_semantic_debug:
            raise ValueError(
                f"{FD_PREFILL_GROUPED_MLP_ENV}=1 conflicts with "
                f"{FD_EAGER_SEMANTIC_DEBUG_ENV}=1"
            )
        if "prefill" in compact_phases:
            raise ValueError(
                f"{FD_PREFILL_GROUPED_MLP_ENV}=1 conflicts with prefill in "
                f"{FD_COMPACT_PHASES_ENV}"
            )
        if forced_all_run_fastpath or forced_all_run_production_attention:
            raise ValueError(
                f"{FD_PREFILL_GROUPED_MLP_ENV}=1 conflicts with forced "
                "all-RUN execution"
            )
    if conditional_graph:
        required = {
            FD_DEVICE_ROUTE_TAPE_ENV: device_route_tape,
            FD_MASKED_DECODE_ATTN_ENV: masked_decode_attention,
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                f"{FD_CONDITIONAL_GRAPH_ENV}=1 requires " + ", ".join(missing)
            )
        if "decode" not in flexidepth_active_phases(values):
            raise ValueError(
                f"{FD_CONDITIONAL_GRAPH_ENV}=1 requires decode in "
                f"{FD_ACTIVE_PHASES_ENV}"
            )
        if forced_all_run_fastpath:
            raise ValueError(
                f"{FD_CONDITIONAL_GRAPH_ENV}=1 requires dynamic routing"
            )
        if not str(conditional_graph_helper_path() or "").strip():
            raise ValueError(
                f"{FD_CONDITIONAL_GRAPH_ENV}=1 requires "
                f"{FD_CONDITIONAL_GRAPH_HELPER_ENV}"
            )
    if defer_project_kv:
        required = {
            FD_DEVICE_ROUTE_TAPE_ENV: device_route_tape,
            FD_SCHEDULER_CONVERGENCE_ENV: scheduler_convergence,
            FD_MASKED_DECODE_ATTN_ENV: masked_decode_attention,
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                f"{FD_DEFER_PROJECT_KV_ENV}=1 requires " + ", ".join(missing)
            )
    if forced_all_run_fastpath:
        raise ValueError(
            f"{FD_FORCED_ALL_RUN_FASTPATH_ENV}=1 requires "
            f"{FD_FORCE_ROUTE_ENV}=all_run"
        )
    if (
        forced_all_run_fastpath
        and flexidepth_active_phases(values) != {"decode"}
    ):
        raise ValueError(
            f"{FD_FORCED_ALL_RUN_FASTPATH_ENV}=1 currently requires "
            f"{FD_ACTIVE_PHASES_ENV}=decode"
        )
    if forced_all_run_fastpath and eager_semantic_debug:
        raise ValueError(
            f"{FD_FORCED_ALL_RUN_FASTPATH_ENV}=1 conflicts with "
            f"{FD_EAGER_SEMANTIC_DEBUG_ENV}=1"
        )
    if (
        forced_all_run_production_attention
        and not forced_all_run_fastpath
    ):
        raise ValueError(
            f"{FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV}=1 requires "
            f"{FD_FORCED_ALL_RUN_FASTPATH_ENV}=1"
        )
    if layer_counters and not route_accounting:
        raise ValueError(
            f"{FD_LAYER_COUNTERS_ENV}=1 requires {FD_ROUTE_ACCOUNTING_ENV}=1"
        )
    if device_route_digest and not device_route_tape:
        raise ValueError(
            f"{FD_DEVICE_ROUTE_DIGEST_ENV}=1 requires "
            f"{FD_DEVICE_ROUTE_TAPE_ENV}=1"
        )
    if device_route_digest and not route_accounting:
        raise ValueError(
            f"{FD_DEVICE_ROUTE_DIGEST_ENV}=1 requires "
            f"{FD_ROUTE_ACCOUNTING_ENV}=1"
        )
    if scheduler_convergence and not device_route_tape:
        raise ValueError(
            f"{FD_SCHEDULER_CONVERGENCE_ENV}=1 requires "
            f"{FD_DEVICE_ROUTE_TAPE_ENV}=1"
        )
    if scheduler_convergence and not device_route_digest:
        raise ValueError(
            f"{FD_SCHEDULER_CONVERGENCE_ENV}=1 requires "
            f"{FD_DEVICE_ROUTE_DIGEST_ENV}=1"
        )
    full_graph_layer_policies(values)
    compact_enabled = full_graph_compact_config(values)
    if virtual_cohort and not compact_enabled:
        raise ValueError(
            f"{FD_VIRTUAL_COHORT_ENV}=1 requires {FD_COMPACT_ENABLED_ENV}=1"
        )
    if weighted_scatter and not compact_enabled:
        raise ValueError(
            f"{FD_WEIGHTED_SCATTER_ENV}=1 requires {FD_COMPACT_ENABLED_ENV}=1"
        )
    if weighted_scatter and virtual_cohort:
        raise ValueError(
            f"{FD_WEIGHTED_SCATTER_ENV}=1 conflicts with "
            f"{FD_VIRTUAL_COHORT_ENV}=1"
        )
    if fused_evidence:
        required = {
            FD_ROUTE_ACCOUNTING_ENV: route_accounting,
            FD_LAYER_COUNTERS_ENV: layer_counters,
            FD_DEVICE_ROUTE_TAPE_ENV: device_route_tape,
            FD_DEVICE_ROUTE_DIGEST_ENV: device_route_digest,
            FD_SCHEDULER_CONVERGENCE_ENV: scheduler_convergence,
        }
        missing = [name for name, enabled in required.items() if not enabled]
        if missing:
            raise ValueError(
                f"{FD_FUSED_EVIDENCE_ENV}=1 requires " + ", ".join(missing)
            )
        # Lane-2 cut3 item 4 (D-359): the decode-only restriction existed because
        # the fused kernel is decode-layout and capped at 1024 rows. The decision
        # is now made PER PASS in `prepare_full_graph_batch` (fused on a decode
        # pass within the cap, the existing accumulators otherwise), so a
        # both-phase deployment is well-defined. Decode must still be active —
        # without it the fused path could never run and the flag would be inert.
        if "decode" not in flexidepth_active_phases(values):
            raise ValueError(
                f"{FD_FUSED_EVIDENCE_ENV}=1 requires decode in "
                f"{FD_ACTIVE_PHASES_ENV}"
            )
    # [lane-2 knob cleanup, D-578] The compact output-projection body, the
    # split-QKV body and the mapped-decode worker-row sizing are deleted. Their
    # env vars are registered in _REMOVED_FEATURE_ENVS and refused by the
    # removed-feature check below, so nothing is silently ignored here.
    # NOTE: the converse (MASKED=1 => the mask actually reaches a triton decode kernel)
    # is NOT enforceable here — this validator has no server_args/backend in scope, and
    # MASKED=1 with LAYER_ROUTED=0 is legitimate when an explicit --decode-attention-
    # backend=triton pin is present (several sealed repo configs do exactly that). The
    # guard therefore lives in ModelRunner._get_attention_backend, where the resolved
    # backend is visible. See _MASKED_DECODE_REQUIRED_BACKEND.
    if (
        masked_decode_attention
        and flexidepth_execution_mode(values) != FD_EXECUTION_FULL_GRAPH
    ):
        raise ValueError(
            f"{FD_MASKED_DECODE_ATTN_ENV}=1 requires "
            f"{FD_EXECUTION_MODE_ENV}={FD_EXECUTION_FULL_GRAPH}"
        )
    if (
        device_route_tape
        and flexidepth_execution_mode(values) != FD_EXECUTION_FULL_GRAPH
    ):
        raise ValueError(
            f"{FD_DEVICE_ROUTE_TAPE_ENV}=1 requires "
            f"{FD_EXECUTION_MODE_ENV}={FD_EXECUTION_FULL_GRAPH}"
        )
    if execution_mode != FD_EXECUTION_FULL_GRAPH:
        return
    if not loaded_layers:
        if skipper_adapter.requires_flexidepth_weights:
            raise ValueError(
                f"{FD_EXECUTION_MODE_ENV}=full_graph requires "
                "SGLANG_FD_WEIGHTS"
            )
        raise ValueError(
            f"{FD_EXECUTION_MODE_ENV}=full_graph requires routed skipper layers"
        )
    conflicts = [
        name
        for name in _CONFLICTING_FULL_GRAPH_ENV
        if _conflicting_env_enabled(name, values.get(name))
    ]
    if conflicts:
        raise ValueError(
            f"{FD_EXECUTION_MODE_ENV}=full_graph is incompatible with "
            + ", ".join(conflicts)
        )
    removed = [
        name
        for name in _REMOVED_FEATURE_ENVS
        if _conflicting_env_enabled(name, values.get(name))
    ]
    if removed:
        raise ValueError(
            "these knobs name features removed from this build (owner "
            "ruling 2026-08-23; see "
            "codex/asplos-plan/2026-08-21-removed-feature-register.md): "
            + ", ".join(removed)
        )
    if tp_size != 1 or pp_size != 1:
        raise ValueError(
            "VP full_graph currently requires TP=1 and PP=1; "
            f"observed TP={tp_size}, PP={pp_size}"
        )
    if quant_config is not None:
        raise ValueError("VP full_graph currently requires unquantized weights")
