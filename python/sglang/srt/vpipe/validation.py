"""Startup validation of the full-graph model configuration.

Runs once at boot and RAISES on any inconsistency: fail-closed checks, zero
tensor operations. A misconfigured routed deployment must refuse to start
rather than serve a posture nobody declared.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional
import os
from typing import Any, Callable, Optional
from sglang.srt.vpipe.attestation import (
    _conflicting_env_enabled,
    full_graph_commit_overlap_enabled,
    full_graph_conditional_branch_counters_enabled,
    full_graph_conditional_graph_enabled,
    full_graph_conditional_max_rows,
    full_graph_conditional_production_all_run_enabled,
    full_graph_defer_project_kv_diagnostic_stage,
    full_graph_repair_group_size,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_o_proj_min_rows,
    full_graph_compact_phases,
    full_graph_compact_q_proj_enabled,
    full_graph_contiguous_routed_qkv_config,
    full_graph_low_row_policy,
)
from sglang.srt.vpipe.common import (
    flexidepth_execution_mode,
    full_graph_compact_routed_qkv_enabled,
)
from sglang.srt.vpipe.common import (
    flexidepth_active_phases,
)
from sglang.srt.vpipe.common import (
    regime_switch_config,
)
from sglang.srt.vpipe.config import (
    full_graph_compact_config,
    full_graph_compact_o_proj_config,
    full_graph_defer_project_kv_enabled,
    full_graph_device_route_digest_enabled,
    full_graph_device_route_tape_enabled,
    full_graph_eager_semantic_debug_enabled,
    full_graph_forced_all_run_fastpath_enabled,
    full_graph_forced_all_run_production_attention_enabled,
    full_graph_fused_evidence_enabled,
    full_graph_layer_counters_enabled,
    full_graph_layer_policies,
    full_graph_mapped_decode_attention_enabled,
    full_graph_masked_decode_attention_enabled,
    full_graph_prefill_grouped_mlp_enabled,
    full_graph_route_accounting_enabled,
    full_graph_scheduler_convergence_enabled,
    full_graph_virtual_cohort_enabled,
    full_graph_weighted_scatter_enabled,
)
from sglang.srt.vpipe.env import (
    FD_ACTIVE_PHASES_ENV,
    FD_COMMIT_OVERLAP_ENV,
    FD_COMPACT_ENABLED_ENV,
    FD_COMPACT_MIN_ROWS_ENV,
    FD_COMPACT_O_PROJ_ENV,
    FD_COMPACT_O_PROJ_LAYERS_ENV,
    FD_COMPACT_PHASES_ENV,
    FD_COMPACT_Q_PROJ_ENV,
    FD_COMPACT_ROUTED_QKV_ENV,
    FD_CONDITIONAL_BRANCH_COUNTERS_ENV,
    FD_CONDITIONAL_GRAPH_ENV,
    FD_CONDITIONAL_GRAPH_HELPER_ENV,
    FD_CONDITIONAL_MAX_ROWS_ENV,
    FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV,
    FD_CONTIGUOUS_ROUTED_QKV_ENV,
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
    FD_LAYER_COUNTERS_ENV,
    FD_LAYER_POLICIES_ENV,
    FD_LOW_ROW_MAX_ROWS_ENV,
    FD_LOW_ROW_POLICY_ENV,
    FD_MAPPED_DECODE_ATTN_ENV,
    FD_MASKED_DECODE_ATTN_ENV,
    FD_PREFILL_GROUPED_MLP_ENV,
    FD_REPAIR_GROUP_SIZE_ENV,
    FD_ROUTED_QKV_CAPACITIES_ENV,
    FD_ROUTE_ACCOUNTING_ENV,
    FD_SCHEDULER_CONVERGENCE_ENV,
    FD_VIRTUAL_COHORT_ENV,
    FD_WEIGHTED_SCATTER_ENV,
    _CONFLICTING_FULL_GRAPH_ENV,
)
from sglang.srt.vpipe.mlp_compact import (
    full_graph_dual_compact_min_rows,
)
from sglang.srt.vpipe.routing import (
    full_graph_forced_route,
)
from sglang.srt.vpipe.common import (
    resolve_full_graph_skipper,
)
from sglang.srt.vpipe.env import (
    SUBLAYER_EXECUTION,
)
from sglang.srt.vpipe.env import (
    REGIME_SWITCH_ENV,
)


def assert_regime_switch_skipper_capability(adapter: Any) -> None:
    """Fail closed unless the resolved skipper is a binary RUN/PROJECT FlexiDepth.

    Ports the M3.57 prereq-B capability check
    (``v4/production_executor.py:188-206``) to the V3 full-graph activation
    site so the W1 regime switch never runs over an AdaSkip / non-binary-action
    adapter.  The resolvers are imported lazily to keep the config/predicate
    surface dependency-light.
    """

    from sglang.srt.vpipe.env import RUN_PROJECT_EXECUTION
    from sglang.srt.vpipe.types import LogicalAction

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
    if not adapter.requires_flexidepth_weights:
        raise RuntimeError(
            "regime switch requires FlexiDepth router/projector weights; "
            f"adapter {adapter.name!r} does not require them"
        )
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
        routed_qkv_capacities,
    ) = full_graph_contiguous_routed_qkv_config(values)
    repair_group_size = full_graph_repair_group_size(values)
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
    commit_overlap = full_graph_commit_overlap_enabled(values)
    if commit_overlap and not defer_project_kv:
        raise ValueError(
            f"{FD_COMMIT_OVERLAP_ENV}=1 requires "
            f"{FD_DEFER_PROJECT_KV_ENV}=1"
        )
    if commit_overlap and defer_project_kv_stage != "full":
        raise ValueError(
            f"{FD_COMMIT_OVERLAP_ENV}=1 requires "
            f"{FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV}=full; diagnostic "
            "stages have no K/V commit to overlap"
        )
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
    if contiguous_routed_qkv and repair_group_size != 1:
        raise ValueError(
            f"{FD_CONTIGUOUS_ROUTED_QKV_ENV}=1 requires "
            f"{FD_REPAIR_GROUP_SIZE_ENV}=1"
        )
    if repair_group_size > 1 and not compact_routed_qkv:
        raise ValueError(
            f"{FD_REPAIR_GROUP_SIZE_ENV}>1 requires "
            f"{FD_COMPACT_ROUTED_QKV_ENV}=1"
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
    compact_enabled, compact_min_rows, _, _ = full_graph_compact_config(values)
    compact_phases = full_graph_compact_phases(values)
    low_row_policy, low_row_max_rows = full_graph_low_row_policy(values)
    route_accounting = full_graph_route_accounting_enabled(values)
    layer_counters = full_graph_layer_counters_enabled(values)
    device_route_tape = full_graph_device_route_tape_enabled(values)
    device_route_digest = full_graph_device_route_digest_enabled(values)
    scheduler_convergence = full_graph_scheduler_convergence_enabled(values)
    forced_route = full_graph_forced_route(values)
    if forced_route is not None and skipper_adapter.requires_stable_request_ids:
        raise ValueError(
            f"{FD_FORCE_ROUTE_ENV} is unsupported by {skipper_adapter.name}"
        )
    forced_all_run_fastpath = full_graph_forced_all_run_fastpath_enabled(values)
    forced_all_run_production_attention = (
        full_graph_forced_all_run_production_attention_enabled(values)
    )
    virtual_cohort = full_graph_virtual_cohort_enabled(values)
    full_graph_dual_compact_min_rows(values)  # load-time validation
    weighted_scatter = full_graph_weighted_scatter_enabled(values)
    fused_evidence = full_graph_fused_evidence_enabled(values)
    masked_decode_attention = full_graph_masked_decode_attention_enabled(values)
    compact_o_proj, compact_o_layers = full_graph_compact_o_proj_config(values)
    full_graph_compact_o_proj_min_rows(values)
    compact_q_proj = full_graph_compact_q_proj_enabled(values)
    mapped_decode_attention = full_graph_mapped_decode_attention_enabled(values)
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
        if compact_phases != {"decode"}:
            raise ValueError(
                f"{FD_LOW_ROW_POLICY_ENV} currently requires "
                f"{FD_COMPACT_PHASES_ENV}=decode"
            )
        if "decode" not in active_phases:
            raise ValueError(
                f"{FD_LOW_ROW_POLICY_ENV} requires decode in "
                f"{FD_ACTIVE_PHASES_ENV}"
            )
        if low_row_max_rows >= compact_min_rows:
            raise ValueError(
                f"{FD_LOW_ROW_MAX_ROWS_ENV} must be smaller than "
                f"{FD_COMPACT_MIN_ROWS_ENV}"
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
        if forced_route is not None or forced_all_run_fastpath:
            raise ValueError(
                f"{FD_CONDITIONAL_GRAPH_ENV}=1 requires dynamic routing"
            )
        if not str(values.get(FD_CONDITIONAL_GRAPH_HELPER_ENV, "")).strip():
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
    if forced_all_run_fastpath and forced_route is not True:
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
    compact_enabled, _, _, _ = full_graph_compact_config(values)
    if skipper_adapter.execution_kind == SUBLAYER_EXECUTION:
        if not device_route_tape:
            raise ValueError(
                f"{skipper_adapter.name} sublayer execution requires "
                f"{FD_DEVICE_ROUTE_TAPE_ENV}=1"
            )
        incompatible = {
            FD_EAGER_SEMANTIC_DEBUG_ENV: eager_semantic_debug,
            FD_CONDITIONAL_GRAPH_ENV: conditional_graph,
            FD_DEFER_PROJECT_KV_ENV: defer_project_kv,
            FD_COMPACT_ROUTED_QKV_ENV: compact_routed_qkv,
            FD_CONTIGUOUS_ROUTED_QKV_ENV: contiguous_routed_qkv,
            FD_COMPACT_ENABLED_ENV: compact_enabled,
            FD_VIRTUAL_COHORT_ENV: virtual_cohort,
            FD_WEIGHTED_SCATTER_ENV: weighted_scatter,
            FD_FUSED_EVIDENCE_ENV: fused_evidence,
            FD_COMPACT_O_PROJ_ENV: compact_o_proj,
            FD_COMPACT_Q_PROJ_ENV: compact_q_proj,
            FD_MAPPED_DECODE_ATTN_ENV: mapped_decode_attention,
            FD_SCHEDULER_CONVERGENCE_ENV: scheduler_convergence,
            FD_FORCED_ALL_RUN_FASTPATH_ENV: forced_all_run_fastpath,
            FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV: (
                forced_all_run_production_attention
            ),
            FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV: (
                conditional_production_all_run
            ),
        }
        enabled = sorted(name for name, active in incompatible.items() if active)
        if enabled:
            raise ValueError(
                f"{skipper_adapter.name} sublayer execution does not yet support "
                + ", ".join(enabled)
            )
        if forced_route is not None:
            raise ValueError(
                f"{skipper_adapter.name} sublayer execution does not support "
                f"{FD_FORCE_ROUTE_ENV}"
            )
        if full_graph_layer_policies(values):
            raise ValueError(
                f"{skipper_adapter.name} sublayer execution does not support "
                f"{FD_LAYER_POLICIES_ENV}"
            )
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
        if flexidepth_active_phases(values) != {"decode"}:
            raise ValueError(
                f"{FD_FUSED_EVIDENCE_ENV}=1 currently requires "
                f"{FD_ACTIVE_PHASES_ENV}=decode"
            )
    if compact_o_proj and not compact_enabled:
        raise ValueError(
            f"{FD_COMPACT_O_PROJ_ENV}=1 requires {FD_COMPACT_ENABLED_ENV}=1"
        )
    if compact_o_proj and not masked_decode_attention:
        raise ValueError(
            f"{FD_COMPACT_O_PROJ_ENV}=1 requires {FD_MASKED_DECODE_ATTN_ENV}=1"
        )
    if compact_q_proj and not compact_o_proj:
        raise ValueError(
            f"{FD_COMPACT_Q_PROJ_ENV}=1 requires {FD_COMPACT_O_PROJ_ENV}=1"
        )
    if mapped_decode_attention and not compact_o_proj:
        raise ValueError(
            f"{FD_MAPPED_DECODE_ATTN_ENV}=1 requires "
            f"{FD_COMPACT_O_PROJ_ENV}=1"
        )
    if mapped_decode_attention and not masked_decode_attention:
        raise ValueError(
            f"{FD_MAPPED_DECODE_ATTN_ENV}=1 requires "
            f"{FD_MASKED_DECODE_ATTN_ENV}=1"
        )
    # NOTE: the converse (MASKED=1 => the mask actually reaches a triton decode kernel)
    # is NOT enforceable here — this validator has no server_args/backend in scope, and
    # MASKED=1 with LAYER_ROUTED=0 is legitimate when an explicit --decode-attention-
    # backend=triton pin is present (several sealed repo configs do exactly that). The
    # guard therefore lives in ModelRunner._get_attention_backend, where the resolved
    # backend is visible. See _MASKED_DECODE_REQUIRED_BACKEND.
    unknown_compact_o_layers = sorted(set(compact_o_layers) - set(loaded_layers))
    if unknown_compact_o_layers:
        raise ValueError(
            f"{FD_COMPACT_O_PROJ_LAYERS_ENV} names unloaded layers: "
            + ", ".join(str(layer_id) for layer_id in unknown_compact_o_layers)
        )
    if contiguous_routed_qkv:
        configured_layers = set(routed_qkv_capacities)
        loaded_layer_set = set(loaded_layers)
        missing_layers = sorted(loaded_layer_set - configured_layers)
        unknown_layers = sorted(configured_layers - loaded_layer_set)
        if missing_layers:
            raise ValueError(
                f"{FD_ROUTED_QKV_CAPACITIES_ENV} omits loaded layers: "
                + ", ".join(str(layer_id) for layer_id in missing_layers)
            )
        if unknown_layers:
            raise ValueError(
                f"{FD_ROUTED_QKV_CAPACITIES_ENV} names unloaded layers: "
                + ", ".join(str(layer_id) for layer_id in unknown_layers)
            )
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
    if (
        forced_route is not None
        and flexidepth_execution_mode(values) != FD_EXECUTION_FULL_GRAPH
    ):
        raise ValueError(
            f"{FD_FORCE_ROUTE_ENV} requires "
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
    if tp_size != 1 or pp_size != 1:
        raise ValueError(
            "VP full_graph currently requires TP=1 and PP=1; "
            f"observed TP={tp_size}, PP={pp_size}"
        )
    if quant_config is not None:
        raise ValueError("VP full_graph currently requires unquantized weights")
