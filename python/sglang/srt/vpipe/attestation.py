"""Runtime attestation — prove what EXECUTED, not what was configured.

A set flag is not an active treatment and an unset flag is not an inactive
mechanism. These surfaces publish realised counters and route digests through
/server_info so a deployment's posture is evidence rather than inference.

Mutable counters must never enter the deployment identity hash: they advance
during warmup and capture, so a manifest snapshot and a later re-read can never
agree. The config-deterministic half stays in the identity; the counters stay
visible beside it."""

from __future__ import annotations

from sglang.srt.vpipe.design import (    
    moe_config_dir,
    SERVED_CONDITIONAL_GRAPH,
    mechanism,
    SERVED_MASKED_DECODE_ATTENTION,
)

from typing import Mapping, Sequence
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
import torch
from sglang.srt.vpipe.common import (
    regime_switch_config,
    resolved_design_attestation,
)
from sglang.srt.vpipe.env import (
    COMPACT_ACCOUNTING_CAPACITY_FRACTION,
    COMPACT_CAPACITY_MULTIPLE,
    FD_CONDITIONAL_BRANCH_COUNTERS_ENV,
    FD_CONDITIONAL_GRAPH_ENV,
    FD_CONDITIONAL_MAX_ROWS_ENV,
    FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV,
    FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV,
    FD_EXECUTION_FULL_GRAPH,
    FULL_GRAPH_CAPTURE_SYNTHETIC_RID_BASE,
    _VALUE_TYPED_CONFLICT_ENVS,
    _BINARY_COHORT_CONFIG_DIGEST,
    _BINARY_COHORT_CUBLAS_PASSES,
    _PREFILL_FALLBACK,
    _BINARY_COHORT_LAYERS,
    _BINARY_COHORT_STATS,
    _ROUTE_DECIDE_STATS,
    _MASKED_DECODE_REQUIRED_BACKEND,
    _VALID_DEFER_PROJECT_KV_DIAGNOSTIC_STAGES,
)
from sglang.srt.vpipe.kernel import (
    weighted_scatter,
)
from sglang.srt.vpipe.common import (
    resolve_full_graph_skipper,
)
from sglang.srt.vpipe.skipper import (
    route_digest_uses_logical_request_ids,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_phases,
    full_graph_gate_mode,
    full_graph_low_row_policy,
)
from sglang.srt.vpipe.common import (
    flexidepth_execution_mode,
)
from sglang.srt.vpipe.common import (
    flexidepth_active_phases,
)
from sglang.srt.vpipe.config import (
    full_graph_batched_commit_enabled,
    full_graph_commit_overlap_enabled,
    full_graph_compact_config,
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
from sglang.srt.vpipe.mlp_compact import (
    full_graph_dual_compact_min_rows,
)
from sglang.srt.vpipe.common import (
    full_graph_compact_routed_qkv_enabled,
)
from sglang.srt.vpipe.coverage import (
    coverage_dense_armed,
)
from sglang.srt.vpipe.coverage import (
    coverage_dense_counters,
)
from sglang.srt.vpipe.regime import (
    regime_switch_zero_counters,
)


_cached: Optional[bool] = None
_VP_ENV_PREFIXES = ("SGLANG_FD_", "SGLANG_VP_")
def regime_switch_attestation(
    environ: Optional[Mapping[str, str]] = None,
    *,
    counters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the ``regime_switch`` runtime-attestation block.

    Present on every endpoint (prefill-only and integrated).  When the switch
    is off the static block is ``{"enabled": False}`` so the OFF state stays
    diffable.  A nested ``counters`` sub-block carries per-phase-by-body pass
    counts (zeros placeholder until I5/I6 wire real counters); it is stripped
    from the deployment identity and must be omitted from expectation subsets.
    """

    config = regime_switch_config(environ)
    if config is None:
        block: dict[str, Any] = {"enabled": False}
    else:
        block = {
            "enabled": True,
            "version": config.version,
            "prefill": {
                "enabled": config.prefill.enabled,
                "min_tokens": config.prefill.min_tokens,
                "row_correction_alpha": config.prefill.row_correction_alpha,
                "include_mixed": config.prefill.include_mixed,
                # The engagement gate was CONFIGURED but never ATTESTED, so
                # /server_info could not confirm it and the launch gate could not check
                # it. A design parameter that cannot be read back is the exact hole that
                # let a rejected configuration run for eighteen hours.
                "engagement_min": config.prefill.engagement_min,
                "engagement_probe_every": config.prefill.engagement_probe_every,
            },
            "decode": {
                "enabled": config.decode.enabled,
                "enter_rows": config.decode.enter_rows,
                "exit_rows": config.decode.exit_rows,
                "low_body": config.decode.low_body,
                "high_body": config.decode.high_body,
            },
        }
        if config.decode.kv_criterion:
            # Lane-2 cut1: attested only when active, so rows-criterion
            # deployments keep their byte-identical block.
            block["decode"]["enter_kv_tokens"] = config.decode.enter_kv_tokens
            block["decode"]["exit_kv_tokens"] = config.decode.exit_kv_tokens
        # [D-849] per-request body pinning, as resolved for this arm.
        block["admission"] = {
            "enabled": config.admission.enabled,
            "criterion": config.admission.criterion,
            "cold_start": config.admission.cold_start,
            "prefill_demotion": config.admission.prefill_demotion,
            "mixed_step": config.admission.mixed_step,
        }
    block["counters"] = (
        regime_switch_zero_counters() if counters is None else counters
    )
    return block
def per_boot_ladder_hash(capture_bs) -> str:
    """Stable hash of the REALIZED capture ladder (boot-contingency evidence).

    Independent of how the ladder was derived: the runner's realized capture_bs
    is the coverage oracle, and this pins it per boot.
    """

    encoded = json.dumps([int(value) for value in capture_bs]).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def coverage_dense_runtime_attestation(model_runner: Any) -> Optional[dict[str, Any]]:
    """Build the ``fd_c3`` attestation block, or ``None`` when the mechanism
    is not armed (production and gate-OFF boots stay byte-identical)."""

    if not coverage_dense_armed():
        return None
    runner = model_runner.decode_cuda_graph_runner
    if runner is not None:
        capture_bs = [int(value) for value in runner.capture_bs]
        decode_capture_bs_max = int(runner.max_bs)
        ladder_hash = per_boot_ladder_hash(capture_bs)
    else:
        # Fail-closed dense boot (graphs disabled / runner absent): the arm is
        # VOID for coverage cells; attest the absence loudly (R-E).
        capture_bs = []
        decode_capture_bs_max = None
        ladder_hash = None
    pool_size = int(model_runner.req_to_token_pool.size)
    coverage_ratio = (
        decode_capture_bs_max / pool_size
        if decode_capture_bs_max is not None and pool_size
        else 0.0
    )
    counters = coverage_dense_counters()
    return {
        "enabled": True,
        "kill_switch_env": "SGLANG_VP_COVERAGE_DENSE",
        # The REALIZED ladder only. The self-sizing derivation and its
        # provenance (ladder_source, target_max_bs, stop_reason, trim events,
        # reserve) were removed: that feature was opt-in, unexercised by any
        # arm, and its defaults attested "self_sized" on boots that never
        # derived one. What remains is measured from the runner, which is the
        # coverage oracle by construction.
        "ladder": {
            "decode_capture_bs_max": decode_capture_bs_max,
            "capture_bs": capture_bs,
            "per_boot_ladder_hash": ladder_hash,
            "req_to_token_pool_size": pool_size,
            "coverage_ratio": coverage_ratio,
            "cuda_graph_padding_enabled": (
                not model_runner.server_args.disable_cuda_graph_padding
            ),
        },
        "counters": {
            "eager_skip_decode_layer_calls": counters.eager_skip_decode_layer_calls,
            "dense_overflow_decisions": counters.dense_overflow_decisions,
            "dense_overflow_rows": counters.dense_overflow_rows,
            "dense_body_passes": counters.dense_body_passes,
            "dense_body_rows": counters.dense_body_rows,
            "overflow_reason": dict(counters.overflow_reason),
            "fd_tokens_skip_body": counters.fd_tokens_skip_body,
            "fd_tokens_prod_allrun_band": counters.fd_tokens_prod_allrun_band,
            "fd_tokens_dense_overflow": counters.fd_tokens_dense_overflow,
            "dispatch_graph_steps": counters.dispatch_graph_steps,
            "regime_observe_eager": counters.regime_observe_eager,
            "coverage_stamp_w1_composed": counters.coverage_stamp_w1_composed,
            "graph_lifecycle_sections": counters.graph_lifecycle_sections,
            "recapture_events": list(counters.recapture_events),
        },
    }
def scheduler_runtime_attestation(scheduler: Any) -> dict[str, Any]:
    model_runner = scheduler.tp_worker.model_runner
    model = model_runner.model
    model_attestation = getattr(model, "vp_runtime_attestation", None)
    if callable(model_attestation):
        model_state = model_attestation()
    else:
        model_state = {
            "model_family": type(model).__name__,
            "flexidepth": {
                "loaded": False,
                "loaded_layers": [],
                "execution": "disabled",
            },
        }

    model_flexidepth = model_state.get("flexidepth", {})
    v3_full_graph_enabled = str(
        model_flexidepth.get("execution", "")
    ).startswith("full_graph_")
    v3_full_graph = None
    if v3_full_graph_enabled:

        v3_full_graph = model_runner_runtime_attestation(
            model_runner,
            loaded_layer_count=len(model_flexidepth.get("loaded_layers", ())),
        )



    # [W1] Thread real per-body regime-switch pass counts into the (identity-
    # stripped) regime_switch.counters block when the model exposes them; falls
    # back to zeros for models/paths without the W1 forward integration.
    model_regime_counters = getattr(model, "vp_regime_switch_counters", None)
    regime_counters = (
        model_regime_counters() if callable(model_regime_counters) else None
    )
    # The prefill leg (I5) counts on the model; the decode leg (I6) counts on the
    # decode cuda-graph runner (which owns the hysteresis + graph dispatch).
    # Overlay the runner's per-body decode counts into regime_switch.counters.decode.
    decode_graph_runner = getattr(model_runner, "decode_cuda_graph_runner", None)
    decode_regime_counters = getattr(
        decode_graph_runner, "vp_regime_switch_decode_counters", None
    )
    if regime_counters is not None and callable(decode_regime_counters):
        decode_counts = decode_regime_counters()
        if decode_counts is not None:
            regime_counters["decode"] = decode_counts
    # [P4] The prefill leg dispatches at the prefill cuda-graph runner (the
    # model-side stamp only executes on eager passes), so overlay the runner's
    # replay-level per-body counts additively onto the stamp counts.
    prefill_graph_runner = getattr(model_runner, "prefill_cuda_graph_runner", None)
    prefill_variant_counters = getattr(
        prefill_graph_runner, "vp_regime_switch_prefill_counters", None
    )
    if regime_counters is not None and callable(prefill_variant_counters):
        prefill_counts = prefill_variant_counters()
        if prefill_counts is not None:
            for body, count in prefill_counts.items():
                regime_counters["prefill"][body] += count
    # [ ->] These counters read {dense: 0, fd: 0} in every campaign cell
    # because the runner's variant machinery was gated on an env var had
    # deleted (see prefill_cuda_graph_runner.__init__). With the gate on the design
    # predicate the runner overlay above carries the real per-body pass counts, so
    # the block is reported again -- and a zero here on a routed arm is a defect.
    # [R2] Realized engagement evidence of the prefill escape gate (EMA,
    # folded samples, forced synchronisations) — runtime evidence inside the
    # identity-stripped counters block, so gates can compare it across trees.
    prefill_engagement = getattr(
        prefill_graph_runner, "vp_regime_switch_prefill_engagement", None
    )
    if regime_counters is not None and callable(prefill_engagement):
        engagement = prefill_engagement()
        if engagement is not None:
            regime_counters["prefill_engagement"] = engagement
    # [D-849] Model-runner-side pinned-dispatch evidence (partitioned steps,
    # rows per sub-pass, pin violations) overlays the zero placeholder.
    split_counters = getattr(model_runner, "vp_admission_split_counters", None)
    if regime_counters is not None and callable(split_counters):
        split_counts = split_counters()
        if split_counts is not None:
            regime_counters["admission"] = split_counts

    # [D-849] Scheduler-side admission evidence: pins handed out, band flips seen
    # by the scheduler's mirror, deferrals, mixed steps, retract re-entries and
    # the cross-body prefix-reuse witness (MUST be 0). Runtime evidence,
    # identity-stripped like the regime counters; never assert in expectations.
    pinner_counts = scheduler.vp_pinner.counters()
    admission_pins = {"enabled": pinner_counts is not None}
    if pinner_counts is not None:
        admission_pins.update(pinner_counts)
        admission_pins.update(
            {
                "retract_reentries": int(scheduler.vp_pin_retract_reentries),
                "admission_deferrals": int(scheduler.vp_pin_admission_deferrals),
                "mixed_steps": int(scheduler.vp_pin_mixed_steps),
                "mixed_step_rows_stock": int(scheduler.vp_pin_mixed_step_rows_stock),
                "mixed_step_rows_fd": int(scheduler.vp_pin_mixed_step_rows_fd),
                "mix_withheld": int(scheduler.vp_pin_mix_withheld),
                "cross_body_prefix_reuse": int(scheduler.vp_pin_cross_body_prefix_reuse),
                "prefix_hit_tokens_stock": int(scheduler.vp_pin_prefix_hit_tokens_stock),
                "prefix_hit_tokens_fd": int(scheduler.vp_pin_prefix_hit_tokens_fd),
            }
        )

    result = {
        "schema_version": 1,
        "v3_full_graph_enabled": v3_full_graph_enabled,
        "v3_full_graph": v3_full_graph,
        "regime_switch": regime_switch_attestation(counters=regime_counters),
        # Host-side per-step batch composition from Scheduler.run_batch —
        # counts replayed and eager passes alike (model-side counters only
        # see eager passes). Runtime evidence, identity-stripped like the
        # regime counters; never assert in expectations.
        "batch_composition": {
            "prefill_passes": int(scheduler.vp_bc_prefill_passes),
            "prefill_tokens": int(scheduler.vp_bc_prefill_tokens),
            "mixed_passes": int(scheduler.vp_bc_mixed_passes),
            "mixed_decode_rows": int(scheduler.vp_bc_mixed_decode_rows),
            "mixed_prefill_tokens": int(scheduler.vp_bc_mixed_prefill_tokens),
            "decode_passes": int(scheduler.vp_bc_decode_passes),
        },
        "admission_pins": admission_pins,
        "model": model_state,
        # [, Codex F7/F8] TOP-LEVEL and RESOLVED. The first attempt put these inside
        # the seam's dict, which lands under "model" -- so the gate looked one level too
        # shallow and would have rejected every correct server. And it published the
        # DECLARED constants, which the gate then compared against the same constants: a
        # tautology. This is what the resolvers actually returned this boot.
        "served_design": resolved_design_attestation(),
    }
    # (c3) coverage-dense evidence block (design 2026-08-04). Emitted ONLY when
    # the mechanism is armed (FD skip-decode deployed AND the
    # SGLANG_VP_COVERAGE_DENSE kill switch ON — the F14 joint scope), so
    # production and gate-OFF boots keep a byte-identical attestation. The
    # bidirectional launch gate demands: expectation attests fd_c3.enabled
    # <=> the boot armed it.
    fd_c3 = coverage_dense_runtime_attestation(model_runner)
    if fd_c3 is not None:
        result["fd_c3"] = fd_c3
    return result
class ReqVPMixin:
    def init_vp(self: "Req", enabled: bool = False) -> None:
        self.vp_enabled: bool = enabled
        # block-position state (advanced by VPDecodeManager); valid only when vp_enabled.
        self.current_block: int = 0
        self.skip_blocks_remaining: int = 0
        # [D-849] The body this request is pinned to for its whole lifetime
        # ("stock" | "fd"), decided once at admission by the scheduler's
        # AdmissionPinner; None until admitted (or forever when pinning is off).
        # Survives a retract/re-admit round trip: the pin is never re-decided.
        self.vp_body: Optional[str] = None


def _env_scan() -> bool:
    for key, value in os.environ.items():
        if key.startswith(_VP_ENV_PREFIXES) and str(value).strip():
            return True
    return False
def vp_runtime_enabled() -> bool:
    """One boot-cached answer to "is any VP/FD machinery requested?"."""

    global _cached
    if _cached is None:
        _cached = _env_scan()
    return _cached
def full_graph_conditional_graph_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether decode uses sequential device conditional stages.

    A design constant -- but a DECODE-phase one. The prefill-only arm has no
    decode phase, so asserting it there makes the arm unservable (validation rejects
    "conditional graph without decode", correctly). The old arm_env_vpre_binarycohort.sh
    simply never exported it; that phase-dependence was implicit in the scripts and has
    to become explicit now that the scripts are gone.
    """

    value = str(
        "1" if mechanism(SERVED_CONDITIONAL_GRAPH, decode_only=True) else "0"
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_CONDITIONAL_GRAPH_ENV} must be a boolean value")
def full_graph_conditional_production_all_run_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Use production attention and dense MLP for device-proven all-RUN stages."""

    values = os.environ if environ is None else environ
    value = str(
        values.get(FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV, "0")
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"{FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV} must be a boolean value"
    )
def full_graph_conditional_max_rows(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[int]:
    """Return the largest row bucket assigned to conditional stage graphs."""

    values = os.environ if environ is None else environ
    raw_value = str(values.get(FD_CONDITIONAL_MAX_ROWS_ENV, "") or "").strip()
    if not raw_value:
        return None
    try:
        max_rows = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{FD_CONDITIONAL_MAX_ROWS_ENV} must be a positive integer"
        ) from error
    if max_rows <= 0:
        raise ValueError(
            f"{FD_CONDITIONAL_MAX_ROWS_ENV} must be a positive integer"
        )
    return max_rows
def full_graph_defer_project_kv_diagnostic_stage(
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Return the cumulative repair stage used by correctness diagnostics."""

    values = os.environ if environ is None else environ
    stage = str(
        values.get(FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV, "full")
    ).strip().lower()
    if stage not in _VALID_DEFER_PROJECT_KV_DIAGNOSTIC_STAGES:
        choices = ", ".join(
            sorted(_VALID_DEFER_PROJECT_KV_DIAGNOSTIC_STAGES)
        )
        raise ValueError(
            f"{FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV} must be one of "
            f"{choices}; got {stage!r}"
        )
    return stage
def full_graph_capture_synthetic_request_ids(
    max_bs: int, device: Any = None
) -> torch.Tensor:
    """Deterministic stable request-id placeholders for graph capture.

    CUDA-graph capture dummy batches carry no real requests, but with
    ``SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_DIGEST=1`` the route-digest/tape
    plumbing keys per-request evidence by ``forward_batch.rids_int`` and
    :func:`_route_policy_request_ids` fails closed without it (the 2026-08-05
    V-rich-c3 boot death during prefill capture). Capture only needs int64
    identities of the right shape wired through static buffers so the baked
    digest plumbing is valid; replay supplies the live hashed rids.
    """

    if max_bs <= 0:
        raise ValueError(
            "full-graph capture synthetic request ids need max_bs >= 1"
        )
    return FULL_GRAPH_CAPTURE_SYNTHETIC_RID_BASE - torch.arange(
        max_bs, dtype=torch.int64, device=device
    )
def binary_cohort_attestation() -> dict:
    """Realized binary_cohort evidence for the runtime attestation."""

    realized = {}
    for device, stats in _BINARY_COHORT_STATS.items():
        calls, run_rows, project_rows = (
            int(value) for value in stats.detach().cpu().tolist()
        )
        total = run_rows + project_rows
        realized[str(device)] = {
            "executor_calls": calls,
            "run_rows": run_rows,
            "project_rows": project_rows,
            "engagement": (project_rows / total) if total else None,
        }
    #: per-layer prefill break-even decisions (eager passes) are realized
    # counters too, so they live under `realized` (stripped from the identity).
    for device, (checked, fallen) in _PREFILL_FALLBACK.items():
        block = realized.setdefault(
            str(device),
            {
                "executor_calls": 0,
                "run_rows": 0,
                "project_rows": 0,
                "engagement": None,
            },
        )
        block["prefill_fallback_layers_checked"] = int(checked)
        block["prefill_fallback_layers_fallen"] = int(fallen)
    # Lane-2 Track B / F1: fused route decisions EXECUTED (device counter
    # incremented inside the kernel, so graph replays are counted) -- realized
    # evidence, never a flag. Zero here with routed layers active means the
    # unfused path ran (forced/explicit routes or no device tape).
    for device, stats in _ROUTE_DECIDE_STATS.items():
        block = realized.setdefault(
            str(device),
            {
                "executor_calls": 0,
                "run_rows": 0,
                "project_rows": 0,
                "engagement": None,
            },
        )
        block["route_decide_fused_calls"] = int(stats.detach().cpu().item())
    return {
        "enabled": bool(_BINARY_COHORT_LAYERS),
        "layers": sorted(_BINARY_COHORT_LAYERS),
        "config_artifacts": dict(_BINARY_COHORT_CONFIG_DIGEST),
        "realized": realized,
        "prefill_cublas": {
            device: {"passes": passes, "rows": rows}
            for device, (passes, rows) in _BINARY_COHORT_CUBLAS_PASSES.items()
        },
    }
def _conflicting_env_enabled(name: str, value: Optional[str]) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    if not normalized:
        return False
    if name in _VALUE_TYPED_CONFLICT_ENVS:
        # A value-typed knob (a path, a list, an integer) counts as set for
        # any non-off value; an explicit off value is allowed (round-3
        # lesson: never refuse an explicit disable).
        return normalized not in {"0", "false", "no", "off"}
    return normalized in {"1", "true", "yes", "on"}
def full_graph_conditional_branch_counters_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether correctness smokes count selected graph bodies."""

    values = os.environ if environ is None else environ
    value = str(
        values.get(FD_CONDITIONAL_BRANCH_COUNTERS_ENV, "0")
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"{FD_CONDITIONAL_BRANCH_COUNTERS_ENV} must be a boolean value"
    )
def full_graph_compact_evidence_specs(
    loaded_layers: tuple[int, ...],
    environ: Optional[Mapping[str, str]] = None,
) -> list[tuple[float, float, float]]:
    """Encode compact accounting without materializing per-layer tensors."""

    values = os.environ if environ is None else environ
    compact_enabled = full_graph_compact_config(values)
    policies = full_graph_layer_policies(values)
    specs: list[tuple[float, float, float]] = []
    for layer_id in loaded_layers:
        mode = 0.0
        run_fraction = -1.0
        project_fraction = -1.0
        policy = policies.get(layer_id)
        if compact_enabled and policy is not None:
            name, run_value, project_value = policy
            if name == "project_base":
                mode, run_fraction = 1.0, float(run_value)
            elif name == "project_filtered_run_compact":
                mode, run_fraction = 1.0, float(run_value)
            elif name == "run_base":
                mode, project_fraction = 1.0, float(project_value)
            elif name == "dual_compact":
                mode = 1.0
                run_fraction = float(run_value)
                project_fraction = float(project_value)
        elif compact_enabled and not policies:
            mode = 2.0
            run_fraction = COMPACT_ACCOUNTING_CAPACITY_FRACTION
            project_fraction = COMPACT_ACCOUNTING_CAPACITY_FRACTION
        specs.append((mode, run_fraction, project_fraction))
    return specs
def record_model_runner_dispatch(
    model_runner: Any, forward_batch: Any, can_run_graph: bool
) -> None:
    """Record host-visible whole-step graph coverage using logical decode rows."""

    if flexidepth_execution_mode() != FD_EXECUTION_FULL_GRAPH:
        return
    forward_mode = getattr(forward_batch, "forward_mode", None)
    is_decode = getattr(forward_mode, "is_decode", None)
    if not callable(is_decode) or not is_decode():
        return
    rows = int(getattr(forward_batch, "batch_size", 0))
    model_runner._fd_full_graph_decode_dispatches = (
        getattr(model_runner, "_fd_full_graph_decode_dispatches", 0) + 1
    )
    model_runner._fd_full_graph_decode_rows = (
        getattr(model_runner, "_fd_full_graph_decode_rows", 0) + rows
    )
    suffix = "graph" if can_run_graph else "eager"
    dispatch_name = f"_fd_full_graph_decode_{suffix}_dispatches"
    row_name = f"_fd_full_graph_decode_{suffix}_rows"
    setattr(model_runner, dispatch_name, getattr(model_runner, dispatch_name, 0) + 1)
    setattr(model_runner, row_name, getattr(model_runner, row_name, 0) + rows)
def low_row_policy_attestation(
    policy: Optional[str] = None,
) -> dict[str, Any]:
    """The routed-MLP body posture, as a standalone (testable) evidence block.

    `native_dense` (lane-2 cut3 item 1) means the routed MLP is the
    model's OWN dense feed-forward at every occupancy plus the projector,
    selected by the route mask — no count-adaptive or grouped dispatch is
    reachable from such a deployment. Reported so the posture is read off
    `/server_info` rather than inferred from the environment.
    """

    if policy is None:
        policy = full_graph_low_row_policy()
    gate_mode = full_graph_gate_mode()
    return {
        "enabled": policy != "off",
        "policy": policy,
        "active_phases": ["decode"] if policy != "off" else [],
        "logical_routes_preserved": True,
        "physical_work": (
            "dense_and_project_all_graph_rows"
            if policy == "native_dense"
            else "unchanged"
        ),
        "flop_savings_claim_allowed": policy == "off",
        "applies_to_all_rows": policy == "native_dense",
        "routed_mlp_kernels": (
            "model_native_dense_only"
            if policy == "native_dense"
            else "policy_dispatch"
        ),
        # The gate arithmetic the routed MLP actually applied, so a reader of
        # /server_info can tell whether a checkpoint was served under the gate
        # it was trained with instead of inferring it from the environment.
        "gate_mode": gate_mode,
        "gate_branch_scaling": (
            "run_times_w_project_times_one_minus_w"
            if gate_mode == "released"
            else "hard_selection_no_w_scaling"
        ),
    }
def model_runner_runtime_attestation(
    model_runner: Any, *, loaded_layer_count: int
) -> dict[str, Any]:
    from sglang.srt.vpipe.common import (
        fdvp_fused_project_input_enabled,
        fdvp_fused_project_input_shared_storage_enabled,
    )

    total_dispatches = getattr(model_runner, "_fd_full_graph_decode_dispatches", 0)
    total_rows = getattr(model_runner, "_fd_full_graph_decode_rows", 0)
    graph_dispatches = getattr(
        model_runner, "_fd_full_graph_decode_graph_dispatches", 0
    )
    graph_rows = getattr(model_runner, "_fd_full_graph_decode_graph_rows", 0)
    eager_dispatches = getattr(
        model_runner, "_fd_full_graph_decode_eager_dispatches", 0
    )
    eager_rows = getattr(model_runner, "_fd_full_graph_decode_eager_rows", 0)
    coverage = 100.0 * graph_rows / total_rows if total_rows else 0.0
    decode_config = model_runner.server_args.cuda_graph_config.decode
    prefill_config = model_runner.server_args.cuda_graph_config.prefill
    backend = getattr(decode_config.backend, "value", str(decode_config.backend))
    prefill_backend = getattr(
        prefill_config.backend, "value", str(prefill_config.backend)
    )
    # host path from the committed config
    external_moe_config = bool(moe_config_dir().strip())
    skipper_adapter = resolve_full_graph_skipper()
    skipper_attestation = skipper_adapter.attestation()
    skipper_attestation["routed_layer_count"] = loaded_layer_count
    if "skipped_depth_ratio" in skipper_attestation:
        skipper_attestation["project_layer_count"] = min(
            loaded_layer_count,
            math.ceil(
                skipper_attestation["skipped_depth_ratio"]
                * loaded_layer_count
            ),
        )
    compact_enabled = full_graph_compact_config()
    low_row_policy = full_graph_low_row_policy()
    virtual_cohort = full_graph_virtual_cohort_enabled()
    weighted_scatter = full_graph_weighted_scatter_enabled()
    fused_evidence = full_graph_fused_evidence_enabled()
    forced_all_run_fastpath = full_graph_forced_all_run_fastpath_enabled()
    forced_all_run_production_attention = (
        full_graph_forced_all_run_production_attention_enabled()
    )
    masked_decode_attention_configured = (
        full_graph_masked_decode_attention_enabled()
    )
    masked_decode_attention = (
        masked_decode_attention_configured and not forced_all_run_fastpath
    )
    # The run mask is read only by the triton decode kernel
    # (triton_backend.py -> decode_attention.py row_active gating); flashinfer has no
    # fd_full_graph reader. A configured mask on any other decode backend is INERT,
    # and because 653c21ec7f elided the Python-side multiply it is also UNSAFE: jump
    # rows keep a non-zero attention output that post_attention_layernorm folds into
    # the residual. Resolve the effective state once, here, so the attestation reports
    # what ran rather than what was requested.
    _masked_decode_active_backend = getattr(
        model_runner, "decode_attention_backend_str", None
    )
    _masked_decode_effective = (
        masked_decode_attention
        and _masked_decode_active_backend == _MASKED_DECODE_REQUIRED_BACKEND
    )
    compact_phases = sorted(full_graph_compact_phases())
    route_accounting_enabled = full_graph_route_accounting_enabled()
    layer_counters_enabled = full_graph_layer_counters_enabled()
    device_route_tape_enabled = full_graph_device_route_tape_enabled()
    device_route_digest_enabled = full_graph_device_route_digest_enabled()
    logical_route_digest = (
        device_route_digest_enabled
        and route_digest_uses_logical_request_ids(skipper_adapter)
    )
    scheduler_convergence_enabled = full_graph_scheduler_convergence_enabled()
    eager_semantic_debug = full_graph_eager_semantic_debug_enabled()
    prefill_grouped_mlp = full_graph_prefill_grouped_mlp_enabled()
    conditional_graph_enabled = full_graph_conditional_graph_enabled()
    conditional_production_all_run = (
        full_graph_conditional_production_all_run_enabled()
    )
    deferred_project_kv = full_graph_defer_project_kv_enabled()
    deferred_project_kv_stage = (
        full_graph_defer_project_kv_diagnostic_stage()
    )
    deferred_project_kv_semantic = (
        not deferred_project_kv or deferred_project_kv_stage == "full"
    )
    commit_overlap = full_graph_commit_overlap_enabled()
    batched_commit = full_graph_batched_commit_enabled()
    compact_routed_qkv = full_graph_compact_routed_qkv_enabled()
    decode_graph_runner = getattr(model_runner, "decode_cuda_graph_runner", None)
    decode_graph_backend = getattr(decode_graph_runner, "backend", None)
    conditional_attestation = getattr(decode_graph_backend, "attestation", None)
    conditional_graph_state = (
        conditional_attestation()
        if conditional_graph_enabled and callable(conditional_attestation)
        else {
            "enabled": conditional_graph_enabled,
            "backend": "unavailable" if conditional_graph_enabled else "disabled",
        }
    )
    scheduler_kv_completion = (
        "graph_side_project_compute_overlapped_cache_commit_graph_end_fence"
        if deferred_project_kv and deferred_project_kv_semantic and commit_overlap
        else "graph_side_project_compute_joined_cache_commit_before_logits"
        if deferred_project_kv and deferred_project_kv_semantic
        else "diagnostic_incomplete_project_kv"
        if deferred_project_kv
        else "inline_after_own_layer_attention"
        if scheduler_convergence_enabled
        else "implicit_inline"
    )
    layer_policies = full_graph_layer_policies()
    active_phases = sorted(flexidepth_active_phases())
    prefill_decision_granularity = skipper_attestation["decision_granularity"]
    prefill_decision = skipper_attestation["decision"]
    return {
        "enabled": flexidepth_execution_mode() == FD_EXECUTION_FULL_GRAPH,
        "eager_semantic_debug": eager_semantic_debug,
        "whole_step_cuda_graph_required": not eager_semantic_debug,
        "conditional_graph": conditional_graph_state,
        "conditional_production_all_run": {
            "enabled": conditional_production_all_run,
            "predicate": "all_valid_rows_run_device_int32",
            "route_mask_graph_key": False,
            "host_route_readback": False,
            "attention": (
                "production_flashinfer_full_exact"
                if conditional_production_all_run
                else "masked_routed_attention"
            ),
            "mlp": (
                "dense_exact_weighted"
                if conditional_production_all_run
                else "filtered_run_only"
            ),
            "mixed_body": "unchanged_dynamic_route_policy",
            "kv_completion": "inline_all_rows_own_layer_projection",
        },
        "decode_backend": backend,
        "decode_graph_max_batch_size": getattr(decode_config, "max_bs", None),
        "prefill_backend": prefill_backend,
        "attention_backends": {
            "decode": getattr(
                model_runner, "decode_attention_backend_str", None
            ),
            "prefill": getattr(
                model_runner, "prefill_attention_backend_str", None
            ),
        },
        "active_phases": active_phases,
        "skipper_adapter": skipper_attestation,
        "prefill_execution": (
            "grouped_variable_cohort"
            if "prefill" in active_phases and prefill_grouped_mlp
            else "masked_fixed_topology"
            if "prefill" in active_phases
            else "vanilla"
        ),
        "prefill_routing": {
            "enabled": "prefill" in active_phases,
            "granularity": (
                prefill_decision_granularity
                if "prefill" in active_phases
                else "disabled"
            ),
            "decision": (
                prefill_decision
                if "prefill" in active_phases
                else "disabled"
            ),
            "kv_completion": (
                "own_layer_projection_all_tokens"
                if "prefill" in active_phases
                else "vanilla"
            ),
        },
        **(
            {
                "prefill_grouped_mlp": {
                    "enabled": True,
                    "scope": "prefill_only",
                    "partition": "dynamic_run_project_token_cohorts",
                    "execution": "native_dense_and_projector_gemm",
                    "merge": "weighted_index_copy",
                    "partition_sync": (
                        "two_cuda_nonzero_dynamic_shape_syncs_per_routed_layer"
                    ),
                    "filtered_moe_kernels": 0,
                    "attention": "unchanged_full_batch",
                    "kv_completion": "own_layer_projection_all_tokens",
                    "decode_execution": "unchanged",
                }
            }
            if prefill_grouped_mlp
            else {}
        ),
        "fixed_topology": not prefill_grouped_mlp,
        "weighted_scatter": {
            "enabled": weighted_scatter,
            "weight_source": "physical_route_row",
            "materialized_packed_weights": not weighted_scatter,
        },
        "fused_evidence": {
            "enabled": fused_evidence,
            "aggregation_kernels_per_replay": 2 if fused_evidence else None,
            "materialized_inline_kv_readiness": not fused_evidence,
            "readiness_proof": (
                "stream_order_after_all_own_layer_attention"
                if fused_evidence
                else "per_layer_valid_row_matrix"
            ),
            # Lane-2 cut3 item 4 : the fused kernel is decode-layout and
            # row-capped, so the mode is chosen PER PASS. A both-phase
            # deployment therefore carries BOTH evidence definitions — fused on
            # decode passes within the cap, the per-pass accumulators on prefill
            # passes and on any decode pass above it. Stated here so a readout
            # cannot silently attribute one definition to the whole run.
            "scope": (
                "decode_passes_within_row_cap"
                if fused_evidence
                else "all_passes"
            ),
            "row_cap": 1024 if fused_evidence else None,
            "unfused_fallback": (
                "prefill_passes_and_decode_above_row_cap"
                if fused_evidence
                else None
            ),
        },
        "device_resident_routes": True,
        "full_batch_attention": True,
        "kv_complete": deferred_project_kv_semantic,
        "route_accounting_enabled": route_accounting_enabled,
        "device_route_tape": {
            "enabled": device_route_tape_enabled,
            "storage": (
                "graph_static_layer_by_row_bool"
                if device_route_tape_enabled
                else "capture_time_tensor_references"
            ),
            "action_codes": {"project_only": 0, "run": 1},
            "storage_action_codes": {"project_only": 0, "run": 1},
            "logical_action_codes": skipper_attestation[
                "logical_action_codes"
            ],
            "policy_row_identity": (
                skipper_attestation.get("online_state_key")
                or "stable_request_hash_token_epoch"
                if (
                    logical_route_digest
                    or skipper_adapter.requires_stable_request_ids
                )
                else "hidden_state_row"
            ),
            "row_identity": (
                "stable_request_hash_token_epoch_layer_action"
                if logical_route_digest
                else "request_slot_token_epoch_cache_position"
                if device_route_digest_enabled
                else "disabled"
            ),
            "digest_enabled": device_route_digest_enabled,
            "digest_algorithm": (
                "batching_invariant_logical_action_int64_weighted_"
                "fingerprint"
                if logical_route_digest
                else "ordered_dual_int64_weighted_fingerprint"
                if device_route_digest_enabled
                else None
            ),
            "hot_path_host_readback": False,
            "hot_path_route_host_syncs": 0,
            "kv_completion": (
                "foreground_run_plus_graph_side_project_compute_overlapped_cache_commit"
                if deferred_project_kv
                and deferred_project_kv_semantic
                and commit_overlap
                else "foreground_run_plus_graph_side_project_compute_joined_cache_commit"
                if deferred_project_kv and deferred_project_kv_semantic
                else "diagnostic_incomplete_project_kv"
                if deferred_project_kv
                else "inline_all_rows_own_layer_projection"
            ),
            "repair_input_storage": (
                "per_routed_stage_graph_static_full_shape"
                if deferred_project_kv
                else "not_materialized_until_grouped_repair"
            ),
            "repair_projection": (
                "mapped_project_rows_kv_columns_only_above_min_rows"
                if deferred_project_kv and compact_routed_qkv
                else "all_rows_released_shape"
                if deferred_project_kv
                else "inline"
            ),
            **(
                {
                    "repair_diagnostic_stage": deferred_project_kv_stage,
                    "repair_semantic_kv_complete": (
                        deferred_project_kv_semantic
                    ),
                }
                if deferred_project_kv
                else {}
            ),
        },
        "scheduler_full_graph_convergence": {
            "enabled": scheduler_convergence_enabled,
            "scheduler_owner": (
                "req_pool_indices" if scheduler_convergence_enabled else "disabled"
            ),
            "token_epoch": (
                "scheduler_logical_token_position"
                if scheduler_convergence_enabled
                else "disabled"
            ),
            "cache_position": (
                "out_cache_loc" if scheduler_convergence_enabled else "disabled"
            ),
            "readiness_key": (
                "request_slot_token_epoch_layer_cache_position"
                if scheduler_convergence_enabled
                else "disabled"
            ),
            "kv_completion": scheduler_kv_completion,
            "per_layer_host_dispatches": 0,
            "graph_key_depends_on_route_mask": False,
            "deferred_repair": deferred_project_kv,
            "commit_overlap": commit_overlap,
            "commit_batching": (
                "single_cross_layer_kernel_launch_per_step"
                if batched_commit
                else "per_layer_masked_pool_writes"
            ),
            "repair_topology": (
                "route_prefix_fork_side_compute_overlap_commit_suffix_evidence_join"
                if deferred_project_kv
                and deferred_project_kv_semantic
                and commit_overlap
                else "route_prefix_fork_side_compute_join_cache_commit_suffix"
                if deferred_project_kv and deferred_project_kv_semantic
                else "route_prefix_fork_diagnostic_side_compute_suffix_join"
                if deferred_project_kv
                else "disabled"
            ),
        },
        "conditional_kernel": (
            "forced_all_run_dense_exact"
            if forced_all_run_fastpath
            else "layer_policy_virtual_cohort_swiglu"
            if compact_enabled and layer_policies and virtual_cohort
            else "layer_policy_mixed_exact_conditional"
            if compact_enabled and layer_policies
            else (
                "mapped_bounded_compact_cublas_with_filtered_overflow"
                if compact_enabled
                else "triton_one_expert_pair"
            )
        ),
        "forced_all_run_fastpath": {
            "enabled": forced_all_run_fastpath,
            "route": "all_run" if forced_all_run_fastpath else None,
            "attention": (
                "production_flashinfer_full_exact"
                if forced_all_run_production_attention
                else "full_exact_no_route_compaction"
                if forced_all_run_fastpath
                else "dynamic_route_policy"
            ),
            "mlp": (
                "dense_exact_weighted"
                if forced_all_run_fastpath
                else "dynamic_route_policy"
            ),
        },
        "bounded_compact": {
            "enabled": compact_enabled,
            "active_phases": compact_phases,
            # These three were env knobs; they are now fixed
            # constants (accounting/allocation, never policy).
            "capacity_fraction": COMPACT_ACCOUNTING_CAPACITY_FRACTION,
            "capacity_multiple": COMPACT_CAPACITY_MULTIPLE,
            "dual_compact_min_rows": full_graph_dual_compact_min_rows(),
            # Honest-evidence flag (it3/it4 review finding): with a
            # per-bucket threshold active, sub-threshold passes run the
            # fused body while route-derived compact accounting still
            # encodes the layer as compact — coverage/overflow evidence is
            # APPROXIMATE for those buckets until bucket-aware accounting
            # lands. Analysis must not treat compact coverage as exact
            # when this flag is true.
            "sub_threshold_accounting_approximate": (
                full_graph_dual_compact_min_rows() > 0
                or any(
                    policy[0] == "project_filtered_run_compact"
                    and (policy[2] or 1.0) > 1.0
                    for policy in full_graph_layer_policies().values()
                )
            ),
        },
        "low_row_policy": low_row_policy_attestation(low_row_policy),
        "virtual_cohort": {
            "enabled": virtual_cohort,
            "mapping": "device_row_map",
            "explicit_hidden_gather": not virtual_cohort,
            "explicit_route_weight_gather": not virtual_cohort,
            "explicit_output_scatter": not virtual_cohort,
            "packed_activation_intermediate": virtual_cohort,
        },
        "masked_decode_attention": {
            "enabled": masked_decode_attention,
            "configured": masked_decode_attention_configured,
            "required_backend": _MASKED_DECODE_REQUIRED_BACKEND,
            "active_backend": _masked_decode_active_backend,
            # Only the triton decode kernel reads fd_full_graph_attention_run_mask;
            # flashinfer has no reader, so a mask configured on any other backend is
            # INERT. These three fields must therefore report the EFFECTIVE state, not
            # the env flag: keying them on the flag alone asserted suppression on an
            # arm that suppressed nothing, and that is what let a broken config pass a
            # whole campaign (2026-08-05 v_rich_c3).
            "effective": _masked_decode_effective,
            "backend_mismatch": (
                masked_decode_attention
                and _masked_decode_active_backend != _MASKED_DECODE_REQUIRED_BACKEND
            ),
            "kv_write": "complete",
            "jump_row_attention_reads": (
                "suppressed" if _masked_decode_effective else "full"
            ),
            "jump_row_output": "zero" if _masked_decode_effective else "unmasked",
        },
        "project_input_fusion": {
            "enabled": fdvp_fused_project_input_enabled(),
            "shared_storage": (
                fdvp_fused_project_input_shared_storage_enabled()
            ),
        },
        "per_layer_route_counters_enabled": layer_counters_enabled,
        "layer_policies": {
            str(layer_id): {
                "policy": policy,
                "capacity_fraction": (
                    run_fraction
                    if policy == "project_base"
                    else project_fraction if policy == "run_base" else None
                ),
                "run_capacity_fraction": (
                    run_fraction
                    if policy in {"dual_compact", "project_filtered_run_compact"}
                    else None
                ),
                "project_capacity_fraction": (
                    project_fraction
                    if policy in {"dual_compact", "run_base"}
                    else None
                ),
                "compact_min_rows": (
                    project_fraction
                    if policy == "project_filtered_run_compact"
                    else None
                ),
            }
            for layer_id, (policy, run_fraction, project_fraction) in sorted(
                layer_policies.items()
            )
        },
        "conditional_kernel_config": (
            "external_tuned" if external_moe_config else "sglang_default"
        ),
        "materialized_gather_bytes": (
            0 if virtual_cohort else None if compact_enabled else 0
        ),
        "materialized_scatter_bytes": (
            0 if virtual_cohort else None if compact_enabled else 0
        ),
        "materialization_accounting": (
            "virtual_row_map_explicit_cohort_io_only"
            if virtual_cohort
            else "model_full_graph_routes"
            if compact_enabled
            else "none"
        ),
        "counters": {
            "decode_dispatches_total": total_dispatches,
            "decode_rows_total": total_rows,
            "whole_step_graph_replays": graph_dispatches,
            "whole_step_graph_rows": graph_rows,
            "eager_dispatches": eager_dispatches,
            "eager_rows": eager_rows,
            "whole_step_graph_coverage_pct": coverage,
            "routed_layer_rows_graph_replayed": graph_rows * loaded_layer_count,
            "routed_layer_rows_eager": eager_rows * loaded_layer_count,
        },
    }
