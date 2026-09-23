"""Fail-closed helpers for QPS deployment identity and VP activation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


SERVER_IDENTITY_FIELDS = (
    "model_path",
    "tokenizer_path",
    "served_model_name",
    "revision",
    "version",
    "context_length",
    "max_req_input_len",
    "max_total_num_tokens",
    "max_running_requests",
    "mem_fraction_static",
    "dtype",
    "quantization",
    "kv_cache_dtype",
    "tp_size",
    "dp_size",
    "pp_size",
    "attention_backend",
    "decode_attention_backend",
    "prefill_attention_backend",
    "disable_cuda_graph",
    "cuda_graph_config",
    "disable_overlap_schedule",
    "disable_radix_cache",
    "chunked_prefill_size",
)


def runtime_attestations(
    server_info: dict[str, Any], *, require: bool = True
) -> list[dict[str, Any]]:
    internal_states = server_info.get("internal_states")
    if not isinstance(internal_states, list) or not internal_states:
        raise ValueError("/server_info has no scheduler internal states")
    attestations = []
    missing = []
    for index, state in enumerate(internal_states):
        if not isinstance(state, dict) or not isinstance(state.get("vp_runtime"), dict):
            missing.append(index)
            continue
        attestations.append(state["vp_runtime"])
    if missing:
        if attestations or require:
            raise ValueError(
                "/server_info is missing VP runtime attestation for scheduler "
                f"ranks {missing}"
            )
        return []
    if any(value != attestations[0] for value in attestations[1:]):
        raise ValueError("VP runtime attestation differs across scheduler ranks")
    return attestations


def assert_json_subset(observed: Any, expected: Any, path: str = "runtime") -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise ValueError(f"{path}: expected an object, observed {type(observed).__name__}")
        for key, value in expected.items():
            if key not in observed:
                raise ValueError(f"{path}.{key}: missing from observed runtime")
            assert_json_subset(observed[key], value, f"{path}.{key}")
        return
    if observed != expected:
        raise ValueError(f"{path}: expected {expected!r}, observed {observed!r}")


def validate_expected_runtime(
    server_info: dict[str, Any], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    mode = expected.get("attestation")
    if mode not in ("present", "absent"):
        raise ValueError("runtime expectation must set attestation=present or absent")
    if mode == "absent":
        attestations = runtime_attestations(server_info, require=False)
        if attestations:
            raise ValueError("expected no VP runtime attestation, but one is present")
        return []
    runtime_expected = expected.get("runtime")
    if not isinstance(runtime_expected, dict) or not runtime_expected:
        raise ValueError("present runtime expectation requires a non-empty runtime object")
    attestations = runtime_attestations(server_info)
    for index, observed in enumerate(attestations):
        assert_json_subset(observed, runtime_expected, f"runtime[{index}]")
    return attestations


def stable_server_identity(server_info: dict[str, Any]) -> dict[str, Any]:
    states = server_info.get("internal_states") or []
    effective_limits = [
        state.get("effective_max_running_requests_per_dp")
        for state in states
        if isinstance(state, dict)
    ]
    stable_runtime = deepcopy(runtime_attestations(server_info, require=False))
    for runtime in stable_runtime:
        v3_full_graph = runtime.get("v3_full_graph")
        if isinstance(v3_full_graph, dict):
            v3_full_graph.pop("counters", None)
            conditional_graph = v3_full_graph.get("conditional_graph")
            if isinstance(conditional_graph, dict):
                conditional_graph.pop("branch_counts", None)
                conditional_graph.pop("run_body_executions", None)
                conditional_graph.pop("mixed_body_executions", None)
                conditional_graph.pop(
                    "production_all_run_body_executions", None
                )
                conditional_graph.pop(
                    "filtered_all_run_body_executions", None
                )
                # The decode conditional-graph block carries its own copy of
                # the binary_cohort attestation; its `realized` counters
                # advance with traffic exactly like the top-level copies
                # stripped below (first tripped by the vp_final integrated
                # arm, 2026-09-01: identity drifted between reps).
                cg_binary_cohort = conditional_graph.get("binary_cohort")
                if isinstance(cg_binary_cohort, dict):
                    cg_binary_cohort.pop("realized", None)
        regime_switch = runtime.get("regime_switch")
        if isinstance(regime_switch, dict):
            regime_switch.pop("counters", None)
        # batch_composition: live per-step host counters (Scheduler.run_batch)
        # — runtime evidence like the counter blocks above, never identity.
        runtime.pop("batch_composition", None)
        # admission_pins (D-849): the scheduler's per-request body-pin
        # counters (admitted_*, steps_observed, band_flips, mixed_steps,
        # prefix hits) advance with traffic — runtime evidence, never identity
        # (first tripped 2026-09-21 13:42Z: every fork natural cell failed the
        # manifest re-read). The admission DESIGN stays in the identity through
        # regime_switch.admission.
        runtime.pop("admission_pins", None)
        # binary_cohort (D-302): the REALIZED dispatch counters advance on
        # every executor call (warmup and capture included), so they cannot
        # sit in the identity hash — the manifest snapshot and the runner's
        # re-read would never match. The config-deterministic half (enabled,
        # layer set, tuned-artifact digest) stays in the identity.
        for container in (runtime, runtime.get("model") or {}):
            if not isinstance(container, dict):
                continue
            flexidepth = container.get("flexidepth")
            if isinstance(flexidepth, dict):
                binary_cohort = flexidepth.get("binary_cohort")
                if isinstance(binary_cohort, dict):
                    binary_cohort.pop("realized", None)
            binary_cohort = container.get("binary_cohort")
            if isinstance(binary_cohort, dict):
                binary_cohort.pop("realized", None)
        # (c3): counters are runtime evidence; the realized ladder is
        # boot-contingent (per-boot hash / trim variance is measured evidence,
        # SS7.7 of the design) — neither may move the deployment identity SHA.
        # Config-deterministic ladder facts (source, reserve, padding, target)
        # stay in the identity.
        fd_c3 = runtime.get("fd_c3")
        if isinstance(fd_c3, dict):
            fd_c3.pop("counters", None)
            fd_c3_ladder = fd_c3.get("ladder")
            if isinstance(fd_c3_ladder, dict):
                for key in (
                    "decode_capture_bs_max",
                    "capture_bs",
                    "per_boot_ladder_hash",
                    "coverage_ratio",
                ):
                    fd_c3_ladder.pop(key, None)
        model = runtime.get("model")
        if isinstance(model, dict):
            flexidepth = model.get("flexidepth")
            if isinstance(flexidepth, dict):
                flexidepth.pop("full_graph_routes", None)
        v2 = runtime.get("v2")
        if isinstance(v2, dict):
            v2.pop("counters", None)
            host_timing = v2.get("host_timing")
            if isinstance(host_timing, dict):
                host_timing.pop("counters", None)
                host_timing.pop("route_stages", None)
            movement = v2.get("movement_accounting")
            if isinstance(movement, dict):
                movement.pop("by_lane", None)
            trace = v2.get("trace")
            if isinstance(trace, dict):
                trace.pop("selected_tensors", None)
                trace.pop("flushed", None)
            stage_graphs = v2.get("stage_graphs")
            if isinstance(stage_graphs, dict):
                stage_graphs.pop("graph_pool_count", None)
                stage_graphs.pop("captured_keys", None)
                stage_graphs.pop("capture_first_state_audits", None)
                stage_graphs.pop("counters", None)
                stage_graphs.pop("key_counters", None)
                stage_graphs.pop("lane_counters", None)
                stage_graphs.pop("policy_lane_counters", None)
                stage_graphs.pop("rejection_reasons", None)
                stage_graphs.pop("eager_fallback_reasons", None)
                stage_graphs.pop("eager_fallback_lane_counters", None)
                stage_graphs.pop("state_debug_audits", None)
                eager_compare = stage_graphs.get("eager_compare")
                if isinstance(eager_compare, dict):
                    eager_compare.pop("audited_keys", None)
                    eager_compare.pop("audits", None)
        v4 = runtime.get("v4")
        if isinstance(v4, dict):
            v4.pop("counters", None)
            v4.pop("state_contracts", None)
            v4.pop("state_actions", None)
            device_rebatching = v4.get("device_rebatching")
            if isinstance(device_rebatching, dict):
                device_rebatching.pop("admission_gate", None)
                device_rebatching.pop("completion_registry", None)
                device_rebatching.pop("foreground_epoch_slots", None)
                device_rebatching.pop("lease_rows", None)
                device_rebatching.pop("pending_rounds", None)
                executor = device_rebatching.get("executor")
                if isinstance(executor, dict):
                    executor.pop("accounting", None)
                    repair = executor.get("project_kv_repair")
                    if isinstance(repair, dict):
                        for key in (
                            "replay_max_abs_diffs",
                            "replay_max_ulp_diffs",
                            "replay_stats",
                            "round_joins",
                            "stats",
                            "unresolved",
                        ):
                            repair.pop(key, None)
                        successor = repair.get("successor_commit")
                        if isinstance(successor, dict):
                            for key in (
                                "bank_pending_at_reuse",
                                "bank_ready_at_reuse",
                                "bank_reuse_fences",
                                "empty_tick_fences",
                                "fences",
                                "handoffs",
                                "launches",
                                "pending",
                                "pending_at_fence",
                                "pending_banks",
                                "ready_at_fence",
                                "registered_owners",
                            ):
                                successor.pop(key, None)
            host_timing = v4.get("host_timing")
            if isinstance(host_timing, dict):
                host_timing.pop("counters", None)
                host_timing.pop("route_stages", None)
            movement = v4.get("movement_accounting")
            if isinstance(movement, dict):
                movement.pop("by_lane", None)
            trace = v4.get("trace")
            if isinstance(trace, dict):
                trace.pop("selected_tensors", None)
                trace.pop("flushed", None)
            stage_graphs = v4.get("stage_graphs")
            if isinstance(stage_graphs, dict):
                stage_graphs.pop("graph_pool_count", None)
                stage_graphs.pop("captured_keys", None)
                stage_graphs.pop("capture_first_state_audits", None)
                stage_graphs.pop("counters", None)
                stage_graphs.pop("key_counters", None)
                stage_graphs.pop("lane_counters", None)
                stage_graphs.pop("policy_lane_counters", None)
                stage_graphs.pop("rejection_reasons", None)
                stage_graphs.pop("eager_fallback_reasons", None)
                stage_graphs.pop("eager_fallback_lane_counters", None)
                stage_graphs.pop("state_debug_audits", None)
                eager_compare = stage_graphs.get("eager_compare")
                if isinstance(eager_compare, dict):
                    eager_compare.pop("audited_keys", None)
                    eager_compare.pop("audits", None)
    return {
        "server": {key: server_info.get(key) for key in SERVER_IDENTITY_FIELDS},
        "effective_max_running_requests_per_dp": effective_limits,
        "vp_runtime": stable_runtime,
    }


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
