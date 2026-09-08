"""Process-level policy selection — dependency-free, no policy code imported.

Which skipper a process runs is decided from the environment before any adapter
is constructed, so the selection itself never drags policy modules into the
import graph. Also owns the two identity predicates: whether the selected policy
needs stable logical request identities (RandomSkip hashes on them) and
whether route digests do.

Unknown boolean values RAISE rather than defaulting -- a typo in a deployment
env must fail closed, not silently select the other branch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Optional,
    Mapping,
)
import msgspec
import math
import os
from sglang.srt.vpipe.env import (
    FD_COMPACT_ENABLED_ENV,
    FD_BATCHED_COMMIT_ENV,
    FD_COMMIT_OVERLAP_ENV,
    FD_DEFER_PROJECT_KV_ENV,
    FD_DEVICE_ROUTE_DIGEST_ENV,
    FD_DEVICE_ROUTE_TAPE_ENV,
    FD_EAGER_SEMANTIC_DEBUG_ENV,
    FD_EXECUTION_FULL_GRAPH,
    FD_FORCED_ALL_RUN_FASTPATH_ENV,
    FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV,
    FD_FUSED_EVIDENCE_ENV,
    FD_LAYER_COUNTERS_ENV,
    FD_LAYER_POLICIES_ENV,
    FD_MASKED_DECODE_ATTN_ENV,
    FD_PREFILL_GROUPED_MLP_ENV,
    FD_ROUTE_ACCOUNTING_ENV,
    FD_SCHEDULER_CONVERGENCE_ENV,
    FD_VIRTUAL_COHORT_ENV,
    FD_WEIGHTED_SCATTER_ENV,
    _VALID_LAYER_POLICIES,
    FD_EXECUTION_DIRECT_EAGER,
    FD_EXECUTION_MODE_ENV,
    _VALID_EXECUTION_MODES,
    FD_ACTIVE_PHASES_ENV,
    _VALID_ACTIVE_PHASES,
    DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER,
    FLEXIDEPTH_FULL_GRAPH_SKIPPER,
    DEVICE_ROUTE_DIGEST_ENV,
)
from sglang.srt.vpipe.common import (
    _strict_bool,
    DECODE_BODY_HIGH,
    DECODE_BODY_LOW,
    _DECODE_BODIES,
    RegimeSwitchConfig,
)
from sglang.srt.vpipe.skipper import (
    configured_full_graph_skipper_name,
)


def full_graph_policy_identity_required(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether the selected policy needs stable logical request identities."""

    return (
        configured_full_graph_skipper_name(environ)
        == DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER
    )
def full_graph_request_identity_required(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether policy execution or route evidence needs request identities."""

    values = os.environ if environ is None else environ
    if full_graph_policy_identity_required(values):
        return True
    digest = str(values.get(DEVICE_ROUTE_DIGEST_ENV, "0")).strip().lower()
    if digest in {"1", "true", "yes", "on"}:
        return configured_full_graph_skipper_name(values) == (
            FLEXIDEPTH_FULL_GRAPH_SKIPPER
        )
    if digest in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{DEVICE_ROUTE_DIGEST_ENV} must be a boolean value")
def full_graph_eager_semantic_debug_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether the trace-only eager execution gate is enabled."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_EAGER_SEMANTIC_DEBUG_ENV)
def full_graph_defer_project_kv_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether PROJECT-row K/V completion runs as graph side work."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_DEFER_PROJECT_KV_ENV)
def full_graph_batched_commit_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether per-layer deferred commits fuse into one launch."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_BATCHED_COMMIT_ENV)
def full_graph_commit_overlap_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether the deferred K/V commit overlaps the logits suffix."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_COMMIT_OVERLAP_ENV)
def full_graph_forced_all_run_fastpath_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Bypass conditional branches in the sealed all-RUN control only."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_FORCED_ALL_RUN_FASTPATH_ENV)
def full_graph_weighted_scatter_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether compact branches fuse route weighting into scatter."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_WEIGHTED_SCATTER_ENV)
def full_graph_layer_policies(
    environ: Optional[Mapping[str, str]] = None,
) -> dict[int, tuple[str, Optional[float], Optional[float]]]:
    """Parse explicit FlexiDepth layer policies from deployment state."""

    values = os.environ if environ is None else environ
    raw = str(values.get(FD_LAYER_POLICIES_ENV, "") or "").strip()
    if not raw:
        return {}
    policies = {}
    for entry in raw.split(","):
        fields = [value.strip() for value in entry.split(":")]
        if len(fields) not in {2, 3, 4}:
            raise ValueError(
                f"invalid {FD_LAYER_POLICIES_ENV} entry: {entry!r}"
            )
        try:
            layer_id = int(fields[0])
        except ValueError as error:
            raise ValueError(
                f"invalid layer id in {FD_LAYER_POLICIES_ENV}: {fields[0]!r}"
            ) from error
        if layer_id < 0 or layer_id in policies:
            raise ValueError(
                f"duplicate or negative layer in {FD_LAYER_POLICIES_ENV}: {layer_id}"
            )
        policy = fields[1]
        if policy not in _VALID_LAYER_POLICIES:
            raise ValueError(
                f"unsupported policy in {FD_LAYER_POLICIES_ENV}: {policy!r}"
            )
        fractions = []
        for raw_fraction in fields[2:]:
            try:
                fractions.append(float(raw_fraction))
            except ValueError as error:
                raise ValueError(
                    f"invalid capacity in {FD_LAYER_POLICIES_ENV}: {raw_fraction!r}"
                ) from error
        if policy == "binary_cohort" and fractions:
            raise ValueError(
                f"binary_cohort in {FD_LAYER_POLICIES_ENV} takes no "
                "capacity fractions (count-adaptive by design)"
            )
        if policy == "project_filtered_run_compact":
            if (
                len(fractions) not in (1, 2)
                or not 0.0 < fractions[0] < 1.0
                or (
                    len(fractions) == 2
                    and not (
                        math.isfinite(fractions[1]) and fractions[1] >= 1.0
                    )
                )
            ):
                raise ValueError(
                    f"{policy} in {FD_LAYER_POLICIES_ENV} requires a RUN "
                    "capacity in (0, 1) and an optional min-rows "
                    "threshold >= 1"
                )
            # min-rows rides in the project slot (unused by this policy);
            # rows >= threshold selects the compact body per capture bucket,
            # below it the plain fused project_filtered_run body runs.
            run_fraction = fractions[0]
            project_fraction = fractions[1] if len(fractions) == 2 else 1.0
        elif policy in {"project_base", "run_base"}:
            if len(fractions) != 1 or not 0.0 < fractions[0] < 1.0:
                raise ValueError(
                    f"{policy} in {FD_LAYER_POLICIES_ENV} requires capacity in (0, 1)"
                )
            if policy == "project_base":
                run_fraction, project_fraction = fractions[0], None
            else:
                run_fraction, project_fraction = None, fractions[0]
        elif policy == "dual_compact":
            if len(fractions) != 2 or any(
                not 0.0 < fraction <= 1.0 for fraction in fractions
            ):
                raise ValueError(
                    f"dual_compact in {FD_LAYER_POLICIES_ENV} requires RUN and PROJECT capacities in (0, 1]"
                )
            run_fraction, project_fraction = fractions
        elif fractions:
            raise ValueError(
                f"{policy} in {FD_LAYER_POLICIES_ENV} cannot specify capacity"
            )
        else:
            run_fraction, project_fraction = None, None
        policies[layer_id] = (policy, run_fraction, project_fraction)
    return policies
def full_graph_prefill_grouped_mlp_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether prefill uses variable-size RUN/PROJECT GEMM cohorts."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_PREFILL_GROUPED_MLP_ENV)
def full_graph_virtual_cohort_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether compact branches use mapped virtual-tensor I/O."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_VIRTUAL_COHORT_ENV)
def full_graph_compact_config(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether the routed bodies compact, failing closed on a bad value.

    [lane-2 knob cleanup, D-578] This used to return a four-tuple carrying a
    capacity fraction, a minimum row count and a rounding multiple. After D-574
    deleted the capacity-based bodies, the fraction and the min-row count no
    longer reached any computation — they survived only as inputs to attestation
    counters, i.e. they changed what we *reported* about a pass, never what the
    pass did. Reporting constants that look like policy knobs are exactly what
    the cleanup is removing, so both are gone and the rounding multiple is now
    the module constant ``COMPACT_CAPACITY_MULTIPLE`` (an allocation
    granularity, not a policy). Compaction itself stays a single on/off.
    """

    values = os.environ if environ is None else environ
    enabled_value = str(values.get(FD_COMPACT_ENABLED_ENV, "0")).strip().lower()
    true_values = {
        "1",
        "true",
        "yes",
        "on",
    }
    false_values = {"0", "false", "no", "off", ""}
    if enabled_value not in true_values | false_values:
        raise ValueError(
            f"{FD_COMPACT_ENABLED_ENV} must be a boolean value"
        )
    return enabled_value in true_values
def full_graph_scheduler_convergence_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether scheduler identity and inline K/V readiness are attested."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_SCHEDULER_CONVERGENCE_ENV)
def full_graph_layer_counters_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    values = os.environ if environ is None else environ
    value = str(values.get(FD_LAYER_COUNTERS_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_LAYER_COUNTERS_ENV} must be a boolean value")

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_LAYER_COUNTERS_ENV)
def full_graph_route_accounting_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether route and compact-work counters run inside graph replay."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_ROUTE_ACCOUNTING_ENV)
def full_graph_device_route_tape_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether routed actions use one graph-stable device tensor."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_DEVICE_ROUTE_TAPE_ENV)
def full_graph_device_route_digest_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether graph replay accumulates a device-only action digest."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_DEVICE_ROUTE_DIGEST_ENV)
def full_graph_fused_evidence_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether replay evidence is reduced by two fixed-shape kernels."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_FUSED_EVIDENCE_ENV)
def full_graph_forced_all_run_production_attention_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Use the production decode backend in the sealed all-RUN body."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV)
def full_graph_masked_decode_attention_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether Triton decode attention suppresses JUMP-row reads."""

    values = os.environ if environ is None else environ
    return _strict_bool(values, FD_MASKED_DECODE_ATTN_ENV)
