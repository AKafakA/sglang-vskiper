"""Process-level policy selection — dependency-free, no policy code imported.

Which skipper a process runs is decided from the environment before any adapter
is constructed, so the selection itself never drags policy modules into the
import graph. Also owns the two identity predicates: whether the selected policy
needs stable logical request identities (RandomSkip and AdaSkip hash on them)
and whether route digests do.

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
    FD_COMPACT_CAPACITY_FRACTION_ENV,
    FD_COMPACT_CAPACITY_MULTIPLE_ENV,
    FD_COMPACT_ENABLED_ENV,
    FD_COMPACT_MIN_ROWS_ENV,
    FD_COMPACT_O_PROJ_ENV,
    FD_COMPACT_O_PROJ_LAYERS_ENV,
    FD_COMPACT_ROUTED_QKV_ENV,
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
    FD_MAPPED_DECODE_ATTN_ENV,
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
    ADASKIP_FULL_GRAPH_SKIPPER,
    DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER,
    FLEXIDEPTH_FULL_GRAPH_SKIPPER,
    DEVICE_ROUTE_DIGEST_ENV,
)
from sglang.srt.vpipe.common import (
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
        in {
            DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER,
            ADASKIP_FULL_GRAPH_SKIPPER,
        }
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
    value = str(values.get(FD_EAGER_SEMANTIC_DEBUG_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_EAGER_SEMANTIC_DEBUG_ENV} must be a boolean value")
def full_graph_defer_project_kv_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether PROJECT-row K/V completion runs as graph side work."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_DEFER_PROJECT_KV_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_DEFER_PROJECT_KV_ENV} must be a boolean value")
def full_graph_forced_all_run_fastpath_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Bypass conditional branches in the sealed all-RUN control only."""

    values = os.environ if environ is None else environ
    value = str(
        values.get(FD_FORCED_ALL_RUN_FASTPATH_ENV, "0")
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"{FD_FORCED_ALL_RUN_FASTPATH_ENV} must be a boolean value"
    )
def full_graph_weighted_scatter_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether compact branches fuse route weighting into scatter."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_WEIGHTED_SCATTER_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_WEIGHTED_SCATTER_ENV} must be a boolean value")
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
    value = str(values.get(FD_PREFILL_GROUPED_MLP_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_PREFILL_GROUPED_MLP_ENV} must be a boolean value")
def full_graph_virtual_cohort_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether compact branches use mapped virtual-tensor I/O."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_VIRTUAL_COHORT_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_VIRTUAL_COHORT_ENV} must be a boolean value")
def full_graph_compact_config(
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[bool, int, float, int]:
    """Return the bounded-compaction policy and fail closed on invalid values."""

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
    enabled = enabled_value in true_values
    try:
        min_rows = int(values.get(FD_COMPACT_MIN_ROWS_ENV, "256"))
        capacity_fraction = float(
            values.get(FD_COMPACT_CAPACITY_FRACTION_ENV, "0.625")
        )
        capacity_multiple = int(
            values.get(FD_COMPACT_CAPACITY_MULTIPLE_ENV, "16")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "invalid FlexiDepth full-graph compact configuration"
        ) from error
    if min_rows <= 0:
        raise ValueError(f"{FD_COMPACT_MIN_ROWS_ENV} must be positive")
    if not 0.0 < capacity_fraction < 1.0:
        raise ValueError(
            f"{FD_COMPACT_CAPACITY_FRACTION_ENV} must be in (0, 1)"
        )
    if capacity_multiple <= 0:
        raise ValueError(f"{FD_COMPACT_CAPACITY_MULTIPLE_ENV} must be positive")
    return enabled, min_rows, capacity_fraction, capacity_multiple
def full_graph_scheduler_convergence_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether scheduler identity and inline K/V readiness are attested."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_SCHEDULER_CONVERGENCE_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_SCHEDULER_CONVERGENCE_ENV} must be a boolean value")
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
def full_graph_route_accounting_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether route and compact-work counters run inside graph replay."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_ROUTE_ACCOUNTING_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_ROUTE_ACCOUNTING_ENV} must be a boolean value")
def full_graph_device_route_tape_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether routed actions use one graph-stable device tensor."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_DEVICE_ROUTE_TAPE_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_DEVICE_ROUTE_TAPE_ENV} must be a boolean value")
def full_graph_device_route_digest_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether graph replay accumulates a device-only action digest."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_DEVICE_ROUTE_DIGEST_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_DEVICE_ROUTE_DIGEST_ENV} must be a boolean value")
def full_graph_fused_evidence_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether replay evidence is reduced by two fixed-shape kernels."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_FUSED_EVIDENCE_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_FUSED_EVIDENCE_ENV} must be a boolean value")
def full_graph_forced_all_run_production_attention_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Use the production decode backend in the sealed all-RUN body."""

    values = os.environ if environ is None else environ
    value = str(
        values.get(FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV, "0")
    ).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"{FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV} must be a boolean value"
    )
def full_graph_masked_decode_attention_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether Triton decode attention suppresses JUMP-row reads."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_MASKED_DECODE_ATTN_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_MASKED_DECODE_ATTN_ENV} must be a boolean value")
def full_graph_compact_o_proj_config(
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[bool, dict[int, float]]:
    """Return exact compact output-projection capacities by layer."""

    values = os.environ if environ is None else environ
    raw_enabled = str(values.get(FD_COMPACT_O_PROJ_ENV, "0")).strip().lower()
    if raw_enabled in {"1", "true", "yes", "on"}:
        enabled = True
    elif raw_enabled in {"0", "false", "no", "off", ""}:
        enabled = False
    else:
        raise ValueError(f"{FD_COMPACT_O_PROJ_ENV} must be a boolean value")

    raw_layers = str(values.get(FD_COMPACT_O_PROJ_LAYERS_ENV, "") or "").strip()
    layers: dict[int, float] = {}
    if raw_layers:
        for entry in raw_layers.split(","):
            fields = [field.strip() for field in entry.split(":")]
            if len(fields) != 2:
                raise ValueError(
                    f"invalid {FD_COMPACT_O_PROJ_LAYERS_ENV} entry: {entry!r}"
                )
            try:
                layer_id = int(fields[0])
                fraction = float(fields[1])
            except ValueError as error:
                raise ValueError(
                    f"invalid {FD_COMPACT_O_PROJ_LAYERS_ENV} entry: {entry!r}"
                ) from error
            if layer_id < 0 or layer_id in layers:
                raise ValueError(
                    f"duplicate or negative layer in "
                    f"{FD_COMPACT_O_PROJ_LAYERS_ENV}: {layer_id}"
                )
            if not 0.0 < fraction < 1.0:
                raise ValueError(
                    f"capacity in {FD_COMPACT_O_PROJ_LAYERS_ENV} must be in (0, 1)"
                )
            layers[layer_id] = fraction
    if enabled and not layers:
        raise ValueError(
            f"{FD_COMPACT_O_PROJ_ENV}=1 requires "
            f"{FD_COMPACT_O_PROJ_LAYERS_ENV}"
        )
    if not enabled and layers:
        raise ValueError(
            f"{FD_COMPACT_O_PROJ_LAYERS_ENV} requires "
            f"{FD_COMPACT_O_PROJ_ENV}=1"
        )
    return enabled, layers
def full_graph_mapped_decode_attention_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether sparse RUN rows use persistent virtual-row attention."""

    values = os.environ if environ is None else environ
    value = str(values.get(FD_MAPPED_DECODE_ATTN_ENV, "0")).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_MAPPED_DECODE_ATTN_ENV} must be a boolean value")
