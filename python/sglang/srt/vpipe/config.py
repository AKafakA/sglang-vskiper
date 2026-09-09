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

from sglang.srt.vpipe.design import (  # [D-609] the design is code
    SERVED_COMPACT_ENABLED, SERVED_CONDITIONAL_GRAPH, SERVED_DEVICE_ROUTE_DIGEST,
    SERVED_DEVICE_ROUTE_TAPE, SERVED_LAYER_COUNTERS, SERVED_LAYER_POLICY,
    SERVED_MASKED_DECODE_ATTENTION, SERVED_PREFILL_GROUPED_MLP,
    SERVED_ROUTE_ACCOUNTING, SERVED_ROUTED_LAYERS, SERVED_SCHEDULER_CONVERGENCE,
    SERVED_WEIGHTED_SCATTER, SERVED_DEFER_PROJECT_KV,
    mechanism, skipper_deployed,
)

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
    FD_LAYER_COUNTERS_ENV,
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
    # [D-609] design constant, not an environment read
    digest = "1" if mechanism(SERVED_DEVICE_ROUTE_DIGEST) else "0"
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

    # [D-609] design constant, not an environment read
    return False  # debug knob, deleted (D-609)
def full_graph_defer_project_kv_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether PROJECT-row K/V completion runs as graph side work."""

    # [D-609] design constant, not an environment read
    return mechanism(SERVED_DEFER_PROJECT_KV)
def full_graph_batched_commit_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether per-layer deferred commits fuse into one launch."""

    # [D-609] design constant, not an environment read
    return False  # feature dropped (D-578); refused by validation
def full_graph_commit_overlap_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether the deferred K/V commit overlaps the logits suffix."""

    # [D-609] design constant, not an environment read
    return False  # feature dropped (D-578); refused by validation
def full_graph_forced_all_run_fastpath_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Bypass conditional branches in the sealed all-RUN control only."""

    # [D-609] design constant, not an environment read
    return False  # gate-arm only, never served
def full_graph_weighted_scatter_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether compact branches fuse route weighting into scatter."""

    # [D-609] design constant, not an environment read
    return mechanism(SERVED_WEIGHTED_SCATTER)
def full_graph_layer_policies(
    environ: Optional[Mapping[str, str]] = None,
) -> dict[int, tuple[str, Optional[float], Optional[float]]]:
    """Parse explicit FlexiDepth layer policies from deployment state."""

    # [D-609] The design is a constant: every routed layer runs binary_cohort.
    # This used to come from a 16-entry env string; an arm that failed to export it
    # silently served ZERO routed layers -- the mechanism off, looking configured.
    return {
        layer: (SERVED_LAYER_POLICY, None, None)
        for layer in (SERVED_ROUTED_LAYERS if skipper_deployed() else ())
    }
def full_graph_prefill_grouped_mlp_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether prefill uses variable-size RUN/PROJECT GEMM cohorts."""

    # [D-609] design constant, not an environment read
    return mechanism(SERVED_PREFILL_GROUPED_MLP)
def full_graph_virtual_cohort_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether compact branches use mapped virtual-tensor I/O."""

    # [D-609] design constant, not an environment read
    return False  # superseded by binary_cohort (D-574)
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

    # [D-609] design constant
    return mechanism(SERVED_COMPACT_ENABLED)
def full_graph_scheduler_convergence_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether scheduler identity and inline K/V readiness are attested."""

    # [D-609] design constant, not an environment read
    return mechanism(SERVED_SCHEDULER_CONVERGENCE)
def full_graph_layer_counters_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    # [D-609] design constant
    value = "1" if mechanism(SERVED_LAYER_COUNTERS) else "0"
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{FD_LAYER_COUNTERS_ENV} must be a boolean value")

    # [D-609] design constant, not an environment read
def full_graph_route_accounting_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether route and compact-work counters run inside graph replay."""

    # [D-609] design constant, not an environment read
    return mechanism(SERVED_ROUTE_ACCOUNTING)
def full_graph_device_route_tape_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether routed actions use one graph-stable device tensor."""

    # [D-609] design constant, not an environment read
    return mechanism(SERVED_DEVICE_ROUTE_TAPE)
def full_graph_device_route_digest_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether graph replay accumulates a device-only action digest."""

    # [D-609] design constant, not an environment read
    return mechanism(SERVED_DEVICE_ROUTE_DIGEST)
def full_graph_fused_evidence_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether replay evidence is reduced by two fixed-shape kernels."""

    # [D-609] design constant, not an environment read
    return False  # gate-arm only, never served
def full_graph_forced_all_run_production_attention_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Use the production decode backend in the sealed all-RUN body."""

    # [D-609] design constant, not an environment read
    return False  # gate-arm only, never served
def full_graph_masked_decode_attention_enabled(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return whether Triton decode attention suppresses JUMP-row reads."""

    # [D-609] design constant, not an environment read
    return mechanism(SERVED_MASKED_DECODE_ATTENTION)
