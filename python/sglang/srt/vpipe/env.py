"""Environment knob names and process-level constants.

The frozen tree referenced 202 env knobs accumulated across four superseded VP
generations; the four canonical deployment arms set 25. Only the names the
served path actually reads travel here -- the rest were gates on experiments the
decision log records as rejected or not-promoted.

Names only. Reading and validation live with the code that owns each knob, so
that a knob and its fail-closed parsing stay together.
"""

from __future__ import annotations

import os


# Operator declaration of the staged checkpoint's revision. Needed because
# a locally staged snapshot exposes no commit hash, and a directory name is
# not identity -- renaming any checkpoint would defeat it. General identity
# machinery (consumed by the llama attach site), not a skipper feature.
SERVED_MODEL_REVISION_ENV = "SGLANG_VP_SERVED_MODEL_REVISION"
DEVICE_ROUTE_DIGEST_ENV = "SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_DIGEST"
FULL_GRAPH_MOCK_SEED_ENV = "SGLANG_VP_FULL_GRAPH_MOCK_SEED"
FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO_ENV = (
    "SGLANG_VP_FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO"
)
FULL_GRAPH_MOCK_TOKEN_SKIP_RATE_ENV = (
    "SGLANG_VP_FULL_GRAPH_MOCK_TOKEN_SKIP_RATE"
)
FULL_GRAPH_SKIPPER_ENV = "SGLANG_VP_FULL_GRAPH_SKIPPER"
REGIME_SWITCH_ENV = "SGLANG_VP_REGIME_SWITCH"
_MOCK_CONFIG_ENVS = (
    FULL_GRAPH_MOCK_TOKEN_SKIP_RATE_ENV,
    FULL_GRAPH_MOCK_SKIPPED_DEPTH_RATIO_ENV,
    FULL_GRAPH_MOCK_SEED_ENV,
)
_FD_PARITY_TRACE_ENABLED = bool(
    os.environ.get("SGLANG_FD_PARITY_TRACE_RID", "").strip()
)
FD_EXECUTION_MODE_ENV = "SGLANG_FD_EXECUTION_MODE"
FD_ACTIVE_PHASES_ENV = "SGLANG_FD_ACTIVE_PHASES"
FD_EXECUTION_DIRECT_EAGER = "direct_eager"
FD_EXECUTION_FULL_GRAPH = "full_graph"
_MASKED_DECODE_REQUIRED_BACKEND = "triton"
FD_EAGER_SEMANTIC_DEBUG_ENV = "SGLANG_FD_FULL_GRAPH_EAGER_SEMANTIC_DEBUG"
FD_PREFILL_GROUPED_MLP_ENV = "SGLANG_FD_FULL_GRAPH_PREFILL_GROUPED_MLP"
FD_COMPACT_ENABLED_ENV = "SGLANG_FD_FULL_GRAPH_COMPACT"
FD_COMPACT_PHASES_ENV = "SGLANG_FD_FULL_GRAPH_COMPACT_PHASES"
FD_COMPACT_MIN_ROWS_ENV = "SGLANG_FD_FULL_GRAPH_COMPACT_MIN_ROWS"
FD_DUAL_COMPACT_MIN_ROWS_ENV = "SGLANG_FD_FULL_GRAPH_DUAL_COMPACT_MIN_ROWS"
FD_COMPACT_CAPACITY_FRACTION_ENV = (
    "SGLANG_FD_FULL_GRAPH_COMPACT_CAPACITY_FRACTION"
)
FD_COMPACT_CAPACITY_MULTIPLE_ENV = (
    "SGLANG_FD_FULL_GRAPH_COMPACT_CAPACITY_MULTIPLE"
)
FD_LOW_ROW_POLICY_ENV = "SGLANG_FD_FULL_GRAPH_LOW_ROW_POLICY"
FD_LOW_ROW_MAX_ROWS_ENV = "SGLANG_FD_FULL_GRAPH_LOW_ROW_MAX_ROWS"
FD_ROUTE_ACCOUNTING_ENV = "SGLANG_FD_FULL_GRAPH_ROUTE_ACCOUNTING"
FD_LAYER_COUNTERS_ENV = "SGLANG_FD_FULL_GRAPH_LAYER_COUNTERS"
FD_DEVICE_ROUTE_TAPE_ENV = "SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_TAPE"
FD_DEVICE_ROUTE_DIGEST_ENV = "SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_DIGEST"
FD_SCHEDULER_CONVERGENCE_ENV = (
    "SGLANG_FD_FULL_GRAPH_SCHEDULER_CONVERGENCE"
)
FD_FORCE_ROUTE_ENV = "SGLANG_FD_FULL_GRAPH_FORCE_ROUTE"
FD_FORCED_ALL_RUN_FASTPATH_ENV = (
    "SGLANG_FD_FULL_GRAPH_FORCED_ALL_RUN_FASTPATH"
)
FD_FORCED_ALL_RUN_PRODUCTION_ATTN_ENV = (
    "SGLANG_FD_FULL_GRAPH_FORCED_ALL_RUN_PRODUCTION_ATTENTION"
)
FD_LAYER_POLICIES_ENV = "SGLANG_FD_FULL_GRAPH_LAYER_POLICIES"
FD_VIRTUAL_COHORT_ENV = "SGLANG_FD_FULL_GRAPH_VIRTUAL_COHORT"
FD_WEIGHTED_SCATTER_ENV = "SGLANG_FD_FULL_GRAPH_WEIGHTED_SCATTER"
FD_FUSED_EVIDENCE_ENV = "SGLANG_FD_FULL_GRAPH_FUSED_EVIDENCE"
FD_MASKED_DECODE_ATTN_ENV = "SGLANG_FD_FULL_GRAPH_MASKED_DECODE_ATTENTION"
FD_COMPACT_O_PROJ_ENV = "SGLANG_FD_FULL_GRAPH_COMPACT_O_PROJ"
FD_COMPACT_O_PROJ_LAYERS_ENV = "SGLANG_FD_FULL_GRAPH_COMPACT_O_PROJ_LAYERS"
FD_COMPACT_O_PROJ_MIN_ROWS_ENV = "SGLANG_FD_FULL_GRAPH_COMPACT_O_PROJ_MIN_ROWS"
FD_COMPACT_Q_PROJ_ENV = "SGLANG_FD_FULL_GRAPH_COMPACT_Q_PROJ"
FD_MAPPED_DECODE_ATTN_ENV = "SGLANG_FD_FULL_GRAPH_MAPPED_DECODE_ATTENTION"
FD_CONDITIONAL_GRAPH_ENV = "SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH"
FD_CONDITIONAL_PRODUCTION_ALL_RUN_ENV = (
    "SGLANG_FD_FULL_GRAPH_CONDITIONAL_PRODUCTION_ALL_RUN"
)
FD_CONDITIONAL_GRAPH_HELPER_ENV = (
    "SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER"
)
FD_CONDITIONAL_MAX_ROWS_ENV = (
    "SGLANG_FD_FULL_GRAPH_CONDITIONAL_MAX_ROWS"
)
FD_CONDITIONAL_BRANCH_COUNTERS_ENV = (
    "SGLANG_FD_FULL_GRAPH_CONDITIONAL_BRANCH_COUNTERS"
)
FD_DEFER_PROJECT_KV_ENV = "SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV"
FD_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE_ENV = (
    "SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE"
)
FD_COMMIT_OVERLAP_ENV = "SGLANG_FD_FULL_GRAPH_COMMIT_OVERLAP"
FD_BATCHED_COMMIT_ENV = "SGLANG_FD_FULL_GRAPH_BATCHED_COMMIT"
FD_COMPACT_ROUTED_QKV_ENV = "SGLANG_FD_FULL_GRAPH_COMPACT_ROUTED_QKV"
FD_CONTIGUOUS_ROUTED_QKV_ENV = (
    "SGLANG_FD_FULL_GRAPH_CONTIGUOUS_ROUTED_QKV"
)
FD_ROUTED_QKV_CAPACITIES_ENV = (
    "SGLANG_FD_FULL_GRAPH_ROUTED_QKV_CAPACITIES"
)
FD_ROUTED_QKV_MIN_ROWS_ENV = "SGLANG_FD_FULL_GRAPH_ROUTED_QKV_MIN_ROWS"
FD_ROUTED_QKV_CAPACITY_MULTIPLE_ENV = (
    "SGLANG_FD_FULL_GRAPH_ROUTED_QKV_CAPACITY_MULTIPLE"
)
_VALID_LAYER_POLICIES = frozenset(
    (
        "binary_cohort",
        "dense_filtered_project",
        "dual_compact",
        "filtered",
        "full_dual",
        "project_base",
        "project_filtered_run",
        "project_filtered_run_compact",
        "run_base",
    )
)
_VALID_EXECUTION_MODES = frozenset(
    (FD_EXECUTION_DIRECT_EAGER, FD_EXECUTION_FULL_GRAPH)
)
# `native_dense` (lane-2 cut3 item 1, D-358) is `full_dual` without a row bound:
# the model's OWN dense feed-forward for every row plus the projector, selected
# by the route mask, at every occupancy. It is the serving arm's routed-MLP
# posture — the count-adaptive and grouped paths stay in the tree for their
# gate arms but are not reachable from a `native_dense` deployment.
_VALID_LOW_ROW_POLICIES = frozenset(("off", "full_dual", "native_dense"))
# Routed-MLP gate arithmetic, which is a property of the CHECKPOINT, not a tuning
# knob. `released` is the published FlexiDepth gate: `w * MLP` on RUN rows and
# `(1 - w) * PROJECT` on the rest. `hard_mask` is the straight-through gate
# (`m = hard + w - w.detach()`, trained 2026-09-03 for the Qwen3 family): the
# forward is a hard selection with NO `w` scaling on either branch. Serving an
# `ste_hard` checkpoint under `released` applies scaling it was never trained
# with and fails silently — the same defect that voided the 09-03 quality gates.
FD_GATE_MODE_ENV = "SGLANG_FD_GATE_MODE"
_VALID_GATE_MODES = frozenset(("released", "hard_mask"))
# Sentinel bound for `native_dense`; larger than any capturable decode bucket
# (the req-to-token pool ceiling is 4096 rows) so the body always resolves.
_NATIVE_DENSE_UNBOUNDED_ROWS = 1 << 30
_VALID_ACTIVE_PHASES = frozenset(("decode", "prefill", "both"))
_VALID_FORCED_ROUTES = frozenset(("off", "all_run", "all_project"))
_VALID_DEFER_PROJECT_KV_DIAGNOSTIC_STAGES = frozenset(
    ("full", "qkv_only", "qkv_rope", "readiness_only")
)
_CONFLICTING_FULL_GRAPH_ENV = (
    "SGLANG_VP_V2_CONFIG",
    "SGLANG_FD_VP_PROJECT",
    "SGLANG_FD_VP_STAGE_ROUTE",
    "SGLANG_FD_VP_ASYNC_KV",
    "SGLANG_VP_ASYNC_KV_BATCHED",
)
# Knobs whose features were DROPPED from this build (owner ruling 2026-08-23:
# AdaSkip sublayer skipper; batched K/V commit + commit overlap; routed-QKV
# compact/contiguous staging). Setting one is a stale config, refused loudly
# at validation -- see codex/asplos-plan/2026-08-21-removed-feature-register.md
# for revival. An explicit off value ("0"/"false"/"no"/"off") is allowed.
_REMOVED_FEATURE_ENVS = (
    "SGLANG_VP_ADASKIP_PROFILE",
    "SGLANG_VP_ADASKIP_DENSE_REFERENCE_MLP",
    "SGLANG_VP_ADASKIP_MAX_GRAPH_ROWS",
    "SGLANG_VP_ADASKIP_MAX_REQUEST_SLOTS",
    "SGLANG_FD_FULL_GRAPH_REPAIR_GROUP_SIZE",
)
# Value-typed names among the conflict/removed sets: any non-off value counts
# as set (a path, a list, an integer); the rest are boolean-ish.
_VALUE_TYPED_CONFLICT_ENVS = frozenset(
    (
        "SGLANG_VP_V2_CONFIG",
        "SGLANG_VP_ADASKIP_PROFILE",
        "SGLANG_VP_ADASKIP_MAX_GRAPH_ROWS",
        "SGLANG_VP_ADASKIP_MAX_REQUEST_SLOTS",
        "SGLANG_FD_FULL_GRAPH_REPAIR_GROUP_SIZE",
    )
)
FULL_GRAPH_CAPTURE_SYNTHETIC_RID_BASE = -(1 << 62)
_BINARY_COHORT_SCRATCH: dict = {}
_BINARY_COHORT_STATS: dict = {}
_BINARY_COHORT_LAYERS: set = set()
_BINARY_COHORT_CONFIG_DIGEST: dict = {}
DETERMINISTIC_MOCK_FULL_GRAPH_SKIPPER = "deterministic_mock"
FLEXIDEPTH_FULL_GRAPH_SKIPPER = "flexidepth"
RUN_PROJECT_EXECUTION = "run_project"
