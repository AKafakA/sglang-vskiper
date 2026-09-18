"""Startup gate for the deferred-commit overlap knob (Lane K, W2 step 2).

SGLANG_FD_FULL_GRAPH_COMMIT_OVERLAP=1 moves the deferred PROJECT-K/V cache
commit off the logits critical path (commit ∥ suffix inside the composed
conditional graph; the graph-end node is the step-boundary fence). The knob
is meaningless without the deferral it overlaps and without the full repair
commit, so both misconfigurations must be refused AT STARTUP — a silently
inert flag is the attested-but-not-running class (R6).

CPU-safe: validator calls with injected environ mappings only.
"""
import sys
from contextlib import ExitStack
from unittest.mock import patch

from vskipper.runtime.validation import validate_full_graph_model_configuration

ROUTED = list(range(16, 32))

# The canonical vdec_fd defer posture (the sealed arm env, host paths elided).
DEFER_POSTURE = {
    "SGLANG_VP_REGIME_SWITCH": "off",  # isolated defer seam; switching requires foreground K/V
    "SGLANG_FD_WEIGHTS": "/x.pt",
    "SGLANG_FD_EXECUTION_MODE": "full_graph",
    "SGLANG_FD_ACTIVE_PHASES": "decode",
    "SGLANG_FD_FULL_GRAPH_COMPACT": "1",
    "SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_TAPE": "1",
    "SGLANG_FD_FULL_GRAPH_ROUTE_ACCOUNTING": "1",
    "SGLANG_FD_VP_FUSED_PROJECT_INPUT": "1",
    "SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV": "1",
    "SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH": "1",
    "SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER": "/x.so",
    "SGLANG_FD_FULL_GRAPH_MASKED_DECODE_ATTENTION": "1",
    "SGLANG_FD_FULL_GRAPH_SCHEDULER_CONVERGENCE": "1",
    "SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_DIGEST": "1",
}


def call(env, label):
    try:
        from vskipper.runtime import design, validation
        # Exercise the dependency guards as unit seams. Overlap and batched
        # commit are pinned off in serving; their real readers are checked below.
        with ExitStack() as stack:
            stack.enter_context(patch.dict(design._ARM_CACHE, name="vskipper"))
            stack.enter_context(patch.object(design, "_host", return_value={
                "flexidepth_weights": "/fixture/router.pt",
                "conditional_graph_helper": "/fixture/helper.so",
            }))
            for reader, key in (
                ("full_graph_defer_project_kv_enabled", "SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV"),
                ("full_graph_commit_overlap_enabled", "SGLANG_FD_FULL_GRAPH_COMMIT_OVERLAP"),
                ("full_graph_batched_commit_enabled", "SGLANG_FD_FULL_GRAPH_BATCHED_COMMIT"),
            ):
                stack.enter_context(patch.object(validation, reader,
                    side_effect=lambda values, key=key: values.get(key) == "1"))
            validate_full_graph_model_configuration(
                loaded_layers=ROUTED, loaded_flexidepth_layers=ROUTED,
                tp_size=1, pp_size=1, quant_config=None, environ=env,
                cuda_graph_enabled=True,
            )
        return ("no-raise", label)
    except Exception as e:
        return (f"{type(e).__name__}: {str(e)[:110]}", label)


results = [
    call(dict(DEFER_POSTURE), "A defer posture, overlap unset (control)"),
    call(dict(DEFER_POSTURE, SGLANG_FD_FULL_GRAPH_COMMIT_OVERLAP="1"),
         "B defer posture + overlap"),
    call({k: v for k, v in dict(
        DEFER_POSTURE, SGLANG_FD_FULL_GRAPH_COMMIT_OVERLAP="1").items()
        if k != "SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV"},
         "C overlap without defer"),
    call(dict(
        DEFER_POSTURE,
        SGLANG_FD_FULL_GRAPH_COMMIT_OVERLAP="1",
        SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE="qkv_only",
    ), "D overlap at a diagnostic stage"),
    call(dict(DEFER_POSTURE, SGLANG_FD_FULL_GRAPH_BATCHED_COMMIT="1"),
         "E defer posture + batched commit"),
    call({k: v for k, v in dict(
        DEFER_POSTURE, SGLANG_FD_FULL_GRAPH_BATCHED_COMMIT="1").items()
        if k != "SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV"},
         "F batched commit without defer"),
    call(dict(
        DEFER_POSTURE,
        SGLANG_FD_FULL_GRAPH_BATCHED_COMMIT="1",
        SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV_DIAGNOSTIC_STAGE="qkv_only",
    ), "G batched commit at a diagnostic stage"),
    call(dict(DEFER_POSTURE, SGLANG_FD_FULL_GRAPH_COMPACT_ROUTED_QKV="1"),
         "H defer posture + compact routed qkv"),
    call({k: v for k, v in dict(
        DEFER_POSTURE, SGLANG_FD_FULL_GRAPH_COMPACT_ROUTED_QKV="1").items()
        if k != "SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV"},
         "I compact routed qkv without defer"),
    call(dict(
        DEFER_POSTURE,
        SGLANG_FD_FULL_GRAPH_CONTIGUOUS_ROUTED_QKV="1",
        SGLANG_FD_FULL_GRAPH_ROUTED_QKV_CAPACITIES="16:0.6:0.6",
    ), "J contiguous without compact"),
    call(dict(
        DEFER_POSTURE,
        SGLANG_FD_FULL_GRAPH_COMPACT_ROUTED_QKV="1",
        SGLANG_FD_FULL_GRAPH_CONTIGUOUS_ROUTED_QKV="1",
    ), "K contiguous without capacities"),
    call(dict(
        DEFER_POSTURE,
        SGLANG_FD_FULL_GRAPH_COMPACT_ROUTED_QKV="1",
        SGLANG_FD_FULL_GRAPH_CONTIGUOUS_ROUTED_QKV="1",
        SGLANG_FD_FULL_GRAPH_ROUTED_QKV_CAPACITIES="16:0.6:0.6",
    ), "L contiguous with partial coverage"),
    call(dict(
        DEFER_POSTURE,
        SGLANG_FD_FULL_GRAPH_COMPACT_ROUTED_QKV="1",
        SGLANG_FD_FULL_GRAPH_CONTIGUOUS_ROUTED_QKV="1",
        SGLANG_FD_FULL_GRAPH_ROUTED_QKV_CAPACITIES=",".join(
            f"{layer}:0.6:0.6" for layer in ROUTED
        ),
    ), "M contiguous with full coverage"),
]

ok = True
for outcome, label in results:
    print(f"  {label:40s} -> {outcome}")
outcomes = dict((label, outcome) for outcome, label in results)
if outcomes["A defer posture, overlap unset (control)"] != "no-raise":
    print("FAIL: the existing defer posture must be unaffected"); ok = False
if outcomes["B defer posture + overlap"] != "no-raise":
    print("FAIL: the treated posture must boot"); ok = False
if "requires SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV=1" not in outcomes[
        "C overlap without defer"]:
    print("FAIL: overlap without defer must be refused"); ok = False
if "requires the full repair commit" not in outcomes[
        "D overlap at a diagnostic stage"]:
    print("FAIL: overlap at a diagnostic stage must be refused"); ok = False
if outcomes["E defer posture + batched commit"] != "no-raise":
    print("FAIL: the batched-commit posture must boot"); ok = False
if "requires SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV=1" not in outcomes[
        "F batched commit without defer"]:
    print("FAIL: batched commit without defer must be refused"); ok = False
if "requires the full repair commit" not in outcomes[
        "G batched commit at a diagnostic stage"]:
    print("FAIL: batched commit at a diagnostic stage must be refused"); ok = False
if outcomes["H defer posture + compact routed qkv"] != "no-raise":
    print("FAIL: the compact-routed-qkv posture must boot"); ok = False
if "requires SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV=1" not in outcomes[
        "I compact routed qkv without defer"]:
    print("FAIL: compact without defer must be refused"); ok = False
if "requires SGLANG_FD_FULL_GRAPH_COMPACT_ROUTED_QKV=1" not in outcomes[
        "J contiguous without compact"]:
    print("FAIL: contiguous without compact must be refused"); ok = False
if "requires" not in outcomes["K contiguous without capacities"]:
    print("FAIL: contiguous without capacities must be refused"); ok = False
if "is missing routed layers" not in outcomes[
        "L contiguous with partial coverage"]:
    print("FAIL: partial capacity coverage must be refused at startup"); ok = False
if outcomes["M contiguous with full coverage"] != "no-raise":
    print("FAIL: full capacity coverage must boot"); ok = False
print("COMMIT-OVERLAP GATE:", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    sys.exit(0 if ok else 1)


def test_commit_overlap_startup_gate():
    assert outcomes["A defer posture, overlap unset (control)"] == "no-raise", \
        "the existing defer posture must be unaffected by the new knob"
    assert outcomes["B defer posture + overlap"] == "no-raise", \
        "the treated overlap posture must boot"
    assert "requires SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV=1" in outcomes[
        "C overlap without defer"], \
        "overlap without the deferral it overlaps must be refused"
    assert "requires the full repair commit" in outcomes[
        "D overlap at a diagnostic stage"], \
        "overlap without the full repair commit must be refused"
    assert outcomes["E defer posture + batched commit"] == "no-raise", \
        "the batched-commit posture must boot"
    assert "requires SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV=1" in outcomes[
        "F batched commit without defer"], \
        "batched commit without the deferral must be refused"
    assert "requires the full repair commit" in outcomes[
        "G batched commit at a diagnostic stage"], \
        "batched commit without the full repair commit must be refused"
    assert outcomes["H defer posture + compact routed qkv"] == "no-raise", \
        "the compact-routed-qkv posture must boot"
    assert "requires SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV=1" in outcomes[
        "I compact routed qkv without defer"], \
        "compact routed qkv without the deferral must be refused"
    assert "requires SGLANG_FD_FULL_GRAPH_COMPACT_ROUTED_QKV=1" in outcomes[
        "J contiguous without compact"], \
        "the contiguous lane without compact must be refused"
    assert "requires" in outcomes["K contiguous without capacities"], \
        "the contiguous lane without capacities must be refused"
    assert "is missing routed layers" in outcomes[
        "L contiguous with partial coverage"], \
        "partial capacity coverage must be refused at startup, not at capture"
    assert outcomes["M contiguous with full coverage"] == "no-raise", \
        "full capacity coverage must boot"


def test_retired_commit_modes_cannot_be_enabled_by_environment():
    from vskipper.runtime.config import (
        full_graph_commit_overlap_enabled, full_graph_batched_commit_enabled,
    )
    for values in ({}, {"SGLANG_FD_FULL_GRAPH_COMMIT_OVERLAP": "1",
                       "SGLANG_FD_FULL_GRAPH_BATCHED_COMMIT": "1"}):
        assert full_graph_commit_overlap_enabled(values) is False
        assert full_graph_batched_commit_enabled(values) is False
