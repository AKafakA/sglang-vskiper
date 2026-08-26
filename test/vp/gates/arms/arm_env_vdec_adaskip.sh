# AdaSkip decode arm — POSTURE-ONLY adaptation of the sealed CSD3 arm env
# (the artifact copy keeps the CSD3 host-env block; this in-repo copy carries
# exactly the SGLANG_* posture + the runner switches). AdaSkip serves
# WEIGHTLESS (requires_flexidepth_weights=False) — validation refuses loaded
# FD weights — and its profile is remapped by the runner to the tree's
# fixture unless VP_GATE_ADASKIP_PROFILE overrides.
export VP_GATE_NO_FD_WEIGHTS=1
export SGLANG_FD_ACTIVE_PHASES=both
export SGLANG_FD_EXECUTION_MODE=full_graph
export SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_TAPE=1
export SGLANG_FD_FULL_GRAPH_LAYER_COUNTERS=1
export SGLANG_FD_FULL_GRAPH_MASKED_DECODE_ATTENTION=1
export SGLANG_FD_FULL_GRAPH_ROUTE_ACCOUNTING=1
export SGLANG_VP_ADASKIP_PROFILE=TREE_FIXTURE
export SGLANG_VP_FULL_GRAPH_SKIPPER=adaskip
