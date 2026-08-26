# V-pre BINARY-COHORT posture (D-302/D-303; sourced after the env scrub).
# The it4 baseline with the deep routed layers switched to the
# count-adaptive binary_cohort executor (pack -> tuned count-GEMM ->
# silu_mul -> count-GEMM -> route-weighted scatter). Deep layers only:
# the calibration measured 16-21 engaging at 5-16% on EVERY workload
# (universal loss regime -> keep them cheap/dense-ish via run_base),
# 22-30 carry the real engagement. No capacity fractions: binary_cohort
# is count-adaptive by construction.
export SGLANG_FD_ACTIVE_PHASES=prefill
export SGLANG_FD_WEIGHTS=/rds/user/wd312/hpc-work/llm/vpipe-csd3/serving/flexidepth_router_weights.pt
export SGLANG_FD_EXECUTION_MODE=full_graph
export SGLANG_FD_FULL_GRAPH_COMPACT=1
export SGLANG_FD_FULL_GRAPH_COMPACT_CAPACITY_FRACTION=0.625
export SGLANG_FD_FULL_GRAPH_COMPACT_CAPACITY_MULTIPLE=16
export SGLANG_FD_FULL_GRAPH_COMPACT_MIN_ROWS=32
export SGLANG_FD_FULL_GRAPH_COMPACT_PHASES=prefill
export SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_TAPE=1
export SGLANG_FD_FULL_GRAPH_ROUTE_ACCOUNTING=1
export SGLANG_FD_VP_FUSED_PROJECT_INPUT=1
export SGLANG_MOE_CONFIG_DIR=/rds/user/wd312/hpc-work/llm/vpipe-csd3/serving/moe-configs-a100-largeM-r1
export SGLANG_FD_FULL_GRAPH_LAYER_POLICIES="16:run_base:0.125,17:run_base:0.0625,18:run_base:0.0625,19:run_base:0.25,20:run_base:0.125,21:run_base:0.125,22:binary_cohort,23:binary_cohort,24:binary_cohort,25:binary_cohort,26:binary_cohort,27:binary_cohort,28:binary_cohort,29:binary_cohort,30:binary_cohort,31:run_base:0.125"
