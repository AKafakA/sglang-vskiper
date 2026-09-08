# V-pre CONVERGED posture (sourced by csd3_debug_cell.sh after env scrub):
# THRESHOLDED dual-compact (iteration 4, commit f416d77001): dual_compact on 22-30 with per-bucket min-rows 4096 — buckets <4096 capture the FUSED body (std posture), >=4096 capture dual_compact. gsm8k lives <4096 (guard = structural), coqa >=4096.
# prefill-only FD routing, full-graph compact, merged large-M MoE configs.
export SGLANG_FD_ACTIVE_PHASES=both
export SGLANG_FD_WEIGHTS=/rds/user/wd312/hpc-work/llm/vpipe-csd3/serving/flexidepth_router_weights.pt
export SGLANG_FD_EXECUTION_MODE=full_graph
export SGLANG_FD_FULL_GRAPH_COMPACT=1
# [D-578] the three compact capacity knobs are gone: the multiple and the
# accounting fraction are internal constants at these exact values, and the
# min-row count no longer reaches any code. Setting them now fails the boot.
export SGLANG_FD_FULL_GRAPH_COMPACT_PHASES=prefill
export SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_TAPE=1
export SGLANG_FD_FULL_GRAPH_ROUTE_ACCOUNTING=1
export SGLANG_FD_VP_FUSED_PROJECT_INPUT=1
export SGLANG_MOE_CONFIG_DIR=/rds/user/wd312/hpc-work/llm/vpipe-csd3/serving/moe-configs-a100-largeM-r1
export SGLANG_FD_FULL_GRAPH_LAYER_POLICIES="16:run_base:0.125,17:run_base:0.0625,18:run_base:0.0625,19:run_base:0.25,20:run_base:0.125,21:run_base:0.125,22:dual_compact:0.625:0.5,23:dual_compact:0.375:0.75,24:dual_compact:0.625:0.5,25:dual_compact:0.625:0.5,26:dual_compact:0.625:0.5,27:dual_compact:0.625:0.5,28:dual_compact:0.625:0.5,29:dual_compact:0.625:0.5,30:dual_compact:0.625:0.5,31:run_base:0.125"
export SGLANG_FD_FULL_GRAPH_DUAL_COMPACT_MIN_ROWS=4096
export SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV=1
export SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH=1
export SGLANG_FD_FULL_GRAPH_MASKED_DECODE_ATTENTION=1
export SGLANG_FD_FULL_GRAPH_SCHEDULER_CONVERGENCE=1
export SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_DIGEST=1
export SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER=/rds/user/wd312/hpc-work/llm/vpipe-csd3/serving/helper-sm80/libvpipe_cuda_conditional_graph.so
