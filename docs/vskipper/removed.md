# What this branch removed

The development tree's `test/vp/` held 194 tracked files and 33,544 lines. This branch keeps the two reproduction
paths and the verification layer, and drops the rest.

## How the keep-set was derived

Not by taste. A file is kept if **any** of these holds:

1. it is reachable by Python import from a stage-2 or stage-3 entry point (the transitive closure, which pulled in
   exactly two modules nobody had listed);
2. it is named by something kept, by a campaign driver, by the paper's regeneration script, or by the paper itself;
3. it is a gate, in `gates/` or `gate_tests/`;
4. it is a root `test_*.py`, which together are the numerical proof for the kernel and cohort paths the refactor
   touches;
5. it is prompt data needed to rebuild a suite (`bbh_cot_fewshot/`).

Rule 2 is what rescued nine files an earlier by-eye list had dropped, among them the work-identity code gate, the
tree-equivalence gate, the workload builder, the quality-2x2 driver, the script that produces the roofline
evidence, and the CUDA helper build that stage 1 needs.

| | files | lines |
|---|---|---|
| kept | 157 | 24,840 |
| removed | 37 | 8,705 |

## What was removed, and why

### Exploratory and superseded tools

- `test/vp/analyze_termination.py`
- `test/vp/binary_cohort_reference.py`
- `test/vp/build_fair_output_workload.py`
- `test/vp/build_production_watchdog_workload.py`
- `test/vp/calib_layer_engagement.py`
- `test/vp/check_length_calibration.py`
- `test/vp/compare_fd_parity_traces.py`
- `test/vp/launch_qps_server.py`
- `test/vp/longbench_eval.py`
- `test/vp/run_fair_evaluation_contract_direct.py`
- `test/vp/stage_scrolls.py`
- `test/vp/steady_state_metrics.py`
- `test/vp/summarize_profile_trace.py`
- `test/vp/tune_fd_full_graph_moe.py`
- `test/vp/validate_runtime_after.py`
- `test/vp/vp_stream.py`

### Host-specific or superseded drivers

- `test/vp/csd3_debug_cell.sh`
- `test/vp/run_production_calibration.sh`
- `test/vp/run_qps_development.sh`
- `test/vp/vast_a100_ladder.sh`

### Microbenchmarks and oracles

- `test/vp/bench_cohort_gemm_paths.py`
- `test/vp/bench_fd_conditional_oracle.py`
- `test/vp/bench_fd_layer_policy_oracle.py`
- `test/vp/eos_margin_probe.py`
- `test/vp/flexidepth_oracle_runner.py`
- `test/vp/flexidepth_upstream_gsm8k_probe.py`
- `test/vp/test_flexidepth_oracle_runner.py`

### One-off A/B rig

- `test/vp/refactor_ab/README.md`
- `test/vp/refactor_ab/box_perf_ab.sh`
- `test/vp/refactor_ab/csd3_ab_arm.sh`
- `test/vp/refactor_ab/csd3_ab_chain.sh`
- `test/vp/refactor_ab/csd3_ab_compare.py`
- `test/vp/refactor_ab/csd3_ab_run.sh`
- `test/vp/refactor_ab/csd3_preflight.sh`
- `test/vp/refactor_ab/perf_client.py`
- `test/vp/refactor_ab/prove_endpoint_owner.py`

### Unused fixtures

- `test/vp/labeled_workload.cap.json`

Nothing in this list executes inside a served cell, so removing it cannot change a measured number. The
check that it cannot change a *reported* number is stage 3's own comparison: regenerating every table and figure
from the data pack with only the kept files, and diffing byte-for-byte against the shipped set.
