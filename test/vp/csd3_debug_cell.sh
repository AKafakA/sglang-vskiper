#!/usr/bin/env bash
# CSD3 DEBUG HARNESS — one workload x rate x arm cell at the STANDARD cell
# contract, derived from the full experiment harness components (sealed
# runner + live deployment manifest), with LENGTH-CALIBRATION CHECKING as
# the only post-step (no quality lane). Owner-ordered 2026-08-19.
#
# Contract enforcement (fail-closed):
#   - request count per cell = v1_cell_counts.json (D-220) or the D-277-A5
#     extension below; the pinned suite MUST match the contract count.
#   - radix policy per workload: gsm8k/coqa/gsm8k_cot ON (D-181/D-277-A5);
#     ifeval/humaneval OFF (repetition sampling).
#
# Env inputs:
#   WORKLOAD RATE ARM_ID EXPECTED_RUNTIME  (required)
#   ARM_ENV   — file sourced before server launch (the arm's export block);
#               empty/absent = dense arm (all SGLANG_FD*/regime vars unset)
#   SUITE_DIR — pinned suite dir (default $AB/workloads-pinned-v3)
#   RESULTS_TAG (default dbg1) · REPS (default 1) · PORT (default 30290)
#   EXTRA_SERVER_ARGS — appended verbatim (e.g. --chunked-prefill-size N)
set -uo pipefail
R=/rds/user/wd312/hpc-work/llm/vpipe-csd3
. /etc/profile.d/modules.sh; module purge
GREAL=/usr/local/software/spack/csd3/opt-2025-06-01/linux-rocky8-zen3/gcc-14.3.0/gcc-14.3.0-vlhhcp6mk32jxxqtnhkkmlrf2rpwwkrd
V=$R/envs/sglang-serve-w2-r1/bin/python
export CUDA_HOME=$R/envs/sglang-serve-w2-r1/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$GREAL/bin:$R/envs/sglang-serve-w2-r1/bin:/usr/bin:/bin"
export CC=$GREAL/bin/gcc CXX=$GREAL/bin/g++
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$GREAL/lib64"
export CPATH="$R/envs/sglang-serve-w2-r1/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
AB=$R/serving/prefill-ab
MODEL=$R/models/Meta-Llama-3-8B-Instruct-53346005
TREE=$R/serving/tree-w2

WORKLOAD=${WORKLOAD:?}; RATE=${RATE:?}; ARM_ID=${ARM_ID:?}; EXPECTED_RUNTIME=${EXPECTED_RUNTIME:?}
SUITE_DIR=${SUITE_DIR:-$AB/workloads-pinned-v3}
RESULTS_TAG=${RESULTS_TAG:-dbg1}
REPS=${REPS:-1}
PORT=${PORT:-30290}
ARM_ENV=${ARM_ENV:-}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}

# --- contract: cell count (D-220 + D-277-A5 extension) ---
COUNT=$($V - "$WORKLOAD" <<'PYEOF'
import json, sys
w = sys.argv[1]
counts = json.load(open("/rds/user/wd312/hpc-work/llm/vpipe-csd3/serving/tree-w2/test/vp/v1_cell_counts.json"))["counts"]
# D-277-A5: gsm8k_cot = 2400 (v1.2 gsm8k pool re-rendered); ifeval cells are
# repetition-sized per measured cell time — pass an explicit SUITE at build.
counts.setdefault("gsm8k_cot", 2400)
# DECLARED 2026-08-20 (owner prefill-dataset directive; pending D-220-style
# ratification): prefill_core cell = 7699 (coqa 20-min class).
counts.setdefault("prefill_core", 7699)
# DECLARED 2026-08-20 (owner in-session): mmlu_pro native 6400 / cot 2400.
counts.setdefault("mmlu_pro", 6400)
counts.setdefault("mmlu_pro_cot", 2400)
counts.setdefault("mixed", 7699)
if w not in counts:
    raise SystemExit(f"no contract count for workload {w!r}")
print(counts[w])
PYEOF
) || { echo "CONTRACT: unknown workload $WORKLOAD"; exit 1; }

SUITE_N=$(wc -l < $SUITE_DIR/$WORKLOAD.requests.jsonl) || { echo "CONTRACT: suite missing for $WORKLOAD"; exit 1; }
if [ "$SUITE_N" -ne "$COUNT" ]; then
  echo "CONTRACT VIOLATION: suite has $SUITE_N requests, contract requires $COUNT for $WORKLOAD"; exit 1
fi

# --- contract: radix policy (D-181 / D-277-A5) ---
case "$WORKLOAD" in
  gsm8k|coqa|gsm8k_cot|prefill_core|mmlu_pro|mmlu_pro_cot|mixed) RADIX_ARGS="" ;;
  ifeval|humaneval)     RADIX_ARGS="--disable-radix-cache" ;;
  *) echo "CONTRACT: no radix policy for $WORKLOAD"; exit 1 ;;
esac

OUT=$AB/arm_$ARM_ID; mkdir -p $OUT/deployment
RESULTS=$OUT/results-$RESULTS_TAG
[ -e "$RESULTS" ] && { echo "refusing to reuse $RESULTS"; exit 1; }
QCFG=$OUT/qps_${WORKLOAD}_${RATE}.json
printf '{"duration_s": 1, "workloads": {"%s": [%s]}}\n' "$WORKLOAD" "$RATE" > $QCFG
SRCREV=$(cat $TREE/SOURCE_REV 2>/dev/null || echo unknown)

echo "=== DBGCELL $ARM_ID $WORKLOAD@$RATE n=$COUNT reps=$REPS radix_args='$RADIX_ARGS' ($(date -u +%FT%TZ))"
# Clean posture: scrub every FD/VP knob BEFORE the arm env, so cells never
# inherit a previous arm's posture when chained in one shell.
for v in $(env | grep -oE "^(SGLANG_FD[A-Z_]*|SGLANG_VP[A-Z_]*|SGLANG_MOE_CONFIG_DIR)"); do unset $v; done
if [ -n "$ARM_ENV" ]; then . "$ARM_ENV"; fi
$V -m sglang.launch_server --model-path $MODEL --port $PORT \
  --attention-backend triton --prefill-attention-backend triton --decode-attention-backend triton \
  $RADIX_ARGS $EXTRA_SERVER_ARGS > $OUT/deployment/server.$RESULTS_TAG.log 2>&1 &
SPID=$!
for i in $(seq 1 200); do curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || break; sleep 6; done
curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || { echo "DBGCELL $ARM_ID SERVER FAILED"; tail -8 $OUT/deployment/server.$RESULTS_TAG.log; kill $SPID 2>/dev/null; exit 1; }
cd $TREE
$V test/vp/make_deployment_manifest.py --deployment-id $ARM_ID --system-id $ARM_ID \
  --model NousResearch/Meta-Llama-3-8B-Instruct --model-revision 53346005fb0ef11d3b6a83b12c895cca40156b6c \
  --client-tokenizer-path $MODEL --source-revision "$SRCREV" \
  --launch-command-file $R/serving/tree-w2/test/vp/csd3_debug_cell.sh --port $PORT \
  --artifact router_weights=$R/serving/flexidepth_router_weights.pt \
  --expected-runtime-json $EXPECTED_RUNTIME --out $OUT/deployment/manifest.$RESULTS_TAG.json || { kill $SPID; exit 1; }
$V test/vp/run_qps_evaluation.py --experiment $ARM_ID \
  --deployment-manifest $OUT/deployment/manifest.$RESULTS_TAG.json \
  --workload-dir $SUITE_DIR --qps-config $QCFG --min-prompts $COUNT \
  --output-dir $RESULTS --evidence-class development --reps $REPS \
  --runner-source-revision "$SRCREV" --port $PORT 2>&1 | tail -$((REPS + 2))
curl -s -m 10 http://127.0.0.1:$PORT/server_info > $OUT/server_info.$RESULTS_TAG.json
kill $SPID 2>/dev/null; sleep 8

# --- the ONLY post-step: length-calibration check ---
$V test/vp/check_length_calibration.py \
  --cell-jsonl $RESULTS/${WORKLOAD}_qps*_rep[0-9].jsonl \
  --suite-metadata $SUITE_DIR/$WORKLOAD.metadata.jsonl || { echo "DBGCELL $ARM_ID LENGTH-CALIBRATION FAILED"; exit 2; }
echo "=== DBGCELL $ARM_ID DONE ($(date -u +%FT%TZ))"
