#!/usr/bin/env bash
# Part C cell #1 -- the CLIENT-EQUIVALENCE GATE (D-796 C3, rehearsed on CSD3 per D-800).
#
# Question: does the sglang bench client reproduce lm-eval's score on the SAME served arm and the
# SAME documents? Table 2's arms C/D are driven by lm-eval; Part C's loaded rows would be driven by
# the bench. The difference-of-differences is robust to a UNIFORM client change (anything common to
# C and D cancels in D-C), but not to a model-dependent one -- so this must be measured, not
# assumed. It is currently asserted only in a code comment, and there is precedent for that failing
# silently (Gate B passed for weeks on lanes differing by exactly the BOS token it stripped).
#
# Runs INSIDE an INTR allocation. Never on a login node.
set -euo pipefail
R=${VPIPE_CSD3_ROOT:?set VPIPE_CSD3_ROOT to the campaign root on the cluster}
P=$R/partc
TREE=$P/tree
SUITES=$P/suites
SERVE=$R/envs/sglang-serve-c39f15c7-r1/bin/python
LMEVAL=$R/envs/lmeval-hf-r1/bin/python
MODEL=$R/models/Meta-Llama-3-8B-Instruct-53346005
OUT=${1:?usage: partc_client_gate.sh <out-dir> [arm] [workload]}
ARM=${2:-upstream}
WL=${3:-gsm8k}
PORT=${PORT:-33077}
mkdir -p "$OUT"
# The venv bin must be on PATH, not just its python: CUDA-graph capture shells out to
# `ninja` (present at $SERVE's sibling, 1.13.0) and dies FileNotFoundError without it.
export PATH="$(dirname "$SERVE"):$PATH"
# The node's default nvcc is CUDA 11.4, which rejects -std=c++20, and the newest module is
# 12.1 -- still short of the 13.0 torch was built against. The venv bundles a matching
# toolkit, so point CUDA_HOME at that rather than at a mismatched system one.
CU13="$(dirname "$SERVE")/../lib/python3.12/site-packages/nvidia/cu13"
if [ -x "$CU13/bin/nvcc" ]; then
  export CUDA_HOME="$(cd "$CU13" && pwd)"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
LOG(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/gate.log"; }

case "$WL" in
  gsm8k)   SUITE=gsm8k.qual;   TASK=gsm8k ;;
  coqa)    SUITE=coqa.qual;    TASK=coqa ;;
  bbh_cot) SUITE=bbh_cot.qual; TASK=bbh_cot_fewshot ;;
  *) LOG "unknown workload $WL"; exit 2 ;;
esac

LOG "=== Part C client-equivalence gate: arm=$ARM workload=$WL suite=$SUITE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$OUT/gate.log"

# --- serve the arm -------------------------------------------------------------
# D-609: the arm is NAMED in a durable file the served path reads, never exported.
echo "$ARM" > "$TREE/deploy/active_arm"
export SGLANG_VP_HOST_CONFIG="$TREE/deploy/hosts/csd3.json"
LOG "active_arm=$(cat "$TREE/deploy/active_arm") host_config=$SGLANG_VP_HOST_CONFIG"
LOG "booting $ARM on :$PORT"
PYTHONPATH=$TREE/python \
  $SERVE -u -m sglang.launch_server --model-path "$MODEL" \
  --revision 53346005fb0ef11d3b6a83b12c895cca40156b6c \
  --port "$PORT" --dtype float16 --mem-fraction-static 0.8 \
  --attention-backend triton --sampling-backend pytorch \
  > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { LOG "FATAL server died during boot"; tail -20 "$OUT/server.log"; exit 3; }
  if [ $(( i % 12 )) -eq 0 ]; then
    LOG "  boot $(( i*5 ))s: log=$(stat -c %s "$OUT/server.log" 2>/dev/null)B gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)MiB cpu=$(ps -o time= -p $SRV 2>/dev/null | tr -d ' ')"
  fi
  sleep 5
done
curl -s -m 5 "http://127.0.0.1:$PORT/server_info" > "$OUT/server_info.json" || true
LOG "server up"

# --- client A: lm-eval (the harness Table 2 uses) -------------------------------
LOG "client A -- lm-eval on $TASK"
PYTHONPATH=$TREE/test/vp $LMEVAL "$TREE/test/vp/run_lmeval_quality.py" \
  --workload "$WL" --arm "$ARM" --suite-dir "$SUITES" --suite-name "$SUITE" \
  --output-dir "$OUT/lmeval" --base-url "http://127.0.0.1:$PORT" \
  --tokenizer "$MODEL" --num-concurrent 16 --lmeval-python "$LMEVAL" \
  >> "$OUT/gate.log" 2>&1 || LOG "client A returned $?"

# --- client B: the sglang bench at negligible offered rate ----------------------
LOG "client B -- sglang bench at rate 0.5 (unloaded)"
PYTHONPATH=$TREE/test/vp $SERVE "$TREE/test/vp/run_qps_evaluation.py" \
  --requests "$SUITES/$SUITE.requests.jsonl" \
  --metadata "$SUITES/$SUITE.metadata.jsonl" \
  --base-url "http://127.0.0.1:$PORT" --request-rate 0.5 \
  --output-dir "$OUT/bench" \
  >> "$OUT/gate.log" 2>&1 || LOG "client B returned $?"

# --- score client B with lm-eval's own filters, eval-split rows ONLY ------------
LOG "scoring client B with lm-eval filters (source_split rows only, D-797)"
PYTHONPATH=$TREE/test/vp $LMEVAL "$TREE/test/vp/score_natural_lane_lmeval.py" \
  --suites "$SUITES" --cells "$OUT/bench"/*.jsonl --out "$OUT/bench_scores.json" \
  >> "$OUT/gate.log" 2>&1 || LOG "scorer returned $?"

LOG "=== GATE INPUTS COLLECTED -- compare $OUT/lmeval vs $OUT/bench_scores.json ==="
LOG "bar: the bench must reproduce lm-eval's score for this arm on the same documents"
LOG PARTC-GATE-DONE
