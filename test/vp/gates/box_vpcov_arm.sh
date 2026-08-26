#!/usr/bin/env bash
# Reachability coverage for ONE canonical arm on the box (sm75).
#
# Sources the sealed arm env verbatim from gates/arms/, remaps the CSD3-absolute
# paths to their box equivalents, applies the Turing serve-time overrides from
# box_t0_smoke.sh, and runs a 32-request smoke with the vp coverage tracer on
# PYTHONPATH (when a tracer dir is present). Reachability only -- NEVER a
# performance number (sm75, and the A100 MoE tuning dir is deliberately
# dropped).
#
# Repo-resident since dev-h-harness-selfcontained (D-307 Lane H); logic is the
# sealed 2026-08-24 gate driver verbatim -- only path resolution is
# parameterized. Host-staged inputs (not in the repo): the model snapshot, the
# router weights, the sm75 helper .so, the frozen request suite, the serve
# venv, and (optionally) the coverage tracer package.
#
# Usage: box_vpcov_arm.sh <arm_name>   e.g. vpre_binarycohort
set -uo pipefail

ARM="${1:?usage: box_vpcov_arm.sh <arm_name>}"
GATES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$GATES_DIR/../../.." && pwd)"

W="${VP_GATE_WORKDIR:-/local/scratch/tmp/wd312}"
V="${VP_GATE_PYTHON:-$W/serve-venv-d248/bin/python}"
PORT="${VP_GATE_PORT:-30250}"
MODEL="${VP_GATE_MODEL:-$W/models/Meta-Llama-3-8B-Instruct-53346005}"
SUITE="${VP_GATE_SUITE:-$W/suites/gsm8k.first100.requests.jsonl}"
OUT="${VP_GATE_OUT:-$W/vpcov-out}/$ARM"
ENVFILE="${VP_GATE_ARM_DIR:-$GATES_DIR/arms}/arm_env_${ARM}.sh"
TRACER_DIR="${VPCOV_TRACER_DIR:-$W/vpcov}"

[ -f "$ENVFILE" ] || { echo "FATAL: no arm env $ENVFILE"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"

# --- the arm's own posture, verbatim -----------------------------------------
set -a
# shellcheck disable=SC1090
source "$ENVFILE"
set +a

# --- box remaps (the only edits to the sealed posture) ------------------------
export SGLANG_FD_WEIGHTS="${VP_GATE_FD_WEIGHTS:-$W/flexidepth_router_weights.pt}"
if [ -n "${SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER:-}" ]; then
  export SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER="${VP_GATE_SM_HELPER:-$W/helper-sm75/libvpipe_cuda_conditional_graph.so}"
fi
# A100 large-M MoE tunings are meaningless on Turing and would only perturb
# kernel selection, not which vp/ functions execute. Declared deviation.
unset SGLANG_MOE_CONFIG_DIR

# --- Turing serve-time overrides (identical to box_t0_smoke.sh) ---------------
export SGLANG_IS_FLASHINFER_AVAILABLE=false

# --- the tracer (optional: reachability audit; absent dir = attestation-only) --
export VP_COVERAGE_OUT=$OUT
export VP_COVERAGE_LABEL=$ARM
if [ -d "$TRACER_DIR" ]; then
  export PYTHONPATH=$TRACER_DIR:${PYTHONPATH:-}
fi

{
  echo "=== VPCOV $ARM start $(date -u +%FT%TZ)"
  echo "--- posture (SGLANG_* actually exported) ---"
  env | grep -E '^SGLANG_' | sort
} > "$OUT/posture.txt"

$V -m sglang.launch_server --model-path "$MODEL" \
  --port $PORT --dtype float16 \
  --attention-backend triton --prefill-attention-backend triton --decode-attention-backend triton \
  --disable-radix-cache > "$OUT/server.log" 2>&1 &
SPID=$!

for _ in $(seq 1 120); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || break
  sleep 6
done

if ! curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "VPCOV $ARM SERVER FAILED" | tee -a "$OUT/posture.txt"
  tail -25 "$OUT/server.log" | tee -a "$OUT/posture.txt"
  kill $SPID 2>/dev/null
  # A boot failure is a RESULT (this arm is untraceable on sm75), not a crash.
  exit 3
fi

curl -s -m 10 "http://127.0.0.1:$PORT/server_info" > "$OUT/server_info.json"

$V "$REPO_ROOT/test/vp/fdpre_label_probe.py" --url "http://127.0.0.1:$PORT" \
  --requests-jsonl "$SUITE" --first-n 32 \
  --concurrency 8 --max-new-tokens 128 --topk 20 \
  --output-dir "$OUT/probe" 2>&1 | tail -3 | tee -a "$OUT/posture.txt"

# SIGTERM so the tracer's handler flushes before the interpreter goes away.
kill $SPID 2>/dev/null
sleep 8
kill -9 $SPID 2>/dev/null

echo "=== VPCOV $ARM done $(date -u +%FT%TZ)  traces=$(ls "$OUT"/vpcov-*.json 2>/dev/null | wc -l)" | tee -a "$OUT/posture.txt"
