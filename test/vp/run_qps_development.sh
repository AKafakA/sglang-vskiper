#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/python:$ROOT/test/vp${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:?set PYTHON_BIN to the existing remote SGLang venv Python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a fresh result directory}"
WORKLOAD_DIR="${WORKLOAD_DIR:?set WORKLOAD_DIR to the frozen workload suite}"
TOKENIZER_SNAPSHOT="${TOKENIZER_SNAPSHOT:?set TOKENIZER_SNAPSHOT}"
SERVER_ROOT="${SERVER_ROOT:?set SERVER_ROOT to the exact server source tree}"
SERVER_REVISION="${SERVER_REVISION:?set SERVER_REVISION}"
SERVER_SOURCE_ARCHIVE="${SERVER_SOURCE_ARCHIVE:?set SERVER_SOURCE_ARCHIVE}"
RUNNER_REVISION="${RUNNER_REVISION:?set RUNNER_REVISION}"
RUNNER_SOURCE_ARCHIVE="${RUNNER_SOURCE_ARCHIVE:?set RUNNER_SOURCE_ARCHIVE}"
SYSTEM_ID="${SYSTEM_ID:?set SYSTEM_ID}"
EXPECTED_RUNTIME="${EXPECTED_RUNTIME:?set EXPECTED_RUNTIME}"
QPS_CONFIG="${QPS_CONFIG:?set QPS_CONFIG}"
HF_HOME="${HF_HOME:?set HF_HOME to the complete model cache}"

HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_REVISION="${MODEL_REVISION:-53346005fb0ef11d3b6a83b12c895cca40156b6c}"
SERVER_PROFILE="${SERVER_PROFILE:-production}"
SYSTEM_CONFIG="${SYSTEM_CONFIG:-}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
DURATION_S="${DURATION_S:-180}"
MIN_PROMPTS="${MIN_PROMPTS:-256}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-8}"
REPS="${REPS:-1}"
REP_START="${REP_START:-1}"
CAMPAIGN_TOTAL_REPS="${CAMPAIGN_TOTAL_REPS:-$REPS}"
MAX_WALLTIME_S="${MAX_WALLTIME_S:-21600}"
RESUME_COMPLETED="${RESUME_COMPLETED:-0}"
CONTINUE_AFTER_ACCOUNTING_REJECTION="${CONTINUE_AFTER_ACCOUNTING_REJECTION:-0}"
BACKEND="${BACKEND:-sglang-native}"
ALLOW_CODE_EXECUTION="${ALLOW_CODE_EXECUTION:-0}"
EVIDENCE_CLASS="${EVIDENCE_CLASS:-development}"
PERFORMANCE_PHASE="${PERFORMANCE_PHASE:-}"
CONTRACT_LANE="${CONTRACT_LANE:-}"
EVALUATION_CONTRACT="${EVALUATION_CONTRACT:-}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "ERROR: refusing to reuse OUTPUT_ROOT=$OUTPUT_ROOT" >&2
  exit 1
fi
for path in \
  "$WORKLOAD_DIR" \
  "$TOKENIZER_SNAPSHOT" \
  "$SERVER_ROOT" \
  "$SERVER_SOURCE_ARCHIVE" \
  "$RUNNER_SOURCE_ARCHIVE" \
  "$EXPECTED_RUNTIME" \
  "$QPS_CONFIG" \
  "$ROOT/test/vp/launch_qps_server.py" \
  "$ROOT/test/vp/run_qps_evaluation.py" \
  "$ROOT/test/vp/validate_runtime_after.py"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path does not exist: $path" >&2
    exit 1
  fi
done
if [[ "$RESUME_COMPLETED" != "0" && "$RESUME_COMPLETED" != "1" ]]; then
  echo "ERROR: RESUME_COMPLETED must be 0 or 1" >&2
  exit 1
fi
if [[ "$CONTINUE_AFTER_ACCOUNTING_REJECTION" != "0" && \
  "$CONTINUE_AFTER_ACCOUNTING_REJECTION" != "1" ]]; then
  echo "ERROR: CONTINUE_AFTER_ACCOUNTING_REJECTION must be 0 or 1" >&2
  exit 1
fi
if [[ "$ALLOW_CODE_EXECUTION" != "0" && "$ALLOW_CODE_EXECUTION" != "1" ]]; then
  echo "ERROR: ALLOW_CODE_EXECUTION must be 0 or 1" >&2
  exit 1
fi
if [[ "$EVIDENCE_CLASS" != "smoke" && \
  "$EVIDENCE_CLASS" != "development" && \
  "$EVIDENCE_CLASS" != "performance" ]]; then
  echo "ERROR: EVIDENCE_CLASS must be smoke, development, or performance" >&2
  exit 1
fi
if [[ "$EVIDENCE_CLASS" == "smoke" && \
  ( "$REPS" != "1" || "$REP_START" != "1" || \
    "$CAMPAIGN_TOTAL_REPS" != "1" || "$MIN_PROMPTS" -gt 32 ) ]]; then
  echo "ERROR: smoke requires one repetition and min-prompts<=32" >&2
  exit 1
fi
if [[ "$EVIDENCE_CLASS" == "smoke" ]] && \
  ! awk -v duration="$DURATION_S" 'BEGIN { exit !(duration <= 60) }'; then
  echo "ERROR: smoke requires duration<=60" >&2
  exit 1
fi
if [[ "$EVIDENCE_CLASS" == "performance" && -z "$PERFORMANCE_PHASE" ]]; then
  echo "ERROR: performance evidence requires PERFORMANCE_PHASE" >&2
  exit 1
fi
if [[ "$EVIDENCE_CLASS" == "performance" && -z "$EVALUATION_CONTRACT" ]]; then
  echo "ERROR: performance evidence requires EVALUATION_CONTRACT" >&2
  exit 1
fi
if [[ -n "$EVALUATION_CONTRACT" && ! -f "$EVALUATION_CONTRACT" ]]; then
  echo "ERROR: evaluation contract does not exist: $EVALUATION_CONTRACT" >&2
  exit 1
fi

evaluation_args=(
  --experiment "$SYSTEM_ID"
  --deployment-manifest "$OUTPUT_ROOT/deployment/deployment_manifest.json"
  --workload-dir "$WORKLOAD_DIR"
  --qps-config "$QPS_CONFIG"
  --output-dir "$OUTPUT_ROOT/results"
  --evidence-class "$EVIDENCE_CLASS"
  --backend "$BACKEND"
  --runner-source-revision "$RUNNER_REVISION"
  --runner-source-archive "$RUNNER_SOURCE_ARCHIVE"
  --duration-s "$DURATION_S"
  --min-prompts "$MIN_PROMPTS"
  --warmup-requests "$WARMUP_REQUESTS"
  --reps "$REPS"
  --rep-start "$REP_START"
  --campaign-total-reps "$CAMPAIGN_TOTAL_REPS"
  --max-walltime-s "$MAX_WALLTIME_S"
)
if [[ -n "$PERFORMANCE_PHASE" ]]; then
  evaluation_args+=(--performance-phase "$PERFORMANCE_PHASE")
fi
if [[ -n "$CONTRACT_LANE" ]]; then
  evaluation_args+=(--contract-lane "$CONTRACT_LANE")
fi
if [[ -n "$EVALUATION_CONTRACT" ]]; then
  evaluation_args+=(--evaluation-contract "$EVALUATION_CONTRACT")
fi
if [[ "$RESUME_COMPLETED" == "1" ]]; then
  evaluation_args+=(--resume-completed)
fi
if [[ "$CONTINUE_AFTER_ACCOUNTING_REJECTION" == "1" ]]; then
  evaluation_args+=(--continue-after-accounting-rejection)
fi
if [[ "$ALLOW_CODE_EXECUTION" == "1" ]]; then
  evaluation_args+=(--allow-code-execution)
fi

if [[ -n "$EVALUATION_CONTRACT" ]]; then
  "$PYTHON_BIN" "$ROOT/test/vp/run_qps_evaluation.py" \
    "${evaluation_args[@]}" --contract-preflight-only
fi

model_path_arg=()
if [[ -n "$MODEL_PATH" ]]; then
  if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH is not a directory: $MODEL_PATH" >&2
    exit 1
  fi
  model_path_arg+=(--model-path "$MODEL_PATH")
fi

launch_extra=()
if [[ -n "$SYSTEM_CONFIG" ]]; then
  jq -e '
    (type == "object") and
    ((.server_env // {}) | type == "object") and
    ((.artifacts // {}) | type == "object") and
    ((.server_args // []) | type == "array") and
    ((.server_args // []) | all(type == "string"))
  ' "$SYSTEM_CONFIG" >/dev/null
  launch_extra+=("--artifact=system_config=$SYSTEM_CONFIG")
  while IFS= read -r entry; do
    entry="${entry//\{SERVER_ROOT\}/$SERVER_ROOT}"
    launch_extra+=("--server-env=$entry")
  done < <(jq -r '(.server_env // {}) | to_entries[] | "\(.key)=\(.value)"' "$SYSTEM_CONFIG")
  while IFS= read -r entry; do
    entry="${entry//\{SERVER_ROOT\}/$SERVER_ROOT}"
    launch_extra+=("--artifact=$entry")
  done < <(jq -r '(.artifacts // {}) | to_entries[] | "\(.key)=\(.value)"' "$SYSTEM_CONFIG")
  while IFS= read -r entry; do
    launch_extra+=("--server-arg=$entry")
  done < <(jq -r '(.server_args // [])[]' "$SYSTEM_CONFIG")
fi

mkdir -p "$OUTPUT_ROOT"
launcher_log="$OUTPUT_ROOT/deployment.launcher.log"
launcher_pid=""

stop_launcher() {
  if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "$launcher_pid"
    local started="$SECONDS"
    while kill -0 "$launcher_pid" 2>/dev/null; do
      if (( SECONDS - started >= 60 )); then
        kill -KILL "$launcher_pid" 2>/dev/null || true
        if [[ -s "$OUTPUT_ROOT/deployment/deployment_state.json" ]]; then
          local server_pid
          server_pid="$(jq -r '.server_pid // empty' "$OUTPUT_ROOT/deployment/deployment_state.json")"
          if [[ "$server_pid" =~ ^[0-9]+$ ]]; then
            kill -TERM -- "-$server_pid" 2>/dev/null || true
            sleep 5
            kill -KILL -- "-$server_pid" 2>/dev/null || true
          fi
        fi
        break
      fi
      sleep 1
    done
    wait "$launcher_pid" || true
  fi
  launcher_pid=""
}
trap stop_launcher EXIT

"$PYTHON_BIN" "$ROOT/test/vp/launch_qps_server.py" \
  --deployment-id "${SYSTEM_ID}_${PERFORMANCE_PHASE:-development}" \
  --system-id "$SYSTEM_ID" \
  --model "$MODEL" \
  "${model_path_arg[@]}" \
  --model-revision "$MODEL_REVISION" \
  --client-tokenizer-path "$TOKENIZER_SNAPSHOT" \
  --server-source-root "$SERVER_ROOT" \
  --server-source-revision "$SERVER_REVISION" \
  --server-python "$PYTHON_BIN" \
  --launcher-source-revision "$RUNNER_REVISION" \
  --server-profile "$SERVER_PROFILE" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --expected-runtime "$EXPECTED_RUNTIME" \
  --server-env "HF_HOME=$HF_HOME" \
  --server-env "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
  --artifact "source_archive=$SERVER_SOURCE_ARCHIVE" \
  --server-arg=--random-seed=42 \
  --output-dir "$OUTPUT_ROOT/deployment" \
  "${launch_extra[@]}" >"$launcher_log" 2>&1 &
launcher_pid="$!"

started="$SECONDS"
while [[ ! -s "$OUTPUT_ROOT/deployment/deployment_manifest.json" ]]; do
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    echo "ERROR: launcher exited before deployment became ready" >&2
    wait "$launcher_pid" || true
    exit 1
  fi
  if (( SECONDS - started >= 900 )); then
    echo "ERROR: timed out waiting for deployment readiness" >&2
    exit 1
  fi
  sleep 2
done

"$PYTHON_BIN" "$ROOT/test/vp/run_qps_evaluation.py" "${evaluation_args[@]}"
"$PYTHON_BIN" "$ROOT/test/vp/validate_runtime_after.py" \
  --server-info "$OUTPUT_ROOT/results/server_info.after.json" \
  --expected-runtime "$EXPECTED_RUNTIME" \
  --server-profile "$SERVER_PROFILE" \
  --output "$OUTPUT_ROOT/results/runtime_after.audit.json"

stop_launcher
if pgrep -af "sglang.launch_server.*--port $PORT" >/dev/null; then
  echo "ERROR: server survived development-run shutdown on port $PORT" >&2
  exit 1
fi

echo "QPS_ENDPOINT_COMPLETED system=$SYSTEM_ID output=$OUTPUT_ROOT"
