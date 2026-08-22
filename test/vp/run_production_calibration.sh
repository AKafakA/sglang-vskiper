#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT/python:$ROOT/test/vp${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:?set PYTHON_BIN to the existing remote SGLang venv Python}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new calibration artifact directory}"
WORKLOAD_DIR="${WORKLOAD_DIR:?set WORKLOAD_DIR to the frozen workload suite}"
TOKENIZER_SNAPSHOT="${TOKENIZER_SNAPSHOT:?set TOKENIZER_SNAPSHOT}"
UPSTREAM_SERVER_ROOT="${UPSTREAM_SERVER_ROOT:?set UPSTREAM_SERVER_ROOT}"
UPSTREAM_SERVER_REVISION="${UPSTREAM_SERVER_REVISION:?set UPSTREAM_SERVER_REVISION}"
UPSTREAM_SOURCE_ARCHIVE="${UPSTREAM_SOURCE_ARCHIVE:?set UPSTREAM_SOURCE_ARCHIVE}"
RUNNER_SOURCE_REVISION="${RUNNER_SOURCE_REVISION:?set RUNNER_SOURCE_REVISION}"
RUNNER_SOURCE_ARCHIVE="${RUNNER_SOURCE_ARCHIVE:?set RUNNER_SOURCE_ARCHIVE}"
EVALUATION_CONTRACT="${EVALUATION_CONTRACT:-}"
QPS_CONFIG="${QPS_CONFIG:-$ROOT/test/vp/vanilla_qps_calibration.example.json}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
MODEL_REVISION="${MODEL_REVISION:-53346005fb0ef11d3b6a83b12c895cca40156b6c}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
ALLOW_CODE_EXECUTION="${ALLOW_CODE_EXECUTION:-0}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "ERROR: refusing to reuse OUTPUT_ROOT=$OUTPUT_ROOT" >&2
  exit 1
fi
if [[ "$ALLOW_CODE_EXECUTION" != "0" && "$ALLOW_CODE_EXECUTION" != "1" ]]; then
  echo "ERROR: ALLOW_CODE_EXECUTION must be 0 or 1" >&2
  exit 1
fi
if [[ -n "$EVALUATION_CONTRACT" && ! -f "$EVALUATION_CONTRACT" ]]; then
  echo "ERROR: evaluation manifest does not exist: $EVALUATION_CONTRACT" >&2
  exit 1
fi

# Validate every hard-coded input BEFORE creating output or starting the
# background launcher. The runtime contract below and the default QPS config
# are both passed unconditionally, and both are absent from this tree: without
# this the entry point created its output root, launched a server, and only
# then failed on a path it could have checked in the first second.
EXPECTED_RUNTIME_PATH="$ROOT/test/vp/runtime_expectations/production_upstream.json"
for path in \
  "$ROOT/test/vp/launch_qps_server.py" \
  "$EXPECTED_RUNTIME_PATH" \
  "$QPS_CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path does not exist: $path" >&2
    echo "       (dropped with the evaluation tooling; recover with" >&2
    echo "        'git show c3a1302668:\${path#$ROOT/}' -- see the" >&2
    echo "        removed-feature register)" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
launcher_log="$OUTPUT_ROOT/deployment.launcher.log"
launcher_pid=""

stop_launcher() {
  if [[ -n "$launcher_pid" ]] && kill -0 "$launcher_pid" 2>/dev/null; then
    kill -TERM "$launcher_pid"
    local started="$SECONDS"
    while kill -0 "$launcher_pid" 2>/dev/null; do
      if (( SECONDS - started >= 60 )); then
        echo "ERROR: launcher did not stop within 60 seconds" >&2
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
  --deployment-id production_vanilla_calibration \
  --system-id production_vanilla \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --client-tokenizer-path "$TOKENIZER_SNAPSHOT" \
  --server-source-root "$UPSTREAM_SERVER_ROOT" \
  --server-source-revision "$UPSTREAM_SERVER_REVISION" \
  --server-python "$PYTHON_BIN" \
  --launcher-source-revision "$RUNNER_SOURCE_REVISION" \
  --server-profile production \
  --host 127.0.0.1 \
  --port "$PORT" \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  --expected-runtime "$EXPECTED_RUNTIME_PATH" \
  --artifact "source_archive=$UPSTREAM_SOURCE_ARCHIVE" \
  --output-dir "$OUTPUT_ROOT/deployment" >"$launcher_log" 2>&1 &
launcher_pid="$!"

started="$SECONDS"
while [[ ! -s "$OUTPUT_ROOT/deployment/deployment_manifest.json" ]]; do
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    echo "ERROR: launcher exited before deployment became ready" >&2
    wait "$launcher_pid" || true
    exit 1
  fi
  if (( SECONDS - started >= 900 )); then
    echo "ERROR: timed out waiting for deployment manifest" >&2
    exit 1
  fi
  sleep 2
done

evaluation_args=(
  --experiment production_vanilla
  --deployment-manifest "$OUTPUT_ROOT/deployment/deployment_manifest.json"
  --workload-dir "$WORKLOAD_DIR"
  --qps-config "$QPS_CONFIG"
  --output-dir "$OUTPUT_ROOT/results"
  --evidence-class performance
  --performance-phase calibration
  --runner-source-revision "$RUNNER_SOURCE_REVISION"
  --runner-source-archive "$RUNNER_SOURCE_ARCHIVE"
  --duration-s 180
  --min-prompts 256
  --warmup-requests 32
  --reps 1
  --max-walltime-s 21600
)
if [[ -n "$EVALUATION_CONTRACT" ]]; then
  evaluation_args+=(--evaluation-contract "$EVALUATION_CONTRACT")
fi
if [[ "$ALLOW_CODE_EXECUTION" == "1" ]]; then
  evaluation_args+=(--allow-code-execution)
fi
"$PYTHON_BIN" "$ROOT/test/vp/run_qps_evaluation.py" "${evaluation_args[@]}"

stop_launcher
if pgrep -af "sglang.launch_server.*--port $PORT" >/dev/null; then
  echo "ERROR: SGLang process survived calibration shutdown on port $PORT" >&2
  pgrep -af "sglang.launch_server.*--port $PORT" >&2 || true
  exit 1
fi

echo "PRODUCTION_CALIBRATION_COMPLETED output=$OUTPUT_ROOT"
