#!/usr/bin/env bash
set -euo pipefail

# Vast A100 run ladder for VP ASPLOS validation.
# Defaults to dry-run and refuses non-/workspace checkouts unless explicitly
# overridden for local syntax checks.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${ACTION:-preflight}"
DRY_RUN="${DRY_RUN:-1}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
PROFILE_FILE="${PROFILE_FILE:-$ROOT/test/vp/qwen3_8b_layer_importance.json}"
OUT_BASE="${OUT_BASE:-/workspace}"
SGBENCH_MODES_WAS_SET="${SGBENCH_MODES+x}"
SGBENCH_DATASET_WAS_SET="${SGBENCH_DATASET+x}"
SGBENCH_NUM_PROMPTS_WAS_SET="${SGBENCH_NUM_PROMPTS+x}"
SGBENCH_MAX_CONCURRENCY_WAS_SET="${SGBENCH_MAX_CONCURRENCY+x}"
SGBENCH_BACKEND_WAS_SET="${SGBENCH_BACKEND+x}"
SGBENCH_CONTEXT_LEN_WAS_SET="${SGBENCH_CONTEXT_LEN+x}"
SGBENCH_OUTPUT_LEN_WAS_SET="${SGBENCH_OUTPUT_LEN+x}"
SGBENCH_MAX_RUNNING_REQUESTS_WAS_SET="${SGBENCH_MAX_RUNNING_REQUESTS+x}"
SGBENCH_CHUNKED_PREFILL_SIZE_WAS_SET="${SGBENCH_CHUNKED_PREFILL_SIZE+x}"
SGBENCH_EXTRA_REQUEST_BODY="${SGBENCH_EXTRA_REQUEST_BODY:-}"
SGBENCH_DATASET_PATH="${SGBENCH_DATASET_PATH:-}"
SGBENCH_PORT="${SGBENCH_PORT:-30000}"
SGBENCH_OUT_DIR="${SGBENCH_OUT_DIR:-$OUT_BASE/sgbench_vp}"
# These eight knobs are LIVE: the package reads their SGLANG_ counterparts
# outside any rejection list, and the canonical vdec_fd arm sets
# SGLANG_FD_VP_FUSED_PROJECT_INPUT=1 -- so the ladder must be able to
# reproduce that posture. They were wrongly swept into the removed list by a
# name heuristic; membership is now decided by whether the package actually
# reads the variable.
SGBENCH_FD_VP_FUSED_PROJECT_INPUT="${SGBENCH_FD_VP_FUSED_PROJECT_INPUT:-0}"
SGBENCH_FD_VP_FUSED_ROUTER_DEC_HEAD="${SGBENCH_FD_VP_FUSED_ROUTER_DEC_HEAD:-0}"
SGBENCH_FD_VP_ROUTER_GRAPH="${SGBENCH_FD_VP_ROUTER_GRAPH:-0}"
SGBENCH_FD_VP_ROUTER_GRAPH_MAX_ENTRIES="${SGBENCH_FD_VP_ROUTER_GRAPH_MAX_ENTRIES:-4}"
SGBENCH_FD_VP_ROUTER_GRAPH_MAX_ROWS="${SGBENCH_FD_VP_ROUTER_GRAPH_MAX_ROWS:-64}"
SGBENCH_FD_VP_TRACE="${SGBENCH_FD_VP_TRACE:-0}"
SGBENCH_FD_VP_TRACE_FILE="${SGBENCH_FD_VP_TRACE_FILE:-}"
SGBENCH_FD_VP_TRACE_TIMING="${SGBENCH_FD_VP_TRACE_TIMING:-0}"

# Refuse removed treatment knobs the operator actually SET. The suffix
# subsystem these drove was deleted with the V1 path: the ladder now emits only
# SGLANG_FD_WEIGHTS, SGLANG_FD_EXECUTION_MODE and
# SGLANG_VP_FOREGROUND_STREAM_PRIORITY. Accepting them silently let an operator
# request a treatment, see it echoed back as configured, and get an arm that ran
# identically to the base -- the same experiment-integrity failure the mode
# whitelist exists to prevent. Checked with ${VAR+set} so a knob left unset is
# fine and only an explicit request fails.
_vp_removed_knobs_set=()
for _k in SGBENCH_DECODE_BLOCK_SIZE \
    SGBENCH_DECODE_COHORT \
    SGBENCH_DECODE_GATHER \
    SGBENCH_DECODE_POLICY \
    SGBENCH_DECODE_SPAN \
    SGBENCH_DECODE_STAGE_BLOCK_SYNC \
    SGBENCH_DECODE_STAGE_FUSE_MIXED \
    SGBENCH_DECODE_STAGE_POLICY \
    SGBENCH_DECODE_STAGE_SCHED \
    SGBENCH_DECODE_TOPKS \
    SGBENCH_DECODE_VETO_TAGS \
    SGBENCH_DECODE_VETO_TEXT_RE \
    SGBENCH_DECODE_VP_GRAPH \
    SGBENCH_FD_VP_ALL_SKIP_KV_ONLY_QKV \
    SGBENCH_FD_VP_ASYNC_KV \
    SGBENCH_FD_VP_ASYNC_KV_DEFER_DRAIN \
    SGBENCH_FD_VP_ASYNC_KV_SCOPED \
    SGBENCH_FD_VP_COALESCED_LAYER \
    SGBENCH_FD_VP_COALESCED_MIN_RUN_ROWS \
    SGBENCH_FD_VP_COALESCED_MIN_SKIP_ROWS \
    SGBENCH_FD_VP_DECODE_SUBBATCH_CACHE \
    SGBENCH_FD_VP_GLOBAL_DECODE_SUBBATCH_CACHE \
    SGBENCH_FD_VP_GLOBAL_DECODE_SUBBATCH_CACHE_MAX_ENTRIES \
    SGBENCH_FD_VP_MINIMAL_KV_SUBBATCH \
    SGBENCH_FD_VP_MIXED_DECODE_ASYNC_COHORT_MAX_ENTRIES \
    SGBENCH_FD_VP_MIXED_DECODE_ASYNC_COHORT_TABLE \
    SGBENCH_FD_VP_MIXED_DECODE_ASYNC_KV \
    SGBENCH_FD_VP_MIXED_DECODE_ASYNC_MIN_KEPT_ROWS \
    SGBENCH_FD_VP_MIXED_DECODE_ASYNC_MIN_ROWS \
    SGBENCH_FD_VP_MIXED_DECODE_ASYNC_MIN_SKIP_ROWS \
    SGBENCH_FD_VP_MIXED_DECODE_ASYNC_PATTERN_SCOPE \
    SGBENCH_FD_VP_MIXED_DECODE_ASYNC_PATTERN_WARMUP \
    SGBENCH_FD_VP_MIXED_DECODE_SUBBATCH \
    SGBENCH_FD_VP_MIXED_FULL_BATCH_SPLIT \
    SGBENCH_FD_VP_MIXED_FULL_PROJECT_BASE \
    SGBENCH_FD_VP_MIXED_INPLACE_WEIGHT \
    SGBENCH_FD_VP_MIXED_NEAR_ALL_RUN_FULL_MLP \
    SGBENCH_FD_VP_MIXED_NEAR_ALL_RUN_MAX_SKIP_ROWS \
    SGBENCH_FD_VP_MIXED_PARALLEL_BRANCHES \
    SGBENCH_FD_VP_MIXED_POST_WEIGHT \
    SGBENCH_FD_VP_MIXED_REUSE_OUTPUT_BUFFER \
    SGBENCH_FD_VP_MIXED_REUSE_OUTPUT_BUFFER_MAX_ENTRIES \
    SGBENCH_FD_VP_MIXED_SHAPE_LANE_BUFFERS \
    SGBENCH_FD_VP_MIXED_SHAPE_LANE_BUFFER_MAX_ENTRIES \
    SGBENCH_FD_VP_MIXED_SPLIT_GRAPH \
    SGBENCH_FD_VP_MIXED_SPLIT_GRAPH_MAX_ENTRIES \
    SGBENCH_FD_VP_MIXED_SPLIT_GRAPH_MAX_ROWS \
    SGBENCH_FD_VP_MIXED_SPLIT_GRAPH_MIN_FREE_MB \
    SGBENCH_FD_VP_SCHED_GRAPH \
    SGBENCH_FD_VP_STABLE_MIXED_SPLIT_BUFFERS \
    SGBENCH_FD_VP_STABLE_MIXED_SPLIT_BUFFER_MAX_ENTRIES \
    SGBENCH_FD_VP_STAGE_ROUTE \
    SGBENCH_FD_VP_STAGE_ROUTE_FUSE_HOMOGENEOUS \
    SGBENCH_FD_VP_STAGE_ROUTE_MIN_SPLIT_RUN_ROWS \
    SGBENCH_FD_VP_STAGE_ROUTE_MIN_SPLIT_SKIP_ROWS \
    SGBENCH_FD_VP_STAGE_ROUTE_RUNAHEAD \
    SGBENCH_FD_VP_TRACE_EVERY \
    SGBENCH_FD_VP_TRACE_MASKS \
    SGBENCH_FD_VP_TRACE_MASK_CAP \
    SGBENCH_FD_VP_TRITON_GPU_SUBBATCH \
    SGBENCH_PREFILL_BLOCK_SIZE \
    SGBENCH_PREFILL_POLICY \
    SGBENCH_PREFILL_TOPKS \
    SGBENCH_PREFILL_VETO_TAGS \
    SGBENCH_PREFILL_VETO_TEXT_RE \
    SGBENCH_VP_ASYNC_KV \
    SGBENCH_VP_ASYNC_KV_BATCHED \
    SGBENCH_VP_ASYNC_KV_BATCHED_MAX_ROWS \
    SGBENCH_VP_ASYNC_KV_BATCHED_TOKEN_LAUNCH \
    SGBENCH_VP_ASYNC_KV_DEFER_DRAIN \
    SGBENCH_VP_ASYNC_KV_GROUPED_EVENT \
    SGBENCH_VP_ASYNC_KV_LOOKAHEAD_RELEASE \
    SGBENCH_VP_ASYNC_KV_RELEASE_LAYER \
    SGBENCH_VP_ASYNC_KV_REPAIR_GRAPH \
    SGBENCH_VP_ASYNC_KV_REPAIR_GRAPH_MAX_ENTRIES \
    SGBENCH_VP_ASYNC_KV_REPAIR_GRAPH_MAX_ROWS \
    SGBENCH_VP_ASYNC_KV_SCOPED \
    SGBENCH_VP_KV_ONLY_QKV; do
  if [[ -n "${!_k+set}" ]]; then
    _vp_removed_knobs_set+=("$_k")
  fi
done
if (( ${#_vp_removed_knobs_set[@]} )); then
  echo "FATAL: these treatment knobs drive an implementation that is not in" >&2
  echo "       this build, so setting them would record a treatment that did" >&2
  echo "       not run: ${_vp_removed_knobs_set[*]}" >&2
  echo "       Unset them. See" >&2
  echo "       codex/asplos-plan/2026-08-21-removed-feature-register.md" >&2
  exit 2
fi

SGBENCH_MODES="${SGBENCH_MODES:-vanilla,vanilla_matched,flexidepth}"
SGBENCH_DATASET="${SGBENCH_DATASET:-random}"
SGBENCH_NUM_PROMPTS="${SGBENCH_NUM_PROMPTS:-2048}"
SGBENCH_MAX_CONCURRENCY="${SGBENCH_MAX_CONCURRENCY:-}"
SGBENCH_REQUEST_RATE="${SGBENCH_REQUEST_RATE:-inf}"
SGBENCH_SEED="${SGBENCH_SEED:-42}"
SGBENCH_CONTEXT_LEN="${SGBENCH_CONTEXT_LEN:-8192}"
SGBENCH_OUTPUT_LEN="${SGBENCH_OUTPUT_LEN:-128}"
SGBENCH_RANDOM_INPUT_LEN="${SGBENCH_RANDOM_INPUT_LEN:-512}"
SGBENCH_RANDOM_OUTPUT_LEN="${SGBENCH_RANDOM_OUTPUT_LEN:-128}"
SGBENCH_RANDOM_RANGE_RATIO="${SGBENCH_RANDOM_RANGE_RATIO:-1.0}"
SGBENCH_WARMUP_REQUESTS="${SGBENCH_WARMUP_REQUESTS:-32}"
SGBENCH_BACKEND="${SGBENCH_BACKEND:-sglang-native}"
SGBENCH_REPS="${SGBENCH_REPS:-3}"
SGBENCH_EVIDENCE_CLASS="${SGBENCH_EVIDENCE_CLASS:-diagnostic}"
SGBENCH_LOAD_SAMPLE_INTERVAL_MS="${SGBENCH_LOAD_SAMPLE_INTERVAL_MS:-1000}"
SGBENCH_SERVER_CAP_PURPOSE="${SGBENCH_SERVER_CAP_PURPOSE:-auto}"
SGBENCH_PROFILE="${SGBENCH_PROFILE:-0}"
SGBENCH_PROFILE_BY_STAGE="${SGBENCH_PROFILE_BY_STAGE:-0}"
SGBENCH_PROFILE_NUM_STEPS="${SGBENCH_PROFILE_NUM_STEPS:-}"
SGBENCH_PROFILE_OUTPUT_DIR="${SGBENCH_PROFILE_OUTPUT_DIR:-}"
SGBENCH_PROFILE_PREFIX="${SGBENCH_PROFILE_PREFIX:-}"
SGBENCH_ORDER_POLICY="${SGBENCH_ORDER_POLICY:-counterbalanced}"
SGBENCH_BASELINE_SERVER_PROFILE="${SGBENCH_BASELINE_SERVER_PROFILE:-production}"
SGBENCH_CANDIDATE_SERVER_PROFILE="${SGBENCH_CANDIDATE_SERVER_PROFILE:-breakable_dynamic}"
SGBENCH_VP_FOREGROUND_STREAM_PRIORITY="${SGBENCH_VP_FOREGROUND_STREAM_PRIORITY:-0}"
# The mode named "flexidepth" must actually RUN FlexiDepth. Left unset,
# flexidepth_execution_mode() defaults to direct_eager, whose routed-row
# gather is a device->host sync and illegal under stream capture -- so with
# the default breakable_dynamic profile (graphs ON) startup validation
# refused it and NO treated arm could boot. full_graph is the only live
# production dispatch; direct_eager stays available for the quality
# reference but requires a graphless profile, enforced below.
SGBENCH_FD_EXECUTION_MODE="${SGBENCH_FD_EXECUTION_MODE:-full_graph}"
SGBENCH_MEM_FRACTION_STATIC="${SGBENCH_MEM_FRACTION_STATIC:-0.85}"
SGBENCH_MAX_RUNNING_REQUESTS="${SGBENCH_MAX_RUNNING_REQUESTS:-}"
SGBENCH_CHUNKED_PREFILL_SIZE="${SGBENCH_CHUNKED_PREFILL_SIZE:-4096}"
SGBENCH_ATTENTION_BACKEND="${SGBENCH_ATTENTION_BACKEND:-}"
SGBENCH_CUDA_GRAPH_MAX_BS_DECODE="${SGBENCH_CUDA_GRAPH_MAX_BS_DECODE:-}"
SGBENCH_CUDA_GRAPH_MAX_BS_PREFILL="${SGBENCH_CUDA_GRAPH_MAX_BS_PREFILL:-}"
FD_BASE_MODEL="${FD_BASE_MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
FD_CHECKPOINT="${FD_CHECKPOINT:-xuan-luo/FlexiDepth-Llama-3-8B-Instruct}"
FD_WEIGHTS="${FD_WEIGHTS:-/dev/shm/flexidepth_router_weights.pt}"
SGBENCH_SCORE_FILE="${SGBENCH_SCORE_FILE:-}"

export HF_HOME="${HF_HOME:-/dev/shm/hf_vast}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_cmd() {
  echo
  echo "+ $*"
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

join_by() {
  local IFS="$1"
  shift
  echo "$*"
}

require_vast_checkout() {
  if [[ "${ALLOW_NON_VAST:-0}" != "1" && "$ROOT" != /workspace/sglang-vp* ]]; then
    die "refusing to run outside /workspace/sglang-vp*; set ALLOW_NON_VAST=1 only for local dry-run/syntax checks"
  fi
}

require_a100() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi not found; this must run on the Vast GPU node"
  fi
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  echo "GPU: $gpu_name"
  if [[ "${ALLOW_NON_A100:-0}" != "1" && "$gpu_name" != *"A100"* ]]; then
    die "expected an A100 GPU; set ALLOW_NON_A100=1 only for a smoke check"
  fi
}

require_cache_space() {
  mkdir -p "$HF_HOME"
  local free_kb
  free_kb="$(df -Pk "$HF_HOME" | awk 'NR==2 {print $4}')"
  echo "HF_HOME: $HF_HOME"
  echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
  echo "HF free GiB: $((free_kb / 1024 / 1024))"
  if (( free_kb < 60 * 1024 * 1024 )); then
    die "/dev/shm/hf has less than 60 GiB free; do not use overlay disk for Qwen3"
  fi
}

require_profile_file() {
  [[ -s "$PROFILE_FILE" ]] || die "missing PROFILE_FILE=$PROFILE_FILE"
  echo "PROFILE_FILE: $PROFILE_FILE"
}

activate_env_if_present() {
  if [[ -n "${SGLANG_VENV:-}" && -f "$SGLANG_VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$SGLANG_VENV/bin/activate"
  elif [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
  fi
  export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
  cd "$ROOT"
}

preflight() {
  require_vast_checkout
  require_a100
  require_cache_space
  # require_profile_file is NOT called here. PROFILE_FILE defaults to
  # test/vp/qwen3_8b_layer_importance.json, which is not in this tree, and every
  # remaining consumer sits in an ACTION that require_scripts already refuses or
  # in a refused dynamic* branch. Demanding it unconditionally made preflight --
  # and therefore every action, including the supported vanilla and flexidepth
  # modes -- fail on an artifact nothing live reads.
  activate_env_if_present
  python - <<'PY'
import os, sys
print("python:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
PY
}

sgbench_effective_output_len() {
  case "$SGBENCH_DATASET" in
    random|random-ids) echo "$SGBENCH_RANDOM_OUTPUT_LEN" ;;
    *) echo "$SGBENCH_OUTPUT_LEN" ;;
  esac
}

validate_sgbench_protocol() {
  local output_len
  output_len="$(sgbench_effective_output_len)"
  case "$SGBENCH_EVIDENCE_CLASS" in
    smoke|diagnostic)
      echo "NON_PERFORMANCE_RUN: evidence_class=$SGBENCH_EVIDENCE_CLASS"
      echo "NON_PERFORMANCE_RUN: this artifact cannot accept or reject a performance candidate"
      ;;
    performance|headline)
      die "$SGBENCH_EVIDENCE_CLASS evidence must use the manifested deployment plus test/vp/run_qps_evaluation.py; the generic ladder is diagnostic-only"
      ;;
    *)
      die "unknown SGBENCH_EVIDENCE_CLASS=$SGBENCH_EVIDENCE_CLASS; expected smoke, diagnostic, performance, or headline"
      ;;
  esac
  if [[ -n "$SGBENCH_MAX_RUNNING_REQUESTS" && "$SGBENCH_SERVER_CAP_PURPOSE" != "pressure" ]]; then
    die "an explicit SGBENCH_MAX_RUNNING_REQUESTS requires SGBENCH_SERVER_CAP_PURPOSE=pressure"
  fi
}

stage_qwen3() {
  preflight
  run_cmd python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
}

stage_flexidepth() {
  preflight
  run_cmd python -c "from huggingface_hub import snapshot_download; print('base', snapshot_download('$FD_BASE_MODEL')); print('fd', snapshot_download('$FD_CHECKPOINT'))"
  run_cmd env FD_W_OUT="$FD_WEIGHTS" python test/vp/extract_flexidepth_weights.py
}

smoke_decode() {
  preflight
  run_cmd env \
    PROFILE_MODE=dynamic_row \
    VP_AGREE_MODEL="$MODEL" \
    CONC=8 \
    MAXRUN=8 \
    INLEN=512 \
    NTOK=8 \
    PROFILE_STEPS=2 \
    PROFILE_OUT_DIR="$OUT_BASE/vp_decode_profiles_i231_smoke" \
    PROFILE_ACTIVITIES=CPU,GPU \
    WARMUP_CONC=8 \
    WARMUP_NTOK=4 \
    GRAPH=0 \
    VP_GRAPH=1 \
    VP_SPAN=4 \
    DYN_DECODE_PROFILE_FILE="$PROFILE_FILE" \
    DYN_DECODE_TOPKS=8,12,0 \
    DYN_DECODE_POLICY=round_robin \
    DYN_DECODE_GATHER=1 \
    DYN_DECODE_BLOCK_SIZE=4 \
    python test/vp/profile_decode_modes.py
}

profile_decode() {
  preflight
  run_cmd env \
    PROFILE_MODE=dynamic_row \
    VP_AGREE_MODEL="$MODEL" \
    CONC=64 \
    MAXRUN=128 \
    INLEN=2048 \
    NTOK=24 \
    PROFILE_STEPS=8 \
    PROFILE_OUT_DIR="$OUT_BASE/vp_decode_profiles_i231" \
    PROFILE_ACTIVITIES=CPU,GPU \
    WARMUP_CONC=64 \
    WARMUP_NTOK=4 \
    GRAPH=0 \
    VP_GRAPH=1 \
    VP_SPAN=4 \
    DYN_DECODE_PROFILE_FILE="$PROFILE_FILE" \
    DYN_DECODE_TOPKS=8,12,0 \
    DYN_DECODE_POLICY=round_robin \
    DYN_DECODE_GATHER=1 \
    DYN_DECODE_BLOCK_SIZE=4 \
    python test/vp/profile_decode_modes.py
}

ci_decode() {
  preflight
  run_cmd env \
    VP_AGREE_MODEL="$MODEL" \
    CELLS=2048:64 \
    REPS=3 \
    NTOK_A=8 \
    NTOK_B=72 \
    MAXRUN=128 \
    GRAPH=0 \
    VP_GRAPH=1 \
    WARMUP_CONC=64 \
    WARMUP_NTOK=4 \
    VP_SPAN=4 \
    VP_DECODE_MODE=dynamic \
    DYN_DECODE_PROFILE_FILE="$PROFILE_FILE" \
    DYN_DECODE_TOPKS=8,12,0 \
    DYN_DECODE_POLICY=round_robin \
    DYN_DECODE_GATHER=1 \
    DYN_DECODE_BLOCK_SIZE=4 \
    DYN_DECODE_COHORT=0 \
    python test/vp/headline_ci.py
}

headline_gov() {
  preflight
  run_cmd env \
    MODEL="$MODEL" \
    TASKS=gov_report \
    QN=50 \
    MAXLEN=7000 \
    GENLEN_CAP=64 \
    CONC=12 \
    REPS=3 \
    MODES=vanilla,dynamic_all_jump \
    DYN_SPEC_ENV=dynamic_layer_prefill \
    SGLANG_VP_BLOCK_SIZE=1 \
    SGLANG_VP_LAYER_PROFILE_FILE="$PROFILE_FILE" \
    SGLANG_VP_LAYER_PROFILE_TOPKS=8,12,0 \
    SGLANG_VP_PREFILL_POLICY=external \
    VP_EXTERNAL_PROFILE_ASSIGN=balanced \
    SGLANG_VP_LANEFUSE_GATHER=1 \
    OUTDIR="$OUT_BASE/govreport_rankfile_layer_profile_topk8120_qn50_ci_i231" \
    python test/vp/mixed_longbench_serving.py
}

headline_mixed() {
  preflight
  run_cmd env \
    MODEL="$MODEL" \
    TASKS=hotpotqa,gov_report \
    QN=20 \
    MAXLEN=7000 \
    GENLEN_CAP=64 \
    CONC=12 \
    REPS=3 \
    MODES=vanilla,dynamic_tag_veto \
    DYN_SPEC_ENV=dynamic_layer_prefill \
    SGLANG_VP_BLOCK_SIZE=1 \
    SGLANG_VP_LAYER_PROFILE_FILE="$PROFILE_FILE" \
    SGLANG_VP_LAYER_PROFILE_TOPKS=8,12,0 \
    SGLANG_VP_PREFILL_POLICY=external \
    VP_EXTERNAL_PROFILE_ASSIGN=balanced \
    SGLANG_VP_PREFILL_VETO_TAGS=qa \
    SGLANG_VP_LANEFUSE_GATHER=1 \
    OUTDIR="$OUT_BASE/mixed_hotpot_gov_rankfile_tagveto_topk8120_ci_i231" \
    python test/vp/mixed_longbench_serving.py
}

headline_multinews() {
  preflight
  run_cmd env \
    MODEL="$MODEL" \
    TASKS=multi_news \
    QN=20 \
    MAXLEN=7000 \
    GENLEN_CAP=64 \
    CONC=12 \
    REPS=3 \
    MODES=vanilla,dynamic_all_jump \
    DYN_SPEC_ENV=dynamic_layer_prefill \
    SGLANG_VP_BLOCK_SIZE=1 \
    SGLANG_VP_LAYER_PROFILE_FILE="$PROFILE_FILE" \
    SGLANG_VP_LAYER_PROFILE_TOPKS=8,12,0 \
    SGLANG_VP_PREFILL_POLICY=external \
    VP_EXTERNAL_PROFILE_ASSIGN=balanced \
    SGLANG_VP_LANEFUSE_GATHER=1 \
    OUTDIR="$OUT_BASE/multinews_rankfile_layer_profile_topk8120_qn20_ci_i231" \
    python test/vp/mixed_longbench_serving.py
}

wait_sglang_ready() {
  local port="$1"
  local log_path="$2"
  local timeout_s="${READY_TIMEOUT_S:-300}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before ready; log tail:" >&2
      tail -80 "$log_path" >&2 || true
      return 1
    fi
    if python - "$port" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

port = int(sys.argv[1])
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
    raise SystemExit(0 if resp.status == 200 else 1)
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "server did not become ready within ${timeout_s}s; log tail:" >&2
  tail -80 "$log_path" >&2 || true
  return 1
}

stop_sglang_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "-$SERVER_PID" 2>/dev/null || kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  SERVER_PID=""
}

stop_load_sampler() {
  if [[ -n "${LOAD_SAMPLER_PID:-}" ]] && kill -0 "$LOAD_SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$LOAD_SAMPLER_PID" 2>/dev/null || true
    wait "$LOAD_SAMPLER_PID" 2>/dev/null || true
  fi
  LOAD_SAMPLER_PID=""
}

cleanup_sgbench_mode() {
  stop_load_sampler
  stop_sglang_server
}

capture_server_info() {
  local port="$1"
  local output_path="$2"
  echo "+ capture http://127.0.0.1:${port}/server_info -> $output_path"
  if [[ "$DRY_RUN" == "0" ]]; then
    python - "$port" "$output_path" <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
output_path = sys.argv[2]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/server_info", timeout=10) as response:
    info = json.load(response)
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(info, output, indent=2, sort_keys=True)
    output.write("\n")
states = [item.get("internal_state", item) for item in info.get("internal_states") or []]
resolved = [state.get("effective_max_running_requests_per_dp") for state in states]
capacity = [(state.get("memory_usage") or {}).get("token_capacity") for state in states]
print("SGBENCH_RESOLVED_MAX_RUNNING_REQUESTS_PER_DP:", resolved)
print("SGBENCH_TOKEN_CAPACITY_PER_DP:", capacity)
PY
  fi
}

start_load_sampler() {
  local port="$1"
  local output_path="$2"
  echo "+ python test/vp/sample_sglang_load.py --base-url http://127.0.0.1:$port --output $output_path --interval-ms $SGBENCH_LOAD_SAMPLE_INTERVAL_MS &"
  if [[ "$DRY_RUN" == "0" ]]; then
    python test/vp/sample_sglang_load.py \
      --base-url "http://127.0.0.1:$port" \
      --output "$output_path" \
      --interval-ms "$SGBENCH_LOAD_SAMPLE_INTERVAL_MS" &
    LOAD_SAMPLER_PID=$!
  fi
}

launch_sglang_server() {
  local raw_mode="$1"
  local mode="$raw_mode"
  local matched_vanilla=0
  local port="$2"
  local log_path="$3"
  local vp_foreground_stream_priority="$SGBENCH_VP_FOREGROUND_STREAM_PRIORITY"
  # _fgpriority / _no_fgpriority are the ONE override that still does anything:
  # they set SGLANG_VP_FOREGROUND_STREAM_PRIORITY, which model_runner.py:581
  # reads to change the foreground CUDA stream priority. Strip EXACTLY ONE,
  # BEFORE vanilla_matched is normalised -- normalisation matched the bare name
  # only, so vanilla_matched_fgpriority survived it and was then refused by the
  # whitelist even though the help advertises the suffix on every base mode.
  # That would have been the fourth over-broad refusal in this refactor.
  case "$mode" in
    *_no_fgpriority)
      mode="${mode%_no_fgpriority}"
      vp_foreground_stream_priority=0
      ;;
    *_fgpriority)
      mode="${mode%_fgpriority}"
      vp_foreground_stream_priority=-1
      ;;
  esac
  # A second suffix is ambiguous, never intended, and previously resolved
  # silently to whichever appeared leftmost -- so flexidepth_fgpriority_no_fgpriority
  # ran at priority -1 under a name that also says otherwise. Refuse it.
  case "$mode" in
    *_fgpriority|*_no_fgpriority)
      echo "FATAL: mode '$raw_mode' repeats or contradicts the fgpriority" \
           "override. Use exactly one of _fgpriority or _no_fgpriority." >&2
      return 2
      ;;
  esac

  if [[ "$mode" == "vanilla_matched" ]]; then
    mode=vanilla
    matched_vanilla=1
  fi

  # WHITELIST, not a blacklist. A blacklist cannot be complete here: the
  # parser below still accepts ~50 override suffixes whose ONLY emission site
  # was the deleted flexidepth_vp block, so each one now yields an arm
  # byte-identical to plain flexidepth while carrying a name that claims a
  # treatment. That is the same "arm whose name is a lie" failure as the
  # removed async modes, just spread across every suffix.
  #
  # Exactly three modes are distinguishable in this build (vanilla_matched is
  # normalised to vanilla above), so name them and refuse everything else.
  case "$mode" in
    vanilla|flexidepth) ;;
    dynamic|dynamic_*|flexidepth_vp|flexidepth_vp_*)
      echo "FATAL: mode '$raw_mode' selects the V1 generic-VP / inline" \
           "VP-project scheduler, which is not in this build. Its knobs are" \
           "inert, so the arm would run as vanilla under a routed name. Use" \
           "'flexidepth' (full-graph) instead; see the removed-feature" \
           "register for what V1 did and how to revive it." >&2
      return 2
      ;;
    *_scopedasync|*_no_scopedasync|*_batchedasync|*_tokenasync|\
    *_lookaheadasync|*_no_lookaheadasync|*_streamasync|*_kvonly|*_no_kvonly|\
    *_mixed_async|*_no_mixed_async)
      echo "FATAL: mode '$raw_mode' selects the removed batched/async K/V" \
           "subsystem. That implementation is not in this build, so the arm" \
           "would not run the treatment its name claims. Drop the suffix, or" \
           "reimplement async K/V first (see the removed-feature register)." >&2
      return 2
      ;;
    *)
      echo "FATAL: mode '$raw_mode' is not available in this build. Its" \
           "override suffix has no effect here -- the environment variables" \
           "the suffixes set are emitted only by code removed with the V1" \
           "path, so the arm would be identical to plain 'flexidepth' under a" \
           "name claiming otherwise. Supported: vanilla, vanilla_matched," \
           "flexidepth." >&2
      return 2
      ;;
  esac
  local env_args=(
    HF_HOME="$HF_HOME"
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
    HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
    PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"
    SGLANG_FD_WEIGHTS=
    SGLANG_FD_VP_PROJECT=
    SGLANG_FD_VP_TRACE=
    SGLANG_FD_VP_TRACE_TIMING=
    SGLANG_FD_VP_TRACE_EVERY=
    SGLANG_FD_VP_TRACE_FILE=
    SGLANG_FD_VP_TRACE_MASKS=
    SGLANG_FD_VP_TRACE_MASK_CAP=
    SGLANG_FD_VP_MIXED_DECODE_SUBBATCH=
    SGLANG_FD_VP_DECODE_SUBBATCH_CACHE=
    SGLANG_FD_VP_MINIMAL_KV_SUBBATCH=
    SGLANG_FD_VP_TRITON_GPU_SUBBATCH=
    SGLANG_FD_VP_ASYNC_KV=
    SGLANG_FD_VP_ASYNC_KV_DEFER_DRAIN=
    SGLANG_FD_VP_ASYNC_KV_SCOPED=
    SGLANG_FD_VP_MIXED_DECODE_ASYNC_KV=
    SGLANG_FD_VP_MIXED_DECODE_ASYNC_MIN_ROWS=
    SGLANG_FD_VP_MIXED_DECODE_ASYNC_MIN_KEPT_ROWS=
    SGLANG_FD_VP_MIXED_DECODE_ASYNC_MIN_SKIP_ROWS=
    SGLANG_FD_VP_MIXED_DECODE_ASYNC_PATTERN_WARMUP=
    SGLANG_FD_VP_MIXED_DECODE_ASYNC_PATTERN_SCOPE=
    SGLANG_FD_VP_MIXED_DECODE_ASYNC_COHORT_TABLE=
    SGLANG_FD_VP_MIXED_DECODE_ASYNC_COHORT_MAX_ENTRIES=
    SGLANG_FD_VP_GLOBAL_DECODE_SUBBATCH_CACHE=
    SGLANG_FD_VP_GLOBAL_DECODE_SUBBATCH_CACHE_MAX_ENTRIES=
    SGLANG_FD_VP_MIXED_FULL_BATCH_SPLIT=
    SGLANG_FD_VP_MIXED_POST_WEIGHT=
    SGLANG_FD_VP_MIXED_INPLACE_WEIGHT=
    SGLANG_FD_VP_MIXED_PARALLEL_BRANCHES=
    SGLANG_FD_VP_MIXED_NEAR_ALL_RUN_FULL_MLP=
    SGLANG_FD_VP_MIXED_NEAR_ALL_RUN_MAX_SKIP_ROWS=
    SGLANG_FD_VP_MIXED_FULL_PROJECT_BASE=
    SGLANG_FD_VP_MIXED_REUSE_OUTPUT_BUFFER=
    SGLANG_FD_VP_MIXED_REUSE_OUTPUT_BUFFER_MAX_ENTRIES=
    SGLANG_FD_VP_MIXED_SHAPE_LANE_BUFFERS=
    SGLANG_FD_VP_MIXED_SHAPE_LANE_BUFFER_MAX_ENTRIES=
    SGLANG_FD_VP_MIXED_SPLIT_GRAPH=
    SGLANG_FD_VP_MIXED_SPLIT_GRAPH_MAX_ENTRIES=
    SGLANG_FD_VP_MIXED_SPLIT_GRAPH_MAX_ROWS=
    SGLANG_FD_VP_MIXED_SPLIT_GRAPH_MIN_FREE_MB=
    SGLANG_FD_VP_STABLE_MIXED_SPLIT_BUFFERS=
    SGLANG_FD_VP_STABLE_MIXED_SPLIT_BUFFER_MAX_ENTRIES=
    SGLANG_FD_VP_ALL_SKIP_KV_ONLY_QKV=
    SGLANG_FD_VP_FUSED_ROUTER_DEC_HEAD=
    SGLANG_FD_VP_ROUTER_GRAPH=
    SGLANG_FD_VP_ROUTER_GRAPH_MAX_ROWS=
    SGLANG_FD_VP_ROUTER_GRAPH_MAX_ENTRIES=
    SGLANG_FD_VP_FUSED_PROJECT_INPUT=
    SGLANG_FD_VP_STAGE_ROUTE=
    SGLANG_FD_VP_STAGE_ROUTE_FUSE_HOMOGENEOUS=
    SGLANG_FD_VP_STAGE_ROUTE_RUNAHEAD=
    SGLANG_FD_VP_STAGE_ROUTE_MIN_SPLIT_SKIP_ROWS=
    SGLANG_FD_VP_STAGE_ROUTE_MIN_SPLIT_RUN_ROWS=
    SGLANG_FD_VP_COALESCED_LAYER=
    SGLANG_FD_VP_COALESCED_MIN_SKIP_ROWS=
    SGLANG_FD_VP_COALESCED_MIN_RUN_ROWS=
    SGLANG_VP_MODE=
    SGLANG_VP_BLOCK_SIZE=
    SGLANG_VP_SCHED=
    SGLANG_VP_SPAN=
    SGLANG_VP_GRAPH=
    SGLANG_VP_GRAPH_SPAN_STARTS=
    SGLANG_VP_HIT_RATE=
    SGLANG_VP_DISABLE_ROUTER=
    SGLANG_VP_DET_LAYER_PATTERN=
    SGLANG_VP_DET_LAYER_PATTERN_FILE=
    SGLANG_VP_LANESPLIT=
    SGLANG_VP_LANEFUSE=
    SGLANG_VP_LANEFUSE_GATHER=
    SGLANG_VP_LAYER_PROFILE_FILE=
    SGLANG_VP_LAYER_PROFILE_TOPKS=
    SGLANG_VP_PREFILL_POLICY=
    SGLANG_VP_PREFILL_VETO_TAGS=
    SGLANG_VP_PREFILL_VETO_TEXT_RE=
    SGLANG_VP_DECODE_LAYER_PROFILE_SPECS=
    SGLANG_VP_DECODE_LAYER_PROFILE_FILE=
    SGLANG_VP_DECODE_LAYER_PROFILE_TOPKS=
    SGLANG_VP_DECODE_PROFILE_POLICY=
    SGLANG_VP_DECODE_PROFILE_COHORT=
    SGLANG_VP_DECODE_VETO_TAGS=
    SGLANG_VP_DECODE_VETO_TEXT_RE=
    SGLANG_VP_STAGE_SCHED=
    SGLANG_VP_STAGE_POLICY=
    SGLANG_VP_STAGE_BLOCK_SYNC=
    SGLANG_VP_STAGE_FUSE_MIXED=
    SGLANG_VP_ASYNC_KV=
    SGLANG_VP_ASYNC_KV_DEFER_DRAIN=
    SGLANG_VP_ASYNC_KV_SCOPED=
    SGLANG_VP_ASYNC_KV_BATCHED=
    SGLANG_VP_ASYNC_KV_BATCHED_MAX_ROWS=
    SGLANG_VP_ASYNC_KV_BATCHED_TOKEN_LAUNCH=
    SGLANG_VP_ASYNC_KV_LOOKAHEAD_RELEASE=
    SGLANG_VP_ASYNC_KV_RELEASE_LAYER=
    SGLANG_VP_KV_ONLY_QKV=
    SGLANG_VP_FOREGROUND_STREAM_PRIORITY=
  )
  if [[ "$vp_foreground_stream_priority" != "0" ]]; then
    env_args+=(
      SGLANG_VP_FOREGROUND_STREAM_PRIORITY="$vp_foreground_stream_priority"
    )
  fi
  if [[ "$mode" == "flexidepth" ]]; then
    [[ -s "$FD_WEIGHTS" ]] || die "missing FD_WEIGHTS=$FD_WEIGHTS; run ACTION=stage-flexidepth first"
    env_args+=(
      SGLANG_FD_WEIGHTS="$FD_WEIGHTS"
      SGLANG_FD_EXECUTION_MODE="$SGBENCH_FD_EXECUTION_MODE"
    )
    # Live full-graph knobs: emitted only when the operator changed them from
    # the default, so a default run stays byte-identical to before. Accepting
    # a knob the ladder never emits would be its own kind of lie -- the arm
    # would record a treatment it did not receive.
    if [[ "${SGBENCH_FD_VP_FUSED_PROJECT_INPUT}" != "0" ]]; then
      env_args+=(SGLANG_FD_VP_FUSED_PROJECT_INPUT="${SGBENCH_FD_VP_FUSED_PROJECT_INPUT}")
    fi
    if [[ "${SGBENCH_FD_VP_FUSED_ROUTER_DEC_HEAD}" != "0" ]]; then
      env_args+=(SGLANG_FD_VP_FUSED_ROUTER_DEC_HEAD="${SGBENCH_FD_VP_FUSED_ROUTER_DEC_HEAD}")
    fi
    if [[ "${SGBENCH_FD_VP_ROUTER_GRAPH}" != "0" ]]; then
      env_args+=(SGLANG_FD_VP_ROUTER_GRAPH="${SGBENCH_FD_VP_ROUTER_GRAPH}")
    fi
    if [[ "${SGBENCH_FD_VP_ROUTER_GRAPH_MAX_ENTRIES}" != "4" ]]; then
      env_args+=(SGLANG_FD_VP_ROUTER_GRAPH_MAX_ENTRIES="${SGBENCH_FD_VP_ROUTER_GRAPH_MAX_ENTRIES}")
    fi
    if [[ "${SGBENCH_FD_VP_ROUTER_GRAPH_MAX_ROWS}" != "64" ]]; then
      env_args+=(SGLANG_FD_VP_ROUTER_GRAPH_MAX_ROWS="${SGBENCH_FD_VP_ROUTER_GRAPH_MAX_ROWS}")
    fi
    if [[ "${SGBENCH_FD_VP_TRACE}" != "0" ]]; then
      env_args+=(SGLANG_FD_VP_TRACE="${SGBENCH_FD_VP_TRACE}")
    fi
    if [[ "${SGBENCH_FD_VP_TRACE_FILE}" != "" ]]; then
      env_args+=(SGLANG_FD_VP_TRACE_FILE="${SGBENCH_FD_VP_TRACE_FILE}")
    fi
    if [[ "${SGBENCH_FD_VP_TRACE_TIMING}" != "0" ]]; then
      env_args+=(SGLANG_FD_VP_TRACE_TIMING="${SGBENCH_FD_VP_TRACE_TIMING}")
    fi
  elif [[ "$mode" != "vanilla" ]]; then
    die "unknown SGBENCH mode $raw_mode; expected vanilla, vanilla_matched, flexidepth, or flexidepth with override suffixes. The V1 modes (dynamic*, flexidepth_vp*) are not in this build and are refused earlier"
  fi

  local server_profile="$SGBENCH_CANDIDATE_SERVER_PROFILE"
  if [[ "$mode" == "vanilla" && "$matched_vanilla" == "0" ]]; then
    server_profile="$SGBENCH_BASELINE_SERVER_PROFILE"
  fi
  # direct_eager cannot coexist with captured graphs: its routed-row gather is
  # a device->host sync, illegal under stream capture. Refuse here -- with the
  # profile actually resolved -- rather than let startup validation reject it
  # after a model load. Checked against $server_profile, which does not exist
  # earlier in this function.
  if [[ "$SGBENCH_FD_EXECUTION_MODE" == "direct_eager" && "$mode" != "vanilla" ]]; then
    case "$server_profile" in
      graphless_overlap|matched_eager) ;;
      *)
        echo "FATAL: SGBENCH_FD_EXECUTION_MODE=direct_eager needs a graphless" \
             "server profile (graphless_overlap or matched_eager); got" \
             "'$server_profile'." >&2
        return 2
        ;;
    esac
  fi
  local server_profile_args=()
  case "$server_profile" in
    production)
      ;;
    graphless_overlap)
      # Dynamic data-dependent routing is not capturable by SGLang's stock
      # decode/prefill CUDA graphs, but it can still retain the production
      # overlap scheduler and radix cache. This profile isolates that graph
      # constraint from the stricter matched-eager debugging control below.
      server_profile_args+=(
        --disable-cuda-graph
      )
      ;;
    breakable_dynamic)
      # Capture fixed model regions while FlexiDepth routed layers execute at
      # explicit eager graph breaks. This retains production overlap/radix and
      # is the first production-graph compatibility gate for dynamic routing.
      server_profile_args+=(
        --cuda-graph-backend-decode breakable
        --cuda-graph-backend-prefill breakable
      )
      ;;
    matched_eager)
      server_profile_args+=(
        --disable-cuda-graph
        --disable-radix-cache
        --disable-overlap-schedule
      )
      ;;
    *)
      die "unknown server profile $server_profile; expected production, breakable_dynamic, graphless_overlap, or matched_eager"
      ;;
  esac

  local cmd=(
    env "${env_args[@]}"
    python -m sglang.launch_server
    --model-path "$MODEL"
    --port "$port"
    --host 127.0.0.1
    --mem-fraction-static "$SGBENCH_MEM_FRACTION_STATIC"
  )
  if [[ -n "$SGBENCH_MAX_RUNNING_REQUESTS" ]]; then
    cmd+=(--max-running-requests "$SGBENCH_MAX_RUNNING_REQUESTS")
  fi
  cmd+=("${server_profile_args[@]}")
  if [[ -n "$SGBENCH_CHUNKED_PREFILL_SIZE" ]]; then
    cmd+=(--chunked-prefill-size "$SGBENCH_CHUNKED_PREFILL_SIZE")
  fi
  if [[ -n "$SGBENCH_ATTENTION_BACKEND" ]]; then
    cmd+=(--attention-backend "$SGBENCH_ATTENTION_BACKEND")
  fi
  if [[ -n "$SGBENCH_CUDA_GRAPH_MAX_BS_DECODE" ]]; then
    cmd+=(--cuda-graph-max-bs-decode "$SGBENCH_CUDA_GRAPH_MAX_BS_DECODE")
  fi
  if [[ -n "$SGBENCH_CUDA_GRAPH_MAX_BS_PREFILL" ]]; then
    cmd+=(--cuda-graph-max-bs-prefill "$SGBENCH_CUDA_GRAPH_MAX_BS_PREFILL")
  fi

  echo
  echo "+ setsid $(join_by ' ' "${cmd[@]}") > $log_path 2>&1 &"
  if [[ "$DRY_RUN" == "0" ]]; then
    setsid "${cmd[@]}" >"$log_path" 2>&1 &
    SERVER_PID=$!
    wait_sglang_ready "$port" "$log_path"
  fi
}

write_sgbench_manifest() {
  local manifest="$SGBENCH_OUT_DIR/run_manifest.txt"
  mkdir -p "$SGBENCH_OUT_DIR"
  {
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "git_head=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "python=$(command -v python)"
    python --version
    echo
    echo "[git-status]"
    git -C "$ROOT" status --short 2>/dev/null || echo "unavailable (no git checkout)"
    echo
    echo "[source-sha256]"
    sha256sum \
      "$ROOT"/python/sglang/srt/vpipe/*.py \
      "$ROOT"/python/sglang/srt/model_executor/forward_batch_info.py \
      "$ROOT"/python/sglang/srt/model_executor/model_runner.py \
      "$ROOT"/python/sglang/srt/mem_cache/common.py \
      "$ROOT"/python/sglang/srt/managers/scheduler.py \
      "$ROOT"/python/sglang/srt/managers/tp_worker.py \
      "$ROOT"/python/sglang/srt/models/llama.py \
      "$ROOT"/python/sglang/srt/models/qwen2.py \
      "$ROOT"/python/sglang/srt/models/qwen3.py \
      "$ROOT"/test/vp/analyze_sgbench_pairs.py \
      "$ROOT"/test/vp/sample_sglang_load.py \
      "$ROOT"/test/vp/vast_a100_ladder.sh
    echo
    echo "[sgbench-config]"
    echo "MODEL=$MODEL"
    echo "PROFILE_FILE=$PROFILE_FILE"
    declare -p | sort | awk '/^declare .* (SGBENCH_|SGLANG_|FD_)/'
    env | sort | awk '/^(HF_HOME|TRANSFORMERS_CACHE|HF_HUB_DISABLE_XET)=/'
    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  } >"$manifest"
  echo "SGBENCH_MANIFEST: $manifest"
}

sgbench_dataset_args() {
  case "$SGBENCH_DATASET" in
    longbench_v2)
      local args=(--dataset-name longbench_v2 --sharegpt-output-len "$SGBENCH_OUTPUT_LEN")
      [[ -n "$SGBENCH_CONTEXT_LEN" ]] && args+=(--sharegpt-context-len "$SGBENCH_CONTEXT_LEN")
      echo "${args[@]}"
      ;;
    sharegpt)
      local args=(--dataset-name sharegpt --sharegpt-output-len "$SGBENCH_OUTPUT_LEN")
      [[ -n "$SGBENCH_CONTEXT_LEN" ]] && args+=(--sharegpt-context-len "$SGBENCH_CONTEXT_LEN")
      echo "${args[@]}"
      ;;
    random|random-ids)
      echo --dataset-name "$SGBENCH_DATASET" --random-input-len "$SGBENCH_RANDOM_INPUT_LEN" --random-output-len "$SGBENCH_RANDOM_OUTPUT_LEN" --random-range-ratio "$SGBENCH_RANDOM_RANGE_RATIO"
      ;;
    autobench)
      [[ -n "$SGBENCH_DATASET_PATH" ]] || die "SGBENCH_DATASET_PATH is required for SGBENCH_DATASET=autobench"
      local args=(--dataset-name autobench --dataset-path "$SGBENCH_DATASET_PATH")
      if [[ -n "$SGBENCH_OUTPUT_LEN_WAS_SET" ]]; then
        args+=(--sharegpt-output-len "$SGBENCH_OUTPUT_LEN")
      fi
      echo "${args[@]}"
      ;;
    *)
      die "unsupported SGBENCH_DATASET=$SGBENCH_DATASET; use longbench_v2, sharegpt, autobench, random, or random-ids"
      ;;
  esac
}

sgbench_mode_once() {
  local mode="$1"
  local rep="$2"
  local port="$SGBENCH_PORT"
  local label="${SGBENCH_DATASET}_${mode}_rep${rep}"
  local profile_output_dir=""
  if [[ -n "$SGBENCH_PROFILE_OUTPUT_DIR" ]]; then
    profile_output_dir="${SGBENCH_PROFILE_OUTPUT_DIR%/}/$label"
  fi
  local log_path="$SGBENCH_OUT_DIR/server_${label}.log"
  local output_file="$SGBENCH_OUT_DIR/${label}.jsonl"
  local server_info_file="$SGBENCH_OUT_DIR/server_info_${label}.json"
  local load_file="$SGBENCH_OUT_DIR/load_${label}.jsonl"
  mkdir -p "$SGBENCH_OUT_DIR"
  SERVER_PID=""
  LOAD_SAMPLER_PID=""
  trap cleanup_sgbench_mode EXIT
  launch_sglang_server "$mode" "$port" "$log_path"
  capture_server_info "$port" "$server_info_file"
  local dataset_args
  read -r -a dataset_args <<<"$(sgbench_dataset_args)"
  local bench_cmd=(
    python -m sglang.benchmark.serving
    --backend "$SGBENCH_BACKEND"
    --host 127.0.0.1
    --port "$port"
    --model "$MODEL"
    --tokenizer "$MODEL"
    --num-prompts "$SGBENCH_NUM_PROMPTS"
    --request-rate "$SGBENCH_REQUEST_RATE"
    --seed "$SGBENCH_SEED"
    --warmup-requests "$SGBENCH_WARMUP_REQUESTS"
    --ready-check-timeout-sec 300
    --disable-tqdm
    --output-details
    --output-file "$output_file"
    "${dataset_args[@]}"
  )
  if [[ -n "$SGBENCH_MAX_CONCURRENCY" ]]; then
    bench_cmd+=(--max-concurrency "$SGBENCH_MAX_CONCURRENCY")
  fi
  if [[ -n "$SGBENCH_EXTRA_REQUEST_BODY" ]]; then
    bench_cmd+=(--extra-request-body "$SGBENCH_EXTRA_REQUEST_BODY")
  fi
  if [[ "$SGBENCH_PROFILE" == "1" ]]; then
    bench_cmd+=(--profile)
    if [[ "$SGBENCH_PROFILE_BY_STAGE" == "1" ]]; then
      bench_cmd+=(--profile-by-stage)
    fi
    if [[ -n "$SGBENCH_PROFILE_NUM_STEPS" ]]; then
      bench_cmd+=(--profile-num-steps "$SGBENCH_PROFILE_NUM_STEPS")
    fi
    if [[ -n "$profile_output_dir" ]]; then
      bench_cmd+=(--profile-output-dir "$profile_output_dir")
    fi
    if [[ -n "$SGBENCH_PROFILE_PREFIX" ]]; then
      bench_cmd+=(--profile-prefix "$SGBENCH_PROFILE_PREFIX")
    fi
  fi
  start_load_sampler "$port" "$load_file"
  run_cmd "${bench_cmd[@]}"
  stop_load_sampler
  stop_sglang_server
  trap - EXIT
}

sgbench() {
  preflight
  validate_sgbench_protocol
  echo "SGBENCH_OUT_DIR: $SGBENCH_OUT_DIR"
  echo "SGBENCH_DATASET: $SGBENCH_DATASET"
  echo "SGBENCH_MODES: $SGBENCH_MODES"
  echo "SGBENCH_NUM_PROMPTS: $SGBENCH_NUM_PROMPTS"
  echo "SGBENCH_MAX_CONCURRENCY: $SGBENCH_MAX_CONCURRENCY"
  echo "SGBENCH_REPS: $SGBENCH_REPS"
  echo "SGBENCH_EVIDENCE_CLASS: $SGBENCH_EVIDENCE_CLASS"
  echo "SGBENCH_MAX_RUNNING_REQUESTS: ${SGBENCH_MAX_RUNNING_REQUESTS:-auto}"
  echo "SGBENCH_PROFILE: $SGBENCH_PROFILE"
  echo "SGBENCH_PROFILE_BY_STAGE: $SGBENCH_PROFILE_BY_STAGE"
  echo "SGBENCH_PROFILE_NUM_STEPS: $SGBENCH_PROFILE_NUM_STEPS"
  echo "SGBENCH_PROFILE_OUTPUT_DIR: $SGBENCH_PROFILE_OUTPUT_DIR"
  echo "SGBENCH_PROFILE_PREFIX: $SGBENCH_PROFILE_PREFIX"
  echo "SGBENCH_ORDER_POLICY: $SGBENCH_ORDER_POLICY"
  echo "SGBENCH_BASELINE_SERVER_PROFILE: $SGBENCH_BASELINE_SERVER_PROFILE"
  echo "SGBENCH_CANDIDATE_SERVER_PROFILE: $SGBENCH_CANDIDATE_SERVER_PROFILE"
  echo "SGBENCH_VP_FOREGROUND_STREAM_PRIORITY: $SGBENCH_VP_FOREGROUND_STREAM_PRIORITY"
  echo "FD_WEIGHTS: $FD_WEIGHTS"
  write_sgbench_manifest
  local reps="$SGBENCH_REPS"
  IFS=',' read -r -a modes <<<"$SGBENCH_MODES"
  for rep in $(seq 1 "$reps"); do
    local run_modes=("${modes[@]}")
    if [[ "$SGBENCH_ORDER_POLICY" == "counterbalanced" && $((rep % 2)) -eq 0 ]]; then
      run_modes=()
      for ((i=${#modes[@]}-1; i>=0; i--)); do
        run_modes+=("${modes[i]}")
      done
    elif [[ "$SGBENCH_ORDER_POLICY" != "fixed" && "$SGBENCH_ORDER_POLICY" != "counterbalanced" ]]; then
      die "unknown SGBENCH_ORDER_POLICY=$SGBENCH_ORDER_POLICY; expected fixed or counterbalanced"
    fi
    echo "SGBENCH_REP_${rep}_ORDER: ${run_modes[*]}"
    for raw_mode in "${run_modes[@]}"; do
      mode="$(echo "$raw_mode" | xargs)"
      [[ -n "$mode" ]] || continue
      sgbench_mode_once "$mode" "$rep"
    done
  done
}

sgbench_smoke() {
  SGBENCH_EVIDENCE_CLASS=smoke
  SGBENCH_DATASET=random
  SGBENCH_NUM_PROMPTS=8
  SGBENCH_MAX_CONCURRENCY=8
  SGBENCH_RANDOM_INPUT_LEN=512
  SGBENCH_RANDOM_OUTPUT_LEN=8
  SGBENCH_REPS=1
  sgbench
}

sgbench_flexidepth() {
  MODEL="$FD_BASE_MODEL"
  [[ -z "${SGBENCH_MODES_WAS_SET:-}" ]] && \
    SGBENCH_MODES="vanilla,vanilla_matched,flexidepth"
  [[ -z "${SGBENCH_DATASET_WAS_SET:-}" ]] && SGBENCH_DATASET="sharegpt"
  [[ -z "${SGBENCH_BACKEND_WAS_SET:-}" ]] && SGBENCH_BACKEND="sglang"
  [[ -z "${SGBENCH_NUM_PROMPTS_WAS_SET:-}" ]] && SGBENCH_NUM_PROMPTS=1024
  [[ -z "${SGBENCH_MAX_CONCURRENCY_WAS_SET:-}" ]] && SGBENCH_MAX_CONCURRENCY=256
  [[ -z "${SGBENCH_OUTPUT_LEN_WAS_SET:-}" ]] && SGBENCH_OUTPUT_LEN=256
  [[ -z "${SGBENCH_CONTEXT_LEN_WAS_SET:-}" ]] && SGBENCH_CONTEXT_LEN=""
  [[ -z "${SGBENCH_MAX_RUNNING_REQUESTS_WAS_SET:-}" ]] && SGBENCH_MAX_RUNNING_REQUESTS=""
  [[ -z "${SGBENCH_CHUNKED_PREFILL_SIZE_WAS_SET:-}" ]] && SGBENCH_CHUNKED_PREFILL_SIZE=""
  sgbench
}

sgbench_score() {
  preflight
  [[ "$SGBENCH_DATASET" == "longbench_v2" ]] || die "sgbench-score currently supports SGBENCH_DATASET=longbench_v2"
  [[ -n "$SGBENCH_SCORE_FILE" ]] || die "set SGBENCH_SCORE_FILE=/path/to/longbench_v2_*.jsonl"
  run_cmd python test/vp/score_sgbench_longbench_v2.py \
    --bench-jsonl "$SGBENCH_SCORE_FILE" \
    --model "$MODEL" \
    --num-prompts "$SGBENCH_NUM_PROMPTS" \
    --sharegpt-output-len "$SGBENCH_OUTPUT_LEN" \
    --sharegpt-context-len "$SGBENCH_CONTEXT_LEN" \
    --seed "$SGBENCH_SEED"
}

all() {
  stage_qwen3
  smoke_decode
  profile_decode
  ci_decode
  sgbench
}

# Every ACTION below drives a helper script. Several of those helpers are not
# part of this package: the vpipe cleanup kept the mechanism and dropped the
# evaluation tooling that no arm in this build exercises (removed-feature
# register). An ACTION whose helper is absent must fail HERE -- naming what is
# missing -- rather than minutes into a run with a bare "No such file", or
# worse, after a server is already up. Checked against the filesystem, so it
# fails when the file is genuinely gone and passes when it is restored.
require_scripts() {
  local missing=() s
  for s in "$@"; do
    [[ -f "$ROOT/test/vp/$s" ]] || missing+=("$s")
  done
  if (( ${#missing[@]} )); then
    echo "FATAL: ACTION=$ACTION requires helper script(s) absent from this" \
         "build: ${missing[*]}. Recover one with" \
         "'git show c3a1302668:test/vp/<name> > test/vp/<name>' (see the" \
         "removed-feature register), or choose an ACTION that does not need" \
         "it. Refusing rather than half-running." >&2
    exit 2
  fi
}

case "$ACTION" in
  preflight) preflight ;;
  stage-qwen3) stage_qwen3 ;;
  stage-flexidepth) require_scripts extract_flexidepth_weights.py; stage_flexidepth ;;
  smoke-decode) require_scripts profile_decode_modes.py; smoke_decode ;;
  profile-decode) require_scripts profile_decode_modes.py; profile_decode ;;
  ci-decode) require_scripts headline_ci.py; ci_decode ;;
  headline-gov) require_scripts mixed_longbench_serving.py; headline_gov ;;
  headline-mixed) require_scripts mixed_longbench_serving.py; headline_mixed ;;
  headline-multinews) require_scripts mixed_longbench_serving.py; headline_multinews ;;
  sgbench) require_scripts analyze_sgbench_pairs.py; sgbench ;;
  sgbench-smoke) require_scripts analyze_sgbench_pairs.py; sgbench_smoke ;;
  sgbench-flexidepth) require_scripts analyze_sgbench_pairs.py; sgbench_flexidepth ;;
  sgbench-score) require_scripts score_sgbench_longbench_v2.py; sgbench_score ;;
  all) require_scripts profile_decode_modes.py headline_ci.py analyze_sgbench_pairs.py; all ;;
  *)
    cat >&2 <<'EOF'
Usage:
  ACTION=preflight|stage-qwen3|stage-flexidepth|smoke-decode|profile-decode|ci-decode|headline-gov|headline-mixed|headline-multinews|sgbench|sgbench-smoke|sgbench-flexidepth|sgbench-score|all \
  DRY_RUN=1 bash test/vp/vast_a100_ladder.sh

Set DRY_RUN=0 only after the Vast A100 instance is intentionally running.
This script never starts, stops, recycles, or destroys a Vast instance.
For official SGLang serving runs, use ACTION=sgbench and --output-details files
under SGBENCH_OUT_DIR. ACTION=sgbench is diagnostic-only and cannot support a
performance decision. Use test/vp/launch_qps_server.py plus
test/vp/run_qps_evaluation.py for the open-loop labeled workload and omit
client max-concurrency. Each diagnostic
run still saves `/server_info` and sampled running/waiting request occupancy.
Explicit SGBENCH_MAX_RUNNING_REQUESTS values are allowed only with
SGBENCH_SERVER_CAP_PURPOSE=pressure and are never headline evidence.
Set SGBENCH_ATTENTION_BACKEND to compare an explicit SGLang attention backend;
leave it empty to retain SGLang's model/hardware default.
SGBENCH_MODES supports vanilla,vanilla_matched,flexidepth (plus override suffixes).
The V1 modes dynamic,dynamic_decode,dynamic_both,flexidepth_vp,flexidepth_vp_async,
flexidepth_vp_sched,flexidepth_vp_sched_async are REMOVED and refused at preflight.
vanilla uses SGBENCH_BASELINE_SERVER_PROFILE; vanilla_matched uses the same
SGBENCH_CANDIDATE_SERVER_PROFILE as the skipper mode.
The only override suffix this build supports is _fgpriority/_no_fgpriority,
which sets SGLANG_VP_FOREGROUND_STREAM_PRIORITY: model_runner.py reads it and
runs foreground model work at CUDA priority -1. It is parsed before the mode
whitelist precisely so that a live treatment is not refused along with the dead
ones.

Every other suffix that used to exist (_mixed_async, _stablebuf, _postweight,
_inplaceweight, _nearallrun, _fullprojbase, _reuseout, _shapelane, _splitgraph,
_routerfusion, _routergraph, _projfusion, _fdstageroute, _scopedasync,
_batchedasync, _kvonly and the rest) is REFUSED. Their only emission site was
the flexidepth_vp block removed with the V1 path, so an arm carrying one would
have run identically to plain flexidepth under a name claiming a treatment.
See codex/asplos-plan/2026-08-21-removed-feature-register.md.
For FlexiDepth without training, run ACTION=stage-flexidepth once, then
ACTION=sgbench-flexidepth with MODEL=NousResearch/Meta-Llama-3-8B-Instruct.
EOF
    exit 2
    ;;
esac
