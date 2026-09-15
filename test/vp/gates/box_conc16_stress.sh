#!/usr/bin/env bash
# Conc-16 stress gate: N boots of one arm posture at the occupancy shape
# that exposed the batched-commit plan-table corruption (100-req,
# concurrency 16, 192 new tokens — the 2026-08-27 lesson: 32-req smokes
# never reach the multi-bucket drain replays where commit-path bugs
# live). Verdict per boot: SURVIVED (server healthy after the window,
# probe complete) or CRASHED (first CUDA error line quoted).
# Reachability/aliveness only — NEVER a performance number.
#
# Usage: box_conc16_stress.sh <arm> [boots] [tree-dir] [out-root]
# Env overrides mirror box_vpcov_arm.sh (VP_GATE_*).
set -uo pipefail
ARM="${1:?usage: box_conc16_stress.sh <arm> [boots] [tree-dir] [out-root]}"
BOOTS="${2:-3}"
GATES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${VP_GATE_WORKDIR:-/tmp/vpipe-gates}"
TREE="${3:-$(cd "$GATES_DIR/../../.." && pwd)}"
OUTROOT="${4:-$W/conc16-stress-out}"
V="${VP_GATE_PYTHON:-$W/sglang-sm75/.venv/bin/python}"
PORT="${VP_GATE_PORT:-30258}"
MODEL="${VP_GATE_MODEL:-$W/models/Meta-Llama-3-8B-Instruct-53346005}"
SUITE="${VP_GATE_SUITE:-$W/suites/gsm8k.first100.requests.jsonl}"
# [Codex F13] The arm_env_*.sh scripts are deleted (D-609): every canonical invocation of
# this gate exited 2 before launching a server. The arm is now selected by NAME through the
# durable file the served path reads, exactly as box_vpcov_arm.sh does.
TREE="${VP_GATE_TREE:-$(cd "$GATES_DIR/../../.." && pwd)}"
printf '%s\n' "$ARM" > "$TREE/deploy/active_arm"
export SGLANG_VP_HOST_CONFIG="${SGLANG_VP_HOST_CONFIG:-$TREE/deploy/hosts/a100.json}"
CRASHES=0
for BOOT in $(seq 1 "$BOOTS"); do
  # Scrub the treatment namespace EVERY iteration (the r2 env-leak lesson).
  while read -r v; do unset "$v"; done < <(env | grep -Eo '^SGLANG_(FD|VP)_[A-Z0-9_]+')
  set -a
  # shellcheck disable=SC1090
  set +a
  if [ "${VP_GATE_NO_FD_WEIGHTS:-0}" = "1" ]; then
    unset SGLANG_FD_WEIGHTS
  else
    export SGLANG_FD_WEIGHTS="${VP_GATE_FD_WEIGHTS:-$W/flexidepth_router_weights.pt}"
  fi
  if [ -n "${SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER:-}" ]; then
    export SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER="${VP_GATE_SM_HELPER:-$W/helper-sm75/libvpipe_cuda_conditional_graph.so}"
  fi
  unset SGLANG_MOE_CONFIG_DIR
  export SGLANG_IS_FLASHINFER_AVAILABLE=false
  export PYTHONPATH=$TREE/python
  OUT=$OUTROOT/$ARM/boot$BOOT
  rm -rf "$OUT"; mkdir -p "$OUT"
  setsid $V -m sglang.launch_server --model-path "$MODEL" \
    --port $PORT --dtype float16 \
    --attention-backend triton --prefill-attention-backend triton --decode-attention-backend triton \
    --disable-radix-cache > "$OUT/server.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 200); do
    curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || break
    sleep 6
  done
  if ! curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "boot$BOOT BOOT-FAILED"
    tail -5 "$OUT/server.log" | sed 's/^/    /'
    kill -TERM -$SPID 2>/dev/null
    CRASHES=$((CRASHES + 1))
    continue
  fi
  $V "$TREE/test/vp/fdpre_label_probe.py" --url "http://127.0.0.1:$PORT" \
    --requests-jsonl "$SUITE" --first-n 100 \
    --concurrency 16 --max-new-tokens 192 --topk 20 \
    --output-dir "$OUT/probe" 2>&1 | tail -1
  PROBE_RC=${PIPESTATUS[0]}
  ALIVE=0
  for _h in $(seq 1 8); do
    curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ALIVE=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 5
  done
  if [ "$ALIVE" = "1" ] && [ "$PROBE_RC" = "0" ]; then
    echo "boot$BOOT SURVIVED"
  elif [ "$ALIVE" = "1" ]; then
    # A live server with a failed probe is NOT survival — the workload
    # never completed (the review-caught client-failure blind spot).
    echo "boot$BOOT PROBE-FAILED rc=$PROBE_RC (server alive; workload incomplete)"
    CRASHES=$((CRASHES + 1))
  else
    echo "boot$BOOT CRASHED"
    grep -m1 "CUDA error" "$OUT/server.log" | sed 's/^/    /'
    CRASHES=$((CRASHES + 1))
  fi
  kill -TERM -$SPID 2>/dev/null; sleep 8; kill -9 -$SPID 2>/dev/null; sleep 4
done
echo "CONC16-STRESS $ARM: $((BOOTS - CRASHES))/$BOOTS survived"
[ "$CRASHES" = "0" ] || exit 6
