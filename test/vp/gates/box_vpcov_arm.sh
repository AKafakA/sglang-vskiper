#!/usr/bin/env bash
# One canonical arm on the box (sm75): serve, probe 32 greedy requests, attest.
# Reachability/attestation only -- NEVER a performance number.
#
# This is the PROVEN 2026-08-24 v2 driver (box_vpcov_arm_v2.sh) verbatim in
# logic, repo-resident since dev-h-harness-selfcontained (D-307 Lane H) with
# path resolution parameterized. v2's shape, kept exactly: TREE is a parameter
# and sets PYTHONPATH (namespace-package srt override, the proven CSD3 A/B
# pattern); the probe client comes from the tree under test; server_info is
# fetched again AFTER the probe (the attested route counters are post-probe);
# no coverage tracer. Arm postures are sourced verbatim from gates/arms/ and
# only their host-absolute paths are remapped below.
#
# Host-staged inputs (deliberately not in the repo): model snapshot, router
# weights (sha 9a1759...), sm75 helper .so, frozen request suite, serve venv.
#
# Usage: box_vpcov_arm.sh <arm> [tree-dir] [out-root]
set -uo pipefail
ARM="${1:?usage: box_vpcov_arm.sh <arm> [tree-dir] [out-root]}"
GATES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${VP_GATE_WORKDIR:-/local/scratch/tmp/wd312}"
TREE="${2:-$W/tree-a56485bcf8}"
OUTROOT="${3:-$W/vpcov-out}"
V="${VP_GATE_PYTHON:-$W/sglang-sm75/.venv/bin/python}"
PORT="${VP_GATE_PORT:-30250}"
MODEL="${VP_GATE_MODEL:-$W/models/Meta-Llama-3-8B-Instruct-53346005}"
SUITE="${VP_GATE_SUITE:-$W/suites/gsm8k.first100.requests.jsonl}"
OUT=$OUTROOT/$ARM
ENVFILE="${VP_GATE_ARM_DIR:-$GATES_DIR/arms}/arm_env_${ARM}.sh"
[ -f "$ENVFILE" ] || { echo "FATAL: no arm env $ENVFILE"; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"
set -a
# shellcheck disable=SC1090
source "$ENVFILE"
set +a
if [ "${VP_GATE_NO_FD_WEIGHTS:-0}" = "1" ]; then
  # Weightless-skipper arms (AdaSkip: requires_flexidepth_weights=False —
  # validation REFUSES loaded FD weights for them).
  unset SGLANG_FD_WEIGHTS
else
  export SGLANG_FD_WEIGHTS="${VP_GATE_FD_WEIGHTS:-$W/flexidepth_router_weights.pt}"
fi
if [ -n "${SGLANG_VP_ADASKIP_PROFILE:-}" ]; then
  export SGLANG_VP_ADASKIP_PROFILE="${VP_GATE_ADASKIP_PROFILE:-$TREE/test/vp/fixtures/adaskip_fixed_profile.json}"
fi
if [ -n "${SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER:-}" ]; then
  export SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER="${VP_GATE_SM_HELPER:-$W/helper-sm75/libvpipe_cuda_conditional_graph.so}"
fi
unset SGLANG_MOE_CONFIG_DIR
# Stock mode: strip EVERY vpipe knob (including the weights remap above) so
# the server is genuinely stock — the per-family stock-inertness gate serves
# the seamed tree this way and its generated text must equal upstream's.
if [ "${VP_GATE_STOCK:-0}" = "1" ]; then
  while read -r v; do unset "$v"; done < <(env | grep -Eo '^SGLANG_(FD|VP)_[A-Z0-9_]+')
fi
export SGLANG_IS_FLASHINFER_AVAILABLE=false
export PYTHONPATH=$TREE/python
{
  echo "=== VPCOV $ARM start $(date -u +%FT%TZ) tree=$TREE"
  env | grep -E '^SGLANG_' | sort
} > "$OUT/posture.txt"
$V -m sglang.launch_server --model-path "$MODEL" \
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
  echo "VPCOV $ARM SERVER FAILED" | tee -a "$OUT/posture.txt"
  tail -30 "$OUT/server.log" | sed 's/^/    /' | tee -a "$OUT/posture.txt"
  kill $SPID 2>/dev/null
  exit 3
fi
curl -s -m 10 "http://127.0.0.1:$PORT/server_info" > "$OUT/server_info.json"
$V "$TREE/test/vp/fdpre_label_probe.py" --url "http://127.0.0.1:$PORT" \
  --requests-jsonl "$SUITE" --first-n 32 \
  --concurrency 8 --max-new-tokens 128 --topk 20 \
  --output-dir "$OUT/probe" 2>&1 | tail -3 | tee -a "$OUT/posture.txt"
curl -s -m 10 "http://127.0.0.1:$PORT/server_info" > "$OUT/server_info.json"
kill $SPID 2>/dev/null; sleep 8; kill -9 $SPID 2>/dev/null; sleep 4
echo "=== VPCOV $ARM done $(date -u +%FT%TZ)" | tee -a "$OUT/posture.txt"
