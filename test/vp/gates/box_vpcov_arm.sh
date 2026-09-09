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
# [D-609] Harness paths come from the committed host config, like everything else.
# These are plumbing (where to write, which interpreter), not design -- but there is
# no reason for a second mechanism, and a hardcoded dev-box path is how this script
# failed on the A100.
_HC="${SGLANG_VP_HOST_CONFIG:-}"
if [ -n "$_HC" ] && [ -f "$_HC" ]; then
  W="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('gate_workdir',''))" "$_HC")"
  VP_GATE_PYTHON="${VP_GATE_PYTHON:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('serve_python',''))" "$_HC")}"
  VP_GATE_MODEL="${VP_GATE_MODEL:-$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('model_path',''))" "$_HC")}"
fi
W="${W:-${VP_GATE_WORKDIR:-/local/scratch/tmp/wd312}}"
TREE="${2:-$W/tree-a56485bcf8}"
OUTROOT="${3:-$W/vpcov-out}"
V="${VP_GATE_PYTHON:-$W/sglang-sm75/.venv/bin/python}"
PORT="${VP_GATE_PORT:-30250}"
MODEL="${VP_GATE_MODEL:-$W/models/Meta-Llama-3-8B-Instruct-53346005}"
SUITE="${VP_GATE_SUITE:-$W/suites/gsm8k.first100.requests.jsonl}"
OUT=$OUTROOT/$ARM
rm -rf "$OUT"; mkdir -p "$OUT"
# [D-609] Arms are NAMED, not exported. The arm_env_*.sh scripts are deleted: an
# export that fails to reach the server is indistinguishable from one that worked,
# which is how eighteen hours ran on a rejected design. The arm goes into the
# durable file the served path reads, and the design comes from the tree.
python3 - "$ARM" "$TREE" <<'ARMPY'
import importlib.util, pathlib, sys
arm, tree = sys.argv[1], pathlib.Path(sys.argv[2])
# design.py is dependency-free by design; load it BY PATH so this works under bare
# python3 without dragging in the sglang package chain (orjson et al).
spec = importlib.util.spec_from_file_location(
    "vpipe_design", tree / "python" / "sglang" / "srt" / "vpipe" / "design.py")
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
d.resolve_arm(arm)                                    # fails closed on an unknown arm
(tree / "deploy").mkdir(exist_ok=True)
(tree / "deploy" / "active_arm").write_text(arm + "\n")
print(f"active arm -> {arm}")
ARMPY
[ $? -eq 0 ] || { echo "FATAL: arm $ARM is not a canonical arm"; exit 2; }
export SGLANG_VP_HOST_CONFIG="${VP_GATE_HOST_CONFIG:-$TREE/deploy/hosts/vast-a100.json}"
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
