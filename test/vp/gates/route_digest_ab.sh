#!/usr/bin/env bash
# ROUTE-DIGEST EQUALITY: does a tree change what the model DOES?
#
# The box gate proves the tree runs and routes. The perf A/B measured speed.
# Neither proves OUTPUT equality. This does: the same 32 greedy requests through
# both trees, comparing the device route tape and the generated text.
#
#   refactored = $REFACTORED_TREE  (sglang.srt.vpipe)
#   frozen     = $FROZEN_TREE      (c3a1302668, sglang.srt.vp; sha-verified)
#
# Both trees must carry the identical requires_grad fix, so the tree is the
# only variable. Decode is greedy and the request set is fixed, so equal route
# rows + equal text is an equal route for every token of every request.
#
# Known instrument artifact: tape.digest_sum_u64 differs across boots on an
# identical tree (payload encodes physical request_slots); the comparator's
# other fields are the equality signal.
set -uo pipefail
GATES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${VP_GATE_WORKDIR:-/local/scratch/tmp/wd312}"
REFACTORED_TREE="${REFACTORED_TREE:-$W/vpipe-final}"
FROZEN_TREE="${FROZEN_TREE:-$W/baseline-tree}"
RES="${VP_GATE_AB_OUT:-$W/route-digest-ab}"

while pgrep -f "gate_final.sh|gate_r2.sh" >/dev/null; do sleep 30; done
sleep 20
rm -rf "$RES"; mkdir -p "$RES"

for PAIR in "refactored:$REFACTORED_TREE" "frozen:$FROZEN_TREE"; do
  NAME=${PAIR%%:*}; TREE=${PAIR##*:}
  echo "=== $NAME  tree=$TREE  $(date -u +%FT%TZ)" | tee -a "$RES/summary.txt"
  PYTHONPATH=$TREE/python bash "$GATES_DIR/box_vpcov_arm.sh" vdec_fd > "$RES/$NAME.log" 2>&1
  rc=$?
  OUT="${VP_GATE_OUT:-$W/vpcov-out}/vdec_fd"
  cp "$OUT/server_info.json"   "$RES/$NAME.server_info.json" 2>/dev/null
  cp "$OUT/probe/summary.json" "$RES/$NAME.probe.json"       2>/dev/null
  cp "$OUT/probe/labels.jsonl" "$RES/$NAME.labels.jsonl"     2>/dev/null
  echo "  rc=$rc" | tee -a "$RES/summary.txt"
done

python3 "$GATES_DIR/route_digest_compare.py" 2>&1 | tee -a "$RES/summary.txt"
echo "=== ROUTE DIGEST AB DONE $(date -u +%FT%TZ)" | tee -a "$RES/summary.txt"
