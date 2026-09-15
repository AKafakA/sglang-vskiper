#!/usr/bin/env bash
# After the quality lane: prove the D-702 refactor serves the SAME DESIGN.
#
# Why this is queued rather than assumed. The projector/policy split was written to be
# behaviour-preserving -- every consumer keeps evaluating the same predicate, nothing is added
# inside the CUDA-graph capture region, no attestation field moves. That is an argument. Until
# it is checked, the paper's Implementation section describes an interface the MEASURED binary
# does not have, and this project's own rule is that an argument is not a measurement.
#
# ~20 min of card time, no traffic, two trees that differ in exactly five files (verified by
# `diff -rq` at staging: types/skipper/common/validation/seam, all md5-matching the local
# commit). It does NOT touch tree-318bc23929's serving package -- the measured tree is the
# `--tree-a` side and is only read.
set -uo pipefail
echo $$ > /opt/vpipe/chain_equiv_v1.pid
O=/opt/vpipe
A=$O/trees/tree-318bc23929
B=$O/trees/tree-57434b28d5
OUT=$O/campaign/tree-equivalence
LOG(){ echo "[$(date -u +%FT%TZ)] [equiv] $*"; }

qp=$(cat $O/chain_quality_v1.pid 2>/dev/null || echo 0)
if [ "$qp" = "0" ]; then LOG "no quality-chain pid -- refusing to guess"; exit 2; fi
LOG "waiting for the quality chain (pid $qp)"
while kill -0 "$qp" 2>/dev/null; do sleep 180; done
LOG "quality chain finished: $(tail -1 $O/chain_quality_v1.log 2>/dev/null | cut -c1-110)"

for attempt in $(seq 1 10); do
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  procs=$(pgrep -f 'sglang.launch_server' | wc -l)
  LOG "  card check $attempt: ${mem} MiB, $procs sglang processes"
  [ "${mem:-99999}" -lt 2048 ] && [ "$procs" -eq 0 ] && break
  if [ "$attempt" -ge 3 ] && [ "$procs" -gt 0 ]; then
    for p in $(pgrep -f 'sglang.launch_server'); do
      pp=$(awk '{print $4}' /proc/$p/stat 2>/dev/null || echo -1)
      if [ "$pp" = "1" ]; then LOG "  reaping ORPHAN server pid=$p (ppid=1)"; kill -9 "$p" 2>/dev/null; fi
    done
  fi
  sleep 30
done
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
procs=$(pgrep -f 'sglang.launch_server' | wc -l)
if [ "${mem:-99999}" -ge 2048 ] || [ "$procs" -ne 0 ]; then
  LOG "=== STOPPING: card still busy (${mem} MiB, $procs procs) ==="; LOG EQUIV-CHAIN-STOPPED; exit 0
fi
LOG "card free: ${mem} MiB, 0 sglang processes"

LOG "--- the delta between the two trees, re-stated at run time ---"
diff -rq "$A/python/sglang/srt/vpipe" "$B/python/sglang/srt/vpipe" | sed 's|/opt/vpipe/trees/||g' | sed 's/^/    /'

mkdir -p "$OUT"
# NO --allow. A behaviour-preserving refactor should need none, and a permissive default is
# how a gate passes on the day it matters (D-256).
/opt/vpipe/venv/bin/python "$B/test/vp/verify_tree_equivalence.py" \
   --spec "$O/campaign/quality_spec.json" \
   --tree-a "$A" --tree-b "$B" \
   --arm integrated_it4 --arm integrated_randomskip --arm integrated_alwaysskip \
   --port 32097 --out-dir "$OUT" >> "$O/tree_equiv.log" 2>&1
rc=$?
LOG "gate rc=$rc"

LOG "--- verdict, read from the report rather than the exit code ---"
python3 - "$OUT/tree_equivalence.json" <<'PY' 2>/dev/null || LOG "    no tree_equivalence.json -- the gate did not get that far"
import json, sys
d = json.load(open(sys.argv[1]))
for arm, r in sorted(d["arms"].items()):
    mark = "IDENTICAL" if r["identical"] else f"DIFFERS ({len(r['differing'])})"
    print(f"    {arm:28s} {mark} over {r['fields_compared']} fields")
    for path in r["differing"][:12]:
        print(f"        {path}")
PY
# The chain's exit status is the gate's: a missing report or a differing arm is a failure, not a DONE.
[ -s "$OUT/tree_equivalence.json" ] || { LOG "EQUIV-CHAIN-FAILED: no report"; exit 1; }
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if all(r["identical"] for r in d["arms"].values()) else 1)' "$OUT/tree_equivalence.json" \
  || { LOG "EQUIV-CHAIN-FAILED: an arm differs (rc=$rc)"; exit 1; }
[ "$rc" = 0 ] || { LOG "EQUIV-CHAIN-FAILED: gate rc=$rc"; exit 1; }
LOG EQUIV-CHAIN-DONE
