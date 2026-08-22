#!/usr/bin/env bash
# Paired performance A/B: rewritten vpipe tree vs frozen vp tree.
#
# Answers ONE question -- did the refactor regress performance -- by holding
# everything else fixed: same box, same suite, same decode config, same request
# count and concurrency, both trees carrying the identical requires_grad fix.
#
# OPEN-LOOP per the evaluation law: the offered rate is the load control, never
# a client concurrency cap. Every submitted request is awaited and counted.
#
# NOT a headline instrument: this is sm75, and it drives the server with a local
# client rather than the sealed runner (whose contract/manifest/workload
# apparatus lives on CSD3, not here). Absolute numbers are not citable. What IS
# valid is the RELATIVE comparison, because the only variable between the two
# arms is the refactor itself.
#
# Usage: box_perf_ab.sh <arm_name> <tree_root>
set -uo pipefail
ARM="${1:?usage: box_perf_ab.sh <arm> <tree_root>}"
TREE="${2:?usage: box_perf_ab.sh <arm> <tree_root>}"
W=/local/scratch/tmp/wd312
V=$W/serve-venv-d248/bin/python
PORT=30260
# ARM is interpolated into a path that is then rm -rf'd. Unvalidated, an
# argument like ../suites resolves OUTSIDE the result root and deletes a sibling
# tree. Restrict it to the supported identifiers.
case "$ARM" in
  refactored|frozen|rewritten) ;;
  *) echo "ERROR: unsupported arm '$ARM' (expected refactored|frozen|rewritten)" >&2; exit 2 ;;
esac
OUT=$W/perf-ab/$ARM
# belt and braces: the resolved path must still sit under the result root
case "$(readlink -m "$OUT")" in
  "$(readlink -m "$W/perf-ab")"/*) ;;
  *) echo "ERROR: refusing to clean $OUT -- outside $W/perf-ab" >&2; exit 2 ;;
esac
rm -rf "$OUT"; mkdir -p "$OUT"

# --- identical decode posture for both arms (the box T0 config) ---
export SGLANG_IS_FLASHINFER_AVAILABLE=false
export SGLANG_FD_ACTIVE_PHASES=decode
export SGLANG_FD_WEIGHTS=$W/flexidepth_router_weights.pt
export SGLANG_FD_EXECUTION_MODE=full_graph
export SGLANG_FD_FULL_GRAPH_COMPACT=1
export SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_TAPE=1
export SGLANG_FD_FULL_GRAPH_ROUTE_ACCOUNTING=1
export SGLANG_FD_VP_FUSED_PROJECT_INPUT=1
export SGLANG_FD_FULL_GRAPH_DEFER_PROJECT_KV=1
export SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH=1
export SGLANG_FD_FULL_GRAPH_MASKED_DECODE_ATTENTION=1
export SGLANG_FD_FULL_GRAPH_SCHEDULER_CONVERGENCE=1
export SGLANG_FD_FULL_GRAPH_DEVICE_ROUTE_DIGEST=1
export SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER=$W/helper-sm75/libvpipe_cuda_conditional_graph.so
export PYTHONPATH="$TREE/python"

# TCP bind vacancy before spawning: a listening-but-starting or 503-ing server
# is NOT a vacant port, and would otherwise be measured instead of ours.
port_free() {
  ! $V -c "
import socket, sys
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('127.0.0.1', $PORT)); s.close(); sys.exit(1)
except OSError:
    sys.exit(0)
"
}
for _w in $(seq 1 45); do port_free && break; sleep 2; done
if ! port_free; then
  echo "ERROR: port $PORT already bound; refusing to measure a stale server" >&2
  exit 3
fi

echo "=== PERF-AB $ARM  tree=$TREE  $(date -u +%FT%TZ)" | tee "$OUT/run.log"
# setsid puts the server in its OWN process group. Without it the background
# process inherits this script's group in a non-interactive shell, so comparing
# PGIDs proves nothing -- any sibling listener in the same job would match.
setsid $V -m sglang.launch_server --model-path $W/models/Meta-Llama-3-8B-Instruct-53346005 \
  --port $PORT --dtype float16 \
  --attention-backend triton --prefill-attention-backend triton --decode-attention-backend triton \
  --disable-radix-cache > "$OUT/server.log" 2>&1 &
SPID=$!
for _ in $(seq 1 160); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SPID 2>/dev/null || break
  sleep 6
done
# The endpoint must belong to the process WE spawned. Without this a server
# already owning the port ends the readiness loop on its own /health, and its
# workload gets measured even if our process died -- so both arms can compare
# the same stale server. Same assertion as csd3_ab_run.sh.
if ! kill -0 $SPID 2>/dev/null; then
  echo "$ARM SERVER PROCESS DIED before readiness" | tee -a "$OUT/run.log"
  exit 3
fi
SPGID=$(ps -o pgid= -p $SPID 2>/dev/null | tr -d ' ')
LPIDS=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
if [ -z "$SPGID" ] || [ -z "$LPIDS" ]; then
  echo "$ARM ABORT: cannot identify the listener on port $PORT" | tee -a "$OUT/run.log"
  kill $SPID 2>/dev/null; exit 3
fi
# EVERY listener must belong to our isolated group. Accepting "any match" let a
# foreign listener share the endpoint with ours and still pass.
foreign=""
for lp in $LPIDS; do
  if [ "$(ps -o pgid= -p "$lp" 2>/dev/null | tr -d ' ')" != "$SPGID" ]; then
    foreign="$foreign $lp"
  fi
done
if [ -n "$foreign" ]; then
  echo "$ARM ABORT: port $PORT also served by pid(s)$foreign outside our group $SPGID" \
    | tee -a "$OUT/run.log"
  kill $SPID 2>/dev/null; exit 3
fi
if [ "$SPGID" = "$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')" ]; then
  echo "$ARM ABORT: server shares this script's process group; setsid did not isolate it" \
    | tee -a "$OUT/run.log"
  kill $SPID 2>/dev/null; exit 3
fi
if ! curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "$ARM SERVER FAILED" | tee -a "$OUT/run.log"; tail -20 "$OUT/server.log" | tee -a "$OUT/run.log"
  kill $SPID 2>/dev/null; exit 3
fi
curl -s -m 10 "http://127.0.0.1:$PORT/server_info" > "$OUT/server_info.json"

# --- the timed workload: identical for both arms ---
$V $W/perf_client.py --url "http://127.0.0.1:$PORT" \
  --requests-jsonl $W/suites/gsm8k.first100.requests.jsonl \
  --n 60 --rate 0.5 --max-new-tokens 128 \
  --out "$OUT/metrics.json" 2>&1 | tee -a "$OUT/run.log"
client_rc=${PIPESTATUS[0]}

curl -s -m 10 "http://127.0.0.1:$PORT/server_info" > "$OUT/server_info.after.json"
kill $SPID 2>/dev/null; sleep 8; kill -9 $SPID 2>/dev/null

# The measurement IS the run. Without this the client's status dies in the tee
# pipe and the final echo exits 0, so a caller reads "done" as "measured" -- the
# swallowed-failure class csd3_ab_run.sh was rewritten to eliminate.
if [ "$client_rc" -ne 0 ] || [ ! -s "$OUT/metrics.json" ]; then
  echo "=== PERF-AB $ARM FAILED client_rc=$client_rc metrics=$([ -s "$OUT/metrics.json" ] && echo present || echo MISSING)" | tee -a "$OUT/run.log"
  exit 4
fi
echo "=== PERF-AB $ARM done $(date -u +%FT%TZ)" | tee -a "$OUT/run.log"
