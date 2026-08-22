#!/usr/bin/env bash
# CSD3 A/B — did the vpipe refactor regress performance?
#
#   refactored = sglang.srt.vpipe  (this PR)
#   frozen     = sglang.srt.vp     (c3a1302668)
# Both carry the identical requires_grad fix, so the refactor is the ONLY
# variable. Sealed vdec_fd posture, verbatim.
#
# OPEN-LOOP per the evaluation law: Poisson arrivals at a fixed offered rate is
# the load control; there is no client concurrency cap. (The sealed
# fdpre_label_probe is closed-loop and its own docstring marks its timings
# DIRECTIONAL ONLY, so it is not the instrument for this.)
#
# NOT a headline number: there is no frozen QPS suite on CSD3 and building one
# ad hoc would be a contract deviation. This is a REGRESSION CHECK. What is
# valid is the relative comparison, because both arms are identical except for
# the tree.
#
# Each arm: load once -> WARMUP run (discarded; a cold Triton JIT cache
# previously produced a false +34.7% TTFT regression that vanished on re-run)
# -> 3 measured reps.
set -uo pipefail
R=/rds/user/wd312/hpc-work/llm/vpipe-csd3
RUN=RUNDIR_PLACEHOLDER
RES=$RUN/results; mkdir -p $RES

# FRESHNESS. Reps are written to a fixed path per arm, so a re-run into a used
# results dir leaves the previous attempt's rep files in place for any arm that
# fails early -- and the comparator would read them as if they belonged to this
# run. Refuse instead of silently mixing two runs' data.
stale=$(find $RES -name 'rep[0-9].json' 2>/dev/null | head -5)
if [ -n "$stale" ]; then
  echo "FATAL: $RES already contains rep files from an earlier run:" >&2
  echo "$stale" | sed 's/^/  /' >&2
  echo "Use a fresh run directory; comparing across runs is not a controlled A/B." >&2
  exit 5
fi

. /etc/profile.d/modules.sh; module purge
GREAL=/usr/local/software/spack/csd3/opt-2025-06-01/linux-rocky8-zen3/gcc-14.3.0/gcc-14.3.0-vlhhcp6mk32jxxqtnhkkmlrf2rpwwkrd
V=$R/envs/sglang-serve-w2-r1/bin/python
export CUDA_HOME=$R/envs/sglang-serve-w2-r1/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$GREAL/bin:$R/envs/sglang-serve-w2-r1/bin:/usr/bin:/bin"
export CC=$GREAL/bin/gcc CXX=$GREAL/bin/g++
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$GREAL/lib64"
export CPATH="$R/envs/sglang-serve-w2-r1/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# The arm environment is MANDATORY. Sourcing it silently was enough to launch
# two dense (unrouted) arms that then agreed perfectly with each other.
if [ ! -s "$R/serving/arm_env_vdec_fd.sh" ]; then
  echo "FATAL: missing arm env $R/serving/arm_env_vdec_fd.sh" >&2; exit 8
fi
# Scrub the treatment namespace BEFORE sourcing. Anything SGLANG_FD_* or
# SGLANG_VP_* inherited from the caller's shell would silently join the arm
# posture, so the run would measure a configuration nobody declared -- and both
# arms would inherit it identically, so no gate would notice.
for _v in $(env | sed -n 's/^\(SGLANG_FD_[A-Z0-9_]*\)=.*/\1/p; s/^\(SGLANG_VP_[A-Z0-9_]*\)=.*/\1/p'); do
  unset "$_v"
done
set -a; . $R/serving/arm_env_vdec_fd.sh; set +a
if [ -z "${SGLANG_FD_WEIGHTS:-}" ] || [ "${SGLANG_FD_EXECUTION_MODE:-}" != "full_graph" ]; then
  echo "FATAL: arm env did not configure the routed posture" \
       "(weights='${SGLANG_FD_WEIGHTS:-}' mode='${SGLANG_FD_EXECUTION_MODE:-}')" >&2
  exit 8
fi

PORT=30241
RATE=2.0
N=64
MAXNEW=128

# An already-occupied port makes BOTH arms measure the same stale server, with
# zero errors and full token mass, so the comparator reports valid deltas for an
# experiment that never ran.
if curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
  echo "FATAL: port $PORT already serving; refusing to measure a stale server" >&2
  exit 8
fi

for ARM in refactored frozen; do
  TREE=$RUN/$ARM
  export PYTHONPATH=$TREE/python
  OUT=$RES/$ARM; mkdir -p $OUT
  echo "=== $ARM  tree=$TREE  $(date -u +%FT%TZ)" | tee -a $RES/summary.txt
  # ARM IDENTITY, ASSERTED. This block used to only PRINT: it had no assertion
  # and no non-zero exit, so staging both arms from the same tree -- or
  # resolving the package out of site-packages through the namespace package --
  # produced a fully passing A/B that compared identical code against itself.
  # Each arm must resolve its expected package UNDER ITS OWN TREE, and the two
  # identities must differ.
  case "$ARM" in
    refactored) WANT_PKG=sglang.srt.vpipe; DENY_PKG=sglang.srt.vp ;;
    frozen)     WANT_PKG=sglang.srt.vp;    DENY_PKG=sglang.srt.vpipe ;;
  esac
  ARM_ID=$($V -c "
import importlib.util, hashlib, pathlib, sys
tree = pathlib.Path('$TREE/python').resolve()
want = importlib.util.find_spec('$WANT_PKG')
deny = importlib.util.find_spec('$DENY_PKG')
if want is None:
    print('FAIL: $WANT_PKG not importable'); sys.exit(1)
if deny is not None:
    print('FAIL: $DENY_PKG is ALSO importable -- arms would not be distinct'); sys.exit(1)
loc = pathlib.Path(list(want.submodule_search_locations)[0]).resolve()
if tree not in loc.parents:
    print(f'FAIL: $WANT_PKG resolved to {loc}, OUTSIDE this arm tree {tree}'); sys.exit(1)
h = hashlib.sha256()
for p in sorted(loc.rglob('*.py')):
    h.update(p.relative_to(loc).as_posix().encode()); h.update(p.read_bytes())
print(h.hexdigest())" 2>&1)
  if ! printf '%s' "$ARM_ID" | grep -qE '^[0-9a-f]{64}$'; then
    echo "  $ARM IDENTITY CHECK FAILED: $ARM_ID" | tee -a $RES/summary.txt
    exit 9
  fi
  echo "  $ARM package=$WANT_PKG digest=${ARM_ID:0:16} tree=$TREE" | tee -a $RES/summary.txt
  echo "$ARM_ID" > $OUT/arm_identity.sha256

  # Port vacancy before EVERY arm, not once before the loop. If the first arm
  # (or a child of it) survives shutdown, the second arm's PID can be alive and
  # merely loading while the OLD server answers /health -- so the frozen arm
  # would measure the refactored server and every gate would still pass.
  for _w in $(seq 1 30); do
    curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || break
    sleep 2
  done
  if curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    echo "  $ARM ABORT: port $PORT still served by a previous process" | tee -a $RES/summary.txt
    exit 8
  fi

  setsid $V -m sglang.launch_server --model-path $R/models/Meta-Llama-3-8B-Instruct-53346005 \
    --port $PORT --attention-backend triton --prefill-attention-backend triton \
    --decode-attention-backend triton --disable-radix-cache > $OUT/server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 200); do
    curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || break
    sleep 6
  done
  # Health alone is not proof OUR server answered: require the process we
  # spawned to still be alive.
  if ! kill -0 $SPID 2>/dev/null; then
    echo "  $ARM SERVER PROCESS DIED before readiness" | tee -a $RES/summary.txt
    tail -20 $OUT/server.log | sed 's/^/    /' | tee -a $RES/summary.txt
    exit 8
  fi
  if ! curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    echo "  $ARM SERVER FAILED" | tee -a $RES/summary.txt
    tail -20 $OUT/server.log | sed 's/^/    /' | tee -a $RES/summary.txt
    kill $SPID 2>/dev/null; continue
  fi

  if [ ! -f "$RUN/perf_client.py" ]; then
    echo "  FATAL: $RUN/perf_client.py missing -- refusing to run a server for nothing" | tee -a $RES/summary.txt
    kill $SPID 2>/dev/null; exit 4
  fi
  echo "  warmup (discarded, but must SUCCEED)" | tee -a $RES/summary.txt
  # A failed warmup used to be ignored entirely -- no status check, no set -e.
  # The first MEASURED repetition then absorbed the cold Triton JIT compile,
  # which is exactly the false +34.7% TTFT regression this script's header
  # warns about, while the comparator still reported "valid" deltas.
  $V $RUN/perf_client.py --url http://127.0.0.1:$PORT \
     --requests-jsonl $R/serving/gsm8k.first100.requests.jsonl \
     --n 24 --rate $RATE --max-new-tokens $MAXNEW --out $OUT/warmup.json \
     > $OUT/warmup.log 2>&1
  warm_rc=$?
  warm_ok=$($V -c "
import json,sys
try:
    d = json.load(open('$OUT/warmup.json'))['summary']
except Exception as e:
    print('unreadable: %s' % e); sys.exit()
if d['errors'] or d['ok'] != 24 or d['requested_n'] != 24:
    print('ok=%s errors=%s requested=%s' % (d['ok'], d['errors'], d['requested_n']))
elif d['total_output_tokens'] != d['expected_total_output_tokens']:
    print('tokens %s != expected %s' % (d['total_output_tokens'], d['expected_total_output_tokens']))
else:
    print('OK')" 2>&1)
  if [ $warm_rc -ne 0 ] || [ "$warm_ok" != "OK" ]; then
    echo "    WARMUP FAILED rc=$warm_rc ($warm_ok) -- aborting this arm; the first" \
         "measured rep would absorb cold-JIT work" | tee -a $RES/summary.txt
    tail -5 $OUT/warmup.log 2>/dev/null | sed 's/^/      /' | tee -a $RES/summary.txt
    kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null
    exit 7
  fi
  echo "    warmup ok (24/24, zero errors, full token mass)" | tee -a $RES/summary.txt
  # The SERVER that answered must be running THIS arm's code. server_info
  # reports the package the live process loaded; a stale server from the other
  # arm would report the other package.
  SRV_PKG=$(curl -s -m 10 http://127.0.0.1:$PORT/server_info \
            | $V -c "
import json,sys
d=json.load(sys.stdin)
for st in (d.get('internal_states') or []):
    rt = st.get('vp_runtime') or st.get('runtime') or {}
    if rt: print('vpipe'); break
else: print('unknown')" 2>/dev/null)
  echo "    served package family: ${SRV_PKG:-unknown}" | tee -a $RES/summary.txt

  for REP in 1 2 3; do
    $V $RUN/perf_client.py --url http://127.0.0.1:$PORT \
       --requests-jsonl $R/serving/gsm8k.first100.requests.jsonl \
       --n $N --rate $RATE --seed $((1234+REP)) --max-new-tokens $MAXNEW \
       --out $OUT/rep$REP.json > $OUT/rep$REP.log 2>&1
    rep_rc=$?
    if [ $rep_rc -ne 0 ] || [ ! -s $OUT/rep$REP.json ]; then
      # Stop the arm at the FIRST failed repetition. Continuing leaves a
      # partial arm whose remaining reps look healthy, and a partial arm is
      # what let one repetition be compared against three.
      echo "    rep$REP FAILED rc=$rep_rc -- aborting this arm" | tee -a $RES/summary.txt
      tail -5 $OUT/rep$REP.log 2>/dev/null | sed 's/^/      /' | tee -a $RES/summary.txt
      kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null
      exit 6
    fi
    echo "    rep$REP $($V -c "
import json;d=json.load(open('$OUT/rep$REP.json'))['summary']
print('ok=%s err=%s offered=%s achieved=%s toks=%s ttft_mean=%sms tpot_mean=%sms'%(
 d['ok'],d['errors'],d['offered_rate'],d['achieved_rate'],d['total_output_tokens'],
 d['ttft_ms']['mean'],d['tpot_ms']['mean']))" 2>&1)" | tee -a $RES/summary.txt
  done

  curl -s -m 10 http://127.0.0.1:$PORT/server_info > $OUT/server_info.json
  # Tear down the whole PROCESS GROUP and wait for the port to actually free,
  # so the next arm cannot inherit this server.
  kill -TERM -$SPID 2>/dev/null || kill $SPID 2>/dev/null
  sleep 10
  kill -KILL -$SPID 2>/dev/null || kill -9 $SPID 2>/dev/null
  for _w in $(seq 1 30); do
    curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 || break
    sleep 2
  done
done

# The two arms must be DIFFERENT CODE. Without this, staging the same tree
# twice yields a flawless A/B of a tree against itself.
ID_A=$(cat $RES/refactored/arm_identity.sha256 2>/dev/null)
ID_B=$(cat $RES/frozen/arm_identity.sha256 2>/dev/null)
if [ -z "$ID_A" ] || [ -z "$ID_B" ]; then
  echo "FATAL: missing arm identity digest(s) -- cannot prove the arms differ" | tee -a $RES/summary.txt
  exit 9
fi
if [ "$ID_A" = "$ID_B" ]; then
  echo "FATAL: both arms resolved to IDENTICAL code (${ID_A:0:16}) -- this is not an A/B" | tee -a $RES/summary.txt
  exit 9
fi
echo "arm identities distinct: refactored=${ID_A:0:16} frozen=${ID_B:0:16}" | tee -a $RES/summary.txt

$V $RUN/csd3_ab_compare.py $RES 2>&1 | tee -a $RES/summary.txt
gate_rc=${PIPESTATUS[0]}
echo "=== CSD3 AB DONE $(date -u +%FT%TZ) gate_rc=$gate_rc" | tee -a $RES/summary.txt
# The comparator exits nonzero when errors, work identity, saturation or input
# availability fail. Return THAT, not the status of the echo above -- otherwise
# automation accepts an invalid A/B as successful, which is the swallowed-failure
# class this script was written to replace.
exit $gate_rc
