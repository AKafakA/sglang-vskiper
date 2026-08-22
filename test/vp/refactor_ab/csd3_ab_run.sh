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

set -a; . $R/serving/arm_env_vdec_fd.sh; set +a

PORT=30241
RATE=2.0
N=64
MAXNEW=128

for ARM in refactored frozen; do
  TREE=$RUN/$ARM
  export PYTHONPATH=$TREE/python
  OUT=$RES/$ARM; mkdir -p $OUT
  echo "=== $ARM  tree=$TREE  $(date -u +%FT%TZ)" | tee -a $RES/summary.txt
  # sglang.srt is a NAMESPACE package: __file__ is None, so use __path__.
  # Also attest WHICH vp package this arm actually imported -- the whole point
  # of the A/B is that the two arms differ by exactly that.
  $V -c "
import sglang.srt, importlib.util
print('  sglang.srt path:', list(sglang.srt.__path__))
for mod in ('sglang.srt.vpipe', 'sglang.srt.vp'):
    spec = importlib.util.find_spec(mod)
    print(f'  {mod}:', 'PRESENT' if spec else 'absent')
" 2>&1 | tee -a $RES/summary.txt

  $V -m sglang.launch_server --model-path $R/models/Meta-Llama-3-8B-Instruct-53346005 \
    --port $PORT --attention-backend triton --prefill-attention-backend triton \
    --decode-attention-backend triton --disable-radix-cache > $OUT/server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 200); do
    curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break
    kill -0 $SPID 2>/dev/null || break
    sleep 6
  done
  if ! curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
    echo "  $ARM SERVER FAILED" | tee -a $RES/summary.txt
    tail -20 $OUT/server.log | sed 's/^/    /' | tee -a $RES/summary.txt
    kill $SPID 2>/dev/null; continue
  fi

  if [ ! -f "$RUN/perf_client.py" ]; then
    echo "  FATAL: $RUN/perf_client.py missing -- refusing to run a server for nothing" | tee -a $RES/summary.txt
    kill $SPID 2>/dev/null; exit 4
  fi
  echo "  warmup (discarded)" | tee -a $RES/summary.txt
  $V $RUN/perf_client.py --url http://127.0.0.1:$PORT \
     --requests-jsonl $R/serving/gsm8k.first100.requests.jsonl \
     --n 24 --rate $RATE --max-new-tokens $MAXNEW --out $OUT/warmup.json > /dev/null 2>&1

  for REP in 1 2 3; do
    $V $RUN/perf_client.py --url http://127.0.0.1:$PORT \
       --requests-jsonl $R/serving/gsm8k.first100.requests.jsonl \
       --n $N --rate $RATE --seed $((1234+REP)) --max-new-tokens $MAXNEW \
       --out $OUT/rep$REP.json > $OUT/rep$REP.log 2>&1
    echo "    rep$REP $($V -c "
import json;d=json.load(open('$OUT/rep$REP.json'))['summary']
print('ok=%s err=%s offered=%s achieved=%s toks=%s ttft_mean=%sms tpot_mean=%sms'%(
 d['ok'],d['errors'],d['offered_rate'],d['achieved_rate'],d['total_output_tokens'],
 d['ttft_ms']['mean'],d['tpot_ms']['mean']))" 2>&1)" | tee -a $RES/summary.txt
  done

  curl -s -m 10 http://127.0.0.1:$PORT/server_info > $OUT/server_info.json
  kill $SPID 2>/dev/null; sleep 10; kill -9 $SPID 2>/dev/null; sleep 5
done

$V $RUN/csd3_ab_compare.py $RES 2>&1 | tee -a $RES/summary.txt
gate_rc=${PIPESTATUS[0]}
echo "=== CSD3 AB DONE $(date -u +%FT%TZ) gate_rc=$gate_rc" | tee -a $RES/summary.txt
# The comparator exits nonzero when errors, work identity, saturation or input
# availability fail. Return THAT, not the status of the echo above -- otherwise
# automation accepts an invalid A/B as successful, which is the swallowed-failure
# class this script was written to replace.
exit $gate_rc
