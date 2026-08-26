#!/usr/bin/env bash
# Login-node preflight. Every check can FAIL and prints what it saw -- no
# `ls DIR && echo present` (that passes whenever the directory exists, which
# once burned a 10-minute server launch for a client that was never committed).
set -uo pipefail
R="${VPIPE_CSD3_ROOT:-/rds/user/wd312/hpc-work/llm/vpipe-csd3}"
RUN=$1
fail=0
chk_file(){ if [ -s "$1" ]; then echo "  ok   $2 ($(stat -c%s "$1") B)"; else echo "  FAIL $2 MISSING/EMPTY: $1"; fail=1; fi; }
chk_dir(){  if [ -d "$1" ]; then echo "  ok   $2"; else echo "  FAIL $2 MISSING: $1"; fail=1; fi; }

echo "== trees =="
for A in refactored frozen; do
  chk_dir "$RUN/$A/python/sglang/srt" "$A sglang.srt"
done
# mutually exclusive packages: this is the A/B's ONLY intended variable
for spec in "refactored:vpipe:vp" "frozen:vp:vpipe"; do
  IFS=: read -r A want notwant <<<"$spec"
  if [ -d "$RUN/$A/python/sglang/srt/$want" ] && [ ! -d "$RUN/$A/python/sglang/srt/$notwant" ]; then
    echo "  ok   $A has srt.$want and NOT srt.$notwant"
  else
    echo "  FAIL $A package identity wrong (want=$want present=$(ls -d $RUN/$A/python/sglang/srt/{vp,vpipe} 2>/dev/null | tr '\n' ' '))"; fail=1
  fi
done

echo "== A/B parity patch (both arms must carry it, or the arms differ by more than the refactor) =="
for A in refactored frozen; do
  n=$(grep -c 'requires_grad_(False).eval()' "$RUN/$A/python/sglang/srt/models/llama.py" 2>/dev/null || echo 0)
  if [ "$n" = "2" ]; then echo "  ok   $A requires_grad_(False).eval() x2"
  else echo "  FAIL $A requires_grad_(False).eval() x$n, expected 2"; fail=1; fi
done

echo "== harness =="
chk_file "$RUN/perf_client.py"      "perf_client.py"
chk_file "$RUN/csd3_ab_compare.py"  "csd3_ab_compare.py"
chk_file "$RUN/csd3_ab_run.sh"      "csd3_ab_run.sh"
if grep -q RUNDIR_PLACEHOLDER "$RUN/csd3_ab_run.sh" 2>/dev/null; then
  echo "  FAIL csd3_ab_run.sh still contains RUNDIR_PLACEHOLDER"; fail=1
else echo "  ok   RUNDIR substituted"; fi

echo "== sealed assets =="
chk_file "$R/serving/arm_env_vdec_fd.sh"                "arm_env_vdec_fd.sh"
chk_file "$R/serving/flexidepth_router_weights.pt"      "router weights"
chk_file "$R/serving/gsm8k.first100.requests.jsonl"     "gsm8k requests"
chk_file "$R/serving/helper-sm80/libvpipe_cuda_conditional_graph.so" "sm80 conditional-graph helper"
chk_file "$R/envs/sglang-serve-w2-r1/bin/python"        "venv python"
chk_dir  "$R/models/Meta-Llama-3-8B-Instruct-53346005"  "model"

echo
if [ $fail -ne 0 ]; then echo "PREFLIGHT FAIL -- not submitting"; exit 1; fi
echo "PREFLIGHT OK"
