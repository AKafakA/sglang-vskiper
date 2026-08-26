#!/usr/bin/env bash
# CSD3 refactor A/B arm — mirrors serving/w2_ab_arm.sh exactly (same venv,
# module purge, CUDA/GCC env, helper-sm80, probe invocation). The ONLY
# difference between the two arms is PYTHONPATH, which selects the tree:
#   rewritten -> sglang.srt.vpipe
#   frozen    -> sglang.srt.vp   (c3a1302668)
# Both carry the identical requires_grad fix, so the refactor is the sole
# variable. $1 = arm name, $2 = tree root.
set -uo pipefail
ROOT="${VPIPE_CSD3_ROOT:-/rds/user/wd312/hpc-work/llm/vpipe-csd3}"
. /etc/profile.d/modules.sh; module purge
GREAL=/usr/local/software/spack/csd3/opt-2025-06-01/linux-rocky8-zen3/gcc-14.3.0/gcc-14.3.0-vlhhcp6mk32jxxqtnhkkmlrf2rpwwkrd
V=$ROOT/envs/sglang-serve-w2-r1/bin/python
export CUDA_HOME=$ROOT/envs/sglang-serve-w2-r1/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$GREAL/bin:$ROOT/envs/sglang-serve-w2-r1/bin:/usr/bin:/bin"
export CC=$GREAL/bin/gcc CXX=$GREAL/bin/g++
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$GREAL/lib64"
export CPATH="$ROOT/envs/sglang-serve-w2-r1/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
export NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

ARM=$1; TREE=$2
export PYTHONPATH="$TREE/python"

# vdec_fd posture, verbatim from the sealed arm env (defer=1 leg of w2_ab_arm).
export SGLANG_FD_ACTIVE_PHASES=decode
export SGLANG_FD_WEIGHTS=$ROOT/serving/flexidepth_router_weights.pt
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
export SGLANG_FD_FULL_GRAPH_CONDITIONAL_GRAPH_HELPER=$ROOT/serving/helper-sm80/libvpipe_cuda_conditional_graph.so

OUT=$RUNDIR/$ARM; mkdir -p $OUT
echo "=== CSD3-AB $ARM tree=$TREE $(date -u +%H:%M:%S)"
$V -c "import sglang.srt, os; print('sglang from:', os.path.dirname(sglang.srt.__file__))" 2>&1 | tail -1

$V -m sglang.launch_server --model-path $ROOT/models/Meta-Llama-3-8B-Instruct-53346005 --port 30231 \
  --attention-backend triton --prefill-attention-backend triton --decode-attention-backend triton \
  --disable-radix-cache > $OUT/server.log 2>&1 &
SPID=$!
for i in $(seq 1 200); do curl -sf -m 3 http://127.0.0.1:30231/health >/dev/null 2>&1 && break; kill -0 $SPID 2>/dev/null || break; sleep 6; done
curl -sf -m 3 http://127.0.0.1:30231/health >/dev/null 2>&1 || { echo "CSD3-AB $ARM SERVER FAILED"; tail -8 $OUT/server.log; exit 1; }

$V $ROOT/serving/fdpre_label_probe.py --url http://127.0.0.1:30231 \
  --requests-jsonl $ROOT/serving/gsm8k.first100.requests.jsonl --first-n 100 \
  --concurrency 8 --max-new-tokens 256 --topk 20 --stream-timing \
  --output-dir $OUT/probe 2>&1 | tail -2
curl -s -m 10 http://127.0.0.1:30231/server_info > $OUT/server_info.after.json
kill $SPID 2>/dev/null; sleep 8; kill -9 $SPID 2>/dev/null
echo "=== CSD3-AB $ARM DONE $(date -u +%H:%M:%S)"
