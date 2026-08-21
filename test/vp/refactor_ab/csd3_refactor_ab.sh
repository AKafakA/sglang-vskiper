#!/usr/bin/env bash
# CSD3 sealed open-loop A/B — the INSTRUMENT OF RECORD for the vpipe refactor.
#
# Run this AFTER re-authenticating the CSD3 ControlMaster socket:
#
#   ssh -fN -o ControlMaster=yes \
#     -o ControlPath=~/.ssh/sockets/csd3-wd312@login-icelake.hpc.cam.ac.uk-22 \
#     -o ControlPersist=yes wd312@login-icelake.hpc.cam.ac.uk
#
# Everything below is non-interactive once that socket exists. It answers the
# same question as the box A/B but on the measurement host with the sealed
# runner, so its numbers are citable where the box's are not.
#
# Arms differ ONLY by tree:
#   rewritten = sglang.srt.vpipe   (this refactor)
#   frozen    = sglang.srt.vp      (c3a1302668, pre-refactor)
# Both carry the identical requires_grad fix, so the refactor is the only
# variable. Work identity is asserted BEFORE any timing is compared (GR-1a).
set -euo pipefail

SOCK=~/.ssh/sockets/csd3-wd312@login-icelake.hpc.cam.ac.uk-22
CSD3=wd312@login-icelake.hpc.cam.ac.uk
REMOTE=/rds/user/wd312/hpc-work/llm/vpipe-csd3
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$REMOTE/refactor-ab-$STAMP

LOCAL_REWRITTEN=${1:-/home/wd312/Code/llm/vPipe/vpipe-core}
LOCAL_FROZEN=${2:-/tmp/claude-1000/-home-wd312-Code-llm-vPipe/e677d3cb-83f4-43fd-91a5-cb61d6bbf9e8/scratchpad/baseline}

ssh -o ControlPath=$SOCK -o BatchMode=yes $CSD3 true 2>/dev/null || {
  echo "FATAL: CSD3 ControlMaster socket is not alive. Re-auth first (MFA):"
  echo "  ssh -fN -o ControlMaster=yes -o ControlPath=$SOCK -o ControlPersist=yes $CSD3"
  exit 2
}

echo "== staging both trees to $RUN"
ssh -o ControlPath=$SOCK $CSD3 "mkdir -p $RUN/rewritten $RUN/frozen"
(cd "$LOCAL_REWRITTEN" && tar cf - python/sglang test/vp) |
  ssh -o ControlPath=$SOCK $CSD3 "cd $RUN/rewritten && tar xf -"
(cd "$LOCAL_FROZEN"    && tar cf - python/sglang test/vp) |
  ssh -o ControlPath=$SOCK $CSD3 "cd $RUN/frozen && tar xf -"

# The frozen arm needs the same requires_grad fix, or it cannot capture at all.
ssh -o ControlPath=$SOCK $CSD3 "python3 - <<'PY'
import pathlib
p=pathlib.Path('$RUN/frozen/python/sglang/srt/models/llama.py'); s=p.read_text()
if 'requires_grad_(False)' not in s:
    a='''            self.fd_proj.load_state_dict({'''
    i=s.index(a); j=s.index('})', i)+3
    s=s[:j]+'            self.fd_router.requires_grad_(False).eval()\n            self.fd_proj.requires_grad_(False).eval()\n'+s[j:]
    p.write_text(s); print('frozen arm: grad fix applied')
else: print('frozen arm: already patched')
PY"

echo "== submitting the INTR A/B (elapsed-billed, auto-releases on exit)"
ssh -o ControlPath=$SOCK $CSD3 "cat > $RUN/run_ab.sh <<'SH'
#!/usr/bin/env bash
set -uo pipefail
RUN=$RUN
source $REMOTE/serving/arm_env_vdec_fd.sh
export SGLANG_FD_WEIGHTS=$REMOTE/serving/flexidepth_router_weights.pt
for ARM in rewritten frozen; do
  export PYTHONPATH=\\\$RUN/\\\$ARM/python
  echo \"=== \\\$ARM \\\$(date -u +%FT%TZ)\"
  python -m sglang.launch_server --model-path $REMOTE/models/Meta-Llama-3-8B-Instruct-53346005 \\
    --port 31337 --attention-backend triton --prefill-attention-backend triton \\
    --decode-attention-backend triton --disable-radix-cache > \\\$RUN/\\\$ARM.server.log 2>&1 &
  SPID=\\\$!
  for _ in \\\$(seq 1 200); do curl -sf -m 3 http://127.0.0.1:31337/health >/dev/null && break; sleep 6; done
  python \\\$RUN/\\\$ARM/test/vp/run_qps_evaluation.py \\
    --suite $REMOTE/suites/gsm8k.pinned.json --rate 10 --reps 1 \\
    --output-dir \\\$RUN/\\\$ARM.cells > \\\$RUN/\\\$ARM.runner.log 2>&1
  curl -s http://127.0.0.1:31337/server_info > \\\$RUN/\\\$ARM.server_info.json
  kill \\\$SPID 2>/dev/null; sleep 10; kill -9 \\\$SPID 2>/dev/null
done
python \\\$RUN/rewritten/test/vp/cross_arm_work_gate.py \\\$RUN/rewritten.cells \\\$RUN/frozen.cells \\
  > \\\$RUN/work_gate.txt 2>&1 || echo 'WORK GATE FAILED - timings NOT comparable' >> \\\$RUN/work_gate.txt
SH
chmod +x $RUN/run_ab.sh"

ssh -o ControlPath=$SOCK $CSD3 \
  "cd $RUN && srun -N1 -n1 --qos=INTR --gres=gpu:1 -A KALYVIANAKI-SL3-GPU -p ampere \
     -t 1:0:0 bash $RUN/run_ab.sh" 2>&1 | tail -30

echo "== results in $RUN (work_gate.txt first, then the cells)"
