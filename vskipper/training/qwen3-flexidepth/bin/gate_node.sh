#!/usr/bin/env bash
# ON-NODE quality gate for one sealed checkpoint (adapted from harness/debug-20260902/gate_q4b.sh @7b1dce1:
# no relay -- the checkpoint is hardlinked into an inference copy on this node; the checkpoint's OWN model
# code is kept and asserted against the gate its config declares; tokenizer files come from the init dir).
# Runs while the training GPUs are idle (between segments). bf16 = the model's native dtype; the stock
# reference is measured on this node with the same harness (gate_node.sh q8b-stock stock <base model dir>).
# USAGE: gate_node.sh ARM STEP CKPT_DIR            (ARM=q8b-stock STEP=stock CKPT_DIR=<plain base model dir>)
# Output: ${RUN_ROOT}/gates/<ARM>/gate_step<STEP>.json (+ .log, probe json, step<STEP>_router.pt) and one
#         GATELINE appended to ${RUN_ROOT}/gates/GATELINES.txt.
set -uo pipefail
ARM=${1:?arm}; STEP=${2:?step}; SRC=${3:?checkpoint dir or base model dir}
readonly RUN_ROOT=/mydata/q8b
readonly PY=/mydata/venvs/training/bin/python
readonly IFEVAL_PY=/mydata/venvs/ifeval-score/bin/python
readonly H=${RUN_ROOT}/harness
readonly DONOR=${RUN_ROOT}/models/FlexiDepth-Qwen3-8B-init-sealed-ste     # tokenizer files only
readonly GATE_DTYPE=${GATE_DTYPE:-bfloat16}
readonly STOCK_JSON=${RUN_ROOT}/gates/q8b-stock/gate_stepstock.json
G=${RUN_ROOT}/gates/${ARM}; mkdir -p "${G}"
export HF_HOME=/mydata/hf
log() { echo "[$(date -u +%FT%TZ)] gate ${ARM}@${STEP}: $*" | tee -a "${G}/gate_step${STEP}.log"; }
if [ "${ARM}" = q8b-stock ]; then
  DST=${SRC}
else
  test -f "${SRC}/VP_CHECKPOINT_COMPLETE.json" || { log "checkpoint not sealed: ${SRC}"; exit 3; }
  DST=${G}/infer-step${STEP}
  if [ ! -f "${DST}/.ready" ]; then
    rm -rf "${DST}.partial"; mkdir -p "${DST}.partial"
    # hardlink weights/config/code; skip optimizer/scheduler/rng (gate needs none of them)
    ( cd "${SRC}" && find . -maxdepth 1 -type f ! -name 'optimizer*' ! -name 'scheduler*' ! -name 'rng_state*' \
        -exec ln {} "${DST}.partial/" \; )
    for f in tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json vocab.json merges.txt chat_template.jinja generation_config.json; do
      [ -f "${DST}.partial/${f}" ] || { [ -f "${DONOR}/${f}" ] && cp "${DONOR}/${f}" "${DST}.partial/"; }
    done
    for f in configuration_ddqwen3.py modeling_ddqwen3.py; do
      [ -s "${DST}.partial/${f}" ] || { log "REFUSING: checkpoint lacks its own ${f} (no donor code overwrite, D-360)"; exit 4; }
    done
    mv "${DST}.partial" "${DST}"; touch "${DST}/.ready"
  fi
  ( cd "${DST}" && python3 - <<'PY'
import json, pathlib, sys
cfg = json.load(open('config.json'))
mode = cfg.get('router_gate_mode', 'released')
src = pathlib.Path('modeling_ddqwen3.py').read_text()
has_ste = 'masks = masks + weights - weights.detach()' in src
knows_key = 'router_gate_mode' in src
if mode == 'ste_hard' and not has_ste:
    sys.exit('GATE-MODE-MISMATCH: config declares ste_hard but modeling_ddqwen3.py has no straight-through gate')
if mode != 'released' and not knows_key:
    sys.exit('GATE-MODE-MISMATCH: config declares %r but modeling_ddqwen3.py never reads router_gate_mode' % mode)
print('gate-mode consistent:', mode, '| ste-site', has_ste, '| reads-key', knows_key)
PY
  ) | tee -a "${G}/gate_step${STEP}.log" || { log "refusing to gate: code does not implement the declared gate"; exit 4; }
fi
GJ=${G}/gate_step${STEP}.json
if [ -n "${GATE_TASKS:-}" ] && [ -f "${GJ}" ]; then GJ_MERGE_INTO=${GJ}; GJ=${G}/gate_step${STEP}.${GATE_TASKS//,/_}.json; fi
# One GPU per gate, chosen at runtime: two arms may gate at the same time while the other arm still trains on its
# own two GPUs, so take the least-loaded GPU under an flock (never two gates on one 40 GB card).
GPU_A=""; for g in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | cut -d, -f1); do
  exec {LOCKFD}>/tmp/gate_gpu_${g}.lock; if flock -n ${LOCKFD}; then GPU_A=$g; break; fi; done
[ -n "${GPU_A}" ] || { log "no free GPU for the gate"; exit 6; }
log "gate_mixed dtype=${GATE_DTYPE} model=${DST} gpu=${GPU_A}"
( cd "${H}" && CUDA_VISIBLE_DEVICES=${GPU_A} "${PY}" gate_mixed.py --model "${DST}" --val mixed_val_v2.json --output "${GJ}" --dtype "${GATE_DTYPE}" ${GATE_TASKS:+--tasks ${GATE_TASKS}} > "${G}/gate_mixed_step${STEP}.log" 2>&1 ); GM_RC=$?
if [ "${ARM}" != q8b-stock ]; then
  ( cd "${H}" && CUDA_VISIBLE_DEVICES=${GPU_A} "${PY}" behavioural_probe.py --model "${DST}" --output "${G}/probe_step${STEP}.json" > "${G}/probe_step${STEP}.log" 2>&1 \
    && CUDA_VISIBLE_DEVICES=${GPU_A} "${PY}" behavioural_probe.py --model "${DST}" --output "${G}/probe_step${STEP}-chat.json" --chat-template >> "${G}/probe_step${STEP}.log" 2>&1 ); log "probes rc=$?"
fi
grep -E '^GATE' "${G}/gate_mixed_step${STEP}.log" | tail -1 | tee -a "${G}/gate_step${STEP}.log"
grep -E 'Error|Traceback' "${G}/gate_mixed_step${STEP}.log" | tail -2
test -s "${GJ}" || { log "GATE-JSON-MISSING rc=${GM_RC}"; exit 5; }
if [ -n "${GJ_MERGE_INTO:-}" ]; then python3 - "${GJ_MERGE_INTO}" "${GJ}" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); a["tasks"].update(b["tasks"]); json.dump(a, open(sys.argv[1], "w"), indent=1)
print("merged tasks", sorted(b["tasks"]), "into", sys.argv[1])
PY
GJ=${GJ_MERGE_INTO}; fi
# (ifeval and mmlu_pro removed from the gate by the owner 2026-09-16; bbh added -- see gate_mixed.py)
# GATELINE + router delta
"${PY}" - "${ARM}" "${STEP}" "${DST}" "${G}" "${STOCK_JSON}" <<'PY' | tee -a "${G}/gate_step${STEP}.log" "${RUN_ROOT}/gates/GATELINES.txt"
import json, glob, os, sys
tag, step, dst, g, stock = sys.argv[1:6]
gj = json.load(open(f'{g}/gate_step{step}.json'))
def acc(t):
    v = gj['tasks'].get(t)
    return 'na' if not v or v.get('accuracy') is None else f"{v['accuracy']:.2f}(e{v['empties']})"
ref = {}
try:
    r = json.load(open(stock))['tasks']
    ref = {k: (v['accuracy'] if v.get('accuracy') is not None else None) for k, v in r.items()}
except Exception:
    pass
probe = ''
try:
    p = json.load(open(f'{g}/probe_step{step}.json')); c = json.load(open(f'{g}/probe_step{step}-chat.json'))
    sr = p.get('overall_skip_rate'); sc = c.get('chat_skip_rate') or c.get('overall_skip_rate')
    empties = sum(1 for q in c.get('prompts', []) if not (q.get('continuation') or '').strip())
    probe = f" skip_raw={sr and round(sr,3)} skip_chat={sc and round(sc,3)} chat_empty={empties}/{len(c.get('prompts', []))}"
except Exception:
    pass
n = 0
if tag != 'q8b-stock':
    import torch
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(dst + '/*.safetensors')):
        with safe_open(f, framework='pt') as sf:
            for k in sf.keys():
                if '.router.' in k or '.router_proj.' in k:
                    out[k] = sf.get_tensor(k)
    torch.save(out, f'{g}/step{step}_router.pt'); n = len(out)
refs = ' '.join(f'{k}:{v:.2f}' for k, v in ref.items() if v is not None)
print(f'GATELINE arm={tag} step={step} gsm8k={acc("gsm8k")} bbh={acc("bbh")} coqa_mt={acc("coqa_mt")} coqa_flat={acc("coqa_flat")}{probe} router_tensors={n} | stock_ref(q8b,node): {refs}')
PY
log "done"
