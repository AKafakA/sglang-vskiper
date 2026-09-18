#!/usr/bin/env bash
# Segmented training chain for ONE arm: train to the next kill point, gate on-node, apply the FAIL rule,
# continue. State machine with markers (universal R11), never edits a running script, kills nothing.
# USAGE: segment_chain.sh TAG MODEL_DIR PENALTY ACCUM WORLD "GPU LIST" PORT STOP1,STOP2,... [final]
#   STOP list = kill points (sealed checkpoint steps, multiples of --save-steps 1250); the literal word
#   `final` as the last element runs the closing segment to max_steps (27,643) and gates the LAST SEALED
#   checkpoint (27,500 = last multiple of 1,250; --checkpoint-probe is in the contract, so the trainer exports no
#   end-of-run model -- the final deliverable is that checkpoint, declared as such).
# FAIL rule (R/CLAUDE.md band): at STEP >= 5000 any task < stock - 0.25  -> STATE=FAIL-STOP, exit 10.
# HOLD rule: a task < stock - 0.10 is logged (band miss) but the chain continues -- the owner's keep rule
#   ("highest skip whose quality holds") is applied by decide_phase_a.sh / the owner, not here.
set -uo pipefail
TAG=${1:?tag}; MODEL=${2:?model}; PENALTY=${3:?penalty}; ACCUM=${4:?accum}; WORLD=${5:?world}
GPUS=${6:?gpus}; PORT=${7:?port}; STOPS=${8:?stop list}
readonly RUN_ROOT=/mydata/q8b
readonly BIN=${RUN_ROOT}/bin
RUN=${RUN_ROOT}/runs/${TAG}; WORK=${RUN}/work; OUT=${RUN}/out; KEEP=${RUN}/keep
mkdir -p "${RUN}" "${KEEP}"
STATE=${RUN}/STATE; CHAINLOG=${RUN}/chain.log
# schedule end and save interval (the smoke's OVERRIDES file may shorten them; see run_arm.sh)
MAX_STEPS=27643; SAVE_STEPS=1250
# shellcheck disable=SC1090
[ -f "${RUN}/OVERRIDES" ] && . "${RUN}/OVERRIDES"
FINAL_STEP=$(( MAX_STEPS / SAVE_STEPS * SAVE_STEPS ))   # last sealed save <= max_steps (27,500 for the real arms)
log() { echo "[$(date -u +%FT%TZ)] ${TAG}: $*" | tee -a "${CHAINLOG}"; }
state() { printf 'tag=%s phase=%s step=%s at=%s pid=%s\n' "${TAG}" "$1" "${2:-}" "$(date -u +%FT%TZ)" "$$" > "${STATE}"; }
keep_sealed() {  # hardlink every sealed checkpoint (save-total-limit 2 rotates them) -- keeper5 pattern
  for c in "${WORK}"/checkpoint-*; do
    [ -d "${c}" ] && [ -f "${c}/VP_CHECKPOINT_COMPLETE.json" ] || continue
    n=${c##*checkpoint-}; [ -f "${KEEP}/checkpoint-${n}/VP_CHECKPOINT_COMPLETE.json" ] && continue
    cp -al "${c}" "${KEEP}/checkpoint-${n}.partial" && mv "${KEEP}/checkpoint-${n}.partial" "${KEEP}/checkpoint-${n}" && log "kept checkpoint-${n}"
  done
}
run_segment() {  # $1 = stop step or "final"
  local stop=$1 extra=()
  [ "${stop}" != final ] && extra=(--stop-after-checkpoint-step "${stop}")
  if [ -d "${OUT}" ]; then  # the trainer refuses a pre-existing --output-dir; keep the previous export
    local prev; prev=$(ls "${WORK}"/checkpoint-* -d 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
    mv "${OUT}" "${OUT}-stop${prev:-0}-$(date -u +%H%M%S)"
  fi
  state training "${stop}"; log "segment -> ${stop} (world ${WORLD}, gpus '${GPUS}', penalty ${PENALTY}/${ACCUM})"
  "${BIN}/run_arm.sh" "${TAG}" "${MODEL}" "${PENALTY}" "${ACCUM}" "${WORLD}" "${GPUS}" "${PORT}" "${extra[@]}"
  local rc=$?
  log "segment ${stop} exited rc=${rc}"; keep_sealed
  return ${rc}
}
gate_step() {  # $1 = step ; returns 10 on FAIL rule
  local step=$1 ck=${KEEP}/checkpoint-$1
  [ -f "${ck}/VP_CHECKPOINT_COMPLETE.json" ] || ck=${WORK}/checkpoint-$1
  [ -f "${ck}/VP_CHECKPOINT_COMPLETE.json" ] || { log "no sealed checkpoint-${step}"; return 20; }
  state gating "${step}"
  "${BIN}/gate_node.sh" "${TAG}" "${step}" "${ck}" >> "${CHAINLOG}" 2>&1 || { log "gate_node rc=$? at ${step}"; return 21; }
  local line; line=$(grep "GATELINE arm=${TAG} step=${step} " "${RUN_ROOT}/gates/GATELINES.txt" | tail -1)
  log "${line}"
  touch "${RUN}/GATE-${step}-DONE"
  python3 - "${RUN_ROOT}/gates/${TAG}/gate_step${step}.json" "${RUN_ROOT}/gates/q8b-stock/gate_stepstock.json" "${step}" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))['tasks']; s = json.load(open(sys.argv[2]))['tasks']; step = int(sys.argv[3])
fail = []; miss = []
for t, sv in s.items():
    if sv.get('accuracy') is None or g.get(t, {}).get('accuracy') is None: continue
    d = g[t]['accuracy'] - sv['accuracy']
    if d < -0.25: fail.append(f'{t}:{d:+.2f}')
    if d < -0.10: miss.append(f'{t}:{d:+.2f}')
print(f'RULE step={step} band_misses={miss or "none"} fail_class={fail or "none"}')
# [D-836, owner 2026-09-17 07:5xZ] REPORT ONLY: the run finishes and every checkpoint is kept; a miss is logged, the sweep on the A100 selects. (D-828 early stop lifted for this arm.)
print("GATE MISS (report only, D-836)" if miss else "GATE PASS"); sys.exit(0)
PY
  local rc=$?
  [ ${rc} -eq 10 ] && { state FAIL-STOP "${step}"; log "FAIL RULE at ${step}: a task < stock - 25 pp -- chain stopped, owner paged"; }
  return ${rc}
}
log "chain start stops=${STOPS} model=${MODEL}"
keep_sealed   # hardlink anything already sealed (e.g. a segment run by hand before the chain)
IFS=, read -r -a STOP_ARR <<< "${STOPS}"
for s in "${STOP_ARR[@]}"; do
  if [ "${s}" != final ] && [ -f "${RUN}/GATE-${s}-DONE" ]; then log "step ${s} already gated -- skipping"; continue; fi
  if [ "${s}" != final ] && { [ -f "${KEEP}/checkpoint-${s}/VP_CHECKPOINT_COMPLETE.json" ] || [ -f "${WORK}/checkpoint-${s}/VP_CHECKPOINT_COMPLETE.json" ]; }; then
    log "checkpoint-${s} already sealed -- gating only"; keep_sealed
  else
    run_segment "${s}" || { state SEGMENT-FAILED "${s}"; log "segment ${s} FAILED -- chain stopped"; exit 11; }
    touch "${RUN}/SEG-${s}-DONE"
  fi
  step=${s}; [ "${s}" = final ] && step=${FINAL_STEP}
  gate_step "${step}"; grc=$?
  [ ${grc} -eq 10 ] && exit 10
  [ ${grc} -ne 0 ] && { state GATE-ERROR "${step}"; log "gate error rc=${grc} -- chain stopped (checkpoint kept)"; exit 12; }
done
state CHAIN-DONE "${STOP_ARR[-1]}"; touch "${RUN}/CHAIN-DONE"; log "chain complete"
