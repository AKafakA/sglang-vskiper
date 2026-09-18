#!/usr/bin/env bash
# Seed a NEW work dir for a world-size change from a sealed checkpoint (hardlinks, bit-identical), because
# the trainer byte-compares TRAINING_INPUT_CONTRACT.json (world/ga are in it) -- the consolidate_*_w8.sh
# pattern of the Sep-2 campaign. The first launch in the new dir writes the new contract and resumes
# from checkpoint-STEP via --resume-from-checkpoint auto (optimizer/scheduler/RNG restored).
# USAGE: consolidate_w4.sh SRC_TAG DST_TAG STEP
set -euo pipefail
SRC_TAG=${1:?src tag}; DST_TAG=${2:?dst tag}; STEP=${3:?step}
readonly RUN_ROOT=/mydata/q8b
SRC=${RUN_ROOT}/runs/${SRC_TAG}/keep/checkpoint-${STEP}
[ -f "${SRC}/VP_CHECKPOINT_COMPLETE.json" ] || SRC=${RUN_ROOT}/runs/${SRC_TAG}/work/checkpoint-${STEP}
test -f "${SRC}/VP_CHECKPOINT_COMPLETE.json"
for f in trainer_state.json optimizer.pt scheduler.pt; do test -f "${SRC}/${f}"; done
DST=${RUN_ROOT}/runs/${DST_TAG}/work
test ! -e "${RUN_ROOT}/runs/${DST_TAG}"   # never overwrite
mkdir -p "${DST}"
cp -al "${SRC}" "${DST}/checkpoint-${STEP}"
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m=={"global_step":int(sys.argv[2])}, m' "${DST}/checkpoint-${STEP}/VP_CHECKPOINT_COMPLETE.json" "${STEP}"
printf 'seeded_from=%s\nstep=%s\nat=%s\n' "${SRC}" "${STEP}" "$(date -u +%FT%TZ)" > "${RUN_ROOT}/runs/${DST_TAG}/SEEDED_FROM"
echo "seeded ${DST}/checkpoint-${STEP} from ${SRC}"
