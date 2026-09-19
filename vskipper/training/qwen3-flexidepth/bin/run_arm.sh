#!/usr/bin/env bash
# FlexiDepth-Qwen3-8B alignment ARM launcher (campaign 2026-09-16-cloudlab-qwen3-8b-ste).
# Derived from the p6d launcher harness/debug-20260902/run_probe.sh (the only non-collapsing recipe):
# lr 1e-4 (trainer-enforced), micro-batch 4, GLOBAL BATCH 32, seq 2048, seed 42, max steps 27,643,
# save every 1,250, --checkpoint-probe, penalty ACCUM fixed at 4 so the effective coefficient
# (= PENALTY/ACCUM) is invariant when WORLD changes (owner rulings D-823: world/ga are the only
# recipe axes that move; the penalty values are the two ruled ones).
# USAGE: run_arm.sh TAG MODEL_DIR PENALTY ACCUM WORLD "GPU LIST" PORT [extra trainer flags]
#   e.g. run_arm.sh q8b-ste-c1e5 /mydata/q8b/models/FlexiDepth-Qwen3-8B-init-sealed-ste 4e-5 4 2 "0 1" 29631 \
#          --stop-after-checkpoint-step 2500
set -euo pipefail
umask 027
TAG=${1:?tag}; MODEL=${2:?model dir}; PENALTY=${3:?penalty}; ACCUM=${4:?accum}; WORLD=${5:?world}
GPUS=${6:?gpu list}; PORT=${7:?port}; shift 7
readonly RUN_ROOT=/mydata/q8b
readonly MICRO_BATCH=4
readonly GRAD_ACCUM=$((32 / (MICRO_BATCH * WORLD)))
test $((WORLD * MICRO_BATCH * GRAD_ACCUM)) -eq 32   # global batch is sacred (R7)
read -r -a GPU_ARR <<< "${GPUS}"; test "${#GPU_ARR[@]}" -eq "${WORLD}"
readonly ENV_DIR=/mydata/venvs/training
readonly SOURCE_ARCHIVE=${RUN_ROOT}/src/flexidepth-debug-20260902.tar.gz
readonly SOURCE_REVISION=ccb10a4fe31cf3cac01b688d882aaeeda6934f1b
readonly DATA=/mydata/datasets/flexidepth-qwen3-alignment-b14afda6-d117af2f-seq2048-r2
readonly DATA_SHA=7fd1a8f2d0836115e06a084b665b39ab01c44efaef39edba8426b5ef2225ad47
readonly DATA_NAME=allenai/tulu-3-sft-mixture
readonly DATA_REVISION=b14afda60f1bbebe55d5d2fa1e4df5042f97f8be
readonly MODEL_MANIFEST=${MODEL}/INITIALIZATION_MANIFEST.json
MAX_STEPS=27643
SAVE_STEPS=1250
# Smoke runs only: a per-tag OVERRIDES file (on disk, next to the run) may shorten the schedule. The real arms
# have no such file; the values land in the run's contract either way (no env-var experiment config, D-757).
if [ -f "${RUN_ROOT}/runs/${TAG}/OVERRIDES" ]; then
  # shellcheck disable=SC1090
  . "${RUN_ROOT}/runs/${TAG}/OVERRIDES"; echo "OVERRIDES applied from ${RUN_ROOT}/runs/${TAG}/OVERRIDES: MAX_STEPS=${MAX_STEPS} SAVE_STEPS=${SAVE_STEPS}"
fi
readonly MAX_STEPS SAVE_STEPS
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
WORK=${RUN_ROOT}/runs/${TAG}/work
OUTPUT=${RUN_ROOT}/runs/${TAG}/out
test -x "${ENV_DIR}/bin/python"; test -f "${MODEL_MANIFEST}"; test -f "${DATA}/DATASET_SHA256SUMS"; test -f "${SOURCE_ARCHIVE}"
# Sealed-input checks (fail closed): the source archive's shipped sha, the dataset manifest sha.
SOURCE_SHA=$(sha256sum "${SOURCE_ARCHIVE}" | awk '{print $1}')
test -f "${SOURCE_ARCHIVE}.sha256" && ( cd "$(dirname "${SOURCE_ARCHIVE}")" && sha256sum -c --quiet "$(basename "${SOURCE_ARCHIVE}").sha256" )
printf '%s  %s\n' "${DATA_SHA}" "${DATA}/DATASET_SHA256SUMS" | sha256sum -c --quiet -
MODEL_MANIFEST_SHA=$(sha256sum "${MODEL_MANIFEST}" | awk '{print $1}')
# The checkpoint's code must implement the gate its config declares (D-360 silent-wrong-arithmetic trap).
test "$(grep -c 'masks = masks + weights - weights.detach()' "${MODEL}/modeling_ddqwen3.py")" -eq 1
grep -q '"router_gate_mode": "ste_hard"' "${MODEL}/config.json"
# One extracted copy of the sealed source per archive sha.
SOURCE_EXTRACT=${RUN_ROOT}/src/extract-${SOURCE_SHA:0:12}
if [ ! -f "${SOURCE_EXTRACT}/.extracted" ]; then
  mkdir -p "${SOURCE_EXTRACT}"; tar -xzf "${SOURCE_ARCHIVE}" -C "${SOURCE_EXTRACT}"; touch "${SOURCE_EXTRACT}/.extracted"
fi
test -f "${SOURCE_EXTRACT}/train/sft_qwen3.py"
mkdir -p "${RUN_ROOT}/runs/${TAG}"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${PORT}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 OMP_NUM_THREADS=8
export HF_HOME=/mydata/hf VSK_GPUS="${GPUS}"
export PATH="${ENV_DIR}/bin:${PATH}"
LOG=${RUN_ROOT}/runs/${TAG}/segment-${RUN_ID}.log
{
  echo "arm=${TAG} run_id=${RUN_ID} world=${WORLD} gpus='${GPUS}' mb=${MICRO_BATCH} ga=${GRAD_ACCUM} penalty=${PENALTY} accum=${ACCUM} coef=$(python3 -c "print(${PENALTY}/${ACCUM})")"
  echo "model=${MODEL} manifest_sha=${MODEL_MANIFEST_SHA} data=${DATA} source_sha=${SOURCE_SHA} extra='$*'"
  df -h "${RUN_ROOT}" | tail -1
} | tee -a "${LOG}"
set +e
mpirun -H "127.0.0.1:${WORLD}" -npernode "${WORLD}" -np "${WORLD}" --bind-to none \
  -x MASTER_ADDR -x MASTER_PORT -x VSK_GPUS \
  -x TORCH_NCCL_ASYNC_ERROR_HANDLING -x TOKENIZERS_PARALLELISM \
  -x PYTHONHASHSEED -x OMP_NUM_THREADS -x HF_HOME \
  -x PATH -x LD_LIBRARY_PATH \
  "${RUN_ROOT}/bin/launch_rank_gpus.sh" \
  "${ENV_DIR}/bin/python" "${SOURCE_EXTRACT}/train/sft_qwen3.py" \
  --stage alignment \
  --model-path "${MODEL}" \
  --model-manifest-sha256 "${MODEL_MANIFEST_SHA}" \
  --dataset-dir "${DATA}" \
  --dataset-manifest-sha256 "${DATA_SHA}" \
  --dataset-name "${DATA_NAME}" \
  --dataset-revision "${DATA_REVISION}" \
  --source-revision "${SOURCE_REVISION}" \
  --source-archive-sha256 "${SOURCE_SHA}" \
  --work-dir "${WORK}" \
  --output-dir "${OUTPUT}" \
  --per-device-train-batch-size "${MICRO_BATCH}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit 2 \
  --logging-steps 25 \
  --max-steps "${MAX_STEPS}" \
  --checkpoint-probe \
  --resume-from-checkpoint auto \
  --router-penalty="${PENALTY}" \
  --router-penalty-accumulation-steps "${ACCUM}" \
  "$@" >> "${LOG}" 2>&1
RC=$?
echo "probe=EXIT tag=${TAG} run_id=${RUN_ID} rc=${RC}" | tee -a "${LOG}"
exit ${RC}
