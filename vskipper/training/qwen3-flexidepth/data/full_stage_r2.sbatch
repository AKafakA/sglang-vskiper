#!/usr/bin/env bash
#SBATCH --job-name=vpipe-fd-qwen3-data2
#SBATCH --account=<account>
#SBATCH --partition=sapphire
#SBATCH --qos=cpu2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/cluster-work/logs/fd-qwen3-data-r2-%j.out
#SBATCH --error=/path/to/cluster-work/logs/fd-qwen3-data-r2-%j.err

set -euo pipefail
umask 027
readonly STAGE=${STAGE:?submit with STAGE=alignment or STAGE=annealing}
case "${STAGE}" in
  alignment)
    readonly OUTPUT_NAME=flexidepth-qwen3-alignment-b14afda6-d117af2f-seq2048-r2
    ;;
  annealing)
    readonly OUTPUT_NAME=flexidepth-qwen3-annealing-af60f3c1-d117af2f-seq2048-r2
    ;;
  *)
    echo "invalid stage: ${STAGE}" >&2
    exit 2
    ;;
esac

readonly ROOT=/path/to/cluster-work
readonly ENV=${ROOT}/envs/flexidepth-2196aefd-py312-cu128-r1
readonly TOKENIZER=${ROOT}/models/Qwen3-8B-d117af2f
readonly TOKENIZER_REVISION=d117af2f304f02a8647f88fe05b61cfb405a1d9e
readonly TOKENIZER_MANIFEST=c5fdb878dd893ad1415410fa2d806787467acf3e794a5066630367a3bab7a953
readonly SOURCE=${ROOT}/sources/Flexidepth-1ec394d-qwen3-data-rc6.tar.gz
readonly SOURCE_SHA=61e77a3c7baf3afcc1400699151ee18c2197812076a113ea04008f4dcf3f7c7e
readonly SOURCE_REVISION=1ec394d209aadfbbc8e69c6ca7ee3ca4c67bc259
readonly OUTPUT=${ROOT}/datasets/${OUTPUT_NAME}
readonly RUN=${ROOT}/runs/flexidepth-qwen3-data-${STAGE}-r2/${SLURM_JOB_ID}

export HF_HOME=${ROOT}/cache/huggingface-flexidepth-qwen3
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONHASHSEED=0
test -x "${ENV}/bin/python"
printf '%s  %s\n' "${SOURCE_SHA}" "${SOURCE}" | sha256sum -c -
mkdir -p "${TMPDIR}/source" "${RUN}" "${HF_DATASETS_CACHE}" "$(dirname "${OUTPUT}")"
tar -xzf "${SOURCE}" -C "${TMPDIR}/source"
cd "${TMPDIR}/source/Flexidepth"
export PYTHONPATH=${PWD}
exec 9>"${OUTPUT}.lock"
flock -n 9 || { echo "another job owns ${OUTPUT}" >&2; exit 1; }
exec > >(tee -a "${RUN}/run.log") 2>&1

on_exit() {
  local status=$?
  printf 'exit_status=%s\nfinished_utc=%s\n' \
    "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${RUN}/exit-status.txt"
  if [[ ${status} -ne 0 ]]; then
    touch "${RUN}/FAILED"
  fi
  exit "${status}"
}
trap on_exit EXIT

"${ENV}/bin/python" datasets/prepare_qwen3_training_data.py \
  --stage "${STAGE}" \
  --tokenizer "${TOKENIZER}" \
  --tokenizer-revision "${TOKENIZER_REVISION}" \
  --tokenizer-manifest-sha256 "${TOKENIZER_MANIFEST}" \
  --source-revision "${SOURCE_REVISION}" \
  --source-archive-sha256 "${SOURCE_SHA}" \
  --output-dir "${OUTPUT}" \
  --num-proc 32 \
  --max-length 2048 \
  --test-size 0.01 \
  --seed 42
(cd "${OUTPUT}" && sha256sum -c DATASET_SHA256SUMS)
"${ENV}/bin/python" - "${OUTPUT}/PREPROCESSING_MANIFEST.json" "${STAGE}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
assert payload["status"] == "PASS"
assert payload["stage"] == sys.argv[2]
assert payload["is_preflight"] is False
assert payload["source_examples"] == payload["source_total_examples"]
assert payload["train_examples"] + payload["eval_examples"] == payload["retained_examples"]
PY
cp "${OUTPUT}/PREPROCESSING_MANIFEST.json" "${RUN}/"
cp "${OUTPUT}/DATASET_SHA256SUMS" "${RUN}/"
sha256sum "${RUN}/PREPROCESSING_MANIFEST.json" \
  "${RUN}/DATASET_SHA256SUMS" >"${RUN}/SHA256SUMS"
touch "${RUN}/SUCCESS"
echo "flexidepth_qwen3_data_${STAGE}_r2=PASS"
