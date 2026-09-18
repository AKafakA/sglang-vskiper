#!/usr/bin/env bash
# Submit on the assigned GPU cluster with the account/partition supplied to sbatch.
# Arguments: CONTRACT_JSON SERVING_ENV COMPILER_ROOT [NEW_TEST_TOOLS_DIRECTORY]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60
#SBATCH --job-name=vsk-ref-layout
set -euo pipefail
CONTRACT=${1:?contract JSON required}
SERVING_ENV=${2:?existing serving environment required}
COMPILER_ROOT=${3:?existing compiler root required}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
test -n "${SLURM_JOB_ID:-}"
test -x "$SERVING_ENV/bin/python"
test -x "$COMPILER_ROOT/bin/gcc"
export CUDA_HOME="$SERVING_ENV/lib/python3.12/site-packages/nvidia/cu13"
test -x "$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$COMPILER_ROOT/bin:$SERVING_ENV/bin:/usr/bin:/bin"
export CC="$COMPILER_ROOT/bin/gcc" CXX="$COMPILER_ROOT/bin/g++"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$COMPILER_ROOT/lib64"
export CPATH="$SERVING_ENV/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include"
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="$(dirname -- "$CONTRACT")/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$(dirname -- "$CONTRACT")/torchinductor-cache"
if [ -n "${4:-}" ]; then
    test ! -e "$4"
    "$SERVING_ENV/bin/python" -m pip install --no-deps --target "$4" \
        -r "$(dirname -- "$CONTRACT")/candidate/vskipper/requirements-validation.txt"
fi
# Slurm copies this batch script; the contract's sibling candidate is the pinned source.
exec "$SERVING_ENV/bin/python" "$(dirname -- "$CONTRACT")/candidate/vskipper/scripts/validate_reference.py" "$CONTRACT"
