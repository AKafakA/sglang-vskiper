#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_ROOT=${1:?usage: build_cuda_conditional_graph_helper.sh SOURCE_ROOT OUTPUT_DIR}
readonly OUTPUT_DIR=${2:?usage: build_cuda_conditional_graph_helper.sh SOURCE_ROOT OUTPUT_DIR}
readonly NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
readonly SOURCE=${SOURCE_ROOT}/vskipper/src/vskipper/kernels/csrc/cuda_conditional_graph.cu
readonly OUTPUT=${OUTPUT_DIR}/libvpipe_cuda_conditional_graph.so

test -x "${NVCC}"
test -f "${SOURCE}"
mkdir -p "${OUTPUT_DIR}"
"${NVCC}" -std=c++17 -O2 -shared -Xcompiler -fPIC "${SOURCE}" -o "${OUTPUT}"
printf '%s\n' "${OUTPUT}"
