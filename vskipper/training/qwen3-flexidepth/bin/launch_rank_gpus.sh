#!/usr/bin/env bash
# OpenMPI rank -> single-GPU mapper for a SUBSET of the node's GPUs, so two arms can share one
# 4-GPU node (Phase A: world 2 + world 2) or one arm can take all four (Phase B: world 4).
# Same contract as the sealed campaigns/launch_rank.sh (RANK/LOCAL_RANK/WORLD_SIZE/CUDA_VISIBLE_DEVICES),
# with the GPU list taken from VSK_GPUS ("0 1" | "2 3" | "0 1 2 3") instead of the fixed (0 1 2 3).
set -euo pipefail
readonly OMPI_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:?missing OpenMPI local rank}
readonly OMPI_LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE:?missing OpenMPI local size}
readonly OMPI_GLOBAL_RANK=${OMPI_COMM_WORLD_RANK:?missing OpenMPI global rank}
readonly OMPI_WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:?missing OpenMPI world size}
read -r -a GPUS <<< "${VSK_GPUS:?export VSK_GPUS=\"0 1\" (space-separated GPU indices)}"
if (( OMPI_LOCAL_SIZE != ${#GPUS[@]} )); then
  echo "ERROR: ${OMPI_LOCAL_SIZE} ranks on this node but VSK_GPUS lists ${#GPUS[@]} GPUs (${VSK_GPUS})" >&2
  exit 2
fi
if (( OMPI_LOCAL_RANK < 0 || OMPI_LOCAL_RANK >= ${#GPUS[@]} )); then
  echo "ERROR: invalid local rank ${OMPI_LOCAL_RANK}" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=${GPUS[${OMPI_LOCAL_RANK}]}
export RANK=${OMPI_GLOBAL_RANK}
export LOCAL_RANK=0
export WORLD_SIZE=${OMPI_WORLD_SIZE}
exec "$@"
