#!/usr/bin/env bash
set -uo pipefail
export RUNDIR="${RUNDIR:?set RUNDIR to the staged A/B run directory}"
echo "=== CSD3 REFACTOR A/B start $(date -u +%FT%TZ) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash $RUNDIR/csd3_ab_arm.sh rewritten $RUNDIR/rewritten; echo "### rewritten exit=$?"
bash $RUNDIR/csd3_ab_arm.sh frozen    $RUNDIR/frozen;    echo "### frozen exit=$?"
echo "=== CSD3 REFACTOR A/B done $(date -u +%FT%TZ)"
