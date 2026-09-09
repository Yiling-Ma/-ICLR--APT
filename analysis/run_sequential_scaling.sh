#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATASET=${1:?dataset required}
SHARD=${2:-0}
SHARDS=${3:-4}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export APT_XGB_DEVICE=${XGB_DEVICE:-cpu} ONEK1K_XGB_DEVICE=${XGB_DEVICE:-cpu} COMBAT_XGB_DEVICE=${XGB_DEVICE:-cpu}
"$PYTHON" analysis/sequential_scaling_controls.py run --dataset "$DATASET" \
  --seeds 10 --total 6400 --models logistic_regression,xgboost,mlp \
  --shard "$SHARD" --shards "$SHARDS"
