#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/mayiling/projs/apt_agent_onek1k}
PYTHON=${PYTHON:-/home/mayiling/.conda/envs/apt-scaling/bin/python}
PREPARED=${ONEK1K_PREPARED_DIR:-/home/mayiling/data/onek1k/prepared}
SHARDS=${1:-48}
XGB_THREADS=${ONEK1K_XGB_N_JOBS:-4}
LOG_DIR=${LOG_DIR:-/home/mayiling/data/onek1k/run_logs_xgb}

mkdir -p "$LOG_DIR"
cd "$ROOT"
for ((shard = 0; shard < SHARDS; shard++)); do
  nohup env \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ONEK1K_XGB_N_JOBS="$XGB_THREADS" \
    ONEK1K_PREPARED_DIR="$PREPARED" \
    "$PYTHON" analysis/onek1k_scaling.py run \
      --models xgboost \
      --num-shards "$SHARDS" \
      --shard-index "$shard" \
      >"$LOG_DIR/shard_${shard}.log" 2>&1 &
done
