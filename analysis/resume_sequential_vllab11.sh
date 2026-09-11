#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/mayiling/projs/apt_agent/sequential_audit_20260909
cd "$ROOT"
[[ $(hostname) == vllab11 ]] || exit 2
PYTHON=/home/mayiling/opt/apt-ft-baselines/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export APT_XGB_DEVICE=cuda:0 ONEK1K_XGB_DEVICE=cuda:0 COMBAT_XGB_DEVICE=cuda:0
export COMBAT_PREPARED_DIR=${COMBAT_PREPARED_DIR:-/home/mayiling/data/combat/prepared}
export ONEK1K_PREPARED_DIR=${ONEK1K_PREPARED_DIR:-/home/mayiling/data/onek1k/prepared}
OUT=${APT_SEQUENTIAL_OUT:-/ssd3/mayiling/apt_agent_runtime/sequential_scaling}
mkdir -p "$OUT/logs"
exec 9>"$OUT/batch.lock"
flock -n 9 || { echo 'Another batch is running'; exit 1; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DATASETS=${APT_SEQUENTIAL_DATASETS:-"apt combat_rna onek1k"}
for dataset in $DATASETS; do
  pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$shard "$PYTHON" -u analysis/sequential_scaling_controls.py run \
      --dataset "$dataset" --seeds 10 --total 6400 \
      --models logistic_regression,xgboost,mlp --shard "$shard" --shards 4 \
      --output "$OUT/$dataset" \
      >"$OUT/logs/${dataset}_vllab11_${STAMP}_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [[ $failed == 1 ]]; then echo "FAILED $dataset $STAMP"; exit 1; fi
  "$PYTHON" -u analysis/sequential_scaling_controls.py aggregate --dataset "$dataset" --output "$OUT/$dataset"
  echo "COMPLETE $dataset $STAMP"
done
