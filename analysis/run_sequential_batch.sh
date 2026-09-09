#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
export PYTHON
export APT_XGB_DEVICE=${XGB_DEVICE:-cpu} ONEK1K_XGB_DEVICE=${XGB_DEVICE:-cpu} COMBAT_XGB_DEVICE=${XGB_DEVICE:-cpu}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SHARDS=${SHARDS:-2}
mkdir -p outputs/sequential_scaling/logs
exec 9>outputs/sequential_scaling/batch.lock
flock -n 9 || { echo 'Another batch is already running'; exit 1; }
for dataset in apt combat_rna onek1k; do
  pids=()
  for ((shard=0; shard<SHARDS; shard++)); do
    bash analysis/run_sequential_scaling.sh "$dataset" "$shard" "$SHARDS" \
      >"outputs/sequential_scaling/logs/${dataset}_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [[ "$failed" == 1 ]]; then echo "FAILED $dataset"; exit 1; fi
  "$PYTHON" analysis/sequential_scaling_controls.py aggregate --dataset "$dataset"
  echo "COMPLETE $dataset"
done
