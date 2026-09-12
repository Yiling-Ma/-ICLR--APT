#!/usr/bin/env bash
set -euo pipefail
ROOT=/ssd3/mayiling/apt_agent_runtime/unified_v2
PY=/home/mayiling/opt/apt-ft-baselines/bin/python
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$ROOT/results"
# One worker avoids oversubscribing other users' GPUs; stop at the first error.
exec 9>"$ROOT/worker.lock"
flock -n 9 || exit 1
for model in mlp flat cascade hce flat_contrast cascade_contrast lr xgb; do
    for fold in 0 1 2 3 4; do
        "$PY" "$ROOT/code/run.py" --model "$model" --fold "$fold" --output "$ROOT/results" --device cuda
    done
done
printf 'ALL_TRAINING_JOBS_COMPLETED\n'
