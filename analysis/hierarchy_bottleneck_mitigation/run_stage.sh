#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 OUTPUT {select|refit} CONDITION [SEED ...]" >&2
  exit 2
fi

OUTPUT=$1
COMMAND=$2
CONDITION=$3
shift 3
if [[ $# -eq 0 ]]; then
  SEEDS=(17 29 43)
else
  SEEDS=("$@")
fi
PYTHON_BIN=${APT_HBM_PYTHON:-python}
GPU_COUNT=${APT_HBM_GPU_COUNT:-8}
mkdir -p "$OUTPUT/logs"

TASK_SEEDS=()
TASK_FOLDS=()
for SEED in "${SEEDS[@]}"; do
  for FOLD in 0 1 2 3 4; do
    TASK_SEEDS+=("$SEED")
    TASK_FOLDS+=("$FOLD")
  done
done

worker() {
  local GPU=$1
  local INDEX=$GPU
  while [[ $INDEX -lt ${#TASK_SEEDS[@]} ]]; do
    local SEED=${TASK_SEEDS[$INDEX]}
    local FOLD=${TASK_FOLDS[$INDEX]}
    local LOG="$OUTPUT/logs/${CONDITION}_${COMMAND}_s${SEED}_f${FOLD}.log"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON_BIN" -u \
        analysis/hierarchy_bottleneck_mitigation/run.py "$COMMAND" \
        --condition "$CONDITION" --seed "$SEED" --fold "$FOLD" \
        --output "$OUTPUT" --log1p-cache "$OUTPUT/cache/raw_log1p.npy" \
        >"$LOG" 2>&1 || return 1
    INDEX=$((INDEX + GPU_COUNT))
  done
}

PIDS=()
N_WORKERS=$GPU_COUNT
if [[ $N_WORKERS -gt ${#TASK_SEEDS[@]} ]]; then
  N_WORKERS=${#TASK_SEEDS[@]}
fi
for ((GPU=0; GPU<N_WORKERS; GPU++)); do
  worker "$GPU" &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || STATUS=1
done
if [[ $STATUS -ne 0 ]]; then
  echo "At least one ${CONDITION}/${COMMAND} fold failed; inspect $OUTPUT/logs" >&2
  exit 1
fi
