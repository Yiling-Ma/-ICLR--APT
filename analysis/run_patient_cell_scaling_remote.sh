#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?Usage: run_patient_cell_scaling_remote.sh MODEL [SHARDS]}
SHARDS=${2:-8}
EXTRA_ARGS=("${@:3}")
PYTHON=${PYTHON:-/home/mayiling/opt/apt-ft-baselines/bin/python}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT="$ROOT/outputs/patient_cell_scaling"
MASTER_LOG="$OUTPUT/${MODEL}_master.log"

mkdir -p "$OUTPUT/worker_logs"
cd "$ROOT"

if [[ "$MODEL" == "logistic_regression" ]]; then
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
fi

printf '[%s] Starting %s with %s shards\n' "$(date -Iseconds)" "$MODEL" "$SHARDS" | tee -a "$MASTER_LOG"
pids=()
for ((shard = 0; shard < SHARDS; shard++)); do
    log="$OUTPUT/worker_logs/${MODEL}_shard${shard}.log"
    PYTHONPATH=cell_JEPA "$PYTHON" analysis/patient_cell_scaling.py run \
        --models "$MODEL" --shard-index "$shard" --num-shards "$SHARDS" \
        "${EXTRA_ARGS[@]}" \
        >"$log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
printf '[%s] Finished %s; worker_failure=%s\n' "$(date -Iseconds)" "$MODEL" "$failed" | tee -a "$MASTER_LOG"
exit "$failed"
