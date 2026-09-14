#!/usr/bin/env bash
set -euo pipefail
ROOT=/ssd2/mayiling/apt_hierarchy_signal
CODE="$ROOT/analysis/hierarchy_signal"
PY=/home/mayiling/opt/apt-ft-baselines/bin/python
BASE=/ssd2/mayiling/apt_unified_v2/combined
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 8>"$ROOT/finish.lock"
flock -n 8 || exit 1
while ! grep -q ALL_MLP_AND_REANALYSIS_COMPLETE "$ROOT/launch.log"; do
    if grep -q FAILED "$ROOT/launch.log"; then echo 'FAILED initial workers'; exit 1; fi
    if ! kill -0 "${1:?initial launcher PID}" 2>/dev/null; then echo 'FAILED launcher disappeared'; exit 1; fi
    sleep 60
done
"$PY" "$CODE/audit_completed.py" --results "$ROOT/results" --model mlp
"$PY" "$CODE/summarize_signal.py" --source "$BASE" --results "$ROOT/results" --output "$ROOT/summary_mlp" --model mlp
gate=$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["flat_replication_gate"]))' "$ROOT/summary_mlp/validation.json")
if [[ "$gate" == 1 ]]; then
    echo 'PRESPECIFIED_FLAT_GATE_OPEN'
    pids=()
    for worker in 0 1 2; do
        CUDA_VISIBLE_DEVICES="$worker" bash "$CODE/launch_flat.sh" "$worker" >"$ROOT/flat_worker_$worker.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failed=1; fi
    done
    if [[ "$failed" == 1 ]]; then echo 'FAILED Flat replication'; exit 1; fi
    "$PY" "$CODE/audit_completed.py" --results "$ROOT/results" --model flat
    "$PY" "$CODE/summarize_signal.py" --source "$BASE" --results "$ROOT/results" --output "$ROOT/summary_flat" --model flat
else
    echo 'PRESPECIFIED_FLAT_GATE_CLOSED: no replication training'
fi
echo 'ALL_HIERARCHY_SIGNAL_COMPLETE'
