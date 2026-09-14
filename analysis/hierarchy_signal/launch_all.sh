#!/usr/bin/env bash
set -euo pipefail
ROOT=/ssd2/mayiling/apt_hierarchy_signal
mkdir -p "$ROOT/results" "$ROOT/reanalysis"
exec 9>"$ROOT/launch.lock"
flock -n 9 || { echo 'Another launcher holds the lock'; exit 1; }
pids=()
for worker in 0 1 2 3; do
    bash "$ROOT/analysis/hierarchy_signal/launch.sh" "$worker" >"$ROOT/worker_$worker.log" 2>&1 &
    pids+=("$!")
done
/home/mayiling/opt/apt-ft-baselines/bin/python "$ROOT/analysis/hierarchy_signal/analyze_existing.py" --source /ssd2/mayiling/apt_unified_v2/combined --output "$ROOT/reanalysis" >"$ROOT/reanalysis.log" 2>&1 &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" == 1 ]]; then echo 'FAILED: inspect worker logs'; exit 1; fi
echo 'ALL_MLP_AND_REANALYSIS_COMPLETE'
