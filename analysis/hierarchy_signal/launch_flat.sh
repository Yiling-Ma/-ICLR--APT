#!/usr/bin/env bash
set -euo pipefail
ROOT=/ssd2/mayiling/apt_hierarchy_signal
PY=/home/mayiling/opt/apt-ft-baselines/bin/python
seeds=(101 211 307)
shuffle=${seeds[${1:?worker index}]}
for fold in 0 1 2 3 4; do
    "$PY" -u "$ROOT/analysis/hierarchy_signal/run_signal.py" --kind shuffle --backbone flat --shuffle "$shuffle" --fold "$fold" --output "$ROOT/results"
done
echo "FLAT_WORKER_COMPLETE $1"
