#!/usr/bin/env bash
set -euo pipefail
ROOT=/ssd2/mayiling/apt_hierarchy_signal
PY=/home/mayiling/opt/apt-ft-baselines/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
group=${1:?worker index}
export CUDA_VISIBLE_DEVICES="$group"
if [[ "$group" == 3 ]]; then
    kind=conditional; shuffle=0
else
    seeds=(101 211 307); kind=shuffle; shuffle=${seeds[$group]}
fi
for fold in 0 1 2 3 4; do
    "$PY" -u "$ROOT/analysis/hierarchy_signal/run_signal.py" --kind "$kind" --shuffle "$shuffle" --fold "$fold" --output "$ROOT/results"
done
echo "WORKER_COMPLETE $group"
