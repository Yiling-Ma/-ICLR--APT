#!/usr/bin/env bash
set -euo pipefail
ROOT=/ssd2/mayiling/apt_unified_v2
PY=/home/mayiling/opt/apt-ft-baselines/bin/python
worker=${1:?worker id required}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$ROOT/results"
exec 9>"$ROOT/worker_${worker}.lock"
flock -n 9 || exit 1
case "$worker" in
  0) export CUDA_VISIBLE_DEVICES=0; jobs='cascade_contrast:0 cascade_contrast:1'; device=cuda ;;
  1) export CUDA_VISIBLE_DEVICES=1; jobs='cascade_contrast:2 cascade_contrast:3 cascade_contrast:4'; device=cuda ;;
  2) export CUDA_VISIBLE_DEVICES=2; jobs='flat_contrast:1 flat_contrast:2'; device=cuda ;;
  3) export CUDA_VISIBLE_DEVICES=3; jobs='flat_contrast:3 flat_contrast:4'; device=cuda ;;
  lr) export CUDA_VISIBLE_DEVICES=''; jobs='lr:0 lr:1 lr:2 lr:3 lr:4'; device=cpu ;;
  xgb) export CUDA_VISIBLE_DEVICES=''; jobs='xgb:0 xgb:1 xgb:2 xgb:3 xgb:4'; device=cpu ;;
  *) exit 2 ;;
esac
for job in $jobs; do
  "$PY" -u "$ROOT/code/run.py" --model "${job%:*}" --fold "${job#*:}" --output "$ROOT/results" --device "$device"
done
printf 'ASSIGNED_JOBS_COMPLETED worker=%s\n' "$worker"
