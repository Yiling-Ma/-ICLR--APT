#!/usr/bin/env bash
set -euo pipefail
cd /home/mayiling/projs/apt_agent
[[ $(hostname) == vllab11 ]] || exit 2
ROOT=/ssd3/mayiling/apt_agent_runtime/remaining_v1
PY=/home/mayiling/opt/apt-ft-baselines/bin/python
mkdir -p "$ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
case "$1" in
  mlp)
    exec 9>"$ROOT/mlp.lock"
    flock -n 9 || exit 3
    export CUDA_VISIBLE_DEVICES=4
    "$PY" -u analysis/run_full_budget_mlp.py run --output "$ROOT/full_mlp"
    "$PY" -u analysis/run_full_budget_mlp.py aggregate --output "$ROOT/full_mlp"
    ;;
  modality)
    exec 9>"$ROOT/modality.lock"
    flock -n 9 || exit 3
    export CUDA_VISIBLE_DEVICES=''
    for K in 2000 5000; do
      for COMMAND in prepare run aggregate; do
        "$PY" -u analysis/run_modality_repeats.py "$COMMAND" \
          --source /ssd3/mayiling/apt_agent_runtime/paired_information_validation \
          --output "$ROOT/modality_hvg$K" --hvg "$K"
      done
    done
    ;;
  oracle)
    exec 9>"$ROOT/oracle.lock"
    flock -n 9 || exit 3
    export CUDA_VISIBLE_DEVICES=5
    "$PY" -u analysis/export_lineage_oracle.py --output "$ROOT/oracle"
    ;;
  *) exit 2 ;;
esac
