#!/usr/bin/env bash
set -euo pipefail
cd /home/mayiling/projs/apt_agent
[[ $(hostname) == vllab11 ]] || exit 2
OUT=/ssd3/mayiling/apt_agent_runtime/cell_identity_questions_v1/hvg2000
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || exit 3
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/home/mayiling/opt/apt-ft-baselines/bin/python
cp analysis/CELL_IDENTITY_QUESTIONS_V1.md "$OUT/protocol.md"
for COMMAND in run aggregate; do
  "$PY" -u analysis/cell_identity_questions.py "$COMMAND" \
    --source /ssd3/mayiling/apt_agent_runtime/remaining_v1/modality_hvg2000 --output "$OUT"
done
