#!/usr/bin/env bash
set -euo pipefail
cd /home/mayiling/projs/apt_agent
ROOT=/ssd3/mayiling/apt_agent_runtime/patient_context_pilot_v1
PY=/home/mayiling/opt/apt-ft-baselines/bin/python
mkdir -p "$ROOT"
exec 9>"$ROOT/batch.lock"
flock -n 9 || exit 1
export PYTHONPATH=analysis
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PY" -m unittest discover -s analysis -p test_patient_context_pilot.py -v
CUDA_VISIBLE_DEVICES=6 "$PY" -u analysis/patient_context_pilot.py prepare --output "$ROOT"
CUDA_VISIBLE_DEVICES=6 "$PY" -u analysis/patient_context_pilot.py run --output "$ROOT" --shard 0 >"$ROOT/shard0.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=7 "$PY" -u analysis/patient_context_pilot.py run --output "$ROOT" --shard 1 >"$ROOT/shard1.log" 2>&1 &
B=$!
status=0
wait "$A" || status=1
wait "$B" || status=1
if [[ "$status" != 0 ]]; then exit "$status"; fi
"$PY" -u analysis/patient_context_pilot.py aggregate --output "$ROOT"
echo "COMPLETE patient-context development pilot"
