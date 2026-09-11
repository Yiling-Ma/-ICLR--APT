#!/usr/bin/env bash
set -euo pipefail
cd /home/mayiling/projs/apt_agent
[[ $(hostname) == vllab11 ]] || exit 2
export APT_PAIRED_OUT=${APT_PAIRED_OUT:-/ssd3/mayiling/apt_agent_runtime/paired_information_validation}
mkdir -p "$APT_PAIRED_OUT"
exec 9>"$APT_PAIRED_OUT/run.lock"
flock -n 9 || { echo 'Paired validation already running'; exit 1; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON=/home/mayiling/opt/apt-ft-baselines/bin/python
"$PYTHON" -u analysis/validate_paired_information.py prepare
"$PYTHON" -u analysis/validate_paired_information.py run
"$PYTHON" -u analysis/validate_paired_information.py aggregate
