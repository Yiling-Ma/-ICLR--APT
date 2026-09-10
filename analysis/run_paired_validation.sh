#!/usr/bin/env bash
set -euo pipefail
cd /home/mayiling/projs/apt_agent
[[ $(hostname) == vllab11 ]] || exit 2
mkdir -p outputs/paired_information_validation
exec 9>outputs/paired_information_validation/run.lock
flock -n 9 || { echo 'Paired validation already running'; exit 1; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PYTHON=/home/mayiling/opt/apt-ft-baselines/bin/python
"$PYTHON" -u analysis/validate_paired_information.py prepare
"$PYTHON" -u analysis/validate_paired_information.py run
"$PYTHON" -u analysis/validate_paired_information.py aggregate
