#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"

if [[ "${#GPUS[@]}" -lt 8 ]]; then
  echo "GPU_IDS must provide eight comma-separated GPU ids" >&2
  exit 2
fi

run_job() {
  local variant="$1"
  local fold="$2"
  local gpu="$3"
  local log="matched_ft_${variant}_fold${fold}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=cell_JEPA \
    "${PYTHON_BIN}" -m apt_jepa.scripts.train_matched_ft_transformer \
    --variant "${variant}" --fold "${fold}" >"${log}" 2>&1 &
  JOB_PID="$!"
}

wait_for_jobs() {
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

# Wave 1 saturates all eight GPUs.
pids=()
for fold in 0 1 2 3 4; do
  run_job flat_ce "${fold}" "${GPUS[$fold]}"
  pids+=("${JOB_PID}")
done
for fold in 0 1 2; do
  run_job hce "${fold}" "${GPUS[$((fold + 5))]}"
  pids+=("${JOB_PID}")
done
wait_for_jobs

# Wave 2 completes the two remaining HCE folds.
pids=()
for fold in 3 4; do
  run_job hce "${fold}" "${GPUS[$((fold - 3))]}"
  pids+=("${JOB_PID}")
done
wait_for_jobs

PYTHONPATH=cell_JEPA "${PYTHON_BIN}" -m apt_jepa.scripts.pool_matched_ft_transformer
