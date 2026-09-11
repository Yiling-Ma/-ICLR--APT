#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/mayiling/projs/apt_agent
OUT=/ssd3/mayiling/apt_agent_runtime/rna_apt_distillation_v1
PYTHON=/home/mayiling/opt/apt-ft-baselines/bin/python
RSCRIPT=/home/mayiling/opt/apt-rna-r/bin/Rscript
cd "$ROOT"
mkdir -p "$OUT"
exec 9>"$OUT/batch.lock"
flock -n 9 || { echo 'This distillation batch is already running'; exit 1; }
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/analysis:${PYTHONPATH:-}"
"$PYTHON" -m unittest discover -s analysis -p test_rna_apt_distillation.py -v
for stage in pilot validation; do
  extra=()
  if [[ $stage == pilot ]]; then extra=(--pilot --folds 0); fi
  "$PYTHON" analysis/rna_apt_distillation.py prepare --output "$OUT/$stage" "${extra[@]}"
  "$RSCRIPT" analysis/prepare_distillation_rna.R "$ROOT" "$OUT/$stage"
  pids=()
  shards=2
  if [[ $stage == pilot ]]; then shards=1; fi
  for ((shard=0; shard<shards; shard++)); do
    CUDA_VISIBLE_DEVICES=$((6+shard)) "$PYTHON" analysis/rna_apt_distillation.py run \
      --output "$OUT/$stage" --shard "$shard" --shards "$shards" \
      >"$OUT/${stage}_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [[ $failed != 0 ]]; then echo "FAILED $stage"; exit 1; fi
  "$PYTHON" analysis/rna_apt_distillation.py aggregate --output "$OUT/$stage"
  echo "COMPLETE $stage"
done
