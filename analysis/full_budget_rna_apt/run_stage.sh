#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 OUTPUT prepare|select|refit|aggregate [extra args]" >&2
  exit 2
fi
OUT=$1
STAGE=$2
shift 2
PYTHON_BIN=${APT_FULL_PYTHON:-python}
APT_CACHE=${APT_FULL_APT_CACHE:-}
SOURCE_HASH=${APT_FULL_SOURCE_HASH:-unknown}

if [[ $STAGE == prepare ]]; then
  "$PYTHON_BIN" analysis/full_budget_rna_apt/prepare.py audit --output "$OUT"
  for width in 5000 10000; do
    "$PYTHON_BIN" analysis/full_budget_rna_apt/prepare.py rna --output "$OUT" --width "$width" "$@"
  done
elif [[ $STAGE == select ]]; then
  for width in 5000 10000; do
    conditions=(rna rna_apt)
    [[ $width == 10000 ]] && conditions+=(rna_noise patient_shuffle lineage_shuffle)
    for condition in "${conditions[@]}"; do for fold in 0 1 2 3 4; do
      "$PYTHON_BIN" analysis/full_budget_rna_apt/run.py select --output "$OUT" --width "$width" \
        --condition "$condition" --fold "$fold" --apt-cache "$APT_CACHE" "$@"
    done; done
  done
elif [[ $STAGE == refit ]]; then
  for width in 5000 10000; do
    conditions=(rna rna_apt)
    [[ $width == 10000 ]] && conditions+=(rna_noise patient_shuffle lineage_shuffle)
    for condition in "${conditions[@]}"; do for fold in 0 1 2 3 4; do for seed in 17 29 43; do
      "$PYTHON_BIN" analysis/full_budget_rna_apt/run.py refit --output "$OUT" --width "$width" \
        --condition "$condition" --fold "$fold" --seed "$seed" --apt-cache "$APT_CACHE" \
        --source-git-hash "$SOURCE_HASH" "$@"
    done; done; done
  done
elif [[ $STAGE == aggregate ]]; then
  "$PYTHON_BIN" analysis/full_budget_rna_apt/aggregate.py --input "$OUT" --output "$OUT/report" \
    --capped-root outputs "$@"
else
  echo "unknown stage: $STAGE" >&2; exit 2
fi
