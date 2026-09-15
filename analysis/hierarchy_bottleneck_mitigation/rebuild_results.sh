#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SAVED_RUN_ROOT REPORT_OUTPUT" >&2
  exit 2
fi

PYTHON_BIN=${APT_HBM_PYTHON:-python}

"$PYTHON_BIN" analysis/hierarchy_bottleneck_mitigation/aggregate.py \
  --input "$1" --output "$2" --n-boot 2000 --seed 20260914
"$PYTHON_BIN" analysis/hierarchy_bottleneck_mitigation/summarize_selections.py \
  --input "$1" --output "$2"

if [[ -f "$1/provenance/unified_v2_scores.csv" && \
      -f "$1/provenance/hierarchy_signal_scores.csv" ]]; then
  "$PYTHON_BIN" analysis/hierarchy_bottleneck_mitigation/historical_compare.py \
    --ablation "$2/ablation_results.csv" \
    --unified-scores "$1/provenance/unified_v2_scores.csv" \
    --conditional-scores "$1/provenance/hierarchy_signal_scores.csv" \
    --output "$2"
fi
