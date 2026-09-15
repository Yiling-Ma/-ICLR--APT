# Hierarchy Bottleneck Mitigation (HBM-MLP)

This directory contains the deployable APT-only HBM experiment. It is isolated
from historical validated outputs and reuses the frozen APTBench loader, folds,
ontology, and subject-balanced metric from `analysis/patient_cell_scaling.py`
and `analysis/run_full_budget_mlp.py`.

No command in this directory modifies the frozen fold manifest or historical
results. Examples below assume execution from `/home/mayiling/projs/apt_agent`
with the validated environment.

```bash
export APT_HBM_PYTHON=/home/mayiling/opt/apt-ft-baselines/bin/python
export HBM_OUT=/ssd2/mayiling/apt_hbm_v1

# One-time exact log1p(raw-count) cache. The adjacent manifest hashes the raw
# sparse matrix, barcode file, and final annotated cell order.
$APT_HBM_PYTHON analysis/hierarchy_bottleneck_mitigation/prepare_log1p_cache.py \
  --output "$HBM_OUT/cache/raw_log1p.npy"

# Protocol/data/fold/ontology audit
$APT_HBM_PYTHON analysis/hierarchy_bottleneck_mitigation/run.py audit \
  --output "$HBM_OUT" --log1p-cache "$HBM_OUT/cache/raw_log1p.npy"

# A. Exact unified_v2 joint-head Plain MLP reproduction
# The reported run imported the already committed seed-17 validation selections
# (including their histories and source SHA-256) and performed fresh refits:
$APT_HBM_PYTHON analysis/hierarchy_bottleneck_mitigation/import_unified_mlp_selection.py \
  --source /path/to/paper-repo/analysis/unified_v2/results --output "$HBM_OUT"
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" refit plain

# To reproduce the historical validation search itself instead, replace the
# import command with:
# bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" select plain 17

# B and C
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" select parent 17
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" refit parent
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" select parent_cons 17
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" refit parent_cons

# Fold-specific privileged teacher. Selection teacher sees inner-train only;
# refit teacher sees outer-development only. It is frozen before student KD.
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" select teacher 17
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" refit teacher

# D, E, F
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" select kd 17
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" refit kd
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" select parent_kd 17
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" refit parent_kd
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" select full 17
bash analysis/hierarchy_bottleneck_mitigation/run_stage.sh "$HBM_OUT" refit full

# Rebuild all CSV/Markdown/LaTeX tables, paired intervals, and figures without
# retraining. When committed score CSVs are present under `$HBM_OUT/provenance`,
# this also rebuilds the historical-model comparison. The command fails if any
# required HBM OOF artifact is absent.
bash analysis/hierarchy_bottleneck_mitigation/rebuild_results.sh \
  "$HBM_OUT" "$HBM_OUT/report"

# Rebuild the paper-facing table directly from the saved aggregate CSV.
$APT_HBM_PYTHON analysis/hierarchy_bottleneck_mitigation/build_paper_assets.py \
  --results "$HBM_OUT/report/ablation_results.csv" \
  --table-output tables/hbm_ablation.tex
```

Selection is deliberately staged rather than Cartesian. Parent weight is
selected first; consistency then reuses it. KD selects temperature at weight
0.5, then compares weights 0.25/0.5/1.0 at the selected temperature. Combined
conditions reuse those validation-selected values. The outer test labels never
participate in these choices.
