# Full-budget matched RNA/APT sensitivity experiment

This frozen experiment asks whether paired APT improves held-out-patient cell
typing beyond a strong, full-cell-budget RNA reference. Read `PROTOCOL.md`
before running. Existing capped results are never overwritten.

```bash
export APT_FULL_PYTHON=/home/mayiling/opt/apt-ft-baselines/bin/python
export APT_FULL_APT_CACHE=/ssd2/mayiling/apt_hbm_v1/cache/raw_log1p.npy
export APT_FULL_SOURCE_HASH=GITHUB_COMMIT_CONTAINING_THIS_PROTOCOL
export FULL_OUT=/ssd2/mayiling/apt_full_budget_rna_apt_v1

# Freeze/audit and generate training-only 5k/10k features.
bash analysis/full_budget_rna_apt/run_stage.sh "$FULL_OUT" prepare \
  --rscript /home/mayiling/opt/apt-rna-r/bin/Rscript

# Selection must finish before any outer refit is launched.
bash analysis/full_budget_rna_apt/run_stage.sh "$FULL_OUT" select --device cpu
bash analysis/full_budget_rna_apt/run_stage.sh "$FULL_OUT" refit --device cpu

# Rebuild all scores, paired intervals, tables, and the figure without retraining.
bash analysis/full_budget_rna_apt/run_stage.sh "$FULL_OUT" aggregate
```

Each command is resumable and refuses a changed frozen protocol. For parallel
execution, call `run.py` separately for disjoint width/condition/fold/seed
tuples, but never begin a fold/condition refit before its selection JSON exists.
