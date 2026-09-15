# RPH-Distill implementation plan and reuse audit

## Located implementation to reuse

- Frozen patient folds, 361,792-cell alignment, 5/27 ontology, fixed-denominator
  F1, and patient-balanced confusion pooling: `analysis/patient_cell_scaling.py`.
- Exact APT input and cache manifest audit: `analysis/validate_paired_information.py`
  and `analysis/hierarchy_bottleneck_mitigation/prepare_log1p_cache.py`.
- Training-only RNA library normalization, log1p and mean-binned dispersion HVGs:
  `analysis/prepare_fold_rna.R`; paired cell indices are shared with
  `analysis/full_budget_rna_apt/prepare.py`.
- Unified 256/128 MLP, AdamW selection/refit, seeds and saved probabilities:
  `analysis/unified_v2/run.py`.
- APT student, hierarchy mapping, oracle decoding and error diagnostics:
  `analysis/hierarchy_bottleneck_mitigation/hbm_core.py`.
- Matched patient plus seed-slot bootstrap:
  `analysis/unified_v2/summarize.py` and HBM aggregation code.

## New files

- `PROTOCOL.md`: frozen scientific and statistical contract.
- `common.py`: RNA teacher, routing/within KD, alignment, paired-target shuffle,
  data and leakage assertions.
- `run.py`: teacher/student selection and refit with resumable artifacts.
- `aggregate.py`: fail-closed metrics, paired intervals, tables and figures.
- `diagnostics.py`: quantitative representation probes for Plain and Full RPH.
- `test_rph.py`: synthetic loss, gradient, inference and permutation tests.
- `run_stage.sh`, `README.md`: exact multi-GPU commands and rebuild instructions.

## Execution order

1. Freeze/hash protocol before any new outer result.
2. Verify shared fold-specific 5k RNA files and APT cache manifests.
3. Refit the exact Plain APT MLP baseline with seeds 17/29/43; aggregate and
   require reproduction within the prespecified tolerance before continuing.
4. Select/refit fold-specific RNA teachers; freeze each before student use.
5. Run routing KD, within-lineage KD and cosine alignment separately.
6. Run routing+alignment, routing+within, Full RPH, label-weight control, and
   Full RPH with patient-by-lineage-shuffled teacher targets.
7. Fail-closed aggregation, paired bootstrap, representation diagnostics,
   tables, figures, scientific report and manuscript update.

## Ambiguities resolved before execution

- RNA teacher uses 5,000 training-selected HVGs and all development cells. This
  is fixed independently of the separate 10k full-budget sensitivity outcome.
- Teacher and student latent dimensions are both 128, so alignment uses no
  projection. If an implementation projection is later required, it is a new
  protocol rather than an undocumented change.
- Optional InfoNCE, hierarchical contrastive learning, confidence weighting,
  global/expert heads and random-vector alignment are not in the primary run.
- The shuffled-teacher and stronger-lineage controls reuse the paired Full/base
  selections respectively; they receive no separate outcome-driven tuning.
