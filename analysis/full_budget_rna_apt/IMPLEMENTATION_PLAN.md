# Implementation plan and reuse audit

## Located implementation

- Capped paired probe: `analysis/validate_paired_information.py`, repeated at
  2k/5k by `analysis/run_modality_repeats.py`; it caps each training patient at
  1,000 cells, uses a 64-unit sklearn MLP, patient-by-class weights, and saves
  the committed summaries under `outputs/remaining_modality_hvg{2000,5000}_v1/`.
- Conditional pairing controls: `analysis/cell_identity_questions.py` supplies
  whole-vector within-patient and within-patient-by-lineage permutations.
- Frozen folds, ontology, cell alignment, SB pooling, and fixed-denominator F1:
  `analysis/patient_cell_scaling.py` (`load_data`, `load_folds`,
  `fit_label_encoders`, `patient_balanced_matrix`, `f1_from_confusion`).
- RNA preprocessing: `analysis/prepare_fold_rna.R` (10k library normalization,
  log1p, training-only mean-binned dispersion HVGs).
- Strong neural recipe: `analysis/unified_v2/run.py` and
  `analysis/unified_v2/PROTOCOL.md` (256/128 joint-head MLP, AdamW grid,
  patience, refit, seeds).
- Matched uncertainty: `analysis/unified_v2/summarize.py` (paired patient and
  seed-slot bootstrap).

## Identical between RNA and RNA+APT

Patient folds; indexed cells; labels; inner/validation/refit/test partitions;
RNA normalization, HVG list, and RNA scaler; backbone and heads; loss; sample
weighting (none); optimizer and grid; batch size; maximum epochs; patience;
selection metric; selection seed; final seeds; refit procedure; test patients;
and paired bootstrap draws.

## Unavoidable mismatch

RNA+APT has 293 additional input columns and therefore 75,008 additional first-
layer parameters. RNA+noise has the identical input dimension and trainable
parameter count and is the required dimensionality control. The new full-budget
unified_v2-style MLP is intentionally stronger than the historical capped
64-unit sklearn MLP; historical and new rows are labeled as different probes.

## Execution order

1. Write and hash the frozen protocol; audit raw IDs, folds, counts, ontology,
   and feature dimensions without fitting or scoring outer tests.
2. Prepare training-only 5k and 10k HVGs for inner and outer stages.
3. Complete 5k RNA/RNA+APT selection and all three refits.
4. Complete 10k RNA/RNA+APT, then the predeclared noise and two shuffle controls.
5. Fail-closed artifact audit, paired aggregation, tables, and figure.
6. Interpret using the predeclared cases; do not tune after viewing outer scores.
