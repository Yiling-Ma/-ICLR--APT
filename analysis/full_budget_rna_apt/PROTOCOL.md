# Frozen protocol: full-budget matched RNA versus RNA+APT

Frozen before any new outer-test fit or score was produced. The five existing
patient folds have been examined previously, so this is a targeted sensitivity
analysis rather than untouched confirmatory evaluation. All specified runs are
retained irrespective of outcome.

## Scientific estimand

Primary contrasts are RNA+paired APT minus RNA-only at 5,000 and 10,000
training-selected RNA HVGs using every available development cell. The
10,000-HVG experiment additionally compares paired APT with 293 independent
standard-normal noise features, APT shuffled within patient, and APT shuffled
within patient and true lineage. Shuffles retrain the model and are applied
separately to inner train, validation, outer refit, and outer test partitions.

The raw matrix contains 36,294 genes and approximately 5.24e8 nonzero entries.
An all-gene neural fit is omitted from the frozen primary analysis because
materializing fold-specific full-gene matrices and repeating the complete
selection/refit grid has disproportionate shared-filesystem I/O and compute
cost even with the available GPUs. The
optional learning curve and parameter-matched-width control are also deferred;
neither may delay or alter the primary 5k/10k analysis.

## Isolation and data

- 361,792 annotated cells, fixed 27-subtype/5-lineage ontology.
- Frozen five folds from
  `cell_JEPA/outputs/classical_baselines_kfold5/fold_assignment.json`.
- Outer test is fold `f` (8 patients), inner validation is `(f+1)%5`
  (8 patients), inner training is the remaining 24 patients, and outer refit
  uses all 32 development patients.
- The exact same indexed cells are used for every condition at a given stage.
- Test labels never affect preprocessing, feature selection, optimization,
  stopping, model selection, or control construction.

## Preprocessing

RNA reuses `analysis/prepare_fold_rna.R`: per-cell library-size normalization
to 10,000, log1p, mean-binned dispersion HVG selection fitted on the relevant
training partition, and training-only SD scaling without centering. HVGs are
selected independently for inner training and outer refit and gene lists are
saved per fold/stage. APT uses barcode-aligned raw nonnegative counts, log1p,
then a training-only StandardScaler fitted separately from RNA. Concatenation
occurs only after both transformations.

Noise uses 293 independent N(0,1) features generated without labels. For each
shuffle seed, entire standardized APT vectors are permuted only inside the
allowed patient or patient-by-lineage group and never across partitions.

## Matched model and selection

Every condition uses the same generalized unified_v2 MLP: input--256 ReLU--
dropout 0.1--128 ReLU--dropout 0.1, with 27-way fine and 5-way coarse heads.
The loss is unweighted fine CE + 0.5 coarse CE. AdamW, batch size 1,024,
maximum 50 epochs, validation each epoch, patience 8, and earliest tie match
unified_v2. The predefined grid is learning rate {3e-4, 1e-3} by weight decay
{1e-5, 1e-3}; validation fine subject-balanced pooled Macro-F1 selects the
configuration and epoch. Selection seed is 17. Fresh outer refits use seeds
17, 29, and 43 and exactly the selected epoch count. Each condition receives
the same grid and compute budget. Primary same-architecture comparisons differ
only in input features; parameter counts are recorded, and the RNA+noise
control exactly matches RNA+APT input dimensionality and parameter count.
All neural selection and refit jobs run on CUDA GPUs; the implementation rejects
CPU training. CPU use is limited to the existing R-based sparse RNA
normalization/HVG preparation, artifact I/O, and metric aggregation.

## Metrics and uncertainty

Primary evaluation is fixed-ontology subject-balanced pooled Macro-F1: each
patient confusion matrix is normalized to unit mass, matrices are pooled, and
zero-denominator class F1 is zero. Coarse SB-Macro-F1 is secondary. Paired 95%
intervals use 5,000 matched bootstrap draws with RNG seed 20260915, jointly
resampling held-out patients and the same three training-seed slots across
conditions. Intervals are pointwise and conditional on the frozen folds,
finite seed set, and completed validation selections; selection is not repeated.

Final shuffle slots use seeds 73001, 73002, and 73003 paired respectively with
training seeds 17, 29, and 43. The selection run uses the seed-17 realization.
