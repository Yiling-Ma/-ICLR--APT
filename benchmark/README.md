# APT-Bench Contract

APT-Bench is a reference benchmark for patient-disjoint learning from a fixed
293-feature single-cell aptamer panel. The empirical cohort is single-site and
contains 40 participants, so the benchmark does not by itself establish
cross-site clinical generalization.

## Primary Tracks

1. Hierarchical cell typing: predict five coarse lineages and 27 fine subtypes
   from one APT vector per cell.
2. Patient-level disease recognition: produce one six-class prediction per
   participant and report patient-level metrics.
3. Measurement efficiency: repeat patient-level recognition under nested
   aptamer selection and controlled cell subsampling.

Disease novelty detection is exploratory and is not part of the primary
benchmark contract.

## Fixed Splits

`splits/patient_folds.json` records the immutable five-fold outer partition. For
outer fold `i`, fold `i` is test, fold `(i + 1) % 5` is validation, and the
remaining folds are training. Preprocessing, model selection, feature ranking,
and budget selection must use development patients only.

## Required Prediction Interface

Cell-typing submissions must provide pseudonymous `cell_id`, `sample_id`, true
and predicted lineage, and true and predicted subtype. Disease submissions must
provide one row per `sample_id` with the true disease and six class scores.

## Release Status

The paper source, analysis scripts, fixed fold manifest, and selected result
artifacts are currently versioned in this repository. A complete benchmark
release still requires the processed APT matrix and labels, feature and hierarchy
metadata, an end-to-end evaluator, an environment lockfile, checksums, an
archival DOI, and explicit code/data licenses. These items are submission
blockers: until they are available, third parties cannot reproduce the full
benchmark from this repository alone.
