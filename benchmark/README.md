# APT-Bench Contract

APT-Bench is a benchmark for patient-disjoint learning from a fixed
293-feature single-cell aptamer panel. The empirical cohort is single-site and
contains 40 participants, so the benchmark does not by itself establish
cross-site clinical generalization.

## Primary Tracks

1. Hierarchical cell typing: predict five coarse lineages and 27 fine subtypes
   from one APT vector per cell.
2. Two-axis training scale: vary independent training patients and uniformly
   sampled cells per patient under fixed patient-disjoint outer evaluation.
3. Exploratory representation-transfer and disease audit: aggregate strictly nested
   cell-typing predictions into one six-class prediction per participant, report
   patient-level metrics, and compare against gold-composition and technical
   controls.

Leave-one-disease-out generalization is an exploratory stress test. Open-set
subtype discovery is outside the current contract; all primary hierarchy tasks
use the fixed 5-lineage/27-subtype ontology. Compact-panel selection and inference-time cell subsampling are
archived feature-redundancy diagnostics, not assay-efficiency contributions.

The budget controls have different meanings. A training-scale cell budget
subsamples cells before fitting preprocessing and the classifier. An aptamer budget retrains
the model using only the selected input features. A cell budget subsamples
already-computed held-out predictions before patient aggregation; it does not
reduce the cells used to train the model.

## Fixed Splits

`splits/patient_folds.json` records the immutable five-fold outer partition. For
outer fold `i`, fold `i` is test. The Transformer validation convention uses
fold `(i + 1) % 5`, with the remaining three folds for training. Other protocols
use their documented development-only sampling and inner validation; this rule
must not be imposed on every classical or nested disease run. Preprocessing, model selection, feature ranking,
and budget selection must use development patients only.

## Required Prediction Interface

Cell-typing submissions must provide pseudonymous `cell_id`, `sample_id`, true
and predicted lineage, and true and predicted subtype. Disease submissions must
provide one row per `sample_id` with the true disease and six class scores.
The hierarchy evaluator reports coarse and fine Macro-F1, exact-path accuracy,
root-excluded example-averaged hierarchical F1, and subtype-tree distance. Natural
coarse--fine consistency is supplementary because constrained decoding can alter
it without improving correctness.

## Cross-Track Composition Audit

The disease track includes a bridge from cell typing to patient
characterization. Gold and strictly nested OOF-predicted lineage/subtype
probabilities are aggregated into patient composition vectors and evaluated
with the same fixed disease folds. For each outer fold, downstream training
compositions are made by inner patient-disjoint cross-fitting confined to the 32
development patients; one final cell model trained on those 32 patients
constructs compositions for the eight sealed patients. Reproduce or audit the
pipeline with:

```bash
python analysis/oof_composition_bridge.py audit
python analysis/oof_composition_bridge.py plan
python analysis/oof_composition_bridge.py aggregate
python analysis/oof_composition_bridge_downstream.py
```

The scripts validate every inner and outer assignment against the immutable fold
manifest and write predictions, metrics, permutations, bootstrap results, and QA
records to `outputs/oof_composition_bridge/`. The legacy
`analysis/run_composition_bridge.py` pooled-OOF implementation is retained only
for contamination auditing and must not supply headline values.

## Fair Training-Scale Analysis

`analysis/patient_cell_scaling.py` runs the frozen Logistic Regression and
XGBoost models on `P={8,16,32}` and `C={100,200,400,800,1600}` with 20 matched
subset seeds. `analysis/summarize_fair_patient_cell_scaling.py` extracts exact
fixed-total designs at 3,200, 6,400, and 12,800 cells, matched doublings,
patient-clustered uncertainty, and fold-adjusted descriptive response surfaces.
Run `make fair-scaling-run` for the fits and `make revision` to aggregate the
headline artifacts and rebuild the paper.

External replications reuse this contract on COMBAT RNA/ADT and OneK1K RNA.
OneK1K uses 925 evaluation donors, pool- and donor-disjoint folds,
`C={100,200,400,800}`, and the common 3,200/6,400-cell fixed totals; see
`ONEK1K_SCALING_PROTOCOL.md`. `analysis/summarize_cross_cohort_scaling.py`
combines only completed, QA-passing summaries.

## Release Status

The paper source, analysis scripts, fixed fold manifest, and selected result
artifacts are currently versioned in this repository. A complete benchmark
release still requires the processed APT matrix and labels, feature and hierarchy
metadata, an end-to-end evaluator, an environment lockfile, checksums, an
archival DOI, and explicit code/data licenses. A structural hierarchy export,
selected checksums, and local audit environment snapshot are now provided in
`release_audit/`, but do not complete those release requirements. These items are submission
blockers: until they are available, third parties cannot reproduce the full
benchmark from this repository alone.
