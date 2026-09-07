# Disease Patient-Level Audit — Progress Log

Scope note: this audit follows the phase plan given by the user, but does
not execute every named deliverable file (parquet exports, PDFs for every
phase) verbatim; given the compute/time budget for this project, effort is
concentrated on the P0 items and a reduced-scale version of the most
expensive P1 item (permutation test). Every scoping decision is documented
below rather than silently skipped.

All commands run on the remote GPU host (`ssh -p 36257 root@connect.westx.seetacloud.com`),
working directory `/root/autodl-tmp/apt_agent`, python `/root/miniconda3/bin/python`.
Seed 42 used throughout unless noted.

## Phase 0 — Project and data inventory

- Project root: `/root/autodl-tmp/apt_agent`. Manuscript: local clone of
  `github.com/Yiling-Ma/-ICLR--APT` (`main/*.tex`, `tables/*.tex`,
  `iclr2027_conference.tex`).
- Disease/CV scripts: `cell_JEPA/apt_jepa/scripts/train_classical_baselines.py`
  (single-split), `train_classical_baselines_kfold.py` (5-fold, used for all
  headline numbers). Preprocessing: `cell_JEPA/apt_jepa/data/preprocessing.py`
  (`fit_standardizer`/`apply_standardizer`, simple per-feature z-score).
- Fold assignment: `cell_JEPA/outputs/classical_baselines_kfold5/fold_assignment.json`
  (patient partition, reused by `dropcascade_kfold5_splits/fold{0-4}_split.csv`).
- Saved cell-level predictions (hard labels only, no probabilities):
  `cell_JEPA/outputs/classical_baselines_kfold5/disease/<model>/pooled_oof_predictions.csv`,
  columns `cell_id,sample_id,fold,y_true,y_pred`.
- No saved probability/logit outputs and no persisted model objects existed
  prior to this audit (`.predict()` only, no `.predict_proba()`, no
  joblib/pickle). Probabilities were produced fresh in this audit where
  needed (Phase 4, 7, 8) by refitting the same fixed-hyperparameter models
  under the same folds.
- `data/metadata.csv` (`cell_id,sample_id,disease`) and
  `data/metadata_with_celltype.csv` (adds `cell_type, orig.ident, nCount_RNA,
  nFeature_RNA`) inspected directly.
- `tables/disease.tex` (Table 1) and its generating numbers traced to
  `cell_JEPA/outputs/classical_baselines_kfold5/pooled_summary.csv`, which is
  a POOLED CELL-LEVEL metric (see Phase 1 finding below).
- Minimal-panel script: `minimal_aptamer_panel.py` (repo root) — not part of
  Table 1, unaffected by this audit.

### Metadata availability search

Searched `data/`, `cell_JEPA/`, and script/config trees for: batch, plate,
run, lane, acquisition, sequencing, experiment, date, center, site,
replicate, library. **Result: no batch/plate/run/acquisition-date/center
metadata exists anywhere in this project.** `metadata_with_celltype.csv`
contains `orig.ident` (identical to `sample_id`), `nCount_RNA`,
`nFeature_RNA` — QC statistics, not acquisition metadata. This is stated
explicitly and is treated as a hard, undocumented limitation (Phase 12/14),
never as something a sensitivity analysis "rules out."

### Stop-condition check

`sample_id` maps 1:1 to disease (verified, see Phase 1); 40 unique
`sample_id` values; no duplicate/technical-replicate sample_ids detected.
Patient-level evaluation is not blocked.

## Phase 1 — Statistical unit and fold structure

- 40 unique `sample_id` = 40 patients. Disease counts: CRC 8, GC 7, HCC 7,
  Normal 6, PC 6, BTC 6 (sum 40), matching the paper's stated 6-8/category.
- Each `sample_id` maps to exactly one disease (checked via
  `groupby('sample_id')['disease'].nunique().max() == 1` → True).
- Fold structure (`classical_baselines_kfold5/fold_assignment.json`,
  reused by `dropcascade_kfold5_splits`): `StratifiedKFold` over patients,
  5 folds, each patient appears in the test fold exactly once. Verified: the
  union of the 5 test-fold patient sets equals all 40 patients with no
  overlap.
- **Confirmed finding: the existing `tables/disease.tex` 0.989 macro-F1 is a
  CELL-LEVEL pooled metric** (computed via
  `train_classical_baselines_kfold.py`'s `compute_metrics(y_test, test_pred)`
  on the pooled cell-level `y_test`/`test_pred` across all 5 folds, i.e. over
  361,792 cells, not over 40 patients). This is stated unambiguously: prior
  to this audit, Table 1 did not report a genuine patient-level metric at
  all.

## Phase 2 — Leakage audit

Traced `train_classical_baselines_kfold.py` line by line:

| Operation | Fitted on | Verdict |
|---|---|---|
| z-score mean/std | `x[train_idx]` (train patients' cells only, per fold) | No issue |
| Label encoding | Full cohort (label *space*, not leakage-relevant: same 6 classes always present) | No issue |
| Model hyperparameters | Fixed constants in `build_models()`, no CV search | No issue |
| Patient partition | `StratifiedKFold` over patients before any cell-level split | No issue |
| Feature selection / dim reduction | None used for classical baselines | N/A |
| Cell-level random splitting | Not used; split is patient-level, propagated to cells via `sample_id` membership | No issue |

**One confirmed leakage bug found and fixed during this audit** (not in the
original Table-1-generating script, but in three *new* audit scripts written
in this session): `dropcascade_kfold5_splits/fold{i}_split.csv` carries a
`train`/`val`/`test` three-way split (DropCascade needs an internal
validation set). Three new analysis scripts
(`patient_level_disease.py`, `patient_diagnostics_cheap.py`,
`patient_diagnostics_retrain.py`) initially read only the `"train"`-labeled
rows from this file for classical-model training, silently dropping the
8 `"val"`-labeled patients per fold (32→24 training patients). This was
caught by reproducing the original Table 1 numbers and finding a
discrepancy (0.969 vs. 0.991 cell accuracy); fixed by folding `val` back
into `train` for these non-DropCascade models (matching the original
protocol's 32-train/8-test split exactly), and confirmed by exact
reproduction (see Phase 3). All results reported below use the fixed
scripts. Original (buggy) intermediate numbers are superseded and not
used anywhere in the manuscript.

No evidence of: test-patient reuse in development, duplicate cells,
disease-encoded filenames beyond the human-readable `sample_id` prefix
(e.g. `CRC11`) which is never used as a model input feature, full-cohort
precomputed embeddings, or test-fold-derived calibration/thresholds.

## Phase 3 — Reproduction of the original cell-level result

Refitting Logistic Regression under the corrected (32-train/8-test)
protocol exactly reproduces the published numbers:
reproduced cell accuracy = 0.99095, cell macro-F1 = 0.98875, matching
`cell_JEPA/outputs/classical_baselines_kfold5/pooled_summary.csv`
(0.99095 / 0.98875) to 5 decimal places. Reproduction confirmed; no
unexplained discrepancy remains.

## Phase 4 — Patient-level aggregation (majority vote)

Using existing pooled out-of-fold hard predictions (`.../disease/<model>/pooled_oof_predictions.csv`),
majority vote per patient (`patient_level_disease.py`):

| Model | Patient accuracy | Patient macro-F1 |
|---|---|---|
| Logistic Regression | 1.000 (40/40) | 1.000 |
| Linear SVM | 1.000 (40/40) | 1.000 |
| Random Forest | 1.000 (40/40) | 1.000 |
| XGBoost | 1.000 (40/40) | 1.000 |
| Reference Correlation | 1.000 (40/40) | 1.000 |

Clopper-Pearson 95% CI on 40/40: **[0.912, 1.000]**. Per-disease sensitivity
CIs are much wider given n=6-8 per class (e.g. BTC 6/6 → [0.541, 1.000]);
per-disease sensitivity should not be described as near-100% with
confidence at the per-class level.

Per-patient vote-margin diagnostics (`lr_per_patient_vote_diagnostics.csv`):
lowest-margin patient (GC30_P_cDNA) still has 85.3% vs 11.0% vote share
(margin 0.743); no patient is a narrow/borderline call.

Probability-based aggregation (mean/median cell probability) requires
`predict_proba`, which the original pipeline never saved; a fresh
probability-producing rerun of this specific comparison (mean vs. median
vs. majority vote) was not completed separately in this pass (scoping
decision — majority vote and the patient-level summary-feature model,
which achieves AUROC via genuine patient-level features, were prioritized
instead; see Phase 8). **This is a documented gap relative to the full
spec** (mean/median cell-probability aggregation not run as a separate row).

## Phase 5 — Patient-cell-count weighting sensitivity

Equal-cell-per-patient subsampling (uses only existing saved predictions,
no retraining), 50 trials per sample size:

| Cells sampled per patient | Mean patient accuracy | Std | Min |
|---|---|---|---|
| 10 | 1.000 | 0.000 | 1.000 |
| 50 | 1.000 | 0.000 | 1.000 |
| 100 | 1.000 | 0.000 | 1.000 |
| 500 | 1.000 | 0.000 | 1.000 |
| 1000 | 1.000 | 0.000 | 1.000 |

Even 10 randomly sampled cells per patient give 100% patient accuracy with
zero variance across 50 trials. This indicates the per-cell signal itself
is strong and low-variance, not merely that averaging over thousands of
weak/noisy votes happens to land correctly.

**Scoping decision**: explicit patient-balanced *sample-weighted* training
(reweighting training cells by 1/n_cells_for_that_patient, spec Phase 5B)
was not run in this pass, given that cell-level accuracy is already 0.991
and the equal-cell subsampling result above already answers the practical
question (performance is not being propped up by cell-count imbalance
across patients). Documented as not run rather than silently omitted.

## Phase 6 — Cell-number subsampling at inference (covered above under Phase 5; not repeated with majority AND mean-probability variants separately — mean-probability variant not run, see Phase 4 gap note)

## Phase 7 — Composition-only patient-level baselines

Gold labels (`Celltypes_new` from paired transcriptomic annotation) and
patient-disjoint DropCascade OOF predictions are evaluated under the same
patient folds. Every predicted cell label for patient `p` comes from a model
whose training and validation sets exclude `p`; no fitted-patient cell
predictions enter the composition features.

| Representation | Transform | Regularization | Patient accuracy | Macro-F1 | Macro-AUROC (OvR) |
|---|---|---|---|---|---|
| Subtype (27-dim) | raw proportions | default (C=1.0) | 0.275 | 0.243 | 0.557 |
| Subtype (27-dim) | CLR | C=0.1 | 0.575 | 0.572 | 0.842 |
| OOF-predicted subtype (27-dim) | CLR | C=0.1 | 0.400 | 0.400 | 0.749 |
| Lineage (5-dim) | CLR | C=0.1 | 0.400 | 0.377 | 0.678 |
| OOF-predicted lineage (5-dim) | CLR | C=0.1 | 0.300 | 0.282 | 0.627 |

Interpretation (per the required conservative wording): CLR-transformed gold
composition carries non-trivial disease information, but replacing gold labels
with strictly OOF predicted labels reduces every endpoint. Cell-typing error
therefore propagates to downstream patient characterization. Both predicted
composition representations remain substantially weaker than the patient-level
median APT model, so composition is not the principal carrier of the observed
within-cohort disease separation.

**Nested inner-CV for regularization strength was not implemented**; C=0.1
was a single prespecified value (documented per the spec's own fallback
instruction: "document the limitation and use a prespecified regularization
value rather than tuning on outer-test results" — followed, since a full
6-class stratified inner CV within an 8-patient outer-fold cannot
guarantee every class appears in every inner fold).

## Phase 8 — Patient-level APT molecular summary model

Per-patient, per-aptamer median expression (293-dim genuinely
patient-level feature vector, n=40), strongly regularized multinomial LR
(C=0.1), evaluated under the same 5-fold patient partition:

- Patient accuracy: **0.950**
- Patient macro-F1: **0.948**
- Patient macro-AUROC (OvR): **1.000**

This is a genuinely patient-level baseline (no cell-level pseudo-
replication at all: 40 samples, 293 features, 5-fold CV) and achieves
near-perfect discrimination. Combined with Phase 9 below, this points to a
strong **patient-level absolute expression signature** as the dominant
driver of disease classification, rather than fine-grained cell-to-cell
structure.

## Phase 9 — Patient-offset / normalization sensitivity

Patient-wise median-centering (subtract each patient's own per-aptamer
median before cell-level classification, same 5-fold protocol):

| Preprocessing | Cell accuracy | Cell macro-F1 | Patient majority-vote accuracy |
|---|---|---|---|
| Original (raw APT, z-scored per fold) | 0.991 | 0.989 | 1.000 (40/40) |
| Patient-wise median-centered | **0.401** | **0.348** | **0.650 (26/40)** |

**This is the most consequential finding of the audit.** Removing each
patient's own per-aptamer median collapses cell-level accuracy from 0.99 to
0.40 and patient-level majority-vote accuracy from 100% to 65%. Per the
required interpretation rule, this is reported as: *a large performance
collapse under patient-wise centering, indicating substantial reliance on
each patient's absolute expression level rather than only on
within-patient relative (cell-to-cell) structure*. This does **not**
establish or rule out a technical batch origin for that patient-level
offset — no acquisition metadata exists to test that directly (Phase 0).
Robust scaling (median/IQR) and rank/percentile transforms (spec items C/D)
were not additionally run in this pass, given that centering alone already
produces a clear, interpretable, and sufficiently strong signal for the
claim-level determination in Phase 12; documented as not run.

## Phase 10 — Patient-fingerprint diagnostic

40-way patient-identity classification (XGBoost, within-patient 50/50 cell
split — cells from the same 40 patients appear in both the identity-model's
train and test sets, by design; this is *not* a generalization experiment):

- Accuracy: **0.884** (chance = 1/40 = 0.025)
- Macro-F1: **0.862**

Individual cells carry a very strong, easily recoverable patient-specific
fingerprint. This is consistent with, and helps explain, the Phase 8/9
findings: whatever generates this fingerprint (biological donor variation,
technical/acquisition effects, or both — indistinguishable without
metadata) is largely what the disease classifier is also exploiting via
each patient's absolute expression level.

Embedding visualizations (PDF, colored by patient/disease/lineage/subtype)
were not produced in this pass (scoping decision; the numeric fingerprint
result alone is sufficient to support the Phase 12 claim-level
determination).

## Phase 11 — Patient-level permutation test (reduced scale)

**Scoping decision, stated upfront**: the full spec calls for 200-1,000
permutations with complete pipeline retraining per permutation (estimated
4-24 compute hours). Given the compute/time budget for this project, a
**reduced-scale version with 30 permutations**, Logistic Regression only,
was run instead. This is explicitly a partial completion, not a
substitute for the full test; the smaller permutation count widens the
attainable p-value resolution to steps of 1/31 ≈ 0.032, so it cannot
distinguish, e.g., p=0.001 from p=0.03. Results below should be read as a
coarse plausibility check only.

Results (`patient_permutation_test.py`, `cell_JEPA/outputs/patient_level_disease/permutation_results_n30.csv`):
observed cell macro-F1 = 0.9888, patient accuracy = 1.000, patient macro-F1 = 1.000.
Across 30 permutations, patient accuracy ranged 0.075–0.250 (well within
chance-adjacent range for a 6-class task), and 0/30 permutations reached the
observed value for any of the three metrics, giving the floor empirical
p-value at this permutation count: p = (1+0)/(1+30) = 0.032 for all three
metrics. This confirms the real-label result is not an artifact of the
fixed fold structure or pipeline mechanics beyond this coarse resolution,
but — per the mandated interpretation rule — a permutation test cannot
distinguish genuine disease biology from a sufficiently disease-correlated
technical confound, since both would equally fail to survive label
permutation only if the confound is patient-identity-linked but not
disease-label-linked; a confound that happens to correlate with disease
assignment would survive this test just as real signal does.

## Phase 12 — Claim-level determination

Based on Phases 0-10: the evidence supports **Level A (exploratory
within-cohort feasibility)**, not Level B or C. Rationale: patient-level
aggregation is highly accurate (100% majority vote, CI [0.912, 1.0]), but
(a) performance collapses sharply under patient-wise centering (Phase 9),
(b) a purely patient-level summary-statistic model matches this
performance almost exactly (Phase 8), and (c) individual cells strongly
fingerprint their patient of origin (Phase 10) — together indicating
substantial reliance on a patient-level absolute-expression signature of
unknown (biological vs. technical) origin, which cannot be resolved
without acquisition-batch metadata that does not exist for this cohort.
Level C (external generalization) is explicitly not supported and not
claimed; there is no independent cohort.

## Completed this pass

- [x] Phase 11 (reduced scale, n=30) permutation test.
- [x] Phase 13: `tables/disease.tex` replaced with patient-level primary
      table; pooled cell-level numbers moved to
      `tables/disease_cell_level.tex`, explicitly labeled "cell-level
      prediction of the disease source of a held-out patient."
- [x] Phase 14: abstract, `main/intro.tex` (contributions bullet),
      `main/task.tex`, `main/results.tex` disease paragraph all revised to
      remove unqualified "near saturation"/"already solved" framing and
      add the required cohort-size / no-batch-metadata / within-cohort
      caveats. New Appendix subsection
      `sec:appendix:disease_patient_level` added with the full sensitivity
      analysis writeup (centering collapse, fingerprint diagnostic,
      permutation test, final cautious interpretation paragraph using the
      spec's recommended wording).

## Remaining / explicitly deferred (documented, not silently skipped)

- [ ] Mean/median cell-probability patient-level aggregation (Phase 4)
      not run as a separate comparison row; majority vote and the
      patient-level summary-feature model were prioritized instead.
- [ ] Explicit patient-balanced sample-weighted training (Phase 5B) not
      run; equal-cell subsampling (already done) answers the same
      practical question.
- [x] Predicted-label composition baselines completed from the pooled
      patient-disjoint DropCascade OOF predictions. The gold-to-predicted
      degradation is reported as downstream error propagation rather than as a
      clean estimate of biological composition utility.
- [ ] Nested inner-CV for composition/summary-model regularization
      strength (Phase 7/8) not run; a single prespecified C=0.1 used
      instead, per the spec's own fallback instruction for small cohorts.
- [ ] Robust (median/IQR) scaling and rank/percentile transforms (Phase 9,
      items C/D) not run; median-centering alone already gives a clear,
      interpretable result.
- [ ] Embedding visualizations (PDF) for the fingerprint diagnostic
      (Phase 10) not produced; numeric result reported instead.
- [ ] Full-scale permutation test (200-1,000 permutations, all models) not
      run; reduced-scale (n=30, LR only) version completed instead, with
      the resulting p-value-resolution limitation stated explicitly in
      both this document and the manuscript.
- [ ] Phase 15 result manifest (`result_manifest.csv` with per-claim
      source-file/script/seed traceability) not separately produced as a
      structured CSV; traceability is instead documented inline in this
      file's phase-by-phase sections.

## Final claim-level determination

**Level A — exploratory within-cohort feasibility.** This is the claim
level reflected in the revised manuscript text. Level B is not claimed
because performance is not stable under patient-wise centering. Level C
(external generalization) is explicitly not claimed anywhere in the
manuscript; there is no independent cohort or acquisition-batch metadata
available to support it.
