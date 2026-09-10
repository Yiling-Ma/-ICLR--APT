# Remaining experiment execution

Launched on vllab11 on 2026-09-10 UTC through vllab7. Home storage is shared;
completed results remain in place. No old results were deleted.

## Sequential scaling

Root: `/home/mayiling/projs/apt_agent/sequential_audit_20260909`.
Launcher: `/home/mayiling/projs/apt_agent/analysis/resume_sequential_vllab11.sh`.
Log: `outputs/sequential_scaling/batch_vllab11.log` under the scaling root.
Four disjoint job-index shards use physical GPUs 0, 1, 2, 3 separately. Each
process sees one GPU as cuda:0. Existing JSON/NPZ pairs with matching protocol
are skipped. Changing the scheduling shard count does not change experiment
seeds, training manifests, model settings, or evaluation folds.

APT, COMBAT RNA, and OneK1K each require 2,400 fits. The original APT failure
left 1,770 NPZ outputs; completion requires both artifacts for every expected
job, successful aggregation, and paired-manifest QA, not merely absence of
running processes. New logs have `vllab11` and a UTC launch timestamp in their
names. Do not mistake the preserved old `batch.log` for current status.

## Paired RNA/APT validation

Root: `/home/mayiling/projs/apt_agent`.
Launcher: `analysis/run_paired_validation.sh`.
Log: `outputs/paired_validation_vllab11.log`.
Results: `outputs/paired_information_validation`.

This new post-hoc protocol uses five fixed patient-disjoint outer folds,
one patient-disjoint inner validation fold, and at most 1,000 training cells
per patient. RNA normalization is per cell; selection of 2,000 HVGs and
feature scaling are fitted using the respective inner/outer training cells
only. Operational RNA-derived annotations are not independent ground truth.

Inputs are RNA, APT, RNA+APT, RNA+293 independent noise features, and RNA+APT
with cell pairing shuffled within each patient separately in training and
evaluation. Both an SGD logistic classifier and a 64-unit MLP receive two
regularization candidates per input, selected by inner subject-balanced
pooled macro-F1. Models use 40 training epochs; optimization limitations must
be reported rather than presenting this as unrestricted model performance.
These sklearn models run on CPU with four numerical-library threads; GPU
availability does not automatically accelerate them.

There are 100 outer fits and 200 inner fits. Completion requires all 100
outer confusion-matrix artifacts and successful paired aggregation. Bootstrap
intervals are pointwise and conditional on fitted models, not training-seed
uncertainty. The one fixed noise/shuffle realization is a diagnostic control,
not a formal permutation test. Do not claim a validated biological mechanism
or external disease generalization from these comparisons.

The composition-recovery experiment is already complete and in the paper.
Acquisition-aware disease validation remains TODO pending real metadata or
an independent APT cohort.
