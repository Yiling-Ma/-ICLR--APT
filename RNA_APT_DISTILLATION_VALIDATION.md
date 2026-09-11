# RNA-to-APT Distillation: Frozen Validation Protocol

## Purpose and Claim Boundary

This experiment tests whether paired RNA supervision can improve an APT-only
student on held-out patients. It is an experimental candidate, not an established
algorithmic contribution. Neither improved performance nor novelty is assumed.
The present implementation tests cross-patient target-risk-guided hierarchical
shrinkage. It does not yet implement student-utility-based selection, TabM, C2KD,
or BioKD. See `METHOD_RESEARCH_AND_EXPERIMENT_PLAN_ZH.md` for related work.

At deployment the student uses APT alone. RNA, fine labels, and lineage labels
are available during training only; no disease labels enter the training loss.
The RNA-derived annotation creates an asymmetric teacher advantage, so successful
distillation would not independently validate the biological annotation.

## Data and Splits

- Use the existing aligned APT/RNA cell manifest and fixed five patient-disjoint
  folds. Keep the fixed 27-subtype/5-lineage ontology.
- Reuse the paired-information experiment's training subsamples (at most 1,000
  cells per training patient) and held-out evaluation indices.
- Development pilot: outer fold 0 remains untouched; train on 24 patients and
  evaluate on the eight inner-validation patients.
- Full validation: 32 training patients and eight test patients per outer fold;
  pool out-of-fold confusion matrices for all 40 patients.
- Within each student training partition, hold out each patient-fold block G
  while fitting both the teacher and projector on the remaining patients.
- Select 2,000 RNA HVGs separately within each teacher-source partition. Fit
  RNA and APT standardizers only on the corresponding source cells. RNA uses
  per-cell library normalization to 10,000 followed by log1p; APT uses log1p.
- No teacher, projector, HVG selection, or standardizer sees patients receiving
  its cached training targets. The teacher does see the projector's source
  training cells; this is not an additional inner cross-fit within that source.

All six candidates and their hyperparameters are fixed before full validation.
The full run proceeds regardless of the pilot's effect direction. This is a
retrospective analysis on an already studied cohort, not a new prospective test.

## Models and Targets

All models use an MLP with two 128-unit GELU hidden layers and dropout 0.1. The
teacher consumes RNA, while the projector and final student consume APT.
Optimization uses AdamW, learning rate 0.001, weight decay 0.001, batch size 512,
and 60 fixed epochs without checkpoint selection. Cells are weighted inversely
by their patient's training cell count; no class reweighting is applied.

The RNA teacher produces temperature-2 fine-class probabilities q. An APT
projector is trained on source patients to approximate the teacher distribution
using soft cross-entropy, then predicts r for the excluded target patients.
For a subtype k with parent m, the shrinkage target is

    target(k) = q(m) * [alpha(m) q(k | m) + (1-alpha(m)) r(k | m)].

This preserves teacher lineage mass and changes only within-lineage allocation.
It does not recover information absent from APT. Its proposed benefit is
finite-sample regularization of privileged targets, not a new population optimum
for forward-KL distillation.

| Candidate | Training target / control |
| --- | --- |
| supervised | Fine CE plus 0.5 aggregated-lineage CE; no distillation |
| kd | Ordinary teacher probabilities |
| confidence | Ordinary targets weighted by normalized teacher entropy confidence |
| projection | Projected conditional probabilities; teacher lineage mass retained |
| shrink | Fixed alpha = 0.5 in every lineage |
| lineage | Training-only cross-fitted target-risk-selected alpha per lineage |

The last five candidates add 0.5 times temperature-squared KL to the supervised
loss. Confidence weights have donor-weighted mean one to match average KD scale.
The `projection` condition is a target ablation, not direct deployment of the
projector. A direct-projector baseline remains necessary for a stronger claim.

For `lineage`, alpha is selected from {0, 0.25, 0.5, 0.75, 1} using donor-equal
conditional NLL on student-training labels and cross-fitted targets. A penalty
of 0.05 times squared distance to the global alpha shrinks lineage estimates;
lineages with fewer than four contributing donors use the global alpha. This
is a target-risk proxy, not a measured causal effect on student performance.
If it chooses alpha = 1 everywhere, the candidate reduces to ordinary KD.

## Evaluation and Decision Rule

Three matched student seeds (1701, 1702, 1703) use the same initialization and
minibatch schedule across candidates. Teacher/projector seeds are fixed per
source partition. Coarse predictions sum subtype probabilities by parent;
they are not an independent coarse head used in some earlier paper baselines.

Report subject-balanced pooled Macro-F1: normalize each patient's confusion
matrix by its total cell count, average those matrices, then compute Macro-F1
over the fixed ontology. Average the three seed-specific scores, rather than
computing F1 from averaged predictions. Report paired differences against both
supervised learning and ordinary KD.

The 2,000 bootstrap draws jointly resample student seeds and test patients,
using identical draws for every candidate. Intervals are pointwise 95% intervals
conditional on the fixed folds and upstream fits, not full-pipeline uncertainty.
The bootstrap positive fraction is not a posterior probability or a p-value.
There is no multiplicity-adjusted superiority claim in this first screen.

Promotion to a paper method requires more than beating supervised learning:

1. The proposed component must improve on ordinary KD, not just demonstrate
   that RNA is a useful teacher.
2. An adaptive candidate must justify its complexity relative to fixed mixing
   and confidence weighting, with patient- and seed-level stability examined.
3. Compare direct projector deployment and relevant cross-modal KD methods,
   and replicate on independent paired data before claiming a general method.
4. Freeze a narrower confirmatory comparison before additional evaluation;
   do not choose a winner from these six pointwise comparisons and call it
   prespecified superiority.

Negative or unresolved outcomes will be retained. Training completion is not
evidence that the method works. These reduced-budget probes are not directly
comparable with the paper's full-budget Transformer scores.

## Execution and Artifacts

Run from `/home/mayiling/projs/apt_agent` on vllab11:

```bash
bash analysis/run_rna_apt_distillation.sh
```

The launcher runs unit tests, pilot preparation/training/aggregation, and then
five-fold validation. It uses GPU 6 for the pilot and GPUs 6 and 7 for validation.
Other scaling jobs are not stopped. Outputs and RNA intermediates go to
`/ssd3/mayiling/apt_agent_runtime/rna_apt_distillation_v1`, not the home quota.
The launcher uses a lock and protocol-hash-checked caches for safe resumption.

- `batch.log`: stage-level preparation and aggregation.
- `pilot_0.log`, `validation_0.log`, `validation_1.log`: training progress.
- `pilot/protocol.json`, `validation/protocol.json`: exact manifests and hashes.
- Each stage's `summary.csv`: scores and paired differences.
- Each stage's `completion.json`: structural completion checks and limitations.
- Per-fit NPZ files: cell indices, probabilities, patient confusion matrices,
  selected alphas, and protocol hashes. Keep these on SSD rather than Git.

Implementation: `analysis/rna_apt_distillation.py`,
`analysis/prepare_distillation_rna.R`, `analysis/run_rna_apt_distillation.sh`.
Tests: `analysis/test_rna_apt_distillation.py`.

## Development Pilot Results

The eight-patient inner-validation pilot completed all 18 student fits and
passed the artifact audit. These are development results, not main-paper
performance estimates or a basis for choosing the full-validation candidate.

| Candidate | Fine Macro-F1 | Coarse Macro-F1 |
| --- | ---: | ---: |
| Supervised | 0.1195 | 0.3275 |
| Ordinary KD | 0.1221 | 0.3289 |
| Confidence-weighted KD | 0.1215 | 0.3300 |
| Projected conditional target | 0.1232 | 0.3339 |
| Fixed shrinkage | 0.1253 | 0.3339 |
| Lineage-risk shrinkage | 0.1228 | 0.3316 |

Fixed shrinkage minus ordinary KD is 0.0033 fine Macro-F1, with a pointwise
joint-bootstrap 95% interval of [-0.0009, 0.0051]. Lineage-risk shrinkage minus
ordinary KD is 0.0007 [-0.0014, 0.0024]. Neither establishes superiority.
The adaptive coefficients are [0.75, 0.75, 0.50, 1.00, 0.75] in encoded lineage
order. The pilot therefore does not justify adaptive complexity over fixed
mixing. Full five-fold validation retains all six candidates unchanged.

Machine-readable results are in `outputs/rna_apt_distillation_v1/pilot/`.
`analysis/audit_rna_apt_distillation.py` independently checks patient isolation,
target coverage, normalized predictions, and confusion-matrix reconstruction.
