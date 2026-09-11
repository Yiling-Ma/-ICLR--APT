# APT Patient-Context Development Pilot

## Status and Scope

This is a new development experiment, not an established method contribution.
It tests whether an APT-only classifier benefits from composition-aware,
lineage-conditioned patient references beyond simpler context features. No RNA
distillation or disease loss is included, so the context hypothesis can be
tested separately. Reliability is a fixed heuristic, not learned or calibrated
uncertainty. A positive pilot would require stronger follow-up, not immediate
promotion to the main paper.

Use the existing fold-0 inner split: 24 training patients, eight validation
patients. The eight outer-fold-0 patients are excluded entirely. The cohort and
inner-validation patients have been studied before; this is retrospective
development, not independent confirmation. No outer runs start automatically.

## Reference Construction

1. Reuse aligned log1p APT and the paired experiment's sampled training cells
   (up to 1,000 cells per patient). The ontology is fixed at 27 subtypes and five
   lineages. All context features are constructed without target-patient labels.
2. Within the 24 training patients, cross-fit a five-way lineage MLP by existing
   eight-patient blocks. Each source partition contains 16 patients. Fit APT
   scaling and the lineage classifier on source patients only.
3. For each source patient and lineage, calculate an APT mean weighted by the
   classifier's soft lineage scores. Average those means equally across source
   patients to obtain a reference. Source predictions are in-sample; target
   patients are excluded from all upstream fitting and references.
4. Select up to 128 context cells per target patient using a deterministic
   patient-specific seed, identical across methods and model seeds. Context is
   drawn only from that partition's existing cells. If a query belongs to the
   context subset, remove its contribution, leaving 127 peers. Otherwise use
   all 128. A singleton has zero correction.
5. Form each query's leave-one-out global mean, standard deviation, and five
   soft lineage centroids. For lineage m let a_m be the sum of its context
   scores q_im, and let c_m be the sum of squared scores divided by a_m.
   Define rho_m = a_m/(a_m+10) * c_m. This shrinks corrections supported by few
   cells or diffuse routing scores.
6. The uniform shift is the sum of available centroid-minus-reference vectors
   divided by five. The reliable shift weights these vectors by rho_m, again
   dividing by five (not by the sum of reliability weights). This allows the
   total correction to shrink toward zero when evidence is weak.
7. For inner validation, refit the upstream classifier and references on all
   24 training patients. Use only the validation patient's unlabeled APT for
   context. Neither validation labels nor RNA define its reference.

The additive correction is a hypothesis. Within-lineage subtype composition
can still shift centroids, and patient offsets may contain biology as well as
technical effects. This experiment does not identify technical batch effects.

## Matched Controls

| Input scheme | Features |
| --- | --- |
| raw | Original APT only |
| centered | APT minus leave-one-out patient mean |
| global_context | Raw, centered APT, patient mean, patient standard deviation |
| soft_context | Raw APT, mean, standard deviation, five soft lineage centroids |
| hierarchy_uniform | Raw APT, corrected APT, uniform shift, standard deviation |
| hierarchy_reliable | Raw APT, corrected APT, reliable shift, standard deviation |

Each scheme uses CE and Balanced Softmax, each with seeds 1701/1702/1703: 36
student fits in total. Balanced Softmax uses donor-weighted training label
counts with one pseudo-count per class, adding log prior to training logits
only. It is an existing baseline, not a novel component. No auxiliary coarse
loss is used here; this differs from the earlier distillation experiment.
Coarse predictions are obtained by summing fine probabilities by lineage.

All students use a 128-128 GELU MLP, dropout 0.1, AdamW learning rate 0.001,
weight decay 0.001, batch 512, and 60 fixed epochs. Cells receive donor-equal
weights; no sampler changes or early-stopping choices are made. Input scaling
is fitted on student-training features only. Each input has eight 293-feature
slots, with unused slots padded by zero. Nominal network size and training
schedule match, but effective active input capacity and available information
do not. The soft-context control shares the upstream lineage classifier and
has access to all five centroids; superiority to raw alone is insufficient.

## Evaluation and Interpretation

Report mean seed-specific subject-balanced pooled Macro-F1 and paired
differences within each loss against raw, global context, soft context, and
uniform hierarchy correction. Each of 2,000 bootstrap replicates resamples
the three seeds and eight validation patients, identically across methods.
Intervals are pointwise and conditional on the fixed context sample and
upstream fits. Multiple candidates are screened; no confirmatory superiority
or equivalence claim is permitted.

A worthwhile candidate must improve beyond the stronger simple context and
long-tail controls, not only an unbalanced raw model. The reliable component
must justify itself against uniform correction. Subsequent work would need
context-budget and upstream-seed sensitivity, a flexible learned-context
baseline, matched compute/capacity checks, and frozen external validation.

The five unit tests cover self-exclusion, a known additive-shift synthetic
case, singleton fallback, input dimensions, and equal-patient references.
The synthetic unit test is not a simulation study or evidence of real-data
effect recovery.

## Reproduction

```bash
bash analysis/run_patient_context_pilot.sh
```

Host: vllab11, GPUs 6 and 7. Project: `/home/mayiling/projs/apt_agent`.
Outputs: `/ssd3/mayiling/apt_agent_runtime/patient_context_pilot_v1`.
Large inputs and predictions remain on SSD, not in the home quota or Git.
`protocol.json` records the frozen configuration, `upstream_splits.json`
records calibration partitions, and `context_manifest.json` records selected
context cells. `summary.csv` and `completion.json` appear after all 36 fits.
The launcher uses a lock and does not stop other experiments.

Background: [Balanced Softmax](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html)
is an existing long-tail baseline; [ADTnorm](https://pmc.ncbi.nlm.nih.gov/articles/PMC11261982/)
already studies protein normalization under variable cell compositions.
Patient-context normalization alone is therefore not a novelty claim.

## Completed Development Results

All 36 student fits completed on vllab11. All five unit tests and the independent
artifact audit passed. The audit verified actual patient/context membership,
every stored global context mean's self-exclusion, upstream patient isolation,
and all prediction-derived confusion matrices.

| Input scheme | CE fine | CE coarse | Balanced fine | Balanced coarse |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.1200 | 0.3276 | 0.1047 | 0.3103 |
| Centered | 0.1194 | 0.3179 | 0.1165 | 0.3056 |
| Global context | 0.1314 | 0.3438 | 0.1225 | 0.3224 |
| Soft lineage context | 0.1199 | 0.3435 | 0.1117 | 0.3054 |
| Uniform hierarchy correction | 0.1315 | 0.3437 | 0.1219 | 0.3195 |
| Reliable hierarchy correction | 0.1353 | 0.3415 | 0.1173 | 0.3243 |

For reliable hierarchy correction under CE, paired fine Macro-F1 differences
and pointwise seed/patient-bootstrap 95% intervals are:

- Versus raw: +0.0153 [0.0026, 0.0211].
- Versus global context: +0.0039 [-0.0051, 0.0116].
- Versus soft lineage context: +0.0155 [-0.0053, 0.0254].
- Versus uniform hierarchy correction: +0.0038 [-0.0036, 0.0089].

The coarse difference versus raw is +0.0139 [0.0033, 0.0228], but versus global
context it is -0.0024 [-0.0122, 0.0056]. Balanced Softmax does not improve these
schemes under this fixed recipe; this does not rule out other long-tail tuning.

Interpretation: the pilot motivates further patient-context experiments, but
does not establish the new hierarchy/reliability components' incremental value
over simpler context features. Selection among twelve configurations and prior
use of these patients preclude a confirmatory claim. Do not replace the main
paper method or headline performance with this pilot result. No outer-fold
evaluation or independent validation has been run for this candidate.

Next decision: freeze a narrower raw/global-context/uniform/reliable CE
comparison before wider evaluation, with upstream/context-seed sensitivity
and explicit compute/capacity checks. Preserve the ordinary-context baseline
even if it erases the proposed component's apparent advantage.

Small results: `outputs/patient_context_pilot_v1/summary.csv`, `seed_scores.csv`,
`completion.json`, and `audit.json`. Audit code:
`analysis/audit_patient_context_pilot.py`. All large caches stay on SSD.
