# Dual-Constraint Patient-Context Pilot

## Research Question

Can a learned patient offset remain stable when reference-cell composition
changes, while responding equivariantly to an injected common expression
shift? This is a development hypothesis, not an established contribution.
No RNA distillation, disease supervision, or technical/batch-identification
claim is included.

This implementation replaces the previous hand-designed reliability rule
with a learned offset and two explicit constraints. It is not a trained
version of every earlier hierarchy module: no separate lineage router or
prototype library is used. Fine labels organize training interventions only.

## Architecture

A permutation-invariant context encoder receives 128 unlabeled APT cells.
It computes their mean and a 64-dimensional mean-pooled representation of
the centered cells (two 64-unit GELU layers). Centering makes this shape
representation translation-invariant by construction; it is not an identified
biological composition vector.

A 64-unit MLP predicts a 293-dimensional offset from the context mean and shape
representation. Its final layer starts at zero. A shared-form 128-128 GELU MLP
with dropout 0.1 predicts 27 subtypes. Input slots contain the query cell, a
293-dimensional context slot, and the 64-dimensional shape representation:

- Raw: query, zeros, zeros.
- Learned context: query, context mean, shape representation.
- Correction variants: query minus estimated offset, zeros, shape representation.

All models instantiate the same nominal modules, but active pathways differ.
In particular, the ordinary context model receives the uncorrected mean and
can learn a flexible adjustment; it is not deprived of offset information.
The correction models cannot simply pass the raw context mean through the
shape branch, which is centered. No bound on the true offset or identifiable
biological-versus-technical decomposition is assumed.

## Episodes and Objectives

Reuse the 24-patient training and eight-patient validation partitions of the
previous context pilot. Outer fold 0 remains excluded. Standardize log1p APT
using training cells only. This is retrospective development on a previously
studied cohort, not a new confirmatory holdout.

For each of 1,000 steps, sample four training patients uniformly and 32 query
cells per patient. Support cells exclude these queries. Draw a natural
128-cell support and a second support with different subtype proportions but
the same observed subtype set. Training labels select the second view but
never enter the network. Draw a common per-patient Gaussian shift with
standard deviation 0.25 in standardized APT units.

Every method gets the same episodes and three classification views:

1. Original query and natural support.
2. Original query and composition-resampled support.
3. Shifted query and identically shifted natural support.

The base objective is average query CE across these three views. Thus the
constraint variants do not receive exclusive augmentation or extra labeled
query examples. Patient-uniform episodic sampling gives donor-equal weighting
in expectation. No class reweighting or RNA loss is used.

For offset estimator b, the two additional losses are

    L_comp = mean squared difference between b(C1) and b(C2)
    L_equiv = mean squared difference between b(C1+a)-b(C1) and a.

Each active constraint has coefficient one. All correction models also have
a shared 0.001 mean-square offset anchor. The constraints discourage different
degeneracies: an always-zero offset satisfies composition consistency but not
shift equivariance; a raw context mean is shift-equivariant but changes with
composition. These observations are not a proof of identifiability or of
recovery of real technical offsets.

Six variants: raw, learned context, unconstrained correction, composition-only,
equivariance-only, and dual constraints. Seeds: 1701, 1702, 1703. All use AdamW
learning rate 0.001, weight decay 0.001, gradient clipping at five, and 1,000
fixed steps. No early stopping or validation-based hyperparameter selection.

## Natural Evaluation and Stress Tests

Reserve a random 256-cell reference pool per validation patient before using
any labels. All remaining cells form the fixed query set, identical across
methods and conditions. Natural support is the first 128 cells of this pool.
No query cell appears in either support view.

Report four separate conditions: natural, composition-only, injected shift,
and both perturbations. Composition stress uses validation labels solely to
construct a support-matched reweighted view inside the reserved pool. This is
an oracle-constructed diagnostic, not the deployment protocol. Neither those
labels nor stress scores train the model. The diagnostic shift is a fixed
per-patient Gaussian vector of standard deviation 0.5, applied to query and
support together. It is a controlled robustness probe, not a validated model
of real assay noise.

Report subject-balanced pooled coarse/fine Macro-F1, averaged across three
seed-specific scores. Use 2,000 paired seed/patient bootstrap draws and
pointwise 95% intervals. Record each correction model's composition-induced
offset MSE and known offset-increment MSE. Natural performance, stress
robustness, and mechanism diagnostics must not be conflated.

The fixed reference pool changes the query set versus the previous pilot, and
the training schedule/architecture also changes. Do not directly compare its
scores against the earlier 0.1353 result. Comparisons within this experiment
are paired. There are multiple candidates and conditions; no confirmatory
superiority claim or automatic outer evaluation is permitted.

## Decision Criteria

The dual model must offer value over a flexible learned-context baseline and
over the stronger single-constraint ablation. Lower synthetic offset error
alone is not a successful classifier. Improved natural or shifted accuracy
must not hide a large regression in the other condition. Follow-up requires
support/augmentation-scale sensitivity, independent replication, compute and
capacity checks, and comparison with established contextual adaptation.

No universal positive outcome is required. A null/negative result will be
recorded without promoting the candidate to the paper's core method.

## Artifacts

Run `bash analysis/run_dual_context_pilot.sh` on vllab11. GPUs 6 and 7 execute
the 18 student fits. Results and schedules stay under
`/ssd3/mayiling/apt_agent_runtime/dual_context_pilot_v1` to avoid home quota.
The launcher uses a lock and does not interfere with existing scaling jobs.

`protocol.json` records the fixed design and source-manifest hash. Per-seed
schedules record queries, both support views, and injected training shifts.
`summary.csv`, `seed_scores.csv`, `diagnostics.csv`, and `completion.json` are
small deliverables; large prediction arrays remain on SSD.

Six unit tests cover support matching, context permutation invariance, centered
representation shift invariance, two constraint identities, and finite outputs
for every variant. These tests establish implementation properties, not
scientific efficacy.
