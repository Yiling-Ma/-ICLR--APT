# Frozen hierarchy-signal follow-up, 2026-09-14

This is an exploratory follow-up on previously inspected fixed outer folds.
Do not change seeds, shuffle counts or comparisons after inspecting results.
No clinical/data-access TODO is resolved by these experiments.

## Existing-prediction analysis

Use all unified-v2 OOF cells for MLP, Flat, corrected HCE and disease-free
Cascade, training seeds 17/29/43. Correct/cross-parent/within-parent errors
use the parent of the fine argmax, never the coarse head. Compute proportions
inside each patient, then average eligible patients and seeds. Conditional
fractions exclude patients with zero denominator and report their number.
Stratify fine cross-parent errors by coarse-head correctness as a diagnostic.
Check fixed-score oracle invariants. Save ordinary/oracle patient-balanced
confusions, row-normalized visualizations and all subtype metrics; rank
directed oracle confusions by seed-mean patient-balanced off-diagonal mass.
These are not additive decompositions of Macro-F1 or mechanism identification.

## APT-only conditional shuffle

Reuse the completed real-input v2 MLP, not the older provided-expression MLP.
Three shuffle realizations: 101, 211, 307. Within each separate inner,
validation, refit and test partition, permute complete 293-dimensional APT
vectors only inside patient x true-lineage groups. Identity permutations and
singletons remain; do not force derangements. Save full source-row indices.
The group-preserving permutation retains all fit statistics. Training and
validation use true lineage only to construct this privileged control.

Independently select each shuffled recipe per fold with selection seed 17,
four lr/weight-decay candidates, validation ordinary fine SB-F1, max 50 epochs,
patience 8, earliest strict optimum; refit on all 32 development patients for
selected epochs with seeds 17/29/43. The real control already used this rule.
Use v2 raw counts/log1p/fit-only standardization, no caps or weighting. No
test-only corruption masquerades as a retrained shuffled condition.

Primary contrast: real oracle minus mean shuffled oracle, overall 27-class
SB-F1. Five lineage contrasts are exploratory, normalized within eligible
patient-lineage scopes. Report a training-development conditional-proportion
expected-confusion prior with the same privileged lineage. No new chance score.
Do not interpret this contrast as a molecular mechanism or deployable result.

## Conditional MLP

Reuse the v2 256/128 ReLU dropout-.1 encoder and identical 27-leaf/5-parent
heads. Normalize leaf logits separately within each lineage; loss is true
lineage conditional subtype NLL + .5 parent CE. At deployment use
q(parent|x)*r(subtype|parent,x). This changes no effective parameter count.
No contrastive term, disease supervision, added input or new sampling.
Same four-candidate grid, selection seed, patience, refit seeds and all cells;
select by ordinary deployable fine SB-F1. Oracle is diagnostic only.
Primary contrast: deployable conditional MLP minus real ordinary v2 MLP.
Also report ordinary MLP D2, both oracles and paired oracle/deployment gaps.

## Uncertainty and stopping

5,000 percentile paired patient/seed/shuffle bootstrap replicates, RNG seed
20260916. Draw 40 patients and three training-seed slots with replacement;
for each selected seed draw three shuffle slots with replacement and average
the separately scored realizations. All conditions share patient/seed draws.
For lineage scopes, draw only its eligible patients, retaining all child labels.
Compute seed-specific nonlinear F1 before averaging. Zero-denominator F1 is 0.
Never pool predictions over seeds; folds/cells are not biological replicates.
Intervals condition on selected configurations and existing folds; pointwise,
exploratory, not equivalence or multiple-testing-adjusted evidence.

Only if MLP's overall real-minus-shuffle oracle 95% lower bound is strictly
positive, repeat the same three-shuffle design with Flat. This gate does not
apply to existing-prediction reanalysis. Do not add seeds to seek significance.
The optional fuller-budget RNA connection is deferred: do not conflate old
capped RNA experiments with this APT-only protocol or start new RNA fits.
