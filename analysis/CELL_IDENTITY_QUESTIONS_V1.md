# One benchmark, two empirical questions

The questions, not predetermined findings, are:
1. Which hierarchical identities are recoverable from APT in unseen patients?
2. Does paired APT add typing information where an RNA-only reference is uncertain?

This is an exploratory extension frozen before these new controls are scored.
Keep the existing ontology, patient folds, source cell subsets and old runs.
Do not expand to foundation models. Existing full-cell MLP and frozen neural
oracle remain separate from the matched-input 1000-cell-per-training-patient probe.

## Targeted matched-budget control

Use the existing SGD/MLP recipes, inner alpha grids, 40 epochs, three training
seeds and RNA train-only 2000-HVG matrices. Reconstruct RNA/APT probabilities by
refitting at the saved inner-selected alpha, requiring identical original hard
predictions. No new choice uses outer-test scores. Add a conditional MAP prior
from actual training-subset subtype counts within each lineage, with fixed
ontology-order tie breaking (including an unseen lineage). Also report RNA and
APT scores masked to the same true lineage. This uses privileged test labels.
Alongside MAP, use the conditional training proportions as an analytic expected
confusion-matrix reference (uniform fallback for an unseen training lineage).
Its Macro-F1 is computed from expected counts, not claimed to be the exact
finite-sample expectation of randomized Macro-F1. Report both prior controls.

Compare RNA, paired RNA+APT, RNA+noise, patient-shuffled APT and
patient-by-true-lineage-shuffled APT. Permute complete APT vectors, separately
inside train/validation/test partitions. Check every permutation preserves
patient identity and, when conditioned, lineage. Singleton groups cannot be
shuffled; record their counts and the fraction of fixed points, never force a
derangement. True lineage is used only to build this diagnostic control.

For each training seed use three shuffle draws. Draw0 of patient shuffle reuses
the old seed-specific permutation; draws1/2 are new. All three conditional
draws use the same seed schedule, no selected favorable permutation. Refit and
reselect alpha for every new training/shuffle combination. Save every result.

## RNA-difficult cells

Define low RNA confidence as top-two probability margin at or below its inner
validation 20th percentile, then freeze that threshold on outer-test patients.
Only the RNA reference defines membership; never choose cells by APT gain.
Report subset coverage and all five lineage results. Additionally report
RNA-error and RNA-correct strata as retrospective diagnostics: their errors
use test labels and are not prospectively identifiable. Paired accuracy gains
on each stratum must be shown together, so corrected errors are not reported
without newly introduced errors. These model-defined subsets do not establish
that RNA is intrinsically unable to separate those cells.

## Scoring and uncertainty

Compute full27-class confusion matrices even inside a lineage: an ordinary
prediction outside the lineage must still contribute a false negative.
For each scope normalize each eligible patient's CM by its total, average,
compute per-class F1, then average the fixed appropriate ontology classes.
Do not average per-patient F1. Report eligible patients/cells and every lineage.
Joint resampling pairs training seeds, shuffle draws within seed, and test
patients across controls; deterministic modes are not treated as nine fits.
Intervals are exploratory pointwise95%, conditional on fixed subsets/folds.
Primary contrast is paired minus conditional-shuffle, with RNA/noise/patient-
shuffle comparisons retained. Both estimators and all seeds must be reported.

## RNA width and execution

The already running 5000-HVG experiment remains untouched. It independently
selects genes in each training partition and reports all modes. It will test
whether the overall paired increment depends on the original RNA width.
If the targeted 2000-HVG increment is positive in all three MLP seeds and its
joint95% interval excludes zero versus both RNA and conditional shuffle,
repeat this same targeted control at5000HVG; report SGD regardless of direction.
This follow-up gate is explicit, not retrospective selection of a width winner.

New output root: /ssd3/mayiling/apt_agent_runtime/cell_identity_questions_v1.
Use CPU4 threads on vllab11, separate lock and no modifications to old jobs.
Paper states pending comparisons as pending, not results. Release only aggregate
CSV, audit metadata and figures; cell-level outputs stay on remote SSD.
