# Full-budget MLP failure structure

Descriptive analysis specification, fixed before inspecting new failure rates.
Reuse all 15 fine-task OOF score files (3 training seeds x 5 folds), without
training, inference, checkpoint selection or filtering the evaluation set.
The capped RNA/APT analyses and historical Soft-cascade CW results are separate.

Ordinary argmax uses the original saved class-column order. Oracle argmax masks
all scores outside the true subtype's fixed parent, retaining the same order.
Require finite scores, exact cell/patient/label alignment and identical ontology.
Check saved ordinary/oracle patient confusion matrices before aggregation.

Partition cells into correct, cross-parent subtype error, and within-parent
subtype error, using the parent of the predicted subtype, never a coarse head.
Within each patient and scope (all cells or one true lineage), divide category
counts by the scope's cell count; average eligible patients, then seeds. Report
eligible patient counts. For the fraction of cross-parent errors corrected by
oracle, divide corrected cross errors by cross errors within each patient/scope;
exclude zero-cross-error patients from this conditional ratio and report their
eligible counts separately. Unconditional repair rates retain all scope-eligible
patients and are used for the ordinary-correct + repair = oracle-correct check.
These rates are not an additive decomposition of nonlinear Macro-F1.

For each seed, sum patient confusion matrices after dividing by each patient's
total OOF cell count. Save ordinary and oracle 27x27 matrices in fixed
lineage-then-subtype order, including zero rows. Save their row-normalized
versions separately. Figure summaries average seed-specific SB matrices first,
then normalize rows; they are visual summaries, not the mean-seed F1 estimand.
Calculate per-class F1 on each seed's SB matrix and only then average seeds.

Select at most three illustrative directed oracle confusions by descending
off-diagonal mass in the mean-seed SB matrix (divide by 40 for a fraction of
patient-balanced all-cell mass); resolve ties in the fixed display order.
Report their source/target support, patient coverage, and per-seed ranks/mass.
Publish every edge and subtype, not only selected examples. Report descriptive
seed ranges, not new subgroup significance tests or causal claims. Keep the
existing training-only expected-CM prior and verify its all-cell score using
outer-development label counts. No extra bootstrap or training is planned.
