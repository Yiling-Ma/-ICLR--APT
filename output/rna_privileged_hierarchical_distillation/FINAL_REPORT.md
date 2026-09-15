# RNA-Privileged Hierarchical Distillation Report

## Protocol

This experiment used the frozen APTBench patient-disjoint folds, the validated APT preprocessing pipeline, the same final seed slots 17, 29, and 43, and deployable APT-only student inference.  RNA or true lineage was allowed only for training-time teacher/distillation signals.  Aggregation used matched held-out-patient and seed-slot bootstrap with 5,000 replicates.

The aggregation status is PASS.  Tables and figures were generated directly from saved predictions.

## What Was Tested

The experiment evaluated hierarchy-routing distillation, within-lineage knowledge distillation, representation alignment, their combinations, a full RPH-Distill model, a shuffled-teacher control, and a strong lineage-label control.

## Results

The plain APT MLP reached fine SB-F1 0.137, cross-lineage error 0.491, fine-outside-parent given coarse-correct 0.176, oracle F1 0.376, and hierarchy gap 0.239.

The best point estimate was routing plus within-lineage distillation, with fine SB-F1 0.143, cross-lineage error 0.489, fine-outside-parent given coarse-correct 0.167, oracle F1 0.384, and hierarchy gap 0.242.  Its paired fine SB-F1 delta over the plain APT MLP was +0.0057 with 95% CI [-0.0005, 0.0127].

Full RPH-Distill reached fine SB-F1 0.142 with paired delta +0.0049 [-0.0054, 0.0107].  The shuffled-teacher control was worse than the plain APT MLP, with fine SB-F1 0.129 and paired delta -0.0081 [-0.0190, -0.0003].

## Scientific Answer

YES, but only modestly and not decisively.  Privileged RNA/lineage training signals produce small improvements in APT-only deployable subtype prediction and slightly reduce some hierarchy inconsistency diagnostics, but paired intervals still cross zero for the main positive candidates and the oracle--deployable gap remains large.

## Manuscript Recommendation

The manuscript should claim hierarchy bottleneck discovery plus weak or partial mitigation, not strong mitigation.  Recommended wording:

> RNA-privileged hierarchy distillation modestly improves APT-only subtype point estimates, but the paired confidence interval overlaps zero and the true-lineage oracle gap remains large.  This suggests that the hierarchy bottleneck is real and not solved by straightforward privileged distillation or consistency losses.
