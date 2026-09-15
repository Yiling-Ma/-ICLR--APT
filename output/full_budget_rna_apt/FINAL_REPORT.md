# Full-Budget RNA vs RNA+APT Sensitivity Experiment

## Protocol

This experiment used the frozen APTBench patient-disjoint folds, all available development cells, training-only preprocessing, training-only HVG selection, matched MLP architecture, matched optimizer/stopping/model-selection metric, and final seeds 17, 29, and 43.  The primary comparison was RNA-only versus RNA+APT at 5,000 and 10,000 HVGs.  The 10,000-HVG setting also included RNA+noise, APT shuffled within patient, and APT shuffled within patient-lineage controls.  All reported intervals are paired over matched seed slots and held-out patients.

The aggregation status is PASS.  Historical capped summaries were not present on vllab8 at aggregation time, so the generated table contains the full-budget results and records the missing capped paths in `completion.json`.

## Results

At 5,000 HVGs, adding APT did not improve the full-budget RNA reference: fine SB-F1 was 0.727 for RNA-only and 0.724 for RNA+APT, with paired delta -0.0037 and 95% CI [-0.0146, 0.0165].

At 10,000 HVGs, adding APT improved over RNA-only: fine SB-F1 was 0.728 for RNA-only and 0.749 for RNA+APT, with paired delta +0.0204 and 95% CI [0.0004, 0.0374].  RNA+APT also improved over RNA+noise by +0.0220 [0.0008, 0.0400].

The pairing controls limit the interpretation.  RNA+APT did not outperform APT shuffled within patient (-0.0036 [-0.0131, 0.0061]) or within patient-lineage (-0.0017 [-0.0059, 0.0039]).

## Scientific Answer

YES, but modestly and with an important caveat.  APT provides incremental predictive value beyond a strong full-cell-budget 10,000-HVG RNA reference, but the current controls do not establish that the improvement depends on exact cell-level RNA--APT pairing.

## Manuscript Recommendation

The manuscript should claim bottleneck discovery plus cautious multimodal sensitivity evidence, not cell-specific multimodal complementarity.  Recommended wording:

> In a full-budget matched sensitivity analysis, RNA+APT improves over a 10,000-HVG RNA-only MLP reference, indicating that APT-associated signal is not merely an artifact of a capped or weak RNA baseline.  However, patient- and patient-lineage-shuffled APT controls match the paired APT condition, so these data do not establish exact cell-level RNA--APT pairing specificity.
