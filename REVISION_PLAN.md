# APT-Bench ICLR 2027 Revision Plan

This document freezes the execution order for the repository-wide revision. It
distinguishes results already supported by machine-readable artifacts from
claims that require new computation or unavailable metadata.

## Evidence already complete

1. **Strict nested OOF composition bridge.** The cell-typing model and downstream
   disease model are cross-fitted without exposing outer-test patients. The
   committed artifacts in `outputs/oof_composition_bridge/` are the sole source
   for composition-bridge values.
2. **Hierarchy evaluation.** Existing pooled OOF prediction artifacts support
   root-excluded hierarchical F1, exact-path accuracy, subtype-tree distance,
   and the wrong-conditioned cross-lineage null analysis.
3. **Compact-panel controls.** Existing artifacts support 100 Random-B repeats,
   inference-time cell subsampling, patient-level label permutation, and an
   XGBoost-SHAP attribution sensitivity. The current panel result is explicitly
   conditional on an LR ranking computed once on all outer-development patients;
   it is not fully nested inner-fold feature ranking.

## P0 computation

1. Replace the endpoint comparison in the patient-cell scaling analysis with:
   - fixed-total budgets `T in {3200, 6400, 12800}` and
     `P in {8, 16, 32}`, with exact `C=T/P` cells per patient;
   - matched patient doublings and cell-depth doublings on a common finite grid;
   - the immutable five patient-disjoint outer folds and all outer-test cells;
   - 20 nested, disease-stratified patient/cell subset seeds;
   - Logistic Regression and XGBoost for coarse and fine typing;
   - pooled cell-weighted macro-F1 and patient-balanced macro-F1;
   - subset-seed intervals and patient-clustered bootstrap uncertainty;
   - a descriptive response-surface model with patient and cell-depth terms.
2. Preserve all raw predictions, sampled-patient manifests, per-patient
   confusion matrices, summaries, tables, figures, protocol hashes, and QA logs.
3. Base every manuscript number on generated CSV/JSON/TeX artifacts.

## Manuscript decisions after P0

1. Rewrite the abstract, introduction, scaling methods/results, discussion, and
   conclusion from the fair scaling estimates. Do not retain the current claim
   that cell-depth gain is at least as large as patient gain unless the matched
   analysis supports it.
2. Keep DropCascade as a hierarchy-aware reference model, not a universal method
   contribution. Do not claim a component advantage without a full five-fold
   ablation.
3. Keep disease recognition as a within-cohort audit. The available metadata
   contain no acquisition batch, run, panel version, or date, and there is no
   external cohort; clinical, causal, and cross-batch claims remain unsupported.
4. Describe compact panels as conditional measurement-efficiency evidence and
   predictive redundancy. Do not call the selected panel uniquely minimal or a
   validated biomarker signature.
5. Compress the main paper to at most nine pages before references by moving
   implementation details, secondary tables, consistency, LODO, and extended
   controls to the appendix without removing unresolved TODO disclosures.

## Secondary work, attempted only after P0 and page-limit repair

1. Add a stronger baseline or patient-balanced training-seed sensitivity only if
   existing code and compute permit a fair five-fold run.
2. A fully nested compact-panel reranking is desirable but secondary to correcting
   the scaling claim; if not run, the limitation remains explicit.
3. External validation and formal multi-annotator reliability cannot be claimed
   without a compatible external cohort and completed blinded relabeling. The
   paper will provide protocols and TODOs rather than fabricated results.

## Deliverables

- executable analysis code and one-command reproduction entry point;
- machine-readable raw and summary artifacts;
- generated LaTeX tables and publication figures;
- a visually checked final PDF with main text at or below nine pages;
- `REVISION_REPORT.md` mapping every claim to its artifact and recording all
  completed, incomplete, and blocked items.
