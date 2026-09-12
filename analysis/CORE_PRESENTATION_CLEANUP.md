# Core Presentation Cleanup

Baseline: `ed8139e` (complete prior manuscript and PDF remain recoverable in Git).
This revision changes documentation and displays, not trained models, predictions,
bootstrap draws, or result CSVs.

## Editorial Decisions

- Remove the entire old resource comparison from the PDF, not just panel (b).
  Related Work supplies the bounded literature positioning; no new unsupported
  resource-level Yes/No assertions are introduced.
- Remove section 3.4, migration history in the main text, the one-drop disclaimer,
  and the sampling-report discussion. Move RNA-label provenance and training-only
  selection details into matched-input methods. Current cohort, annotation and
  privileged-label limitations remain.
- Archive composition recovery as a whole, including the direct ridge comparison
  and unresolved lineage result. Archive the incomplete LODO analysis and the
  difficulty/rescue/damage paragraph as whole modules. Keep all saved outputs and
  the previous manuscript version. The single Archived Analyses subsection records
  scope; this document supplies the source version.
- Keep the primary table, matched figure and paired-control table, full subtype
  reporting, supporting oracle replays and split comparison. The extra-model table
  now lists only SVM, Random Forest and reference correlation.
- Figure 2 labels use child counts read from the saved subtype-to-lineage mapping:
  B=5, Myeloid=8, NK=2, Other=2, T=10. Its caption points to the matched table's All
  row, not the different full-budget leaderboard. Main text explicitly qualifies
  comparisons between raw F1 values for different lineages.
- Main text points to B.3 for experiment details, B.7 for scoring units, and B.4--B.5
  for full-budget and Transformer oracle support.
- Table 2 appears after a complete paragraph in section 5.3; it cannot float into
  a split word or section 5.2. Appendix table placement is also kept near its text.

## Configuration Sources

- `analysis/paired_rna_oracle.py::patient_class_weights`: mean-normalized product
  of inverse training-patient count and balanced class weights. After cancelling
  common factors this is proportional to `1 / (n_patient * n_class)`. Counts use
  only the actual training subset. Multiplication does not retain exact marginal
  patient or class balance.
- `analysis/validate_paired_information.py`: model grids, 40 iterations, matched
  input scaling, standard-normal augmentation and fit constructors.
- `analysis/prepare_fold_rna.R`: per-cell library scaling to 10,000 and log1p;
  training-only mean-binned dispersion selection. RNA feature SD is fitted in
  `inputs`, without centering the sparse matrix.
- `analysis/run_full_budget_mlp.py`: AdamW, learning rate 0.001, batch 1,024,
  dropout 0.1, hidden widths 256/128, decay grid 0.00001/0.001, inner selection of
  epoch and decay and fresh refit. Provided APT expression is standardized without
  adding a count transform. Its upstream preprocessing is not inferred.
- SGD's implicit `l2` penalty / `optimal` schedule and matched MLP's implicit
  Adam / 0.001 learning-rate defaults were checked against the locally installed
  sklearn constructor source. This does not replace the outstanding original
  environment-lock requirement; no original environment version is invented.

## Display and Verification

`analysis/build_findings_focus.py` generates both Table 2 and
`tables/paired_effect_values.tex` from the unrounded saved contrasts.
The abstract, introduction and results share these macros: MLP 2k increment
0.0180 and 5k increment 0.0165. Differences precede rounding. The conditional
shuffle lower endpoint remains -0.00018 and its interpretation is unresolved.

Six tests in `analysis/test_findings_bootstrap.py` pass, including saved-contrast
values, shared macro precision and ontology counts. The 19-page PDF compiles
without unresolved references/citations or overfull boxes; rendered main results
and the configuration table have been checked. No experiment was added to seek
significance. Owner-dependent release and provenance TODOs are unchanged.
