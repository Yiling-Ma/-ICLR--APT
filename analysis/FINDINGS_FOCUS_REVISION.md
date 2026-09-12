# Findings-Focused Manuscript Revision (2026-09-12)

## Scope

No training, additional seeds, threshold tuning or result selection was run.
The primary patient-by-lineage-shuffle contrast remains unresolved. The
paired-error feature analysis stays in `analysis/paired_error_analysis/`; its
post-hoc associations have not been promoted to a new paper claim.

## Main Evidence and Source Mapping

- Abstract and contributions now state the full-budget oracle/prior result
  (0.385/0.180), heterogeneity, and paired-input fine-F1 increments at both RNA
  widths (0.018/0.016), with the conditional-shuffle limit retained.
- `figures/matched_lineage_identity.pdf` reads the completed targeted
  `outputs/cell_identity_questions_v1/hvg2000/scope_scores.csv`. All five
  lineages, both models, APT/RNA oracle and expected priors are shown.
- `tables/paired_increment_main.tex` reads all 2,000-HVG contrasts from the
  SAME `paired_contrasts.csv`. Both shuffle types have three draws per seed.
  The 5,000-HVG RNA comparison comes from its separately labeled width grid.
- Main text juxtaposes full-budget Other oracle 0.886 with capped-budget
  0.497, and expected priors 0.490/0.487. Budget, architecture, loss weighting,
  optimizer and selection differ; the contrast is not a budget-only ablation.
- Full-budget lineage plot is supplementary; the numerical matched table,
  all lineage contrasts and the wider-RNA table remain available.
- P / SP explicitly distinguish patient-only from seed/patient confidence
  intervals in the main and expanded inventories. The MLP is in both.

Rebuild displays with `python analysis/build_findings_focus.py`; build the
paper with `latexmk -pdf iclr2027_conference.tex`.

## Bootstrap Audit

Read production implementations in `analysis/cell_identity_questions.py`
(`scoped_score`, `bootstrap_scores`, `aggregate`), `run_modality_repeats.py`,
`run_full_budget_mlp.py`, and `summarize_mlp_lineages.py`.

For the targeted contrasts, each replicate has one 40-patient multinomial draw
shared by all paired conditions and seed/shuffle slots, three seed draws with
replacement, and three shuffle draws within each selected seed slot. Compute
F1 separately per fitted condition/realization, then average the three shuffle
scores, subtract from the paired score and average the three seed differences.
Do not pool model confusion matrices before F1. For non-shuffle contrasts there
is only one comparator per seed. Full-budget and width checks use the same
mean-of-seed-scores estimand without a shuffle draw.

Per-lineage normalization divides by that patient's cells of the true lineage,
while the main score divides by the patient's complete evaluated population.
Thus averaging lineage F1 values does not reconstruct the main score.
Outside-lineage predictions retain false negatives; unsupported patients are
excluded. No-support resamples are undefined rather than scored as zero.
The paper now provides the formula, point estimator, and exact resampling order.

`test_findings_bootstrap.py` tests production-vectorized versus explicit patient
resampling, nonlinear seed pooling, outside-lineage errors, the three-by-three
resampling loop, and source-consistent primary table values. It does not claim
that three seeds estimate all training uncertainty or that bootstrap coverage
has been independently established.

## Removed from PDF, Preserved for Provenance

The previous top-level manuscript is archived as
`archive/manuscript_before_findings_focus.tex`; all original inputs and outputs
remain under version control. The complete previous build is recoverable at
commit `61b9a2f`. This source snapshot is provenance, not a second up-to-date
manuscript to compile against rewritten inputs.

| Previous section / item | Current treatment |
| --- | --- |
| B.2 single-seed paired pilot / Table 3 | Archived; repeated target results replace it |
| B.3 chance-adjusted gap / Table 4 | Archived; oracle and conditional priors carry interpretation |
| B.5 disease tables and classification audit | Archived as a whole, with its QC, centering and permutation caveats; no disease accuracy is reported in the PDF |
| B.6 composition | Appendix only, retains direct ridge and unresolved lineage result; misplaced disease permutation paragraph removed with disease analysis |
| B.7 Transformer oracles | Short supporting appendix table retained |
| B.8 marginal support plot/table and error null | Archived; complete subtype table retained and explicitly CW |
| B.9/B.13 split definitions and comparison | One appendix subsection; no stale main-text reference |
| B.10 redundant CW leaderboard/path metrics | Archived; expanded SB inventory includes MLP |
| Low-margin and rescue/damage | Appendix only, with conditioning and uncertainty |
| B.14 matched controls | Methods/uncertainty retained; primary evidence promoted |
| Run counts and follow-up gate log | Reproduction documents rather than paper narrative |

Acquisition/annotation/release TODOs remain. They require data-owner material;
editorial condensation does not resolve them. Disease-label use by Soft-cascade
is retained in its objective disclosure even though disease prediction is removed.

## Directly Relevant Literature Check

- Delley et al., 2018, [Scientific Reports](https://www.nature.com/articles/s41598-018-21153-y):
  joint aptamer/RNA profiling and discrimination in Ramos/3T3 cell mixtures.
- Wu et al., 2025, [JACS / author abstract](https://pubmed.ncbi.nlm.nih.gov/40488675/),
  DOI 10.1021/jacs.5c01296: Aptomics protein/glycan/RNA profiling, with cell-line
  validation and clinical specimen applications. Clinical profiling is therefore
  not claimed as a new concept here.
- Luo et al., 2026, [Science / author abstract](https://pubmed.ncbi.nlm.nih.gov/41477891/),
  DOI 10.1126/science.adv6127: CRISPR-linked aptamer target/kinetic discovery.

These directly related studies motivate a complementary evaluation question:
new-patient hierarchy reconstruction under candidate-set and conditional-pairing
controls. The search does NOT establish an exhaustive absence of prior work.
The manuscript makes no "first" claim. The full JACS/Science publisher pages
were access-limited; descriptions are bounded to the accessible author abstracts.
Do not assert that every analysis in those supplements lacks a particular control.
