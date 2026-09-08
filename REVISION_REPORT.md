# APT-Bench ICLR 2027 Revision Report

Status: **COMPLETE - final manuscript and artifact QA passed.**

## Repository audit

- Main source: `iclr2027_conference.tex`; included main-text sections are in
  `main/`, generated table fragments in `tables/`, and figures in `figures/`.
- Immutable patient folds: `benchmark/splits/patient_folds.json`, mirrored by
  the historical baseline artifact used by the analysis code. The manifest has
  five folds of eight unique patients and has not been changed.
- Strict nested composition artifacts: `outputs/oof_composition_bridge/`.
- Historical patient-cell scaling implementation:
  `analysis/patient_cell_scaling.py`; the previous endpoint comparison is no
  longer considered a fair patient-versus-cell effect comparison.
- Compact-panel controls: `analysis/results/psas_controls_v2/` and
  `analysis/results/patient_permutation_n1000/`.
- No repository `AGENTS.md` was present.
- The named `_ICLR__APT(6).pdf` was not present in the repository or indexed
  local files. The tracked `output/pdf/APT-Bench_ICLR2027.pdf` and its source
  were therefore audited.

## Protocol corrections

### Nested composition bridge

The previous pooled-OOF bridge was cross-stage contaminated: a cell model that
generated a downstream training patient's composition could have trained on the
current disease outer-test patients. The corrected committed pipeline seals each
outer-test fold, constructs the 32 development compositions by inner
patient-disjoint cross-fitting, fits one final cell model on the 32 development
patients for the eight sealed patients, and fits the downstream disease model
only on inner-OOF development compositions. Only the corrected nested artifacts
are used for current claims.

### Fair patient-cell scaling

The old comparison contrasted a fourfold patient increase with a variable and
often much larger `100 -> all` cell increase. The replacement protocol uses
`P={8,16,32}` and `C={100,200,400,800,1600}` with 20 matched subset seeds,
retaining the original outer folds and evaluating all outer-test cells. It
contains exact fixed-total diagonals for 3,200, 6,400, and 12,800 training cells,
matched doublings along each axis, pooled cell-weighted Macro-F1, mean
within-patient Macro-F1, separate subset-seed intervals, paired patient bootstrap
intervals, and a descriptive response surface with outer-fold fixed effects.

## Metadata and scope audit

The merged benchmark metadata contain only `cell_id`, `sample_id`, `disease`,
`cell_subtype`, and derived `coarse_subtype`. Annotation metadata add RNA QC
columns but no acquisition batch, plate, date, operator, sequencing run, reagent
lot, library-preparation batch, recruitment site, disease stage, or treatment
status. No compatible external APT cohort is present in the repository. Disease
results therefore remain within-cohort source discrimination and cannot support
clinical, causal, cross-batch, or external-generalization claims.

## Completed experiments

- Strict nested composition bridge: complete; QA passed; artifacts committed.
- The strict bridge completed 25/25 outer-context/base-model fits, 10,000
  patient-label permutations, 2,000 patient-clustered bootstrap replicates, and
  50 equal-cell sensitivity seeds. Its final QA status is `PASS`.
- Strict soft predicted subtype composition reaches patient macro-F1 0.593;
  soft lineage composition reaches 0.477. The paired difference is +0.116 with
  patient-bootstrap 95% CI [-0.015, 0.262]. The corresponding subtype-minus-log
  cell-count difference is +0.062 with 95% CI [-0.087, 0.218].
- Hierarchy metrics and wrong-conditioned cross-lineage null: complete.
- Compact-panel Random-B, inference-cell, permutation, and XGBoost-SHAP controls:
  complete, but the primary budget selection remains conditional on a ranking
  computed once on each outer-development set.
- Fair patient-cell scaling: complete. Logistic Regression and XGBoost each
  completed 1,500 bundles (five folds x 20 subset seeds x three patient budgets
  x five finite cell caps), with coarse and fine evaluation from every fit.
  `fair_scaling_qa.json` reports `PASS` for grid completeness, 20 seeds per
  group, exact fixed totals, complete paired effects, and finite metrics.
- At fixed totals of 3,200, 6,400, and 12,800 training cells, reallocating from
  8 to 32 patients improves pooled OOF Macro-F1 in all 12 model/task/budget
  comparisons. Gains are 0.012--0.016 (LR coarse), 0.006--0.008 (LR fine),
  0.015--0.019 (XGBoost coarse), and 0.006--0.008 (XGBoost fine). All paired
  patient-bootstrap intervals exclude zero; 8/12 empirical subset-seed
  intervals exclude zero.
- Averaged matched patient-versus-cell doubling gains are 0.011/0.005 (LR
  coarse), 0.007/0.003 (LR fine), 0.016/0.008 (XGBoost coarse), and 0.009/0.006
  (XGBoost fine). Fold-adjusted response surfaces give the same ordering but
  are reported as descriptive, not causal.

## Not completed

- Neural scaling grid: deferred because all `vllab13` GPUs were occupied and the
  priority protocol can be answered first with the two frozen classical models.
- Full five-fold DropCascade component ablation: not required for the revised
  benchmark framing; DropCascade is explicitly a reference model.
- Additional neural patient-cell scaling and patient-level DeepSets/MIL/Set
  Transformer baselines: deferred. With 40 patients these are secondary to the
  leakage and fair-scaling corrections, and available GPUs were occupied by
  unrelated jobs during the priority run.
- External validation: blocked by the absence of a compatible external APT
  cohort in the repository.
- Formal annotation agreement: blocked because independent blinded relabeling
  has not yet been performed. The paper retains an explicit TODO rather than
  inventing annotators or agreement values.
- Fully nested inner-fold compact-panel reranking: secondary to the scaling
  correction and not yet run; the conditional-ranking limitation is explicit.

## Headline changes

The previous claim that within-patient cell depth is at least as influential as
patient count was removed because it compared unequal endpoint changes. The
replacement claim is narrower and directly supported: under three exact
fixed-total-cell budgets, broader patient coverage outperforms deeper sampling
from fewer patients for both frozen classical models and both resolutions. The
matched-doubling and response-surface analyses agree over the evaluated range,
but no causal or universal scaling-law claim is made.

The previous pooled-OOF composition value (predicted subtype Macro-F1 0.400) is
not used as a headline result. The strict nested value is 0.593. This numerical
difference is not interpreted as a leakage effect because the corrected
analysis also changes hard versus soft features, transformation, and downstream
regularization selection; `legacy_vs_nested_audit.csv` records the comparison.
The composition bridge remains predictive and associational because its
increment over lineage composition and log cell count is unresolved.

DropCascade is now a hierarchy-aware reference rather than a supported method
winner. Compact panels are secondary appendix sensitivities supporting
conditional within-cohort redundancy, not clinical performance, a uniquely
minimal panel, or validated biomarkers. Disease prediction is consistently
described as within-cohort and confounding-sensitive.

## Files changed

- Analysis and reproduction: `analysis/oof_composition_bridge.py`,
  `analysis/patient_cell_scaling.py`,
  `analysis/summarize_fair_patient_cell_scaling.py`,
  `analysis/generate_composition_protocol_audit.py`, `Makefile`, and
  `requirements-analysis.txt`.
- Benchmark contract and audit documentation: `README.md`,
  `benchmark/README.md`, `REVISION_PLAN.md`, `CLAIM_ARTIFACT_MAP.md`, and this
  report.
- Manuscript: `iclr2027_conference.tex`, `main/intro.tex`, `main/task.tex`,
  `main/method_summary.tex`, `main/method.tex`, `main/results.tex`,
  `main/discussion.tex`, and generated/updated tables and figures.
- New fair-scaling artifacts: `outputs/patient_cell_scaling_fair/`; raw resumable
  run bundles remain at
  `/home/mayiling/projs/apt_agent/outputs/patient_cell_scaling_fair/runs/`.

## Reproduction

Install `requirements-analysis.txt`, provide the documented local data files,
and run `make revision`. The command regenerates the protocol audit and the
scaling CSV/TeX/figures from the committed sufficient statistics, then compiles
the revised PDF. Use `make fair-scaling-aggregate` when the raw run bundles are
available and must be aggregated again. A full single-process rerun can be
started with `make fair-scaling-run`; production runs may use the script's
resumable `--shard-index/--num-shards` interface.

The strict bridge was generated with:

```bash
python analysis/oof_composition_bridge.py audit
python analysis/oof_composition_bridge.py fit-base --device cuda --n-jobs 8
python analysis/oof_composition_bridge.py aggregate
python analysis/oof_composition_bridge_downstream.py --n-jobs 32 --permutations 10000
```

The fair scaling production run used the same immutable folds and resumable
shards:

```bash
python analysis/patient_cell_scaling.py --output-dir outputs/patient_cell_scaling_fair plan
python analysis/patient_cell_scaling.py --output-dir outputs/patient_cell_scaling_fair run --models logistic_regression --num-shards 20 --shard-index <0..19>
python analysis/patient_cell_scaling.py --output-dir outputs/patient_cell_scaling_fair run --models xgboost --num-shards 28 --shard-index <0..27>
python analysis/patient_cell_scaling.py --output-dir outputs/patient_cell_scaling_fair aggregate
python analysis/summarize_fair_patient_cell_scaling.py --output-dir outputs/patient_cell_scaling_fair
python analysis/generate_composition_protocol_audit.py
make revision
```

## Remaining risks before submission

- The processed benchmark matrix, labels, feature metadata, licenses, archival
  version, and executable evaluator are not yet publicly released. Until that
  is completed, the benchmark cannot be reproduced by an external reviewer.
- Acquisition batch, plate, date, operator, sequencing run, reagent lot,
  library-preparation batch, recruitment site, disease stage, and treatment
  metadata are unavailable. The disease track cannot separate biological from
  disease-correlated technical structure.
- There is no compatible external APT cohort, so there is no external clinical
  validation or cross-site estimate.
- Fine labels remain manual transcriptome-derived reference annotations without
  completed blinded multi-annotator agreement. The red manuscript TODO is
  retained.
- Neural scaling, additional patient-set models, and a full DropCascade component
  ablation were not run. Accordingly, scaling conclusions are limited to LR and
  XGBoost and DropCascade is not claimed as a method winner.
- Compact-panel budget selection is outer-test isolated but conditional on one
  ranking per outer-development set; fully nested inner-fold reranking remains
  future work.

## Final QA

- Final main-text page count: **9 pages**, with Conclusion and non-counted
  statements on page 9 and References beginning on page 10.
- Final PDF path: `output/pdf/APT-Bench_ICLR2027_revised.pdf`.
- Numerical artifact-to-paper audit: **PASS**; scaling and composition claims
  were checked against the CSV/JSON files listed in `CLAIM_ARTIFACT_MAP.md`.
- Code syntax and LaTeX build: **PASS**.
- Rendered-page visual inspection: **PASS** for the main scaling figure,
  disease table, Conclusion/statements boundary, and appendix audit/scaling
  tables; no clipping or unresolved references were observed.
