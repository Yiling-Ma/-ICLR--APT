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

### External scaling replication

The fixed-total protocol is now evaluated in two independent public resources.
COMBAT supplies RNA and matched ADT views from 105 donors. OneK1K supplies a
second RNA cohort with 925 evaluation donors and 1,199,468 cells after excluding
doublets, one calibration pool, and donors below 800 eligible cells. OneK1K
folds are disjoint in both donor and multiplexing pool; 293 genes are selected
only on ten donors from the excluded calibration pool. Its depth audit supports
`C={100,200,400,800}` and exact totals 3,200 and 6,400 without conditioning on
the 116 unusually deep donors that reach 1,600 cells.

### Technical-confounding audit

The disease track now includes a matched nuisance-only baseline built from the
recorded RNA QC fields. For every patient, the analysis summarizes recovered
cell count, median and interquartile-range log UMI count, median and
interquartile-range detected-gene count, and the median detected-gene/UMI
ratio. Logistic Regression regularization is selected by four-fold inner
patient CV inside each 32-patient outer-development set, using the same sealed
eight-patient outer folds as the composition bridge. The comparison therefore
does not benefit from test-patient leakage or a weaker tuning protocol.

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
- The six-feature recorded-QC baseline reaches patient accuracy/macro-F1/AUROC
  0.550/0.566/0.859. Predicted subtype composition exceeds it by only +0.027
  Macro-F1, with paired patient-bootstrap 95% CI [-0.157, 0.215]. This does not
  establish incremental disease information beyond the available nuisance
  summaries, and the missing acquisition variables remain untestable.
- In the conditional audit, QC plus predicted lineage reaches macro-F1 0.583
  and AUROC 0.875, whereas QC plus predicted subtype reaches 0.599 and 0.825.
  The conditional subtype-minus-lineage Macro-F1 increment is +0.016 with 95%
  paired patient-bootstrap CI [-0.120, 0.169]. The paper therefore makes no
  subtype-increment or cross-task error-propagation claim.
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
- OneK1K scaling: complete. LR and XGBoost each completed 1,200 bundles (five
  folds x 20 subset seeds x three donor budgets x four cell caps), and
  `fair_scaling_qa.json` is `PASS`. Seven of eight fixed-total effects favor 32
  over 8 donors (+0.004 to +0.009). XGBoost coarse at 3,200 cells reverses
  (-0.0028; paired donor-bootstrap 95% CI [-0.0036, -0.0017]) and returns
  positive at 6,400 cells (+0.0056). No OneK1K empirical subset-seed interval
  excludes zero.
- Across APT, COMBAT RNA/ADT, and OneK1K RNA, 37 of 38 label-agnostic
  fixed-total comparisons favor broader subject coverage. This is retained only
  as a naive result to audit: it jointly changes subject breadth and the
  marginal training label distribution and is not an isolated
  patient-diversity effect.
- Class-matched fixed-total control: complete for APT, COMBAT RNA, and OneK1K
  RNA. Each dataset completed 1,200/1,200 LR jobs (five folds, 20 seeds, two
  totals, three subject budgets, and two tasks), and every QA file is `PASS`.
  Within each fold/seed/task, exact per-class quotas are frozen from the nested
  P=8 subset and reused at P=16 and P=32. All nominal subjects contribute and
  every paired condition has identical class counts and proportions.
- XGBoost class-matched robustness: complete at T=6,400 with 10 matched seeds.
  Each dataset completed 300/300 jobs and every QA file is `PASS`. The final
  production artifacts use the CPU histogram backend throughout. APT
  coarse/fine effects are +0.0083/-0.0058; COMBAT RNA effects are
  +0.0011/-0.0027; and OneK1K effects are -0.0011/-0.0006.
- The matched result revises the headline interpretation. At T=6,400, APT fine
  and COMBAT fine are negative under both LR and XGBoost, while OneK1K fine is
  near zero and changes sign across models. Coarse effects are also model- and
  cohort-dependent: only APT remains positive under both classifiers.
  A two-stage joint resampling procedure samples one completed training-subset
  seed and then paired outer-test subjects in each of 2,000 replicates. All
  18 joint 95% intervals include zero; Table 15 reports these intervals and
  the probability of a positive effect as the primary uncertainty analysis.
  Seed-only and subject-only intervals are retained only as decompositions.
  Consequently, class matching is the primary scaling result. The 37/38 count
  is retained only as the naive joint effect that motivates the label-distribution
  audit, not as a universal law or headline finding.

## Not completed

- Neural scaling grid: deferred because all `vllab13` GPUs were occupied and the
  priority protocol can be answered first with the two frozen classical models.
- Full five-fold DropCascade component ablation: not required for the revised
  benchmark framing; DropCascade is explicitly a reference model.
- Additional neural patient-cell scaling and patient-level DeepSets/MIL/Set
  Transformer baselines: deferred. With 40 patients these are secondary to the
  leakage and fair-scaling corrections, and available GPUs were occupied by
  unrelated jobs during the priority run.
- External cell-typing scaling replication is complete. External APT disease
  validation remains unavailable because no compatible external APT cohort is
  present.
- Formal annotation agreement: blocked because independent blinded relabeling
  has not yet been performed. The paper retains an explicit TODO rather than
  inventing annotators or agreement values.
- Fully nested inner-fold compact-panel reranking: secondary to the scaling
  correction and not yet run; the conditional-ranking limitation is explicit.

## Headline changes

The previous claim that within-patient cell depth is at least as influential as
patient count was removed because it compared unequal endpoint changes. The
primary scaling finding is instead methodological: unmatched fixed-total
comparisons confound subject breadth with the marginal training label distribution.
The naive analysis favors broader subject coverage in 37 of 38 comparisons,
but after class matching fine point estimates are negative or near zero across
LR and XGBoost, while coarse estimates remain model- and cohort-dependent. All
joint intervals include zero. The paper therefore
identifies and corrects label-distribution confounding rather than claiming a
patient-dominant scaling law.

The previous pooled-OOF composition value (predicted subtype Macro-F1 0.400) is
not used as a headline result. The strict nested value is 0.593. This numerical
difference is not interpreted as a leakage effect because the corrected
analysis also changes hard versus soft features, transformation, and downstream
regularization selection; `legacy_vs_nested_audit.csv` records the comparison.
The composition bridge is now named a cross-task representation-transfer audit.
It remains predictive and associational because its increments over lineage
composition, log cell count, and recorded RNA-QC proxies are unresolved. It is
not presented as evidence that typing errors propagate to disease performance.

The historically named DropCascade implementation is now presented as the
soft-cascade member of a backbone- and budget-matched reference ladder rather
than as a proposed method. The main text and tables no longer use branded styling or
claim an algorithmic contribution. The prespecified patient-bootstrap tests
are framed as an architecture audit, and their null result is reported as a
benchmark finding: the complete soft-cascade recipe has no supported advantage
over the backbone- and budget-matched references. Because only that recipe uses
cross-disease contrastive regularization, this is not a component-level effect.

Direct APT disease classifiers, compact panels, and attribution are no longer
primary disease results. The main table instead reports composition and
recorded-nuisance controls, while direct 40/40 majority vote and the 0.950
median-APT accuracy are confined to an appendix confounding audit. Patient-level
attribution figures were removed. Compact panels are secondary appendix
sensitivities supporting
conditional within-cohort redundancy, not clinical performance, a uniquely
minimal panel, or validated biomarkers. Disease prediction is consistently
described as within-cohort and confounding-sensitive.

## Files changed

- Analysis and reproduction: `analysis/oof_composition_bridge.py`,
  `analysis/patient_cell_scaling.py`,
  `analysis/summarize_fair_patient_cell_scaling.py`,
  `analysis/generate_composition_protocol_audit.py`,
  `analysis/conditional_subtype_increment_audit.py`,
  `analysis/technical_covariate_disease_audit.py`, `Makefile`, and
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
- OneK1K protocol, adapter, and outputs: `ONEK1K_SCALING_PROTOCOL.md`,
  `analysis/prepare_onek1k.py`, `analysis/onek1k_scaling.py`,
  `analysis/run_onek1k_xgb_shards.sh`, and `outputs/onek1k_scaling/`. The 101 MB
  per-donor confusion artifact and raw resumable bundles remain on the experiment
  host; committed summaries and audit manifests support every reported number.
- New confounding-audit artifacts:
  `outputs/technical_covariate_disease_audit/`.

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

The recorded-nuisance audit was generated with:

```bash
python analysis/technical_covariate_disease_audit.py \
  --metadata data/metadata.csv \
  --annotation data/cell_annotation.csv \
  --folds benchmark/splits/patient_folds.json \
  --composition-predictions outputs/oof_composition_bridge/patient_disease_predictions.parquet \
  --permutations 0 \
  --output-dir outputs/technical_covariate_disease_audit
```

The conditional subtype-increment audit uses the committed nested features:

```bash
python analysis/conditional_subtype_increment_audit.py \
  --composition-features outputs/oof_composition_bridge/patient_composition_features.parquet \
  --patient-qc outputs/technical_covariate_disease_audit/patient_qc_features.csv \
  --folds benchmark/splits/patient_folds.json \
  --output-dir outputs/technical_covariate_disease_audit
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

The OneK1K replication is reproduced with the commands in
`ONEK1K_SCALING_PROTOCOL.md`; the production run used 32 CPU shards for each
model family and 2,000 paired donor-bootstrap replicates. Cross-cohort outputs
are regenerated with:

```bash
python analysis/summarize_cross_cohort_scaling.py
cp outputs/cross_cohort_scaling/cross_cohort_scaling.pdf figures/
cp outputs/cross_cohort_scaling/cross_cohort_scaling_table.tex tables/cross_cohort_scaling.tex
make paper
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
- The research main text currently ends on PDF page 10. If the final ICLR 2027
  format enforces a nine-page main-text limit, one page must still be compressed
  or moved to the appendix before submission.

## Final QA

- Final research main-text page count: **10 pages**. Non-counted statements and
  References begin on page 11; the complete PDF has 31 pages.
- Final PDF path: `output/pdf/APT-Bench_ICLR2027_revised.pdf`.
- Numerical artifact-to-paper audit: **PASS**; APT/COMBAT/OneK1K scaling,
  transfer, and conditional subtype-increment claims
  were checked against the CSV/JSON files listed in `CLAIM_ARTIFACT_MAP.md`.
- Code syntax and LaTeX build: **PASS**.
- Rendered-page visual inspection: **PASS** for the main scaling figure,
  disease table, Conclusion/statements boundary, and appendix audit/scaling
  tables; no clipping or unresolved references were observed.

## Clarity and layout pass

- Rewrote the Abstract to 160 words around the statistical-unit problem, the
  patient-disjoint benchmark contract, the two-axis scaling result, the
  composition/confounding audit, and the benchmark's intended significance.
- Replaced the ambiguous novelty-contract statement with an explicit scope
  boundary: open-set subtype discovery is excluded, while all primary hierarchy
  tasks use the fixed five-lineage/27-subtype ontology.
- Reduced the main hierarchy table to Coarse Macro-F1, Fine Macro-F1, Exact Path,
  and Subtype Tree Distance. Fine accuracy, root-excluded hierarchical F1, and
  natural consistency are reported together in the appendix.
- Rebuilt the benchmark-comparison and subtype-diagnostic tables as readable
  stacked panels, enlarged the fixed-total table, strengthened the main scaling
  figure's legend and error bars, and enlarged the appendix scaling and
  compact-panel figures.
- Reordered appendix floats so the full scaling table uses the previously sparse
  page before the scaling figures. A final 31-page render was inspected at normal
  reading scale; no clipping, unresolved references, or overfull boxes remain.
