# Benchmark-first manuscript revision

Historical revision record, superseded by the current prior-controlled
hierarchy and paired APT/RNA findings. The split comparison is now descriptive
appendix background only; see `analysis/SPLIT_AUDIT_RESOLUTION.md`.
The positioning statements below describe an earlier version, not current claims.

## Further focus revision

The paper now centers one contribution: patient-disjoint evaluation for APT
cell typing, with the observed split-protocol sensitivity as its central
empirical evidence. The four historical model-by-resolution controls all give
higher cell-split scores, but do not isolate a causal leakage effect or justify
a universal claim. No unsupported "first benchmark" priority claim is added.
Soft-cascade remains a reference; RNA and disease analyses are brief secondary
context. Scaling sections and simulations have been moved intact into
`supplement/scaling_synthetic.tex` and `supplement/scaling_empirical.tex`, built
through `sampling_report.tex`. Results have not been deleted or rerun.
The public data/permissions/evaluator release gap remains a substantive blocker,
not a concern this writing change resolves.

## Scope

Primary contribution: an APT-only, patient-disjoint cell-typing benchmark.
Soft-cascade Transformer is a documented reference recipe, not a demonstrated
algorithmic advance. The main table compares LR, XGBoost, Flat Transformer,
HCE Transformer, and Soft-cascade using the same frozen prediction artifacts.
Linear SVM, Random Forest, and reference correlation remain in the appendix.

The paired-input RNA/MLP/SGD experiment is a separate modality reference with
a capped training budget. Its MLP row is not a full-budget hierarchy baseline.
An equivalent full-budget MLP evaluation remains missing. No training job was
started or stopped for this manuscript revision.

## Metric change

The main reporting metric was changed retrospectively from cell-weighted
pooled Macro-F1 to subject-balanced pooled Macro-F1. Each subject's confusion
matrix is divided by that subject's total evaluated cells before pooling.
This is neither row normalization nor mean within-subject Macro-F1.
Checkpoints and prediction targets were not changed. Historical cell-weighted
point estimates were independently reconstructed by the new script and agree
with the legacy evaluator. Pointwise intervals resample the same patients
across recipes, conditional on fixed folds and fitted models.

The main table makes descriptive comparisons, not selective model-superiority
tests. Historical Holm-adjusted comparisons remain explicitly restricted to
the old cell-weighted metric. No historical significance marker was copied to
the new metric.

## Reproduction

Run in an environment with numpy, pandas, and scikit-learn:

```sh
python -m unittest discover -s analysis -p test_revise_benchmark_table.py
python analysis/revise_benchmark_table.py \
  --dropcascade "$PREDICTIONS/dropcascade_kfold5/pooled_oof_predictions.csv" \
  --classical-root "$PREDICTIONS/classical_baselines_kfold5" \
  --matched-root "$PREDICTIONS/matched_ft_transformer" \
  --output-dir analysis/generated --table-path tables/main_result.tex
make paper
```

Only non-identifying aggregate metrics and source hashes are committed.
The audit checks patient-by-class target counts across recipes and reconstructs
legacy scores; historical neural pooled exports lack cell IDs, so this check
does not independently establish barcode-level alignment of those exports.
Their original frozen OOF construction is retained, not reconstructed here.
The release still needs its complete prediction-interface provenance.

## Interpretation changes

- Main text: APT cell typing, a separate modality table, and a short secondary
  patient-aggregation application.
- Appendix: complete baseline inventory, historical metrics, path diagnostics,
  disease comparison, and sampling/budget audits.
- RNA descriptions now match the current validation code: HVGs are selected
  independently in inner and outer training subsets. RNA-derived annotations
  remain a limitation, but are not conflated with global HVG preprocessing.
- No claim of beating RNA methods, validated one-drop diagnosis, independent
  disease validation, or significant cascade-component improvements is added.
- Data-owner TODOs, annotation uncertainties, and release requirements remain.
