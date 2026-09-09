# Chance-adjusted resolution audit

Regenerate from the repository root with `python analysis/audit_granularity_chance.py`
(NumPy required). The script writes `audit.json` and `tables/granularity_chance.tex`.

Inputs are the five matched paired-RNA linear-probe runs, not the main neural
model comparison. The JSON records the SHA256 of every source artifact.
Each patient confusion matrix is normalized by its total cell count before
averaging, so scores are subject-balanced pooled Macro-F1, not the average of
within-patient F1 scores.

The chance reference is F1 of the product of true and predicted marginals.
It is not a finite-sample permutation expectation. Intervals use 2,000 paired
patient resamples, recomputing both marginals, and condition on the fitted models.
They do not include training-subset or optimization uncertainty.

Chance adjustment does not match class count, entropy, class boundaries, or
annotation reliability. The residual gap is descriptive, not a new metric or
evidence of a cardinality-independent biological limitation.

True-lineage-conditioned subtype decoding has not been performed. It needs
complete held-out subtype scores with verified class order and parent mapping,
or conditional models trained without test-patient access. Retaining only
already lineage-correct hard predictions is not an equivalent control.
