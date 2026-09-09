# Known-null and independent-target validation

Completed six scenarios, each with 512 reference cohorts x two fits plus 100
evaluation cohorts x three conditions x two fits: 9,744 LR fits total.
All scenarios, reference seeds, evaluation seeds, effects and errors are saved.

Reproduce from repository root:
```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analysis/validate_scaling_ground_truth.py --output outputs/scaling_ground_truth_validation
python analysis/render_scaling_ground_truth.py
```

The target is expected P32-minus-P8 subject-balanced pooled fine Macro-F1
under the overlap-quota C sampling policy, integrating over the random P8
reference quotas. It is **not** total value of recruiting more subjects.
In the no-shift case, features are independent across cells conditional on
class, and quotas depend on labels only: the fitted-model distribution is the
same at both subject budgets. The expected score difference is exactly zero.
For nonzero random effects, the target is estimated by independent Monte Carlo
with its own SE. It is never asserted that high heterogeneity guarantees a
large positive target. The oracle uses a separate direct-index sampling
implementation, not the priority sampler being evaluated.

Bias means error relative to this policy target. A/B target different sampling
policies; their positive effects can be legitimate unrestricted allocation
benefits. Calling those effects type-I errors or proof that subjects are
useless would be incorrect. The independent reference measures recovery of a
specified policy, not discovery of a universal causal subject effect.

Bias intervals use normal Monte Carlo SE across independent simulated cohorts;
for non-null scenarios they propagate independent target-estimation variance.
They are pointwise, not multiplicity-adjusted. This experiment does not assess
coverage of the real-data patient/seed bootstrap, or equivalence-test size.
RMSE uses the estimated reference value and inherits its Monte Carlo error.

Source and sampler hashes are in source_manifest.json. Runtime: Python 3.12,
NumPy 2.5.3, scikit-learn 1.9.0. Simulation ran locally on CPU; no changes to
the ongoing vllab13 real-data GPU experiment were made.
