# Synthetic mechanism check

Completed 8 settings x 20 independent seeds x 4 conditions x 2 subject budgets
= 1,280 LR fits. See `analysis/simulate_sequential_controls.py` for the full
generator and `analysis/render_sequential_simulation.py` for the paper table.

This is synthetic, not semi-synthetic, data. It is not a substitute for the
pending real-cohort A-D experiment. Every setting and seed is retained in
`results.json`. Intervals are empirical seed percentiles, not confidence
intervals of the mean. The complete-support generator does not exercise the
missing-class mechanism, so A and B coincide. The null setting is descriptive;
no type-I-error or coverage validation has been conducted.

Local runtime: Python 3.12; numpy 2.5.3; scikit-learn 1.9.0. CPU only.
