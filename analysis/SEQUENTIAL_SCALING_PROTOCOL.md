# Sequential subject-cell controls

## Scope and status

This is a new experiment, not a relabeling of existing class-matched results.
The old implementation both fixes class quotas and uses within-class donor
round-robin allocation. Its difference from the original patient-wise sampling
must not be attributed entirely to label distribution.

## Frozen design

- Datasets: APT, COMBAT RNA, OneK1K; existing five untouched outer folds.
- Total training cells: 6,400; nested development subject pools: 8 and 32.
- Ten training-subset seeds paired across all conditions and model families.
- Models: existing LR and XGBoost configurations; an additional fixed MLP
  (128/64 ReLU, Adam, batch 256, learning rate .001, 100 epochs, no early stopping).
  This MLP is a cross-family sensitivity check, not an encoder-matched Transformer.
- Feature standardization is fitted only on selected training cells. Existing
  feature-preparation limitations of each dataset still apply.
- Full held-out subjects and all of their cells are used for evaluation.

## Conditions

| Condition | Support | Counts | Within-class allocation |
| --- | --- | --- | --- |
| A | Natural | Natural | Uniform pooled-cell priorities |
| B | P=8 support | Natural after support restriction | One uniform representative per supported class, then uniform remainder |
| C | P=8 support | Exact frozen P=8 quotas | Uniform pooled-cell priorities within class |
| D | P=8 support | Same quotas as C | Capacity-constrained donor round robin |

B explicitly includes a support floor and is not a claim of uniformly sampling
all feasible support-preserving subsets. D randomizes donor traversal order.
All conditions reuse the same random cell priorities. A differs from the legacy
equal-cells-per-subject experiment and must be rerun rather than copied from it.
The support and quota reference is solely the P=8 development subset. This
estimates an overlap-restricted contrast, not the total practical benefit of
recruiting additional subjects with new labels. Actual contributing subjects
may be fewer than nominal P, especially after removing newly available classes.

Each run saves selected cell and subject IDs, quota counts, per-class donor
coverage, class names, a manifest hash, and OOF patient confusion matrices.
Treat these manifests as research data; do not publish identifying metadata.

## Inference

Compute subject-balanced pooled Macro-F1 separately per seed, then average seed
scores. Joint bootstrap draws seed multiplicities and subject multiplicities;
reuse both across P, conditions and models. Recompute F1 before averaging seeds.
Report P32 minus P8 and paired A-B, B-C, C-D differences of these effects.
The positive bootstrap fraction is not a Bayesian posterior probability or
a multiple-testing-corrected p-value. Intervals are pointwise, exploratory and
conditional on the existing folds, cohort and fitted models. Complete five-fold
results for every seed are required; aggregation refuses partial results.
Contrasts depend on control order and are not causal mediation contributions.
Do not use an attenuation ratio with a near-zero unmatched effect.

## Commands

```bash
python -m unittest discover -s analysis -p test_sequential_scaling_controls.py -v
bash analysis/run_sequential_scaling.sh apt 0 4
# Run other shards 1,2,3; repeat for combat_rna and onek1k where data exist.
python analysis/sequential_scaling_controls.py aggregate --dataset apt
python analysis/simulate_sequential_controls.py --seeds 20 --output outputs/sequential_scaling_simulation/results.json
```

## Mechanism check

The batch launcher accepts `SHARDS` (default 2) and `XGB_DEVICE` (default CPU).
For the fresh vllab13 run, `CUDA_VISIBLE_DEVICES=4 XGB_DEVICE=cuda:0 SHARDS=2`:
all XGBoost fits use the same GPU backend, while sklearn LR/MLP remain on CPU
on that host. The device is part of the protocol fingerprint; every XGBoost
fit validates its booster device and aborts rather than silently falling back.
GPU work shares the card with existing users; acceleration is not guaranteed.

The separate simulation is synthetic, not semi-synthetic: six Gaussian class
centers, 48 donors, 400 cells per donor, 12 features. Vary donor-wide additive
offset (0/1.5), donor label-mixture heterogeneity (0/1.5), and class separation
(0.5/1.5). Use 32 development and 16 untouched test donors, total 1,600 cells,
20 independent seeds and unweighted LR. It tests a limited mechanism family;
it does not establish a real-data causal effect, universal bias correction or
interval coverage. Zero offset and fixed mixture give exchangeable donors.
Neither a positive effect nor a reversal is enforced. Empirical seed intervals
describe run variability, not confidence intervals of the mean.
