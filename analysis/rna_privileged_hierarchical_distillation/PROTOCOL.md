# Frozen protocol: RNA-Privileged Hierarchical Distillation (RPH-Distill)

Frozen before new RPH outer-test predictions. Existing outer folds have been
examined repeatedly; this is a targeted exploratory follow-up, not untouched
confirmatory evaluation. Every predeclared condition is retained regardless of
outcome. Deployment always uses 293-dimensional APT only.

## Partitions and preprocessing

- Benchmark: 361,792 annotated paired cells, 40 patients, five immutable
  patient folds, 27 subtypes and five lineages.
- For outer fold `f`, test is `f`, validation is `(f+1)%5`, inner training is
  the remaining 24 patients, and final refit uses all 32 development patients.
- APT: barcode-aligned raw 293-feature counts, log1p, then StandardScaler fitted
  on inner training or outer development only.
- RNA teacher: all cells and 5,000 HVGs. Per-cell library normalization to
  10,000 and log1p are cell-local; mean-binned dispersion HVG selection and SD
  scaling without centering are fitted separately on inner training and outer
  development. Gene lists and aligned cell indices are saved per fold/stage.
- The teacher never receives outer-test cells during fitting. Test RNA is used
  only for teacher diagnostics, never to create student training targets.

## Architectures and optimization

RNA teacher: 5000--256--128 shared encoder, dropout 0.1, 27-way fine and 5-way
coarse heads. Its objective is fine CE + 0.5 coarse CE. APT student: the exact
unified_v2 293--256--128 shared encoder and the same heads/base objective.
Teacher parameters are frozen for distillation. Neural work must run on CUDA;
CPU fallback is forbidden. CPU is limited to sparse preprocessing, I/O and
aggregation.

Teacher selection uses seed 17, AdamW, batch 1,024, maximum 50 epochs, patience
8, earliest tie, and learning-rate {3e-4,1e-3} by weight-decay {1e-5,1e-3}.
Fine validation SB-Macro-F1 selects configuration and epoch. Student base
optimizer settings reuse the committed unified_v2 Plain MLP fold selection;
each RPH condition selects its epoch and specified loss weights on inner
validation only. Final teacher/student refits use matched seeds 17,29,43.

## Losses and staged selection

Every student uses fine CE + 0.5 lineage CE. Routing KD is
`T_route^2 KL(q_R || q_A)` over five lineages. Within-lineage KD restricts both
27-way logits to children of the true training lineage and uses
`T_sub^2 KL(p_R || p_A)`. Cosine alignment is the mean of one minus cosine
between paired, L2-normalized 128-dimensional student and frozen teacher
representations.

Conditions, in fixed order:

1. `plain`: base objective.
2. `route`: choose T_route in {1,2,4} at lambda_route=0.25, then choose
   lambda_route in {0.1,0.25,0.5,1.0} at selected temperature.
3. `within`: analogous staged T_sub/lambda_within search.
4. `align`: lambda_align in {0.01,0.05,0.1,0.25}.
5. `route_align`: reuse selected route and align weights/temperatures.
6. `route_within`: reuse selected route and within settings.
7. `full`: reuse all three selected components.
8. `full_shuffled`: reuse Full RPH settings and epochs, but permute the complete
   teacher logits and representation vectors within patient x true-lineage in
   each training partition. APT and hard labels stay fixed; no cross-patient or
   cross-lineage movement is allowed.
9. `strong_lineage`: label-only control with lineage CE weight 1.0, reusing the
   Plain optimizer grid and validation-only epoch selection.

Combination conditions perform one validation run to select epoch only; they
do not reopen component grids. Full-shuffled reuses paired Full settings and
epoch count and receives no separate selection advantage. Optional InfoNCE,
hierarchical supervised contrastive loss, confidence weighting, expert heads
and random alignment are excluded from this initial frozen experiment.

## Baseline gate

Plain final fits are completed and aggregated first. Their Fine SB-Macro-F1
must be within 0.01 absolute of the committed unified_v2/HBM reproduction
(approximately 0.136--0.137); otherwise all RPH outer refits stop for debugging.

## Metrics and statistics

For every condition: fine/coarse subject-balanced pooled Macro-F1, overall
cross- and within-lineage error, cross-lineage fraction among subtype errors,
fine outside true parent conditional on coarse correct, true-lineage oracle
Fine SB-Macro-F1, and oracle-minus-deployable gap. A smaller gap caused by a
worse oracle is not mitigation.

Primary comparisons use 5,000 paired bootstrap draws (RNG 20260915) with the
same held-out-patient draw and matched seed slots across conditions. Report
paired differences for Fine and Coarse SB-F1, overall cross-lineage error and
hierarchy gap. Intervals are pointwise and conditional on frozen folds, finite
seeds and completed validation selections; selection is not repeated.

Secondary representation diagnostics compare Plain and Full RPH using
development-fitted lineage/subtype probes, within-lineage nearest-neighbor
purity and patient-ID probes. Retrieval diagnostics are restricted to held-out
inner-validation cells during selection. These diagnostics cannot select the
reported outer models.

## Interpretation

Strong mitigation requires a paired-positive Fine-F1 interval, lower
cross-lineage error, no oracle-degradation explanation, and better performance
than shuffled-teacher control. Partial mitigation requires a modest deployable
gain with at least some concordant bottleneck improvement. Otherwise conclude
that RNA-privileged supervision does not resolve the bottleneck under this
protocol. RNA is privileged training information, so RPH belongs to the
RNA+APT-training/APT-inference track and is never described as APT-only training.
