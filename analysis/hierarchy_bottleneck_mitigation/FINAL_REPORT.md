# HBM-MLP final research report

## Scientific answer

**No, not under the tested methods.** The hierarchy bottleneck remains
actionable as a precisely measurable failure mode, but parent-mass
supervision, coarse/fine consistency, within-lineage privileged distillation,
and their combinations did not convert it into a statistically resolved
deployable APT-only gain under the frozen protocol.

## What was implemented

The deployable student retains the exact unified_v2 MLP backbone
(293--256--128, ReLU, dropout 0.1) and the existing 27-way fine and 5-way
coarse heads. Its default prediction is `argmax softmax(fine_logits)` and its
inference API accepts APT only.

- **Parent-mass loss:** stable `logsumexp` aggregation of the 27-way fine head
  into five parent probabilities, followed by NLL on the true training
  lineage. This directly constrains the fine head rather than only the coarse
  head.
- **Coarse/fine consistency:** symmetric KL between the explicit coarse-head
  distribution and the parent distribution implied by the fine head. Neither
  side is detached.
- **Oracle teacher:** a fold-specific MLP with a 16-dimensional true-lineage
  embedding. Its logits are normalized only among children of the true
  training lineage. The teacher is frozen before student distillation.
- **Within-lineage KD:** temperature-scaled `T^2 KL(teacher || student)` after
  masking and renormalizing both heads among the true lineage's children.
- **Representation distillation:** not run; it was optional and was kept out
  of the primary experiment so the tested claim remains about the specified
  hierarchy losses rather than an additional representation objective.

The optional soft inference calibration and selective prediction analyses
were also not run. They were secondary analyses, and the primary method did
not first establish a routing improvement that would justify expanding the
outer experiment family.

## Exact protocol and leakage audit

- 361,792 annotated cells, 293 raw APT features, and the frozen 5-lineage / 27-
  subtype ontology.
- Barcode-aligned raw sparse counts, `log1p`, then `StandardScaler` fit on the
  24-patient inner training partition for selection or the 32-patient outer
  development partition for final refit.
- Frozen five patient-disjoint outer folds; validation fold `(outer + 1) % 5`;
  no patient overlap. The five test folds cover every benchmark cell exactly
  once and all 40 patients.
- Fine subject-balanced pooled Macro-F1 is the selection and primary reporting
  metric. Selection uses seed 17 only; final refits use seeds 17, 29, and 43.
- Unified_v2's batch size 1024, AdamW learning-rate/decay selections, maximum
  50 epochs, patience 8, and earliest strict validation maximum are retained.
- The Plain MLP uses the committed unified_v2 fold-specific selections with
  their full histories and SHA-256 provenance, followed by fresh outer refits.
- Teacher selection sees inner-train/validation patients only. Each final
  teacher sees the outer-development patients for its fold only. Checkpoints
  are fold/seed specific; scaler equality and dev/test patient disjointness are
  asserted before KD.
- Teacher targets are produced only for student training cells. The teacher
  and true lineage are never passed to the deployable student at test
  inference. Test labels are used only to construct post-hoc metrics and the
  explicitly non-deployable oracle diagnostic.
- Loss weights and temperatures are selected on inner validation patients.
  No outer-test score changed the staged grids or any selected setting.

The fresh Plain MLP reproduction obtained fine SB-Macro-F1 0.136010 versus
0.137049 in the committed unified_v2 artifact (difference -0.001039). Its
coarse/oracle scores were 0.334825/0.378361 versus committed
0.332805/0.375571. Architecture equivalence, fold assignments, preprocessing,
ontology, and OOF coverage passed direct tests; the small numerical difference
is retained as run-to-run GPU variation rather than replaced by the historical
score.

## Ablation results

All entries are means over matched seed slots. “Cross-lineage error” is the
patient-balanced fraction of all predictions that are wrong and cross the true
parent. “Cross among errors” and “within among errors” partition subtype
errors. Oracle uses true-lineage masking of the same trained student's fine
scores and is not deployable.

| Model | Fine SB-F1 | Coarse SB-F1 | Cross among errors | Within among errors | Overall cross error | Fine outside parent given coarse correct | Oracle F1 | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain MLP | 0.1360 | 0.3348 | 0.7438 | 0.2562 | 0.4909 | 0.1756 | 0.3784 | 0.2424 |
| + parent-mass | 0.1360 | 0.3342 | 0.7467 | 0.2533 | 0.4963 | 0.1787 | 0.3784 | 0.2424 |
| + parent-mass + consistency | 0.1390 | 0.3371 | 0.7460 | 0.2540 | 0.4947 | 0.1784 | 0.3797 | 0.2407 |
| + KD | **0.1400** | 0.3353 | 0.7433 | 0.2567 | 0.4920 | **0.1723** | **0.3838** | 0.2438 |
| + parent-mass + KD | 0.1390 | 0.3324 | 0.7440 | 0.2560 | 0.4925 | 0.1784 | 0.3753 | 0.2364 |
| Full HBM | 0.1375 | **0.3365** | 0.7458 | 0.2542 | 0.4954 | 0.1797 | 0.3721 | **0.2346** |

### Paired comparisons with the reproduced Plain MLP

Intervals use 2,000 matched draws over training-seed slots and held-out
patients. They are paired differences, not comparisons of marginal intervals.

| Condition | Delta fine SB-F1 (95% CI) | Delta overall cross error (95% CI) | Delta hierarchy gap (95% CI) |
|---|---:|---:|---:|
| Parent-mass | -0.00005 [-0.00580, 0.00558] | +0.00542 [-0.00282, 0.01410] | +0.00007 [-0.00683, 0.00494] |
| Parent-mass + consistency | +0.00299 [-0.00264, 0.00818] | +0.00381 [-0.00376, 0.01172] | -0.00163 [-0.00822, 0.00357] |
| KD | +0.00402 [-0.00182, 0.00870] | +0.00107 [-0.00541, 0.00678] | +0.00144 [-0.00618, 0.00764] |
| Parent-mass + KD | +0.00295 [-0.00125, 0.00791] | +0.00157 [-0.00757, 0.01215] | -0.00599 [-0.01281, 0.00200] |
| Full HBM | +0.00152 [-0.00245, 0.00600] | +0.00446 [-0.00052, 0.00972] | -0.00775 [-0.01699, 0.00052] |

No fine-score paired interval excludes zero. The best new fine point estimate,
KD-only at 0.1400, remains below the committed Flat/HCE/Soft-cascade values
0.1471/0.1463/0.1482 and does not reduce overall cross-lineage error.

## Negative-result diagnosis

1. **Lineage routing remains dominant.** About 74--75% of subtype errors remain
   cross-lineage under every condition. Neither parent-mass nor Full HBM lowers
   the overall cross-lineage error rate.
2. **Consistency does not repair the diagnosed disagreement.** Adding
   consistency raises the fine point estimate by 0.0030, but
   fine-outside-parent given coarse-correct changes from 0.1756 to 0.1784 and
   overall cross error from 0.4909 to 0.4947.
3. **KD shows within-lineage signal without global routing improvement.** It
   has the best ordinary point estimate (+0.0040) and raises oracle F1 by
   +0.0055, while overall cross error is essentially unchanged (+0.0011). The
   hierarchy gap therefore grows slightly rather than closes.
4. **A smaller gap can be misleading.** Full HBM reduces the gap by 0.0077,
   but its ordinary score rises only 0.0015 while oracle F1 falls from 0.3784
   to 0.3721. This is not successful conversion of oracle signal into
   deployable performance.

## Paper recommendation

Use framing **(1) bottleneck discovery only**, with the HBM study reported as a
targeted negative-result/robustness experiment, preferably in the main results
if space permits and with the full ablation in the appendix. Do not claim
partial or strong mitigation. The result strengthens the narrower conclusion
that the diagnosed bottleneck is non-trivial and is not solved by direct
parent-mass supervision, coarse/fine agreement, or straightforward
privileged-information distillation.

### Suggested Results wording

> We next asked whether the true-lineage oracle gap can be converted into a
> deployable APT-only improvement. We augmented the unified MLP with a loss on
> parent probability mass implied by its 27-way fine head, symmetric agreement
> between explicit and fine-implied lineage distributions, and within-lineage
> distillation from a fold-specific teacher conditioned on the training
> lineage. Hyperparameters were selected on inner validation patients and all
> final comparisons used matched seed and patient bootstrap draws. The best
> point estimate was obtained by distillation alone (fine SB-Macro-F1 0.140
> versus 0.136 for the reproduced MLP; paired difference +0.004, 95% interval
> [-0.002, 0.009]), but cross-lineage error did not decrease. The full objective
> reached 0.138 (difference +0.002, [-0.002, 0.006]) and likewise did not reduce
> cross-lineage errors. Thus, under the tested objectives, the oracle signal
> was not converted into a resolved deployable gain.

### Suggested Contributions wording

> We identify and quantify a hierarchy bottleneck: APT contains recoverable
> within-lineage subtype information under true-lineage decoding, yet most
> deployable subtype errors cross lineage boundaries. A targeted study of
> fine-head parent-mass supervision, coarse/fine consistency, and
> training-only privileged distillation fails to close this gap under
> patient-disjoint evaluation, establishing that the bottleneck is not removed
> by these straightforward hierarchy-aware objectives.

These statements do not describe the oracle as deployable, posit a biological
mechanism, claim an assay information ceiling, or generalize to hierarchical
cell typing beyond this benchmark and tested model family.

## Artifacts

The full run is stored at `/ssd2/mayiling/apt_hbm_v1` on
`mayiling@vllab7.ucmerced.edu`. It contains fold/seed selections, histories,
scalers, checkpoints, probabilities, predictions, cell/patient IDs, labels,
confusion matrices, diagnostic counts, protocol/cache audits, committed
historical score sources, and a source snapshot with SHA-256 hashes. The
`rebuild_results.sh` command regenerates all result tables, paired intervals,
and figures from saved predictions without retraining.
