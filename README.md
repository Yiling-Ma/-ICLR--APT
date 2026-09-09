# APT-Bench

This repository contains the APT-Bench ICLR 2027 manuscript, analysis code, and
selected result artifacts. APT-Bench studies patient-disjoint hierarchical cell
typing, two-axis patient/cell training scale, and strict cross-fitted transfer
from cell-model outputs to patient-level characterization. Compact-panel and
inference-cell analyses are secondary measurement sensitivities.

The two-axis scaling protocol is externally replicated on COMBAT CITE-seq RNA
and ADT and on OneK1K RNA. OneK1K construction, split constraints, and rerun
commands are documented in [`ONEK1K_SCALING_PROTOCOL.md`](ONEK1K_SCALING_PROTOCOL.md).
A class-matched fixed-total control additionally freezes exact per-class cell
quotas while varying the number of contributing subjects. Its runner,
summarizer, protocols, QA files, and sufficient statistics are under
`analysis/class_matched_scaling.py` and `outputs/class_matched_scaling/`.
This is the primary scaling audit: unmatched breadth comparisons are retained
as a naive diagnostic because they also change the marginal training label
distribution.
The nonlinear robustness protocol uses the same frozen class quotas with
XGBoost at `T=6,400` for 10 matched seeds. Runs are resumable and can be
sharded with:

```bash
python analysis/class_matched_scaling.py \
  --dataset DATASET \
  --output outputs/class_matched_scaling_xgb/DATASET \
  run --models xgboost --totals 6400 --seeds 10 \
  --num-shards N --shard-index I
```

The formal task and split contract is documented in
[`benchmark/README.md`](benchmark/README.md). The current repository is not yet a
complete public benchmark release: dataset access, an end-to-end evaluator,
environment locking, archival versioning, and licenses must be finalized before
submission.

## Build the Paper

```bash
make paper
```

To aggregate completed fair-scaling bundles, regenerate their machine-readable
summaries and figure, and compile the manuscript, run:

```bash
make revision
```

The revised manuscript is written to
`output/pdf/APT-Bench_ICLR2027_revised.pdf`. See `REVISION_REPORT.md` for the
executed protocols, audit decisions, completed experiments, and remaining
submission risks.
