# APT-Bench

This repository contains the APT-Bench ICLR 2027 manuscript, analysis code, and
selected result artifacts. APT-Bench studies patient-disjoint hierarchical cell
typing, two-axis patient/cell training scale, and strict cross-fitted propagation
from cell predictions to patient-level characterization. Compact-panel and
inference-cell analyses are secondary measurement sensitivities.

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
