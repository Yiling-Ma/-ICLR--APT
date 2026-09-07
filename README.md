# APT-Bench

This repository contains the APT-Bench ICLR 2027 manuscript, analysis code, and
selected result artifacts. APT-Bench studies patient-disjoint hierarchical cell
typing, patient-level disease recognition, and measurement efficiency from
single-cell aptamer profiles.

The formal task and split contract is documented in
[`benchmark/README.md`](benchmark/README.md). The current repository is not yet a
complete public benchmark release: dataset access, an end-to-end evaluator,
environment locking, archival versioning, and licenses must be finalized before
submission.

## Build the Paper

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

The compiled manuscript is also stored at
`output/pdf/APT-Bench_ICLR2027.pdf`.
