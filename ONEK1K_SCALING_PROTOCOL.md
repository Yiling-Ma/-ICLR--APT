# OneK1K External Scaling Replication

## Purpose

OneK1K tests whether the subject-coverage effect transfers to a large healthy
PBMC cohort, where it cannot be attributed to balancing multiple disease
groups. It is an external scaling replication, not an additional APT benchmark
or a validation of disease recognition.

## Frozen Cohort Construction

- Source: CZ CELLxGENE OneK1K H5AD, dataset version
  `1e44db10-b572-46cc-adae-dcc7acd44ca6`.
- Exclude cells labeled `Doublet`.
- Reserve one complete multiplexing pool for calibration; select ten donors
  from that pool for unsupervised feature ranking. No calibration-pool donor is
  used for model fitting or evaluation.
- Retain non-calibration donors with at least 800 eligible cells. This yields
  925 donors and 1,199,468 cells while supporting every grid point exactly.
- Use the released 30-class `predicted.celltype.l2` annotation as the fine
  target. Map it deterministically to T, B, NK, Myeloid, and Other lineages.
- Select 293 protein-coding, non-mitochondrial, non-ribosomal genes by variance
  on calibration donors only.

## Splits and Scaling Grid

- Construct five folds that are disjoint in both donor and multiplexing pool.
- Stratify group-aware fold assignment and nested development-donor sampling by
  sex; disease is constant because OneK1K is a healthy cohort.
- Use `P={8,16,32}` training donors and `C={100,200,400,800}` uniformly sampled
  cells per donor.
- The exact fixed-total comparisons shared with APT and COMBAT are
  `T={3,200,6,400}`, comparing `(P,C)=(8,400)` with `(32,100)` and `(8,800)`
  with `(32,200)`.
- Omitting `C=1,600` is prespecified from the coverage audit: only 116 of 981
  raw donors reach that depth, so requiring it would discard most of the
  cohort and condition the replication on unusually high cell recovery.
- Reuse 20 nested donor/cell subset seeds, training-only standardization, all
  untouched test-donor cells, and 2,000 paired donor-bootstrap replicates.
- Refit the frozen Logistic Regression and 400-tree XGBoost baselines for
  coarse and fine prediction at every grid point.

## Primary Readout

The primary effect is the change in subject-balanced, donor-disjoint OOF
Macro-F1 when the same total number of training cells is reallocated from 8 to
32 donors. Pooled-cell and mean-within-donor Macro-F1, empirical subset-seed
intervals, matched doublings, and response-surface coefficients are
sensitivities. Absolute OneK1K F1 is not compared with APT because the feature
space and fine ontology differ.

## Reproduction

```bash
python analysis/prepare_onek1k.py \
  --input /path/to/onek1k.h5ad \
  --output-dir /path/to/onek1k_prepared

ONEK1K_PREPARED_DIR=/path/to/onek1k_prepared \
  python analysis/onek1k_scaling.py plan

ONEK1K_PREPARED_DIR=/path/to/onek1k_prepared \
  python analysis/onek1k_scaling.py run \
  --num-shards <N> --shard-index <0..N-1>

ONEK1K_PREPARED_DIR=/path/to/onek1k_prepared \
  python analysis/onek1k_scaling.py aggregate

python analysis/summarize_fair_patient_cell_scaling.py \
  --output-dir outputs/onek1k_scaling
```
