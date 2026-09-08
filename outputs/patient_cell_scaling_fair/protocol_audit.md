# Patient-Cell Scaling Protocol Audit

Generated: 2026-09-08T18:47:40.788468+00:00
Project root: `/home/mayiling/projs/apt_agent`
Repository commit: `444694221c8df8c7b3e78656faea7396bd71cf9b`

## Dataset

- Expression: `/home/mayiling/projs/apt_agent/data/apt_expression.parquet` (preferred by the existing loader; CSV fallback exists).
- Metadata: `/home/mayiling/projs/apt_agent/data/metadata.csv`; columns: `['cell_id', 'sample_id', 'disease']`.
- Cell annotation: `/home/mayiling/projs/apt_agent/data/cell_annotation.csv`; columns: `['Sample', 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'Celltypes_new']`.
- Cell ID: `cell_id`; patient ID: `sample_id`; disease label: `disease`.
- Fine subtype source column: `Celltypes_new`, renamed to `cell_subtype` by the loader.
- Coarse lineage: `coarse_subtype`, created from the explicit mapping file.
- Filtered benchmark size: 361,792 cells, 40 patients, 293 APT features.
- Evaluation taxonomy: 5 lineages and 27 subtypes.
- Missing subtype labels and the `Unknown` subtype are excluded, matching the current benchmark.

### Patients per disease

| disease   |   patients |
|:----------|-----------:|
| BTC       |          6 |
| CRC       |          8 |
| GC        |          7 |
| HCC       |          7 |
| Normal    |          6 |
| PC        |          6 |

### Cells per patient

| sample_id   | disease   |   n_cells |
|:------------|:----------|----------:|
| CRC11       | CRC       |      7469 |
| CRC13       | CRC       |      5062 |
| CRC16       | CRC       |      7478 |
| CRC17       | CRC       |      6427 |
| CRC22       | CRC       |      5203 |
| CRC23       | CRC       |      6110 |
| CRC24       | CRC       |      7254 |
| CRC25       | CRC       |      7233 |
| GC28_P_cDNA | GC        |      7320 |
| GC29_P_cDNA | GC        |      5117 |
| GC2_P_cDNA  | GC        |      4443 |
| GC30_P_cDNA | GC        |      7164 |
| GC31_P_cDNA | GC        |      7103 |
| GC32_P_cDNA | GC        |      7179 |
| GC36_P_cDNA | GC        |      7234 |
| HCC-1       | HCC       |      7055 |
| HCC-2-1_1   | HCC       |      6913 |
| HCC-2-2     | HCC       |      7179 |
| HCC-22      | HCC       |      7019 |
| HCC-24      | HCC       |      7039 |
| HCC-3-1_2   | HCC       |      7198 |
| HCC-3-2_1   | HCC       |      7285 |
| PBMC-1      | Normal    |     22144 |
| PBMC-2      | Normal    |     16302 |
| PBMC-3      | Normal    |     21398 |
| PBMC-4      | Normal    |     22627 |
| PBMC-5      | Normal    |     19153 |
| PBMC-6      | Normal    |     15164 |
| PC-1        | PC        |     17977 |
| PC-2        | PC        |      1929 |
| PC-3        | PC        |      4981 |
| PC-4        | PC        |      2060 |
| PC-5        | PC        |      3977 |
| PC-6        | PC        |      6890 |
| PRZ1        | BTC       |      8750 |
| PRZ2        | BTC       |     12428 |
| PRZ3        | BTC       |      9117 |
| PRZ4        | BTC       |      7719 |
| PRZ5        | BTC       |     12944 |
| PRZ6        | BTC       |      8748 |

### Subtype-to-lineage mapping

| cell_subtype      | coarse_subtype   |
|:------------------|:-----------------|
| Atypical Memory B | B                |
| GCB               | B                |
| Memory B-1        | B                |
| Memory B-2        | B                |
| Plasma            | B                |
| CD14 Monocyte-1   | Myeloid          |
| CD14 Monocyte-2   | Myeloid          |
| CD14 Monocyte-3   | Myeloid          |
| CD16 Monocyte-1   | Myeloid          |
| CD16 Monocyte-2   | Myeloid          |
| CD4 Monocyte-4    | Myeloid          |
| cDC2              | Myeloid          |
| pDC               | Myeloid          |
| CD16 NK           | NK               |
| CD56 NK           | NK               |
| Erythrocyte       | Other            |
| platelet          | Other            |
| CD4 + Tcm-1       | T                |
| CD4 + Tcm-2       | T                |
| CD8 + CTL-1       | T                |
| CD8 + CTL-2       | T                |
| CD8 + CTL-3       | T                |
| CD8 Tem-1         | T                |
| CD8 Tem-2         | T                |
| Cycling T         | T                |
| Naive T           | T                |
| Tcm               | T                |

## Immutable Outer Folds

Source: `/home/mayiling/projs/apt_agent/cell_JEPA/outputs/classical_baselines_kfold5/fold_assignment.json`. The file contains 40 unique patients exactly once and eight test patients per fold.

|   outer_fold | patient_id   | disease   |   n_cells |
|-------------:|:-------------|:----------|----------:|
|            0 | CRC11        | CRC       |      7469 |
|            0 | CRC16        | CRC       |      7478 |
|            0 | GC29_P_cDNA  | GC        |      5117 |
|            0 | GC30_P_cDNA  | GC        |      7164 |
|            0 | HCC-2-2      | HCC       |      7179 |
|            0 | PBMC-6       | Normal    |     15164 |
|            0 | PC-3         | PC        |      4981 |
|            0 | PRZ2         | BTC       |     12428 |
|            1 | CRC22        | CRC       |      5203 |
|            1 | CRC24        | CRC       |      7254 |
|            1 | GC31_P_cDNA  | GC        |      7103 |
|            1 | HCC-2-1_1    | HCC       |      6913 |
|            1 | HCC-24       | HCC       |      7039 |
|            1 | PBMC-1       | Normal    |     22144 |
|            1 | PC-5         | PC        |      3977 |
|            1 | PRZ6         | BTC       |      8748 |
|            2 | CRC13        | CRC       |      5062 |
|            2 | CRC23        | CRC       |      6110 |
|            2 | GC32_P_cDNA  | GC        |      7179 |
|            2 | HCC-1        | HCC       |      7055 |
|            2 | HCC-3-2_1    | HCC       |      7285 |
|            2 | PBMC-4       | Normal    |     22627 |
|            2 | PC-1         | PC        |     17977 |
|            2 | PRZ5         | BTC       |     12944 |
|            3 | CRC25        | CRC       |      7233 |
|            3 | GC36_P_cDNA  | GC        |      7234 |
|            3 | HCC-3-1_2    | HCC       |      7198 |
|            3 | PBMC-3       | Normal    |     21398 |
|            3 | PC-4         | PC        |      2060 |
|            3 | PC-6         | PC        |      6890 |
|            3 | PRZ3         | BTC       |      9117 |
|            3 | PRZ4         | BTC       |      7719 |
|            4 | CRC17        | CRC       |      6427 |
|            4 | GC28_P_cDNA  | GC        |      7320 |
|            4 | GC2_P_cDNA   | GC        |      4443 |
|            4 | HCC-22       | HCC       |      7019 |
|            4 | PBMC-2       | Normal    |     16302 |
|            4 | PBMC-5       | Normal    |     19153 |
|            4 | PC-2         | PC        |      1929 |
|            4 | PRZ1         | BTC       |      8750 |

## Preprocessing and Frozen Baselines

- Existing preprocessing fits one feature-wise mean and standard deviation on training cells only; standard deviations below 1e-6 are replaced by 1.0.
- Logistic Regression: scikit-learn `LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced')`.
- XGBoost: 400 trees, depth 6, learning rate 0.1, row/feature subsampling 0.8, histogram tree method, multiclass log-loss, random state 42.
- XGBoost does not use validation-based early stopping; the complete grid therefore uses frozen boosting rounds and no inner validation patients.
- The deterministic P=32, C=all endpoint reuses the existing pooled OOF baseline predictions rather than refitting under a different software stack; all other grid cells are newly fitted.
- The scaling experiment fits preprocessing on sampled training cells only and evaluates all cells from the untouched outer-test patients.
- Model random state remains 42; resampling seeds affect only nested patient and cell subsets.

## Existing Full-Data Results

|   accuracy |   macro_f1 |   weighted_f1 |   balanced_accuracy | task   | model               |   n_pooled_test |   n_pooled_patients |
|-----------:|-----------:|--------------:|--------------------:|:-------|:--------------------|----------------:|--------------------:|
|   0.347857 |   0.299448 |      0.363502 |            0.438205 | coarse | logistic_regression |          361792 |                  40 |
|   0.516612 |   0.322044 |      0.475743 |            0.326108 | coarse | xgboost             |          361792 |                  40 |
|   0.166466 |   0.132316 |      0.184621 |            0.222245 | fine   | logistic_regression |          361792 |                  40 |
|   0.322843 |   0.143446 |      0.277401 |            0.142691 | fine   | xgboost             |          361792 |                  40 |

## Output and Logging Conventions

- Experiment root: `/home/mayiling/projs/apt_agent/outputs/patient_cell_scaling_fair`.
- One compressed sufficient-statistics artifact and one JSON metadata record are written per fold/seed/budget/model bundle.
- Coarse and fine models are fitted independently inside a bundle; paired predictions are reduced to per-patient exact-path, root-excluded hierarchical-F1, and subtype-tree-distance sums.
- Resume logic accepts an artifact only when its status is `success`, its protocol hash matches, and all required arrays are readable.

## Missing or Ambiguous Information

- The original baseline artifact does not record package versions or the exact host used for fitting. A diagnostic refit under the current stack differed by 5.73e-4 in pooled fine Macro-F1 because LBFGS reached its frozen 500-iteration limit; the deterministic endpoint therefore imports the exact existing OOF predictions and is still checked numerically with a prespecified absolute tolerance of 5e-4.
- Peak memory is an approximate process high-water mark; it may include memory retained from earlier jobs in the same worker.
- Seed intervals quantify subset-resampling variability and are not independent-patient confidence intervals.
