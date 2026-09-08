# OOF Composition Bridge Protocol Audit

- Protocol: `oof-composition-bridge-base-v1` (`129c94d3690d9d81`)
- Project root: `/home/mayiling/projs/apt_agent`
- Data matrix: `361,792` cells x `293` aptamers
- Retained benchmark patients: `40`
- Cell ID unique: `True`
- Patient ID column: `sample_id`; disease column: `disease`
- Coarse label column: `coarse_subtype`; fine label column: `cell_subtype`
- Fixed taxonomy: `5` lineages and `27` subtypes
- Immutable fold file: `cell_JEPA/outputs/classical_baselines_kfold5/fold_assignment.json`
- Mapping config: `cell_JEPA/apt_jepa/configs/coarse_lineage_explicit.yaml`
- Frozen baseline config: `cell_JEPA/apt_jepa/configs/classical_baselines.yaml`
- Base-model disease supervision: `No`
- Existing pooled OOF hard labels are patient-disjoint but insufficient for primary stacking; strict nested refitting is required.
- Existing classical OOF artifacts contain hard labels only; probability refits are required.

## Patients Per Disease

| disease   |   patients |
|:----------|-----------:|
| BTC       |          6 |
| CRC       |          8 |
| GC        |          7 |
| HCC       |          7 |
| Normal    |          6 |
| PC        |          6 |

## Cells Per Patient

|       |    cells |
|:------|---------:|
| count |    40    |
| mean  |  9044.8  |
| std   |  5294.82 |
| min   |  1929    |
| 25%   |  6774.25 |
| 50%   |  7215.5  |
| 75%   |  8841.75 |
| max   | 22627    |

## Nested Design

For outer fold f, fold f is untouched outer test. For each other immutable fold g, the base model is fit on the 24 patients outside f and g and predicts fold g. These four inner-OOF blocks form disease-training features. A fifth model is fit on all 32 outer-development patients and predicts fold f.

## Ambiguities and Scope

- No batch, acquisition-run, panel-version, or processing-date field has yet been verified; technical-only analysis defaults to log cell count unless audit finds such fields.
- DropCascade uses disease-aware contrastive supervision and is secondary only.
- This command does not edit the manuscript or overwrite prior benchmark outputs.
