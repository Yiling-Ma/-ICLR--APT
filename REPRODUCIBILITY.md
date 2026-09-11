# Reproducibility status

## What is available

- Immutable outer patient partition: `benchmark/splits/patient_folds.json`.
- A structural copy of the 27-to-5 lookup: `benchmark/release_audit/subtype_to_lineage.csv`.
- Selected result/source SHA256 hashes: `benchmark/release_audit/checksums.json`.
- Current local audit environment: `benchmark/release_audit/local_analysis_environment.json`.
  This is not a lockfile or a claim about original training package versions.
- Claim/result inventory: `CLAIM_ARTIFACT_MAP.md`; archived panel detail: `archive/README.md`.

## Verified artifact-level commands

From the repository root, using Python with numpy, pandas, and matplotlib:

```sh
python analysis/audit_release_inventory.py
make paper
```

These commands validate selected saved artifacts and compile the paper PDF.
They do not retrain models or regenerate every result from
raw observations. The paper build requires LaTeX/latexmk.

The revised main table uses subject-balanced re-evaluation of frozen full-budget
OOF predictions, not the historical cell-weighted metric. Reproduction commands,
source checks, and remaining full-budget MLP coverage are documented in
`BENCHMARK_STORY_REVISION.md`. Run `analysis/revise_benchmark_table.py` to rebuild
the current main table; `analysis/generate_hierarchy_main_table.py` and
`analysis/render_paired_rna_oracle.py` are legacy generators and should not be
used to overwrite the revised tables.

Sampling analyses have been removed from the paper and preserved in a
standalone report. Build it independently with `make sampling-report`;
its PDF is `output/pdf/APT-Bench_Sampling_Report.pdf`. Figure rebuilding uses
`python analysis/plot_scaling_protocol_contrast.py`. Existing experiment
artifacts are retained; moving them does not constitute new validation.

The self-contained simulation can be rerun separately with numpy and scikit-learn:

```sh
python analysis/validate_scaling_ground_truth.py --output outputs/scaling_ground_truth_rerun
```

Defaults are 512 reference and 100 evaluation cohorts per scenario; this is
9,744 LR fits and is not part of the quick artifact check. Frozen simulation
settings, seeds, source hashes, and per-replicate results are in
`outputs/scaling_ground_truth_validation/`.

## What remains unavailable or unverified

Complete raw/processed-data access, licenses, anonymous review hosting, archival
DOI, original RNA-QC software/thresholds, full historical training environment
locks, and a portable raw-data-to-all-results command remain open. Neither
checksums nor a public code repository grants permission to redistribute data.
The independent biological validity of the exported hierarchy remains unverified.
Do not conflate a successful artifact rebuild with complete benchmark reproduction.

The outer fold membership is shared, but training/validation use is model-specific:
Transformer checkpoint selection uses its documented validation fold; classical
and nested disease/scaling protocols follow their own development-only rules.
See the paper and per-experiment protocol JSON files rather than assuming that
every model trains on the same number of development patients.
