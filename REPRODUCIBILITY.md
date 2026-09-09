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
python analysis/plot_scaling_protocol_contrast.py
make paper
```

These commands validate selected saved artifacts, regenerate Figure 1, and
compile the PDF. They do not retrain models or regenerate every result from
raw observations. The paper build requires LaTeX/latexmk.

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
