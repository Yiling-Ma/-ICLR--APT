# Paired Subtype Error Analysis

Start with [Chinese report](results/REPORT_ZH.md) and
[frozen diagnostic protocol](PROTOCOL.md).

This directory contains analysis only. No paper, figures in the paper, or model
weights were changed. All 120 relevant fine-task prediction files were analyzed:
2 RNA widths x 2 models x 3 seeds x 5 folds x 2 paired inputs.

## Reproduce

Run on vllab11 with the existing private data and completed prediction grids:

```sh
Rscript analysis/paired_error_analysis/extract_markers.R \
  /home/mayiling/projs/apt_agent \
  /ssd3/mayiling/apt_agent_runtime/remaining_v1/modality_hvg2000/prepared/cells.csv \
  /ssd3/mayiling/apt_agent_runtime/paired_error_analysis_v1/private

OPENBLAS_NUM_THREADS=1 python analysis/paired_error_analysis/run.py \
  --root /home/mayiling/projs/apt_agent \
  --runtime /ssd3/mayiling/apt_agent_runtime \
  --out /ssd3/mayiling/apt_agent_runtime/paired_error_analysis_v1

python analysis/paired_error_analysis/render.py \
  /ssd3/mayiling/apt_agent_runtime/paired_error_analysis_v1/public

python -m unittest discover -s analysis/paired_error_analysis -p 'test_*.py'
```

Python dependencies: numpy, pandas, scipy; matplotlib for rendering.
R dependency: Matrix. Three unit tests passed on the experiment host.

## Private Cell-Level Linkage

Under the remote output's `private/` directory:

- `cell_index.csv`: verified cell row order, operational annotation, patient ID.
- `cell_features.npy` and `feature_names.json`: same row order; 22 contextual RNA
  genes, 5 QC columns, and 293 APT probes in two normalization views (613 columns).
- `pred_{width}_{model}_{seed}.npz`: same row order; truth, RNA prediction, paired
  prediction, rescued/harmed masks, and class names. Integer labels index `classes`.
- `markers_qc.csv`: barcode-keyed raw-RNA-derived marker/QC extraction.

Thus a cell's predictions and all measurements can be joined by row index without
retraining or re-identifying it in expression files. Do not commit these private
files, patient IDs, or barcodes to GitHub. Only `public/` aggregate files are copied
into `results/` here.

Feature associations are descriptive outcome-conditioned comparisons, not model
attributions or molecular validation. Raw APT target identities are not available.
Repeated seeds, rare-class coverage, different denominators, and pointwise versus
multiplicity-adjusted intervals are explicitly documented in the protocol/report.
