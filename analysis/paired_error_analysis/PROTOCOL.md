# Paired RNA / RNA+APT Error Analysis

Post-hoc diagnostic protocol, frozen before inspecting the new transition and
feature results. This is analysis only: no retraining, model selection, or paper
claims are changed. Use all saved fine-label predictions for 2,000/5,000 HVGs,
SGD and MLP, three training seeds, and the same five patient-disjoint folds.

## Error Accounting

Publish every directed true-subtype -> predicted-subtype edge, not only favorable
examples. Separate rescued errors, newly introduced errors, and wrong-to-wrong
rerouting. Validate the edge conservation identity. Cell counts are descriptive
three-seed means, not independent replicated cells. Confusion rates first divide
each patient's row by the number of cells with that true subtype, then average
eligible patients, then seeds. These rates are NOT pooled Macro-F1.

Intervals resample training seeds and held-out patients jointly with paired draws
across models' predictions. Report 2,000 pointwise percentile bootstrap replicates;
they are exploratory, not multiplicity-adjusted discoveries. Missing true classes
are excluded per patient, not scored as zero. No inferential claim is made for
an individual transition selected from the resulting rankings.

## Cell-Level Feature Linkage

The primary feature analysis is the 2,000-HVG MLP, with 5,000-HVG MLP replication.
Every cell is linked by verified barcode to predictions, RNA markers, all 293 APT
features, and available QC. Store private arrays on the remote local disk only.
Publish aggregates only, not individual barcodes or patient identities.

Compare rescued vs persistent RNA errors within patient x true subtype x RNA
prediction; compare newly harmed vs retained-correct cells within patient x true
subtype. Require at least five cells in each group. Average eligible strata
equally within each patient, then eligible patients and seeds equally. Report all
features and exclusions. Additionally publish edge-specific contrasts where at
least five patients qualify in at least one seed; selection is explicitly post hoc.

RNA marker expression is log1p(count/library_size*10,000). APT is reported both
as log1p(raw count) and library-normalized log1p counts (sensitivity to total APT
yield). QC includes RNA total counts, detected genes, available mitochondrial
fraction, APT total counts and detected probes. Use raw-scale mean differences;
do not call these feature importance or causal effects. Feature categories have
different units and must not be ranked together by unscaled magnitude.

The small broad-PBMC marker panel is contextual, not a complete fine-subtype
annotation handbook. Source: [Seurat PBMC tutorial](https://satijalab.org/seurat/articles/pbmc3k_tutorial.html).
Missing markers must be listed. Aptamer IDs do not imply known protein targets.
QC associations do not identify acquisition batches or resolve confounding.
Outcome-conditioned comparisons are subject to selection effects; marker
associations do not validate RNA-derived annotations independently.

## Outputs

Full transition and feature CSVs; seed-level subtype summaries; confusion-change
heatmaps; bidirectional correction plots; a Chinese interpretation with limitations;
and an alignment/provenance audit. Private cell features and per-seed outcomes
share an identical cell-row index, allowing exact one-to-one follow-up.
