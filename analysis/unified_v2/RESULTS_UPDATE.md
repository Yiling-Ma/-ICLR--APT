# Unified v2 results and paper update

## Scope and provenance

Completed: 90 neural joint predictions, 10 deterministic LR task predictions,
and 30 XGBoost task predictions, covering all five folds and all declared seeds.
The complete OOF population is 361,792 cells from 40 patients. No predictions
were filtered to an intersection. Existing clinical/data-access TODOs are unchanged.

Source roots:
- vllab11: `/ssd3/mayiling/apt_agent_runtime/unified_v2/results` (63 predictions).
- vllab15: `/ssd2/mayiling/apt_unified_v2/results` (67 predictions).
- Combined validation root on vllab15: `/ssd2/mayiling/apt_unified_v2/combined`.

The combination copied completed artifacts, not newly fitted models. Local
`results/` retains JSON/CSV audit records and configuration histories, not
the large cell-level NPZs or checkpoints. These remain on the source/combined
hosts; this record does not claim that raw-data access is publicly complete.
Exact source hashes, stage counts and prediction hashes are in
`artifact_audit.json` and `protocol_inventory.csv`.

## HCE correction

The frozen training exporter normalized the five internal-node logits for
coarse readout; HCE instead trains joint-node subtree probabilities. This
readout discrepancy was corrected by `replay_hce_decoding.py`, using the 15
saved HCE checkpoints and saved outer scalers. Original exports are retained
in the combined run folders under `original_head_exports/` and on the original
host. Fine probabilities were reproduced within tolerance and every fine
argmax agreed; the original fine arrays are retained exactly. Only coarse,
D1 and D2 change. Checkpoint and before/after hashes and replay timings are
recorded in `hce_decoding_correction.json`. New training: **0**; additional
frozen-checkpoint inference: **15**. The frozen `run.py` hash is not rewritten.

## Reproduction order

Use the source data/environment described in PROTOCOL.md and merge the
non-overlapping run folders according to STATUS_ZH.md. Inspect each script's
CLI for the data and result root arguments, then run in this order:

1. `audit_results.py` on the combined predictions and original input data.
2. `replay_hce_decoding.py` to correct the HCE readout, preserving originals.
3. `audit_results.py` again and `summarize.py` on the corrected complete set.
4. Transfer JSON/CSV summaries into this directory's `results/`.
5. From the repository root, run `python3 analysis/unified_v2/build_paper_tables.py`.
6. Build `iclr2027_conference.tex` with latexmk.

The table builder requires PASS records for the complete summary, source audit
and HCE correction. It does not accept uncorrected HCE summary artifacts.
Five thousand paired patient/seed bootstrap replicates use seed 20260915;
fine/coarse F1 is computed separately per training seed before averaging.
LR contributes one deterministic fit and patient-only uncertainty.

## Paper mapping and interpretation

Final checks: both source hosts' worker logs contain no matches for
`Traceback`, `ConvergenceWarning`, `Error` or `ERROR`. Source audit and summary
records report PASS; Python syntax checks and `git diff --check` pass.
The compiled PDF has no undefined-reference or overfull-box warnings. New
main tables and appendix count/configuration/decoding/compute pages were
rendered and visually checked. Existing narrow-table underfull-box warnings
remain; they are not clipped content or failed compilation.

- `main_result.tex`: six common-label, unified-input reference models.
- `unified_ablation.tex`: four recipes, reusing main-table Flat/Cascade;
  disease contrastive variants are additional-supervision experiments.
- `unified_decoding.tex`: all eight recipes and D0/D1/D2/privileged D4.
- `unified_fold_counts.tex`, `unified_compute.tex`: observed stage counts,
  selection budgets and recorded host-specific times.
- `main/unified_v2_appendix.tex`: exact design and uncertainty target.
- `historical_main_result.tex`: old recipes, retained to trace earlier oracle
  and error diagnostics; not pooled with the v2 scores.

Fine SB-F1 is about 0.137 for MLP, 0.147 for Flat, and 0.148 for Cascade.
The paired Cascade-minus-Flat interval includes zero. Neither structural
superiority, component-specific effects nor equivalence is established.
The prior-controlled historical MLP and paired-RNA analyses retain their
original recipes, numbers and limitations. Their prior values must not be
subtracted from v2 oracle scores. The inherited outer folds were examined
previously, so these new comparisons are exploratory, not a new untouched
confirmatory test set.
