# Historical split audit: Route A resolution

Follow-up provenance update (Concern 3): inspection of the five original
patient-disjoint Soft-cascade `best.pt` configurations subsequently recovered
their maximum training budget as 50 epochs. The earlier unavailable statement
below was conditional on the replay `resolved_config.json` files alone.
The cell-split maximum was not reassessed. See `CONCERN3_RESOLUTION_ZH.md` and
`generated/main_protocol_audit.json`; this does not change the split-audit scores
or its descriptive scope.

## Applicability and disposition

Audited against manuscript source at 7299c5a and the subsequent changes in
this revision, not against the historical reviewer table numbering.
The abstract, contributions and main results now center prior-controlled
hierarchy reconstruction and matched-budget APT/RNA comparisons with conditional
shuffle controls. The numerical split comparison is appendix background
(B.7, Table 9 in the input PDF); it is not a primary finding. The old concern's
central-positioning premise has already been removed.

Route A is complete: retain all four historical comparisons, explicitly label
CW rather than SB, disclose unmatched allocations and missing difference
uncertainty, and remove causal-sounding "Inflation"/"splitting raises" wording.
Patient-disjoint evaluation remains the new-patient prediction contract.
The two core empirical contributions are unchanged. New training runs: **0**.
No additional seeds, bootstrap intervals, optional SB rescoring or TODO work
were required. An absent matched split experiment is not a Route A blocker.

## Recovered split facts

Counts below are from surviving manifests and cell/patient prediction IDs,
not inferred from current manuscript defaults. Patient-disjoint test sets
are the same eight patients per fold for both models, but their training
allocations differ. XGBoost uses all other 32 patients and no validation set;
Soft-cascade uses 24/8/8 train/validation/test patients.

| Fold | XGB train cells | XGB validation cells | Test cells | Soft train cells | Soft validation cells |
|---|---:|---:|---:|---:|---:|
| 0 | 294812 | 0 | 66980 | 226431 | 68381 |
| 1 | 293411 | 0 | 68381 | 207172 | 86239 |
| 2 | 275553 | 0 | 86239 | 206704 | 68849 |
| 3 | 292943 | 0 | 68849 | 221600 | 71343 |
| 4 | 290449 | 0 | 71343 | 223469 | 66980 |

Within each patient-disjoint fold all patient and cell intersections between
partitions are empty. Pooled OOF evaluation contains 361792 cells, 40 patients.
Both cell-split manifests specify 253254/54269/54269 train/validation/test
cells, all 40 patients in each partition. Every pairwise patient intersection
is 40; every cell intersection is zero. The cell manifests agree exactly.
This is subtype-stratified 70/15/15 splitting with seed 42 in the saved config.
Cell split evaluates only 54269 cells, not the common test set required for a
matched protocol comparison. Neither folds nor the four related contrasts
are independent biological replications.

XGBoost prediction cell IDs were checked against the corresponding manifests
and OOF universe. Specifically, the two historical Soft-cascade CSV exports
used here are `outputs/dropcascade_kfold5/pooled_oof_predictions.csv` and
`outputs/soft_lineage_cascade_cell_matched/test_predictions.csv`.
Both have patient and true/predicted label fields but **no cell barcode**.
This statement does not apply to all neural predictions or later oracle
replay exports. Patient counts agree with the manifests; the cell-split
ordered patient/true-label sequence also agrees with XGBoost. This does not
independently establish neural prediction-to-barcode linkage. Split-implied
cell sets are recoverable; exact neural prediction cell identities are not.

The zero cell-intersection finding is an actual check of saved cell-ID sets in
`outputs/classical_baselines_cell/cell_split_seed42.csv` and
`outputs/soft_lineage_cascade_cell_matched/split.csv`, not merely a requirement
of the split program. The audit also verified those two manifests agree.
It does not establish barcode linkage for the two old neural CSV exports;
their missing identifiers are not evidence of actual partition overlap.

The separate B.5 oracle replay writes `cell_ids` in its 15 per-fold NPZ files
under `/ssd3/mayiling/apt_agent_runtime/remaining_v1/oracle` on vllab11.
Its implementation is `analysis/export_lineage_oracle.py`; completion metadata
is retained locally in `outputs/remaining_oracle_v1/completion.json`.
These are different artifacts from the historical split-audit CSVs. The
follow-up audit inspected all 15 remote NPZ member lists and confirmed
`cell_ids.npy` is present in each, and read the saved PASS completion record;
no checkpoint was replayed or model retrained in this clarification.
The
replay recovers IDs from the keyed test loader and verifies the original
patient/label/prediction sequence; it does not retroactively add independent
barcodes to the old CSVs. This clarification changes no scores or experiments.

## Training provenance and limits

The JSON records source hashes, complete surviving neural configurations and
training-log summaries. Source files are surviving snapshots, not an immutable
copy of the original execution environment or command line. Consequently the
following implementation facts must not be promoted to stronger provenance.

| Item | XGBoost | Soft-cascade |
|---|---|---|
| Input / transform in surviving source | Provided `apt_expression`; train-only mean/std standardization, no additional log transform | Provided `apt_expression`; train-only mean/std standardization, no additional log transform |
| Training sampling | All allocated training cells in surviving fit code | Saved config disables weighted sampler and fine class weighting; mini-batch shuffle |
| Model / optimization | Fixed 400 trees, depth 6, learning rate .1, subsample .8, column subsample .8, hist method, mlogloss | Width 256, 4 layers, 8 heads, FF 512, dropout .1; AdamW lr .0001, decay .00001, batch 512, clipping 1, AMP |
| Selection | No candidate grid or early stopping in surviving code. Cell validation is scored, not used to select XGB; patient route has no validation set | Maximum validation fine CW Macro-F1 over all 27 classes, patience 10, best checkpoint replay |
| Auxiliary labels / loss | Separate coarse and fine classifiers | Fine/coarse/contrastive weights 1/.5/.2; cross-disease contrastive pairing, temperature .1; projection 128; routing strength .5, epsilon .1 |
| Seed | Cell config/split seed 42; patient source default 42, original per-run CLI not recovered | Saved configurations seed 42 |
| Original maximum training budget | Fixed tree count in surviving source; original serialized fit configuration not recovered | **Not recoverable**: saved configs have epochs=0 from evaluation replay, whereas logs show training |

Neural patient folds log 32, 36, 39, 23, 41 epochs, with best validation epochs
22, 26, 29, 13, 31. The cell run logs 45 epochs, best at 35. Logged epoch counts
are observed training lengths, not the missing original maximum. Configs also
record warmup 5 and contrastive/routing ramps 10. Zero-epoch replay must not be
interpreted as zero training. No original exhaustive tuning/search history,
immutable package environment, original upstream APT transformation provenance
or saved numerical scaler state was recovered in this audit. Train-only
standardization is supported by the surviving implementation, not a replay
verification of every original fitted statistic. No checkpoints were unpickled.

## Scores and metric

Both columns use cell-weighted pooled Macro-F1: sum cell confusion counts,
then average class F1. All five coarse and 27 fine classes are supported in
each aggregate prediction artifact. The surviving sklearn implementation uses
`average='macro', zero_division=0`; the audit independently counts TP, truth
and predicted class totals. A zero denominator contributes zero. These are
not subject-normalized pooled F1 or mean within-patient F1.

| Task / model | Patient-disjoint CW | Cell CW | Cell minus patient CW |
|---|---:|---:|---:|
| Coarse XGBoost | 0.32204425863573916 | 0.3715577728673055 | 0.04951351423156636 |
| Coarse Soft-cascade | 0.32649451937506957 | 0.4168739525617857 | 0.09037943318671615 |
| Fine XGBoost | 0.14344618335716552 | 0.21249910211544856 | 0.06905291875828304 |
| Fine Soft-cascade | 0.15219459584616507 | 0.26041526328055903 | 0.10822066743439396 |

The table renderer computes differences from these unrounded recomputations,
then displays three decimals. No precision was invented from rounded paper
values. No protocol-difference uncertainty has been estimated. Adding a
bootstrap would not repair unmatched training, validation or test allocations.
No exact proportion of performance loss, pure leakage effect, information
ceiling or independent-repeat claim is made.

## Sources and reproducibility

Host: `vllab7.ucmerced.edu`; artifact root:
`/home/mayiling/projs/apt_agent/cell_JEPA`.

- XGBoost patient: `outputs/classical_baselines_kfold5/{coarse,fine}/xgboost/pooled_oof_predictions.csv` and `fold_assignment.json`.
- XGBoost cell: `outputs/classical_baselines_cell/cell_split_seed42.csv`, task-specific `test_predictions.csv` and `test_metrics.json`.
- Neural patient: `outputs/dropcascade_kfold5/pooled_oof_predictions.csv`, each fold's `split.csv`, `resolved_config.json`, `training_log.csv`.
- Neural cell: **`outputs/soft_lineage_cascade_cell_matched/`**, not the older, different `soft_lineage_cascade_cell/` run.
- Training code: `apt_jepa/scripts/train_classical_baselines.py`, `train_classical_baselines_kfold.py`, `train_soft_lineage_cascade.py`; data preprocessing/dataset/split modules and classical cell YAML config.

`generated/historical_split_audit.json` contains aggregate counts, set digests,
all four unrounded contrasts, configs, logs, prediction headers and artifact
SHA-256 hashes. No cell IDs are exported. Run the read-only audit on the host:

```sh
python3 analysis/audit_historical_splits.py /home/mayiling/projs/apt_agent
python3 analysis/render_historical_split_audit.py
python3 -m unittest discover -s analysis -p test_historical_split_audit.py
latexmk -pdf -interaction=nonstopmode -halt-on-error iclr2027_conference.tex
```

The first command emits the JSON; save stdout to the generated JSON path before
rendering locally. Assertions verify partition isolation, source membership,
four prediction-derived scores and saved cell metrics. Local tests verify the
retained rows, score rendering, allocation differences and missing-ID/epoch
limitations. Source/PDF checks cover abstract, contribution list, results,
discussion, conclusion, captions and appendix; legacy standalone reports and
version records are not current manuscript claims.

## Remaining boundary

Final verification: the remote read-only artifact audit passed, all three
dedicated local audit tests passed, and `git diff --check` passed. LaTeX
compiled to 19 pages without undefined references or overfull-box warnings.
Rendered pages 3 and 19 were visually checked. PDF text search places the
numerical historical split comparison on page 19; page 4 only points to the
supporting appendix. An optional legacy leaderboard-test rerun could not
import its dependencies (bundled Python lacks sklearn; system Python lacks
pandas). This unrelated test was not reported as passing; its scoring code
was not changed by this revision.

This closes the descriptive split-audit concern, not unrelated release or
biological confounding TODOs. The original study is not retroactively made a
matched, multi-seed protocol experiment. Those limits remain visible, and do
not prevent retaining the historical comparison as appendix background.
