# Remaining experiments: frozen implementation plan

Baseline commit: 336e087. These are new exploratory experiments, not replacements
for old seeds or selected favorable results. All raw patient outputs stay on SSD.

## E1 full-cell MLP

Three seeds 20260912/13/14, all five frozen outer folds, coarse and fine separately.
All development cells, provided 293-feature APT expression and train-only z-score.
MLP 256/128 ReLU, dropout 0.1, AdamW lr .001, unweighted cross-entropy,
batch 1024, up to 50 epochs. Inner validation is the next frozen patient fold.
Choose decay from 1e-5/.001 and epoch by inner SB pooled Macro-F1, earliest tie.
Refit from scratch on all 32 development patients for the selected epoch count.
No disease-label loss. This matches data budget, not optimizer or supervision.
Save checkpoint, preprocessing, cell IDs, ontology columns, probabilities and CMs.
Report all seeds and joint seed/patient bootstrap, not an ensemble selected on test.

## E2 true-lineage oracle

Mask fine scores to children of the true lineage, then argmax. Evaluate every test
cell, not only lineage-correct cells. Save ordinary and oracle confusion matrices
on identical IDs; report paired differences. Run for E1 and for old checkpoints
only when class order, mapping, preprocessing and prediction reconstruction pass.
This diagnostic has privileged labels and is not a deployable model.

## E3 repeated modality controls and RNA width

Keep the existing frozen cell subsample and train/test folds (1000 training cells
per patient), separate training and shuffle RNGs, and report three paired seeds.
Repeat all existing modes for SGD and MLP, rerunning inner alpha selection.
Use HVG=2000 and an additional fixed HVG=5000 sensitivity, each selected on the
corresponding inner/outer training cells only. Width is a reported sensitivity,
not a test-selected winner. MLP sklearn recipe unchanged; do not relabel a new
GPU implementation as a repeat of the old estimator. Use CPU for these repeats.
Intervals resample paired fitted-seed replicates and patients; only three seeds
provide a limited estimate of fit variability, not population-training uncertainty.

## Execution

Host vllab11 through vllab7, SSD root /ssd3/mayiling/apt_agent_runtime.
GPU 4 for E1; CPU 4 threads for E3. Inspect available capacity before launch.
No deletion or restart of old experiments. New outputs under remaining_v1.
Do not add uncompleted outputs to paper tables. Aggregate only complete grids.
