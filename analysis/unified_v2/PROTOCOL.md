# Unified APT reference recipes v2

Frozen before v2 outer predictions; existing folds have already been inspected.
This is an exploratory follow-up, not an untouched confirmatory evaluation.

Input: barcode-aligned ALL_PBMC_APT_Cell_matrix raw nonnegative counts,
293 columns in immutable source order; log1p then StandardScaler fitted on
inner training, refitted on outer development. Hash raw matrix and barcodes.
No supplied apt_expression values are used. Fixed five folds: test f,
validation (f+1)%5, inner training remaining 24 patients, final refit all32.
All available cells; unweighted CE. Fixed ontology 27/5; zero denominator=0.

Seeds: selection17, final17/29/43. Selection performed once per fold/model;
seed intervals condition on that choice. No replacement seeds or significance
stopping. Main models: LR, XGBoost, MLP, Flat, HCE, Cascade without disease.
Additional ablation models: Flat+contrastive and Cascade+contrastive.

LR has independently selected coarse/fine tasks, C=.01,.1,1,10, max_iter1000,
no class weights. XGBoost independently selects coarse/fine, depth3/6,
learning_rate .03/.1, up to500 trees selected at checkpoints100/300/500,
hist CPU4 threads. Refit exact chosen tree count. LR one deterministic fit
per fold/task; XGB three seeds, subsample/colsample .8.

Neural models: joint fine CE + .5 coarse CE; selected only by fine SB-F1.
Coarse is auxiliary from the same selected checkpoint, not an independently
optimized coarse task. MLP256/128 ReLU/dropout.1. All Transformers use existing
APTTransformerBackbone/tokenizer: hidden128, layers2, heads4, ff256, dropout.1.
These are budget-declared v2 configurations, not replicas of historical256/4.
Flat/Cascade share encoder and coarse/global/projection heads; Flat bypasses
experts and routing. HCE uses existing two-level reachability loss (fine1,
coarse.5), a separately identified objective, not a cascade ablation.
AdamW candidates lr .0003/.001 x decay .00001/.001; max50 epochs,
batch128 Transformer /1024 MLP, validation each epoch, patience8, earliest tie.
Fresh outer refit uses selected epochs, never test-based stopping.
For 2x2 comparisons, same grid, losses, encoder, data and selection rules;
each recipe selects its own validation optimum. Thus contrasts compare
selection procedures for whole structures, not pure routing effects.

Contrastive positives: same subtype and different disease, non-self, within
training mini-batch; all non-self examples in denominator; temperature .1,
coefficient .1. Anchors with no positives omitted; no positive anchors =>0.
Disease labels have no other role. All four variants retain projection head.

Saved-score diagnostics D0 ordinary; D1 mask to predicted coarse argmax;
D2 coarse probability times fine probability normalized within each parent;
D4 true-parent masking privileged only. MLP and Transformer reuse their joint
heads. LR/XGB paired independent task models are explicitly not shared heads.
No D3 calibration or D5 extra training in this run.

Final reports: fixed-ontology SB pooled CM, per-fit F1 then seed average.
5000 paired patient/seed bootstrap draws, RNG20260915; same draws across
conditions, no cells/folds treated as independent repetitions. Primary
descriptive contrasts Flat-MLP and Cascade-Flat. Ablation B-A,C-A,D-C and
(D-C)-(B-A) separately, no component claim from unadjusted pointwise intervals.
Hardware/time per search and refit recorded separately, no CPU/GPU ranking.
Old paper tables remain until all v2 conditions validated and summarized.
