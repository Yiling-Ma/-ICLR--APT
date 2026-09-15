# RNA-Privileged Hierarchical Distillation

RPH-Distill uses paired RNA only while fitting an APT student. Deployment and
all reported student test predictions use 293 APT features only. The immutable
scientific contract is in `PROTOCOL.md`; do not change it after inspecting
outer-test results.

## Verified execution

Run from the repository root. Neural commands deliberately reject CPU devices.
The reported run used the CUDA-enabled environment shown below.

```bash
export RPH_PYTHON=/home/mayiling/opt/apt-ft-baselines/bin/python
export RPH_OUTPUT=/ssd2/mayiling/rph_distill_v1
export RPH_APT_CACHE=/ssd2/mayiling/rph_distill_v1/cache/raw_log1p.npy
export RPH_RNA_ROOT=/home/mayiling/experiments/apt_full_budget_rna_apt_v1/prepared/hvg5000

$RPH_PYTHON analysis/rna_privileged_hierarchical_distillation/test_rph.py
$RPH_PYTHON analysis/rna_privileged_hierarchical_distillation/baseline_gate.py --output "$RPH_OUTPUT"
```

The conservative dispatcher waits for the complete 5k-HVG manifest and for a
GPU to remain below 2 GiB and 10% utilization for two consecutive polls. It
then runs all inner selections before creating any new outer-test predictions.

```bash
nohup $RPH_PYTHON analysis/rna_privileged_hierarchical_distillation/orchestrate.py \
  --output "$RPH_OUTPUT" --apt-cache "$RPH_APT_CACHE" --rna-root "$RPH_RNA_ROOT" \
  --python "$RPH_PYTHON" --gpus 0,1,2,3,4,5,6,7 \
  > "$RPH_OUTPUT/orchestrator.log" 2>&1 &
```

Every fit has its own marker, checkpoint, history and log, so restarting the
same command resumes rather than overwrites completed work. A failed child fit
stops the dispatcher and writes `orchestrator_failure.json`.

Individual teacher or student jobs can also be reproduced with `run.py`:

```bash
CUDA_VISIBLE_DEVICES=0 $RPH_PYTHON analysis/rna_privileged_hierarchical_distillation/run.py select \
  --condition teacher --seed 17 --fold 0 --output "$RPH_OUTPUT" \
  --apt-cache "$RPH_APT_CACHE" --rna-root "$RPH_RNA_ROOT" --device cuda
```

## Rebuild without retraining

After `training_complete.json` exists, this single command rebuilds the CSV,
Markdown/LaTeX table, paired 5,000-draw intervals and diagnostic figures from
saved predictions:

```bash
$RPH_PYTHON analysis/rna_privileged_hierarchical_distillation/aggregate.py \
  --input "$RPH_OUTPUT" \
  --output analysis/rna_privileged_hierarchical_distillation/results \
  --n-boot 5000 --seed 20260915
```

The Plain artifacts are the baseline-gated unified_v2 reproduction. Teacher
test outputs are diagnostics only and are never loaded by student training.
