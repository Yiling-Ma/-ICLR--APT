# Remaining experiment status

Updated 2026-09-12 UTC. Source baseline 336e087; new implementation in this commit.

Host: vllab11, reached through vllab7. Root:
`/ssd3/mayiling/apt_agent_runtime/remaining_v1`.

| Job | Hardware | Output | Expected completion artifacts | Observed startup |
|---|---|---|---|---|
| full-budget MLP | GPU 4 | full_mlp | 30 NPZ + completion.json | first coarse outer fold saved, 41.3 s |
| frozen-model oracle | GPU 5 | oracle | 15 NPZ + completion.json | Flat fold0 passed exact prediction reconstruction |
| modality repeats | CPU, four BLAS threads | modality_hvg2000 | 300 NPZ + completion.json | first SGD fits saved |
| wider RNA sensitivity | same CPU queue | modality_hvg5000 | 300 NPZ + completion.json | queued after HVG2000 |

Logs: mlp.log, oracle.log, modality.log in root. Wrappers hold distinct flock
locks. Each completed fit is saved atomically; restarts skip completed fits.
No old results were deleted. Three unit tests passed on the remote Python env.

Oracle replay initially used batch64 and passed eight checkpoints, then stopped
at HCE fold3 because a fine prediction differed. The original batch size is512;
the replay now uses the checkpoint's batch size and still requires exact ordinary
predictions, without relaxing the check. The failed attempt log is preserved as
oracle_batch64_attempt.log. Its successful retry must be verified in completion
metadata, not inferred from this code change.

Update: all15 oracle checkpoints subsequently passed exact ordinary prediction
reconstruction at completion. summary.csv and completion.json have been fetched
to outputs/remaining_oracle_v1. Flat/HCE/Soft-cascade ordinary SB fine
0.139/0.143/0.145 becomes0.366/0.361/0.358 with privileged parent restriction.
The completed diagnostic is now added to the paper; remaining training jobs
must still complete before their results are added.

2026-09-12 02:02 UTC check: all30 MLP fits completed, but aggregation stopped
because legacy NPZ patient IDs had object dtype. Training is not repeated.
The repair reads string IDs from each existing JSON sidecar, verifies each
cell against keyed original metadata, and reconstructs both ordinary and
oracle confusion matrices from saved probabilities before joint resampling.
No pickle loading is enabled and no original predictions are overwritten.
Recovery log: mlp_aggregate_recovery.log; completion.json must still pass.
At this check HVG2000 had88/300 complete fits; HVG5000 remained queued.

Recovery completed successfully: full_mlp completion.json is PASS for all30
fits, including keyed cell/label alignment and exact reconstruction of stored
ordinary/oracle CMs. Coarse/fine mean-over-three-seed SB-Macro-F1 is
0.33358/0.14098, with joint seed/patient95% intervals [0.31053,0.35545] and
[0.12496,0.15516]. These are not paired superiority tests against other models.
Aggregate-only exports are under outputs/remaining_full_mlp_v1. The original
mlp.log failure is historical and resolved; do not restart training because it
still contains that traceback. At recovery verification HVG2000 was93/300;
HVG5000 remained queued. Paper integration of the remaining training results
will follow completion of the modality grid.

Commands (run on vllab11):

```sh
bash /home/mayiling/projs/apt_agent/analysis/run_remaining_v1.sh mlp
bash /home/mayiling/projs/apt_agent/analysis/run_remaining_v1.sh oracle
bash /home/mayiling/projs/apt_agent/analysis/run_remaining_v1.sh modality
```

Do not run duplicate commands while locks are held. These are three independent
jobs; modality includes the two widths sequentially. No completion is claimed
until the corresponding completion.json is PASS and numerical audits are read.
Private predictions, IDs, and checkpoint tensors remain on remote SSD; only
aggregate summaries, audit metadata, and final figures/tables go to GitHub.
