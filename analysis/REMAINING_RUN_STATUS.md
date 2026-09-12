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
