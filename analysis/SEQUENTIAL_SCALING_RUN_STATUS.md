# Completed delivery (2026-09-12 UTC)

All three cohorts are complete: APT, COMBAT RNA, and OneK1K each have
2,400 fits, for 7,200 total. Final artifacts were inspected on vllab11 at
`/ssd3/mayiling/apt_agent_runtime/sequential_scaling`. The independent metadata
audit passed; source summary and protocol hashes match the downloaded files.
See `outputs/sequential_scaling/completion_audit.json` for the precise scope.

Results are delivered in `sampling_report.tex`, section "Completed Sequential
Controls on Three Cohorts", not restored to the primary benchmark paper.
Exact quotas alone retain positive fine effects in all nine model/cohort
comparisons. Changing within-class donor allocation reduces these effects;
the historical reversal cannot be attributed solely to label matching.
Pointwise intervals are conditional on fixed folds and fitted models and
are not causal mediation, multiplicity-controlled evidence, or equivalence tests.

The following is historical launch documentation, not current run status.

# Restart handoff (2026-09-09)

At the user's explicit request, the old vllab7 sequential experiment was stopped
and its `outputs/sequential_scaling` directory deleted. Nine scoped scheduler,
shell and worker processes were terminated and checked. Raw data, other
experiments and the completed local synthetic validation were preserved.

- Host: `mayiling@vllab13.ucmerced.edu`, reached through vllab7.
- Isolated run root: `/home/mayiling/projs/apt_agent/sequential_audit_20260909`.
- Python: `/home/mayiling/opt/apt-ft-baselines/bin/python`.
- Physical GPU 4 (RTX 6000 Ada), exposed as `cuda:0`; two concurrent shards.
- XGBoost uses GPU. Sklearn LR/MLP run on CPU, also on vllab13.
- GPU smoke test passed: booster reported `cuda:0`; a synthetic 6,400 x 293,
  27-class, 10-tree fit including initialization took 15.6 seconds. This is not
  a measured full-job speedup. Other users are actively using the GPUs.
- COMBAT: `/home/mayiling/data/combat/prepared`.
- OneK1K: `/home/mayiling/data/onek1k/prepared`.
- Batch: `analysis/run_sequential_batch.sh`; two shards; APT then COMBAT RNA
  then OneK1K. Full design is 2,400 fits per cohort (7,200 total).
- Batch log: `outputs/sequential_scaling/batch.log` under the isolated root.
- Launcher PID: 4154611; also recorded in `outputs/sequential_scaling/launcher.pid`.
- Shard logs: `outputs/sequential_scaling/logs/{dataset}_{0,1}.log`.
- Results: `outputs/sequential_scaling/{dataset}/runs/`.
- Automatic aggregation follows completion of all shards for each dataset.
  Any failed shard stops the batch; incomplete results cannot produce inference.
- The fresh batch was launched on vllab13. Full real-cohort results are
  pending, not present in the paper. Check logs rather than treating this
  launch-time note as a live status report.
- LR convergence warnings were observed in the old run at the frozen 500-iteration budget;
  retain and audit them before interpreting completed results. Do not silently
  retune individual conditions after viewing scores.
- Every result records hostname, visible GPU and backend configuration. CPU
  results from the deleted run are not reused or mixed into the fresh run.

The local synthetic experiment is complete (1,280 fits); Appendix C.4 reports
all settings. Local tests cover sampling invariants and paired aggregation.
Neither full real-cohort completion nor Overleaf pulling the GitHub update has
been verified in this launch handoff.
