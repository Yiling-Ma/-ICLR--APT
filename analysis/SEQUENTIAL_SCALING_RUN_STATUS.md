# Launch handoff (2026-09-09)

- Host: `mayiling@vllab7.ucmerced.edu`, CPU only. GPU NVML reports a driver mismatch.
- Isolated run root: `/home/mayiling/projs/apt_agent/sequential_audit_20260909`.
- Python: `/home/mayiling/opt/apt-ft-baselines/bin/python`.
- COMBAT: `/home/mayiling/data/combat/prepared`.
- OneK1K: `/home/mayiling/data/onek1k/prepared`.
- Batch: `analysis/run_sequential_batch.sh`; four shards; APT then COMBAT RNA
  then OneK1K. Full design is 2,400 fits per cohort (7,200 total).
- Batch log: `outputs/sequential_scaling/batch.log` under the isolated root.
- Shard logs: `outputs/sequential_scaling/logs/{dataset}_{0,1,2,3}.log`.
- Results: `outputs/sequential_scaling/{dataset}/runs/`.
- Automatic aggregation follows completion of all shards for each dataset.
  Any failed shard stops the batch; incomplete results cannot produce inference.
- APT completed first coarse LR jobs at launch. Full real-cohort results are
  pending, not present in the paper. Check logs rather than treating this
  launch-time note as a live status report.
- LR convergence warnings were observed at the frozen 500-iteration budget;
  retain and audit them before interpreting completed results. Do not silently
  retune individual conditions after viewing scores.

The local synthetic experiment is complete (1,280 fits); Appendix C.4 reports
all settings. Local tests cover sampling invariants and paired aggregation.
Neither full real-cohort completion nor Overleaf pulling the GitHub update has
been verified in this launch handoff.
