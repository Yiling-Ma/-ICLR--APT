# Cell-identity questions: implementation and running status

## Final update: 2026-09-12

All required experiments for this extension are complete and audited:
full-budget MLP 30/30, frozen neural oracle 15/15, reference modality grids
300/300 at each RNA width, and targeted conditional controls 30/30 blocks.
The targeted 5000-HVG conditional-shuffle extension was not triggered under
the frozen rule; it is intentionally not run, not an unfinished job.
The independent 5000-HVG grid gives MLP fine RNA 0.595669 versus paired
0.612135, delta +0.016467, joint95% interval [0.009308,0.028904].
The overall conditional-shuffle conclusion remains unresolved as documented
below. No scope or bootstrap seed was changed to obtain significance.
The final cross-artifact audit and six unit tests passed. Main results,
abstract, discussion, matched hierarchy table, width table, all-lineage
contrast figure, and retrospective rescue/damage interpretation are updated.
Paper-release/annotation/clinical metadata TODOs are not resolved by these
experiments and remain explicit. Historical progress sections below are
retained for provenance.

## Update: 2026-09-12 05:17 UTC

The targeted 2000-HVG extension is complete: 30/30 final blocks and PASS.
Aggregate CSVs are under outputs/cell_identity_questions_v1/hvg2000.
MLP paired-minus-RNA is +0.017991, joint95% CI [0.009170,0.029660].
MLP paired-minus-patient-by-lineage-shuffle is +0.006444, joint95% CI
[-0.000182,0.011797]. All three seed effects are positive, but the latter
interval includes zero. The frozen targeted wider-RNA gate is NOT TRIGGERED;
do not reroll bootstrap seeds or replace the primary scope to pass it.
This is unresolved evidence, not proof of no within-lineage information.
The existing 5000-HVG reference grid continues (147/300 at this check).
The 2000-HVG reference grid is also complete at 300/300 with PASS; its
aggregates are in outputs/remaining_modality_hvg2000_v1.
Paper/PDF numerical integration of these new completed controls is pending
the remaining width comparison and final joint audit. Earlier sections below
record the launch history, not the current completion status.

## Completed and integrated

- Paper reorganized around one benchmark and two empirical questions, without
  assuming an APT increment. Title unchanged; composition remains secondary.
- Full-cell-budget MLP: 30/30 outer fits, three training seeds. Main table now
  reports coarse/fine subject-balanced pooled Macro-F1 0.334/0.141.
- All-lineage MLP oracle/prior analysis: PASS on 361,792 cells, 40 patients.
  Ordinary/oracle/MAP-prior/expected-prior overall scores are
  0.141/0.385/0.148/0.180. Oracle-minus-expected-prior joint interval is
  [0.174,0.217]. All five lineage results and a main-text figure are retained.
- Prior expected counts use rounded totals for descriptive cell counts, to
  avoid floating-point truncation; scoring uses unrounded matrices.

## Running, not yet evidence

- Host vllab11, accessed via vllab7; output lives on local /ssd3, not home.
- Targeted 2000-HVG job: analysis/run_cell_identity_questions.sh.
  Output: /ssd3/mayiling/apt_agent_runtime/cell_identity_questions_v1/hvg2000.
- Expected 30 final seed/fold/model blocks. Each adds two patient-shuffle
  draws to the existing first draw and three patient-by-lineage shuffle draws.
  RNA/APT oracle, conditional priors, all-lineage and RNA-difficulty scopes
  use the same capped training budgets and selection rules as the source grid.
- The existing 300-fit-per-width modality jobs remain independent and are
  neither stopped nor replaced. New blocks wait for their source predictions.
- No repeated conditional-pairing or 5000-HVG robustness result is claimed.
  See CELL_IDENTITY_QUESTIONS_V1.md for the targeted wider-RNA follow-up gate.

## Replay engineering note

The first targeted attempt stopped before producing new control results:
SGD sigmoid probability saturation introduced argmax ties on 73/66,980
test cells. The original classifier predicts from native decision scores.
The fix verifies model.predict exactly against saved predictions, then uses
native SGD decision scores for oracle masking and verifies their argmax too.
RNA confidence margins still use probabilities as specified. MLP uses its
probabilities. The failed log is retained as
cell_identity_questions_v1_probability_tie_attempt.log; tolerances were not
relaxed. The corrected job was relaunched successfully.
The first additional patient-shuffle control has now been written. Six unit
tests pass, including preservation of expected-prior mass, patient-normalized
scoring, group-preserving permutations, and retention of outside-lineage errors.

Only aggregate CSV/figures/audits are published. Private prediction arrays and
checkpoints stay on the remote SSD. Automated follow-up must include these
new jobs, not declare completion when the earlier modality grid alone ends.
