# Cell-identity questions: implementation and running status

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
