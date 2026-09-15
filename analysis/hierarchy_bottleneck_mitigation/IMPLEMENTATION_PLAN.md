# Audited implementation plan

## Sources inspected before implementation

- GitHub paper repository `main`, commit `bcecab6c`,
  `analysis/unified_v2/run.py`: exact joint-head MLP, selection/refit schedule,
  fixed outer/inner fold convention, seed handling, and artifact format.
- Server code `analysis/patient_cell_scaling.py`: benchmark metadata and frozen
  patient-fold loading.
- Server code `analysis/validate_paired_information.py::raw_apt`: barcode-aligned
  raw sparse APT counts and `log1p` construction.
- Server code `analysis/run_full_budget_mlp.py::{matrices,score}`:
  patient-level confusion matrices and pooled subject-balanced Macro-F1.
- GitHub paper repository `analysis/hierarchy_signal/run_signal.py`: existing
  conditional-subtype MLP/true-parent oracle implementation.
- GitHub paper repository `analysis/hierarchy_signal/analyze_existing.py`:
  existing cross-parent and oracle-repair diagnostics.
- Frozen manifests `benchmark/splits/patient_folds.json` and the server
  `cell_JEPA/outputs/classical_baselines_kfold5/fold_assignment.json`; their
  patient assignments were checked to be identical.

## Reuse and modifications

1. `run.py` calls the existing metadata/raw-data/fold/metric functions above.
   It retains unified_v2's 293-256-128 backbone, dropout 0.1, AdamW grid,
   batch size 1024, patience 8, seed-17 inner validation selection, outer
   development refit, and seeds 17/29/43.
2. `hbm_core.py::HBMStudent` adds no inference input: `forward(x)` returns the
   existing fine and coarse heads. The same file adds stable parent log-mass,
   symmetric coarse/fine KL, and masked within-lineage KD.
3. `hbm_core.py::OracleTeacher` is fold-specific and privileged only during
   teacher/student training. `run.py::load_teacher_for_partition` freezes it
   and verifies its scaler and patient scope before distillation.
4. `aggregate.py` reuses the existing confusion-matrix score and performs
   matched seed-slot/patient bootstrap comparisons directly from saved OOF
   predictions. It generates all tables and both figures without retraining.
5. `test_hbm_core.py` and `test_protocol.py` assert architecture equivalence,
   loss behavior, no lineage argument at student inference, ontology mapping,
   fold isolation, train-only scaling, and exact OOF coverage.

## Execution gates

Run A through F in order. The Plain MLP must first reproduce the committed
unified_v2 score within ordinary stochastic/hardware tolerance. Every new
weight/temperature is selected on inner validation only. Test-fold results are
reported after selection and are never used to revise the candidate grid.
