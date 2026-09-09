# Headline Claim to Artifact Map

| Claim family | Authoritative artifact | Generator | Paper location |
|---|---|---|---|
| Filtered cohort size, patients, APT features | `outputs/patient_cell_scaling_fair/protocol_audit.md` after final run; raw inputs `data/metadata.csv`, `data/cell_annotation.csv`, `data/apt_expression.parquet` | `analysis/patient_cell_scaling.py audit` | Abstract, Problem Setup |
| Fixed patient folds | `benchmark/splits/patient_folds.json`; strict bridge mirror `outputs/oof_composition_bridge/patient_fold_manifest.csv` | Frozen manifest, audited by both pipelines | Problem Setup, all evaluations |
| Coarse/fine and hierarchical model metrics | `analysis/generated/hierarchy_main_metrics.csv`, `analysis/generated/hierarchy_path_metrics.csv` | `analysis/generate_hierarchy_main_table.py` | Main hierarchy table |
| Paired model comparisons | `analysis/generated/hierarchy_paired_comparisons.csv` | `analysis/generate_hierarchy_main_table.py` | Main hierarchy caption/results |
| Per-subtype support and F1 | `analysis/generated/subtype_per_class.csv`, `analysis/generated/subtype_model_summary.csv` | `analysis/generate_subtype_audit.py` | Subtype results and appendix |
| Wrong-conditioned cross-lineage null | `analysis/generated/hierarchy_error_null.csv` | `analysis/generate_subtype_audit.py` | Appendix error analysis |
| Strict nested composition bridge | `outputs/oof_composition_bridge/disease_results_summary.csv`, `outputs/oof_composition_bridge/paired_feature_comparisons.csv` | `analysis/oof_composition_bridge.py`, `analysis/oof_composition_bridge_downstream.py` | Disease/composition results |
| Strict bridge patient predictions | `outputs/oof_composition_bridge/patient_disease_predictions.parquet` | Same as above | Confusion matrices and paired bootstrap |
| Legacy-versus-strict protocol audit | `outputs/oof_composition_bridge/legacy_vs_nested_audit.csv` | `analysis/generate_composition_protocol_audit.py` | Appendix protocol audit; not interpreted as a leakage effect |
| Composition controls and permutations | `outputs/oof_composition_bridge/equal_cell_sensitivity.csv`, `negative_controls.csv`, `permutation_test_results.csv` | `analysis/oof_composition_bridge_downstream.py` | Disease/composition results and appendix |
| Conditional subtype increment | `outputs/technical_covariate_disease_audit/conditional_increment_summary.csv`, `conditional_increment_bootstrap.csv` | `analysis/conditional_subtype_increment_audit.py` | Disease transfer results and appendix |
| Historical descriptive patient-cell surface | Historical `outputs/patient_cell_scaling/` summaries retained on the experiment host; committed table fragments are secondary only | Historical `analysis/patient_cell_scaling.py` protocol v2 | Appendix only |
| Fair fixed-total scaling | `outputs/patient_cell_scaling_fair/fixed_total_results.csv`, `fair_scaling_qa.json` | `analysis/patient_cell_scaling.py`, `analysis/summarize_fair_patient_cell_scaling.py` | Abstract, Results, Appendix |
| Matched patient/cell doublings | `outputs/patient_cell_scaling_fair/matched_doubling_effects.csv` | Same as above | Results, Appendix |
| Fold-adjusted response surface | `outputs/patient_cell_scaling_fair/response_surface_coefficients.csv` | Same as above | Results, Appendix |
| COMBAT external scaling replication | `outputs/combat_citeseq_scaling/`, `outputs/combat_citeseq_scaling_adt/` | `analysis/combat_citeseq_scaling.py`, fair-scaling summarizer | Abstract, Results, Appendix |
| OneK1K external scaling replication | `outputs/onek1k_scaling/fixed_total_results.csv`, `matched_doubling_effects.csv`, `response_surface_coefficients.csv`, `fair_scaling_qa.json`; construction audit in `outputs/onek1k_scaling/audit/` | `analysis/prepare_onek1k.py`, `analysis/onek1k_scaling.py`, fair-scaling summarizer | Abstract, Results, Appendix |
| Cross-cohort fixed-total summary | `outputs/cross_cohort_scaling/cross_cohort_fixed_total_effects.csv`, `qa.json` | `analysis/summarize_cross_cohort_scaling.py` | Abstract, main scaling figure, Appendix table |
| Compact-panel Top-B and Random-B | `analysis/results/psas_controls_v2/top_panel_performance.csv`, `random_panel_summary_n100.csv` | `analysis/run_psas_low_cost_controls.py` | Secondary main text, appendix |
| Compact-panel inference cell sensitivity | `analysis/results/psas_controls_v2/cellcount_sensitivity_summary_50seeds.csv` | `analysis/run_psas_low_cost_controls.py` | Appendix |
| LR versus XGBoost-SHAP panels | `analysis/results/psas_controls_v2/lr_vs_shap_panel_performance.csv`, `lr_shap_jaccard_summary.csv` | `analysis/run_psas_low_cost_controls.py` | Appendix |
| Patient-label permutation | `analysis/results/patient_permutation_n1000/patient_label_permutation_summary.json` | `analysis/run_patient_label_permutation.py` | Disease audit appendix |
| Patient-lineage attribution | `analysis/results/patient_lineage_attribution.csv`, `patient_lineage_attribution_summary.csv` | `analysis/plot_patient_lineage_attribution.py` | Appendix figures |

The APT, COMBAT RNA/ADT, OneK1K, and cross-cohort scaling QA files are `PASS`;
all abstract and main-text scaling values are copied from these machine-readable
artifacts.
