# Secondary compact-panel audit

`compact_panel_audit.tex` preserves the previous detailed appendix text. It is
not included by the main manuscript. This is a reporting-scope change, not
deletion or replacement of unfavorable experimental results.

The original figure and tables remain in `figures/panel_size_curve.png` and
`tables/panel_budget.tex`, `tables/psas_random_controls.tex`,
`tables/psas_cellcount_controls.tex`, and `tables/psas_shap_controls.tex`.
Control outputs remain in `analysis/results/psas_controls_v2`.

These analyses reuse 40 patients and cannot identify biological versus
acquisition-associated signatures. They do not validate biomarkers, a reduced
assay, or a low-cell clinical workflow. Consult the archive for the conditional
inner-ranking limitation and exact training versus inference budget definitions.

# Baseline reporting update

The four additional classical rows in `tables/main_result.tex` use existing
`analysis/generated/hierarchy_main_metrics.csv`, rounded to three decimals.
No model was retrained for this reporting update. Missing joint path metrics
are explicitly marked rather than inferred from marginal scores. Full MLP,
ResNet, additional modern-tabular, and pretrained-encoder hierarchy evaluations
remain missing; ongoing scaling runs must not be substituted for those tasks.

Input-compatibility documentation checked for the manuscript clarification:
- scGPT: https://github.com/bowang-lab/scGPT/blob/main/scgpt/tasks/cell_emb.py
- Geneformer: https://huggingface.co/ctheodoris/Geneformer

These sources describe transcriptomic inputs; they are not evidence that an
APT adapter or an RNA-cohort evaluation would be impossible or ineffective.
