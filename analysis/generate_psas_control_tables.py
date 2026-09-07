"""Generate LaTeX tables for the expanded PSAS controls."""

from pathlib import Path

import pandas as pd


RESULT_DIR = Path("analysis/results/psas_controls_v2")
TABLE_DIR = Path("tables")


def metric_interval(mean: float, lower: float, upper: float) -> str:
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"


def write_random_table() -> None:
    data = pd.read_csv(RESULT_DIR / "random_panel_summary_n100.csv")
    rows = []
    for _, row in data.sort_values(["B", "model"]).iterrows():
        interval = metric_interval(
            row["random_mean"],
            row["random_empirical_2.5pct"],
            row["random_empirical_97.5pct"],
        )
        rows.append(
            f"{int(row['B'])} & {row['model']} & {row['top_B_patient_macro_f1']:.3f} & "
            f"{interval} & {row['top_B_random_percentile_midrank']:.1f} \\\\"
        )
    content = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{cclcc}",
            "\\toprule",
            "$B$ & Model & Top-$B$ & Random mean [95\\% interval] & Percentile \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Compact-panel random controls over 100 panel repeats. "
            "Top-$B$ uses fold-specific outer-development LR rankings. Each repeat draws one "
            "random panel shared across outer folds; both classifiers are refitted in every fold. "
            "The percentile is the empirical midrank of Top-$B$ within the random distribution.}",
            "\\label{tab:psas_random}",
            "\\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "psas_random_controls.tex").write_text(content)


def write_cellcount_table() -> None:
    data = pd.read_csv(RESULT_DIR / "cellcount_sensitivity_summary_50seeds.csv")
    data["n_cells"] = data["n_cells"].astype(str)
    cell_order = ["100", "500", "1000", "5000", "all"]
    rows = []
    for budget, group in data.groupby("B", observed=True):
        lookup = group.set_index("n_cells")
        values = [
            metric_interval(lookup.loc[count, "mean"], lookup.loc[count, "empirical_2_5pct"], lookup.loc[count, "empirical_97_5pct"])
            for count in cell_order
        ]
        rows.append(f"{budget} & " + " & ".join(values) + " \\\\")
    content = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{clllll}",
            "\\toprule",
            "$B$ & 100 cells & 500 cells & 1,000 cells & 5,000 cells & All cells \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{LR patient-level macro-F1 under inference-time equal-cell subsampling. "
            "Entries are mean [empirical 95\\% interval] over 50 seeds; cached outer-test "
            "predictions are subsampled before aggregation. Models are trained once using all "
            "outer-development cells, so this is not a training-time cell-budget experiment.}",
            "\\label{tab:psas_cellcount}",
            "\\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "psas_cellcount_controls.tex").write_text(content)


def write_shap_table() -> None:
    performance = pd.read_csv(RESULT_DIR / "lr_vs_shap_panel_performance.csv")
    overlap = pd.read_csv(RESULT_DIR / "lr_shap_jaccard_summary.csv").set_index("B")
    pivot = performance.pivot(index=["B", "ranking"], columns="classifier", values="patient_macro_f1")
    rows = []
    for budget in [10, 20]:
        jaccard = overlap.loc[budget]
        for index, ranking in enumerate(["LR coefficient", "XGBoost-SHAP"]):
            scores = pivot.loc[(budget, ranking)]
            overlap_text = f"{jaccard['mean']:.3f} $\\pm$ {jaccard['std']:.3f}" if index == 0 else "--"
            rows.append(
                f"{budget} & {overlap_text} & {ranking} & {scores['LR']:.3f} & "
                f"{scores['XGBoost']:.3f} \\\\"
            )
    content = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{cclcc}",
            "\\toprule",
            "$B$ & LR--SHAP Jaccard & Ranking & LR eval. & XGB eval. \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Cross-model attribution control. Jaccard is mean $\\pm$ SD across five "
            "outer-fold Top-$B$ sets. LR-coefficient and XGBoost-SHAP rows use their own selected "
            "features; evaluation columns report pooled patient macro-F1 after refitting LR or "
            "XGBoost. SHAP ranking does not select $B^*$.}",
            "\\label{tab:psas_shap}",
            "\\end{table}",
            "",
        ]
    )
    (TABLE_DIR / "psas_shap_controls.tex").write_text(content)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_random_table()
    write_cellcount_table()
    write_shap_table()


if __name__ == "__main__":
    main()
