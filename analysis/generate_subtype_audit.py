#!/usr/bin/env python3
"""Generate subtype audit tables from pooled patient-disjoint predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_SPECS = [
    ("DropCascade", None, r"\ourmethod{}"),
    ("XGBoost", "xgboost", "XGBoost"),
    ("Logistic Regression", "logistic_regression", "Logistic Regression"),
    ("Linear SVM", "linear_svm", "Linear SVM"),
    ("Random Forest", "random_forest", "Random Forest"),
    ("Reference Correlation", "reference_correlation", "Reference Correlation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropcascade-predictions", type=Path, required=True)
    parser.add_argument("--classical-root", type=Path, required=True)
    parser.add_argument("--pooled-summary", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/generated"))
    parser.add_argument("--table-dir", type=Path, default=Path("tables"))
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, str]:
    mapping = {}
    for line in path.read_text().splitlines():
        if line.startswith("  ") and ": " in line:
            subtype, lineage = line.strip().split(": ", 1)
            mapping[subtype] = lineage
    if not mapping:
        raise ValueError(f"No subtype mapping found in {path}")
    return mapping


def per_class_f1(frame: pd.DataFrame, labels: list[str]) -> np.ndarray:
    scores = []
    for label in labels:
        true = frame["y_true"] == label
        pred = frame["y_pred"] == label
        tp = int((true & pred).sum())
        fp = int((~true & pred).sum())
        fn = int((true & ~pred).sum())
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return np.asarray(scores)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy()
    right_rank = pd.Series(right).rank(method="average").to_numpy()
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def load_predictions(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    raw_dropcascade = pd.read_csv(args.dropcascade_predictions)
    dropcascade = raw_dropcascade[["sample_id", "fine_true", "fine_pred"]].rename(
        columns={"fine_true": "y_true", "fine_pred": "y_pred"}
    )
    models = {"DropCascade": dropcascade}
    for display_name, directory, _ in MODEL_SPECS[1:]:
        path = args.classical_root / directory / "pooled_oof_predictions.csv"
        models[display_name] = pd.read_csv(path)[["sample_id", "y_true", "y_pred"]]

    reference_counts = dropcascade["y_true"].value_counts().sort_index()
    for name, frame in models.items():
        if len(frame) != len(dropcascade):
            raise ValueError(f"{name} has {len(frame)} rows; expected {len(dropcascade)}")
        if not frame["y_true"].value_counts().sort_index().equals(reference_counts):
            raise ValueError(f"{name} does not use the same evaluation labels")
    return models, raw_dropcascade


def write_diagnostics_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Model} & $\boldsymbol{\rho}_{\mathrm{cell}}$ & $\boldsymbol{\rho}_{\mathrm{patient}}$ & $G$ & $R_{\mathrm{sibling}}$ & \textbf{Cross-lineage} \\",
        r"\midrule",
    ]
    tex_names = dict((name, tex) for name, _, tex in MODEL_SPECS)
    for row in summary.itertuples(index=False):
        lines.append(
            f"{tex_names[row.model]} & {row.rho_cell:.3f} & {row.rho_patient:.3f} & "
            f"{row.granularity_gap:.3f} & {row.sibling_error_rate:.3f} & "
            f"{row.cross_lineage_error_rate:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Subtype support, granularity, and error diagnostics on pooled patient-disjoint predictions. The two $\rho$ columns are Spearman associations of per-subtype F1 with log cell count and patient coverage. $G=F_{1}^{\mathrm{coarse}}-F_{1}^{\mathrm{fine}}$. Among incorrect subtype predictions, $R_{\mathrm{sibling}}$ retains the true parent lineage and Cross-lineage is its complement. These are descriptive benchmark findings, not proposed ranking metrics.}",
            r"\label{tab:subtype_diagnostics}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_per_subtype_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.04}",
        r"\begin{longtable}{llrrrrrr}",
        r"\caption{Per-subtype support and pooled out-of-fold F1. Cells/patient is the mean among patients in which that subtype is observed. Head and Tail denote the seven most and seven least abundant subtypes; all others are Mid.}\label{tab:subtype_per_class} \\",
        r"\toprule",
        r"\textbf{Subtype} & \textbf{Lineage} & \textbf{Cells} & \textbf{Patients} & \textbf{Cells/patient} & \textbf{Regime} & \textbf{DropCascade F1} & \textbf{XGB F1} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Subtype} & \textbf{Lineage} & \textbf{Cells} & \textbf{Patients} & \textbf{Cells/patient} & \textbf{Regime} & \textbf{DropCascade F1} & \textbf{XGB F1} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{Continued on next page} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.subtype} & {row.lineage} & {row.cell_count:,} & {row.patient_count} & "
            f"{row.mean_cells_per_patient:.1f} & {row.regime} & {row.dropcascade_f1:.3f} & {row.xgboost_f1:.3f} \\\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    path.write_text("\n".join(lines) + "\n")


def write_tail_table(summary: pd.DataFrame, path: Path) -> None:
    tex_names = dict((name, tex) for name, _, tex in MODEL_SPECS)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Tail Macro-F1} & \textbf{Head Macro-F1} \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"{tex_names[row.model]} & {row.tail_macro_f1:.3f} & {row.head_macro_f1:.3f} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Fine-subtype macro-F1 on the seven least and seven most abundant subtypes under pooled patient-disjoint evaluation. This aggregate is reported as a descriptive complement to the complete per-subtype audit in Table~\ref{tab:subtype_per_class}.}",
            r"\label{tab:tail_head}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    mapping = load_mapping(args.mapping)
    models, raw_dropcascade = load_predictions(args)

    reference = models["DropCascade"]
    counts = reference.groupby("y_true").size().sort_values()
    patient_counts = reference.groupby("y_true")["sample_id"].nunique()
    labels = counts.index.tolist()
    tail = set(labels[:7])
    head = set(labels[-7:])

    per_subtype = pd.DataFrame(
        {
            "subtype": labels,
            "lineage": [mapping[label] for label in labels],
            "cell_count": counts.to_numpy(),
            "patient_count": patient_counts.loc[labels].to_numpy(),
        }
    )
    per_subtype["mean_cells_per_patient"] = (
        per_subtype["cell_count"] / per_subtype["patient_count"]
    )
    per_subtype["regime"] = [
        "Tail" if label in tail else "Head" if label in head else "Mid" for label in labels
    ]
    per_subtype["dropcascade_f1"] = per_class_f1(models["DropCascade"], labels)
    per_subtype["xgboost_f1"] = per_class_f1(models["XGBoost"], labels)

    pooled = pd.read_csv(args.pooled_summary)
    model_rows = []
    for model_name, directory, _ in MODEL_SPECS:
        frame = models[model_name]
        f1_values = per_class_f1(frame, labels)
        errors = frame.loc[frame["y_true"] != frame["y_pred"]]
        sibling_rate = float(
            (errors["y_true"].map(mapping) == errors["y_pred"].map(mapping)).mean()
        )
        if directory is None:
            coarse_f1 = float(
                per_class_f1(
                    raw_dropcascade.rename(
                        columns={"coarse_true": "y_true", "coarse_pred": "y_pred"}
                    ),
                    sorted(raw_dropcascade["coarse_true"].unique()),
                ).mean()
            )
            fine_f1 = float(f1_values.mean())
        else:
            coarse_f1 = float(
                pooled.loc[
                    (pooled["task"] == "coarse") & (pooled["model"] == directory), "macro_f1"
                ].iloc[0]
            )
            fine_f1 = float(
                pooled.loc[
                    (pooled["task"] == "fine") & (pooled["model"] == directory), "macro_f1"
                ].iloc[0]
            )
        model_rows.append(
            {
                "model": model_name,
                "rho_cell": spearman(f1_values, np.log(counts.to_numpy())),
                "rho_patient": spearman(f1_values, patient_counts.loc[labels].to_numpy()),
                "coarse_macro_f1": coarse_f1,
                "fine_macro_f1": fine_f1,
                "granularity_gap": coarse_f1 - fine_f1,
                "sibling_error_rate": sibling_rate,
                "cross_lineage_error_rate": 1.0 - sibling_rate,
                "tail_macro_f1": float(per_class_f1(frame, list(counts.index[:7])).mean()),
                "head_macro_f1": float(per_class_f1(frame, list(counts.index[-7:])).mean()),
            }
        )
    model_summary = pd.DataFrame(model_rows)

    per_subtype.to_csv(args.output_dir / "subtype_per_class.csv", index=False)
    model_summary.to_csv(args.output_dir / "subtype_model_summary.csv", index=False)
    write_diagnostics_table(model_summary, args.table_dir / "subtype_diagnostics.tex")
    write_per_subtype_table(per_subtype, args.table_dir / "subtype_per_class.tex")
    write_tail_table(model_summary, args.table_dir / "tail_head.tex")


if __name__ == "__main__":
    main()
