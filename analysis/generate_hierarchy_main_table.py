#!/usr/bin/env python3
"""Generate the main hierarchy table from pooled patient-disjoint predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "DropCascade",
    "Flat FT-style Transformer",
    "FT-style Transformer + HCE",
    "XGBoost",
    "Linear SVM",
    "Logistic Regression",
    "Random Forest",
    "Reference Correlation",
]

MAIN_MODEL_ORDER = [
    "DropCascade",
    "Flat FT-style Transformer",
    "FT-style Transformer + HCE",
    "XGBoost",
]

PRIMARY_COMPARATORS = {
    "FT-style Transformer + HCE",
    "XGBoost",
}

TEX_NAMES = {
    "DropCascade": r"\ourmethod{}",
    "Flat FT-style Transformer": "Flat FT-style Transformer",
    "FT-style Transformer + HCE": r"FT-style Transformer $+$ HCE",
}

CLASSICAL_NAMES = {
    "xgboost": "XGBoost",
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "reference_correlation": "Reference Correlation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropcascade", type=Path, required=True)
    parser.add_argument("--classical-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/generated"))
    parser.add_argument("--table-path", type=Path, default=Path("tables/main_result.tex"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def standardize(frame: pd.DataFrame, true_col: str, pred_col: str) -> pd.DataFrame:
    result = frame[["sample_id", true_col, pred_col]].copy()
    result.columns = ["sample_id", "y_true", "y_pred"]
    result["sample_id"] = result["sample_id"].astype(str)
    result["y_true"] = result["y_true"].astype(str)
    result["y_pred"] = result["y_pred"].astype(str)
    return result


def load_predictions(args: argparse.Namespace) -> dict[str, dict[str, pd.DataFrame]]:
    drop = pd.read_csv(args.dropcascade)
    models = {
        "coarse": {
            "DropCascade": standardize(drop, "coarse_true", "coarse_pred"),
        },
        "fine": {
            "DropCascade": standardize(drop, "fine_true", "fine_pred"),
        },
    }
    for variant, display_name in (
        ("flat_ce", "Flat FT-style Transformer"),
        ("hce", "FT-style Transformer + HCE"),
    ):
        frame = pd.read_csv(args.matched_root / variant / "pooled_oof_predictions.csv")
        models["coarse"][display_name] = standardize(frame, "coarse_true", "coarse_pred")
        models["fine"][display_name] = standardize(frame, "fine_true", "fine_pred")
    for task in ("coarse", "fine"):
        for directory, display_name in CLASSICAL_NAMES.items():
            frame = pd.read_csv(
                args.classical_root / task / directory / "pooled_oof_predictions.csv"
            )
            models[task][display_name] = standardize(frame, "y_true", "y_pred")
    return models


def patient_confusions(
    frame: pd.DataFrame, patients: list[str], labels: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_index = {value: index for index, value in enumerate(patients)}
    label_index = {value: index for index, value in enumerate(labels)}
    patient_ids = frame["sample_id"].map(patient_index).to_numpy()
    true_ids = frame["y_true"].map(label_index).to_numpy()
    pred_ids = frame["y_pred"].map(label_index).to_numpy()
    if np.isnan(true_ids).any() or np.isnan(pred_ids).any():
        raise ValueError("Prediction contains a label outside the shared class universe")
    true_ids = true_ids.astype(int)
    pred_ids = pred_ids.astype(int)
    shape = (len(patients), len(labels))
    tp = np.zeros(shape, dtype=np.float64)
    fp = np.zeros(shape, dtype=np.float64)
    fn = np.zeros(shape, dtype=np.float64)
    correct = true_ids == pred_ids
    np.add.at(tp, (patient_ids[correct], true_ids[correct]), 1.0)
    wrong = ~correct
    np.add.at(fp, (patient_ids[wrong], pred_ids[wrong]), 1.0)
    np.add.at(fn, (patient_ids[wrong], true_ids[wrong]), 1.0)
    return tp, fp, fn


def macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denominator = 2 * tp + fp + fn
    scores = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return scores.mean(axis=-1)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm-adjust a family of p-values while preserving the original index."""
    ordered = p_values.sort_values()
    adjusted = pd.Series(index=ordered.index, dtype=float)
    running_max = 0.0
    family_size = len(ordered)
    for rank, (index, value) in enumerate(ordered.items()):
        running_max = max(running_max, (family_size - rank) * value)
        adjusted.loc[index] = min(running_max, 1.0)
    return adjusted.reindex(p_values.index)


def evaluate(
    models: dict[str, dict[str, pd.DataFrame]], n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    comparisons = []
    for task, task_models in models.items():
        reference = task_models["DropCascade"]
        patients = sorted(reference["sample_id"].unique())
        labels = sorted(reference["y_true"].unique())
        reference_counts = (
            reference.groupby(["sample_id", "y_true"]).size().sort_index()
        )
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(patients), size=(n_boot, len(patients)))
        draw_counts = np.stack(
            [np.bincount(draw, minlength=len(patients)) for draw in draws]
        )
        boot_scores = {}
        point_scores = {}
        for model_name in MODEL_ORDER:
            frame = task_models[model_name]
            if sorted(frame["sample_id"].unique()) != patients:
                raise ValueError(f"{task}/{model_name} uses a different patient set")
            model_counts = frame.groupby(["sample_id", "y_true"]).size().sort_index()
            if not model_counts.equals(reference_counts):
                raise ValueError(
                    f"{task}/{model_name} uses a different patient-by-class target distribution"
                )
            tp, fp, fn = patient_confusions(frame, patients, labels)
            point_tp, point_fp, point_fn = tp.sum(0), fp.sum(0), fn.sum(0)
            point_f1 = float(macro_f1(point_tp, point_fp, point_fn))
            point_accuracy = float(point_tp.sum() / len(frame))
            sampled_tp = draw_counts @ tp
            sampled_fp = draw_counts @ fp
            sampled_fn = draw_counts @ fn
            samples = macro_f1(sampled_tp, sampled_fp, sampled_fn)
            boot_scores[model_name] = samples
            point_scores[model_name] = point_f1
            lower, upper = np.percentile(samples, [2.5, 97.5])
            metrics.append(
                {
                    "task": task,
                    "model": model_name,
                    "macro_f1": point_f1,
                    "accuracy": point_accuracy,
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                }
            )
        reference_boot = boot_scores["DropCascade"]
        for model_name in MODEL_ORDER[1:]:
            difference = reference_boot - boot_scores[model_name]
            lower, upper = np.percentile(difference, [2.5, 97.5])
            p_lower = (np.sum(difference <= 0) + 1) / (n_boot + 1)
            p_upper = (np.sum(difference >= 0) + 1) / (n_boot + 1)
            comparisons.append(
                {
                    "task": task,
                    "comparison": f"DropCascade - {model_name}",
                    "comparator": model_name,
                    "is_primary": model_name in PRIMARY_COMPARATORS,
                    "point_difference": point_scores["DropCascade"]
                    - point_scores[model_name],
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "p_value_two_sided": min(1.0, 2 * min(p_lower, p_upper)),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["p_value_holm"] = np.nan
    primary = comparison_frame["is_primary"]
    comparison_frame.loc[primary, "p_value_holm"] = holm_adjust(
        comparison_frame.loc[primary, "p_value_two_sided"]
    )
    return pd.DataFrame(metrics), comparison_frame


def significance_marker(task: str, model: str, comparisons: pd.DataFrame) -> str:
    if model not in PRIMARY_COMPARATORS:
        return ""
    row = comparisons.loc[
        (comparisons["task"] == task)
        & (comparisons["comparison"] == f"DropCascade - {model}")
    ].iloc[0]
    if row["p_value_holm"] >= 0.05:
        return r"$^{\mathrm{n.s.}}$"
    return r"$^{\dagger}$" if row["point_difference"] > 0 else r"$^{\ddagger}$"


def write_table(metrics: pd.DataFrame, comparisons: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Task} & \textbf{Model} & \textbf{Macro-F1} & \textbf{Accuracy} & \textbf{95\% CI (Macro-F1)} \\",
        r"\midrule",
    ]
    for task_index, (task, task_label) in enumerate(
        (("coarse", "Coarse lineage"), ("fine", "Fine subtype"))
    ):
        task_rows = metrics.loc[metrics["task"] == task].set_index("model")
        best_f1 = task_rows["macro_f1"].max()
        best_accuracy = task_rows["accuracy"].max()
        for row_index, model in enumerate(MAIN_MODEL_ORDER):
            row = task_rows.loc[model]
            task_cell = rf"\multirow{{{len(MAIN_MODEL_ORDER)}}}{{*}}{{{task_label}}}" if row_index == 0 else ""
            model_cell = TEX_NAMES.get(model, model)
            f1_text = f"{row.macro_f1:.3f}" + significance_marker(task, model, comparisons)
            accuracy_text = f"{row.accuracy:.3f}"
            if np.isclose(row.macro_f1, best_f1):
                f1_text = rf"\textbf{{{f1_text}}}"
            if np.isclose(row.accuracy, best_accuracy):
                accuracy_text = rf"\textbf{{{accuracy_text}}}"
            lines.append(
                f"{task_cell} & {model_cell} & {f1_text} & {accuracy_text} & "
                f"[{row.ci95_lower:.3f}, {row.ci95_upper:.3f}] \\\\"
            )
        if task_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Patient-disjoint 5-fold hierarchy results on pooled out-of-fold predictions from all 40 patients. The primary comparisons are \ourmethod{} versus HCE and versus XGBoost, for coarse and fine macro-F1. Intervals and two-sided tests use the same 2{,}000 paired patient bootstrap resamples; the four primary test $p$-values are Holm-corrected. The matched Flat FT-style Transformer is an encoder-capacity diagnostic, and all other row-wise comparisons are exploratory. The two FT-style controls match \ourmethod{}'s tokenizer, encoder dimensions, optimizer, early stopping, training budget, and checkpoint rule; HCE is defined in \S\ref{sec:method:baselines}. For primary comparisons, $^{\dagger}$ denotes a significantly lower comparator, $^{\ddagger}$ a significantly higher comparator, and $^{\mathrm{n.s.}}$ no Holm-adjusted significance. Coarse--fine consistency is reported only in Appendix~\ref{sec:appendix:consistency}.}",
            r"\label{tab:main}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = load_predictions(args)
    metrics, comparisons = evaluate(models, args.n_boot, args.seed)
    metrics.to_csv(args.output_dir / "hierarchy_main_metrics.csv", index=False)
    comparisons.to_csv(args.output_dir / "hierarchy_paired_comparisons.csv", index=False)
    write_table(metrics, comparisons, args.table_path)


if __name__ == "__main__":
    main()
