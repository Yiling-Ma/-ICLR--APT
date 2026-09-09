#!/usr/bin/env python3
"""Recompute hierarchy metrics after equal-cell sampling within each patient."""

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
]

TEX_NAMES = {
    "DropCascade": r"\ourmethod{}",
    "Flat FT-style Transformer": "Flat FT-style Transformer",
    "FT-style Transformer + HCE": r"FT-style Transformer $+$ HCE",
    "XGBoost": "XGBoost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropcascade", type=Path, required=True)
    parser.add_argument("--classical-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--cells-per-patient", type=int, default=1900)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("analysis/generated/patient_balanced_cell_sensitivity.csv"),
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path("tables/patient_balanced_cell_sensitivity.tex"),
    )
    return parser.parse_args()


def standardized(frame: pd.DataFrame, task: str, classical: bool = False) -> pd.DataFrame:
    if classical:
        result = frame[["sample_id", "y_true", "y_pred"]].copy()
    else:
        result = frame[["sample_id", f"{task}_true", f"{task}_pred"]].copy()
        result.columns = ["sample_id", "y_true", "y_pred"]
    return result.astype(str)


def load_predictions(args: argparse.Namespace) -> dict[str, dict[str, pd.DataFrame]]:
    drop = pd.read_csv(args.dropcascade)
    matched = {
        "Flat FT-style Transformer": pd.read_csv(
            args.matched_root / "flat_ce" / "pooled_oof_predictions.csv"
        ),
        "FT-style Transformer + HCE": pd.read_csv(
            args.matched_root / "hce" / "pooled_oof_predictions.csv"
        ),
    }
    models: dict[str, dict[str, pd.DataFrame]] = {"coarse": {}, "fine": {}}
    for task in models:
        models[task]["DropCascade"] = standardized(drop, task)
        for name, frame in matched.items():
            models[task][name] = standardized(frame, task)
        models[task]["XGBoost"] = standardized(
            pd.read_csv(
                args.classical_root
                / task
                / "xgboost"
                / "pooled_oof_predictions.csv"
            ),
            task,
            classical=True,
        )
    return models


def macro_f1(frame: pd.DataFrame, labels: list[str]) -> float:
    scores = []
    for label in labels:
        true = frame["y_true"] == label
        pred = frame["y_pred"] == label
        tp = int((true & pred).sum())
        fp = int((~true & pred).sum())
        fn = int((true & ~pred).sum())
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def shared_subsamples(
    reference: pd.DataFrame,
    cells_per_patient: int,
    repeats: int,
    seed: int,
) -> list[np.ndarray]:
    groups = {
        patient: indices.to_numpy()
        for patient, indices in reference.groupby("sample_id", sort=True).groups.items()
    }
    minimum = min(len(indices) for indices in groups.values())
    if cells_per_patient > minimum:
        raise ValueError(
            f"Requested {cells_per_patient} cells per patient, but minimum is {minimum}"
        )
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [
                rng.choice(indices, size=cells_per_patient, replace=False)
                for indices in groups.values()
            ]
        )
        for _ in range(repeats)
    ]


def evaluate(
    models: dict[str, dict[str, pd.DataFrame]],
    cells_per_patient: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for task, task_models in models.items():
        reference = task_models["DropCascade"].reset_index(drop=True)
        labels = sorted(reference["y_true"].unique())
        samples = shared_subsamples(reference, cells_per_patient, repeats, seed)
        for model in MODEL_ORDER:
            frame = task_models[model].reset_index(drop=True)
            if not frame[["sample_id", "y_true"]].equals(
                reference[["sample_id", "y_true"]]
            ):
                raise ValueError(f"{task}/{model} is not row-aligned with the reference")
            pooled = macro_f1(frame, labels)
            repeated = np.asarray([macro_f1(frame.iloc[index], labels) for index in samples])
            lower, upper = np.percentile(repeated, [2.5, 97.5])
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "pooled_macro_f1": pooled,
                    "balanced_mean_macro_f1": float(repeated.mean()),
                    "balanced_ci_lower": float(lower),
                    "balanced_ci_upper": float(upper),
                    "balanced_minus_pooled": float(repeated.mean() - pooled),
                    "cells_per_patient": cells_per_patient,
                    "repeats": repeats,
                }
            )
    return pd.DataFrame(rows)


def write_table(results: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
            r"\textbf{Task} & \textbf{Model} & \textbf{CW-pooled M-F1} & \textbf{Equal-cell M-F1 [95\% interval]} & $\boldsymbol{\Delta}$ \\",
        r"\midrule",
    ]
    for task_index, (task, label) in enumerate(
        (("coarse", "Coarse lineage"), ("fine", "Fine subtype"))
    ):
        subset = results.loc[results["task"] == task].set_index("model")
        for model_index, model in enumerate(MODEL_ORDER):
            row = subset.loc[model]
            task_cell = rf"\multirow{{4}}{{*}}{{{label}}}" if model_index == 0 else ""
            lines.append(
                f"{task_cell} & {TEX_NAMES[model]} & {row.pooled_macro_f1:.3f} & "
                f"{row.balanced_mean_macro_f1:.3f} [{row.balanced_ci_lower:.3f}, "
                f"{row.balanced_ci_upper:.3f}] & {row.balanced_minus_pooled:+.3f} \\\\"
            )
        if task_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Equal-cell evaluation sensitivity. For each of 50 seeds, 1{,}900 cells are sampled without replacement from every subject, and cell-weighted pooled Macro-F1 is recomputed on the resulting 76{,}000 cells. All models use identical sampled row indices. The interval is the empirical 2.5th--97.5th percentile across seeds; $\Delta$ is the equal-cell mean minus the original cell-weighted pooled metric. This resampling diagnostic is not the subject-balanced pooled metric, and it does not retrain or reselect models.}",
            r"\label{tab:patient_balanced_cell_sensitivity}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    models = load_predictions(args)
    results = evaluate(
        models, args.cells_per_patient, args.repeats, args.seed
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    write_table(results, args.table_path)


if __name__ == "__main__":
    main()
