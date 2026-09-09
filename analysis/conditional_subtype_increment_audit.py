#!/usr/bin/env python3
"""Test subtype composition beyond recorded QC and predicted lineage controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
QC_COLUMNS = (
    "log_n_cells",
    "median_log1p_nCount_RNA",
    "median_log1p_nFeature_RNA",
    "median_feature_count_ratio",
    "iqr_log1p_nCount_RNA",
    "iqr_log1p_nFeature_RNA",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composition-features", type=Path, required=True)
    parser.add_argument("--patient-qc", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20270908)
    return parser.parse_args()


def load_folds(path: Path) -> dict[int, list[str]]:
    with path.open() as handle:
        payload = json.load(handle)
    return {int(key): [str(value) for value in values] for key, values in payload["folds"].items()}


def make_model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=10_000),
    )


def select_c(frame: pd.DataFrame, development: np.ndarray, outer_fold: int, folds, features, classes):
    best_c, best_score = None, -np.inf
    for c_value in C_GRID:
        truths, predictions = [], []
        for inner_fold in sorted(folds):
            if inner_fold == outer_fold:
                continue
            validation = development & frame["patient_id"].isin(folds[inner_fold]).to_numpy()
            training = development & ~validation
            model = make_model(c_value)
            model.fit(frame.loc[training, features], frame.loc[training, "disease"])
            truths.extend(frame.loc[validation, "disease"])
            predictions.extend(model.predict(frame.loc[validation, features]))
        score = f1_score(truths, predictions, labels=classes, average="macro", zero_division=0)
        if score > best_score:
            best_c, best_score = c_value, score
    return float(best_c)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    composition = pd.read_parquet(args.composition_features)
    qc = pd.read_csv(args.patient_qc)
    folds = load_folds(args.folds)

    qc = qc.rename(columns={column: f"qc::{column}" for column in QC_COLUMNS})
    qc_columns = [f"qc::{column}" for column in QC_COLUMNS]
    lineage_columns = sorted(column for column in composition if column.startswith("pred_soft_lineage::"))
    subtype_columns = sorted(column for column in composition if column.startswith("pred_soft_subtype::"))
    feature_sets = {
        "Recorded QC": qc_columns,
        "Recorded QC + predicted lineage": qc_columns + lineage_columns,
        "Recorded QC + predicted subtype": qc_columns + subtype_columns,
    }
    classes = np.array(sorted(qc["disease"].astype(str).unique()))

    summary_rows, prediction_rows = [], []
    for feature_set, features in feature_sets.items():
        all_true, all_pred, all_probability = [], [], []
        for outer_fold in sorted(folds):
            frame = composition[composition["outer_fold"] == outer_fold].merge(
                qc[["sample_id", *qc_columns]],
                left_on="patient_id",
                right_on="sample_id",
                validate="one_to_one",
            )
            test = frame["role"].eq("outer_test").to_numpy()
            development = ~test
            best_c = select_c(frame, development, outer_fold, folds, features, classes)
            model = make_model(best_c)
            model.fit(frame.loc[development, features], frame.loc[development, "disease"])
            predictions = model.predict(frame.loc[test, features])
            raw_probability = model.predict_proba(frame.loc[test, features])
            probability = np.zeros((test.sum(), len(classes)), dtype=float)
            for index, label in enumerate(model.named_steps["logisticregression"].classes_):
                probability[:, np.where(classes == label)[0][0]] = raw_probability[:, index]
            truths = frame.loc[test, "disease"].astype(str).to_numpy()
            patient_ids = frame.loc[test, "patient_id"].astype(str).to_numpy()
            all_true.extend(truths)
            all_pred.extend(predictions)
            all_probability.extend(probability)
            prediction_rows.extend(
                {
                    "feature_set": feature_set,
                    "outer_fold": outer_fold,
                    "patient_id": patient_id,
                    "true_disease": truth,
                    "predicted_disease": prediction,
                }
                for patient_id, truth, prediction in zip(patient_ids, truths, predictions)
            )

        truth = np.asarray(all_true)
        prediction = np.asarray(all_pred)
        probability = np.asarray(all_probability)
        summary_rows.append(
            {
                "feature_set": feature_set,
                "n_features": len(features),
                "accuracy": accuracy_score(truth, prediction),
                "macro_f1": f1_score(truth, prediction, labels=classes, average="macro", zero_division=0),
                "macro_auroc": roc_auc_score(
                    label_binarize(truth, classes=classes), probability, average="macro", multi_class="ovr"
                ),
                "log_loss": log_loss(truth, probability, labels=classes),
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    summary = pd.DataFrame(summary_rows)
    predictions.to_csv(args.output_dir / "conditional_increment_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "conditional_increment_summary.csv", index=False)

    wide = predictions.pivot(
        index=["outer_fold", "patient_id", "true_disease"],
        columns="feature_set",
        values="predicted_disease",
    ).reset_index()
    rng = np.random.default_rng(args.seed)
    truth = wide["true_disease"].astype(str).to_numpy()
    comparisons = [
        ("Recorded QC + predicted subtype", "Recorded QC + predicted lineage"),
        ("Recorded QC + predicted subtype", "Recorded QC"),
    ]
    comparison_rows = []
    for augmented, baseline in comparisons:
        augmented_prediction = wide[augmented].astype(str).to_numpy()
        baseline_prediction = wide[baseline].astype(str).to_numpy()
        observed = f1_score(
            truth, augmented_prediction, labels=classes, average="macro", zero_division=0
        ) - f1_score(truth, baseline_prediction, labels=classes, average="macro", zero_division=0)
        bootstrap = np.empty(args.bootstrap_replicates, dtype=float)
        for replicate in range(args.bootstrap_replicates):
            indices = rng.integers(0, len(truth), len(truth))
            sampled_truth = truth[indices]
            bootstrap[replicate] = f1_score(
                sampled_truth,
                augmented_prediction[indices],
                labels=classes,
                average="macro",
                zero_division=0,
            ) - f1_score(
                sampled_truth,
                baseline_prediction[indices],
                labels=classes,
                average="macro",
                zero_division=0,
            )
        comparison_rows.append(
            {
                "comparison": f"{augmented} minus {baseline}",
                "metric": "macro_f1",
                "difference": observed,
                "ci_low": np.quantile(bootstrap, 0.025),
                "ci_high": np.quantile(bootstrap, 0.975),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "conditional_increment_bootstrap.csv", index=False
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
