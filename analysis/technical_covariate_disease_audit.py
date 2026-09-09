#!/usr/bin/env python3
"""Audit disease predictability from recorded patient-level QC proxies only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FEATURE_SETS = {
    "Recovered cell count": ["log_n_cells"],
    "RNA QC summaries": [
        "median_log1p_nCount_RNA",
        "median_log1p_nFeature_RNA",
        "median_feature_count_ratio",
    ],
    "All recorded QC proxies": [
        "log_n_cells",
        "median_log1p_nCount_RNA",
        "median_log1p_nFeature_RNA",
        "median_feature_count_ratio",
        "iqr_log1p_nCount_RNA",
        "iqr_log1p_nFeature_RNA",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--composition-predictions", type=Path)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20270908)
    return parser.parse_args()


def iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def build_patient_table(metadata_path: Path, annotation_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, usecols=["cell_id", "sample_id", "disease"])
    annotation = pd.read_csv(
        annotation_path,
        usecols=["Sample", "nCount_RNA", "nFeature_RNA", "Celltypes_new"],
    ).rename(columns={"Sample": "cell_id"})
    cells = metadata.merge(annotation, on="cell_id", how="inner", validate="one_to_one")
    cells = cells[cells["Celltypes_new"].fillna("").str.lower() != "unknown"].copy()
    cells["log1p_nCount_RNA"] = np.log1p(cells["nCount_RNA"].clip(lower=0))
    cells["log1p_nFeature_RNA"] = np.log1p(cells["nFeature_RNA"].clip(lower=0))
    cells["feature_count_ratio"] = cells["nFeature_RNA"] / cells["nCount_RNA"].clip(lower=1)

    grouped = cells.groupby(["sample_id", "disease"], sort=True)
    patients = grouped.agg(
        n_cells=("cell_id", "size"),
        median_log1p_nCount_RNA=("log1p_nCount_RNA", "median"),
        median_log1p_nFeature_RNA=("log1p_nFeature_RNA", "median"),
        median_feature_count_ratio=("feature_count_ratio", "median"),
        iqr_log1p_nCount_RNA=("log1p_nCount_RNA", iqr),
        iqr_log1p_nFeature_RNA=("log1p_nFeature_RNA", iqr),
    ).reset_index()
    patients["log_n_cells"] = np.log(patients["n_cells"])
    return patients


def load_folds(path: Path) -> dict[int, list[str]]:
    with path.open() as handle:
        payload = json.load(handle)
    return {int(k): [str(x) for x in v] for k, v in payload["folds"].items()}


def fit_predict_nested(
    patients: pd.DataFrame,
    folds: dict[int, list[str]],
    features: list[str],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    classes = np.sort(np.unique(labels))
    predictions = np.empty(len(patients), dtype=object)
    probabilities = np.zeros((len(patients), len(classes)), dtype=float)
    selected_c = []
    sample_to_index = {sample: i for i, sample in enumerate(patients["sample_id"])}

    for fold in sorted(folds):
        test_ids = set(folds[fold])
        test_idx = np.array([sample_to_index[x] for x in test_ids])
        development_folds = [inner for inner in sorted(folds) if inner != fold]
        development_ids = set(patients["sample_id"]) - test_ids
        development_idx = np.array([sample_to_index[x] for x in development_ids])

        best_c, best_score = None, -np.inf
        for c_value in C_GRID:
            inner_predictions = np.empty(len(development_idx), dtype=object)
            development_position = {index: position for position, index in enumerate(development_idx)}
            for inner_fold in development_folds:
                val_ids = set(folds[inner_fold])
                val_idx = np.array([sample_to_index[x] for x in val_ids])
                train_ids = development_ids - val_ids
                train_idx = np.array([sample_to_index[x] for x in train_ids])
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=c_value, class_weight="balanced", max_iter=10_000),
                )
                model.fit(patients.loc[train_idx, features], labels[train_idx])
                fold_predictions = model.predict(patients.loc[val_idx, features])
                for index, prediction in zip(val_idx, fold_predictions):
                    inner_predictions[development_position[index]] = prediction
            score = f1_score(
                labels[development_idx], inner_predictions.astype(str),
                average="macro", labels=classes, zero_division=0,
            )
            if score > best_score:
                best_c, best_score = c_value, score

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_c, class_weight="balanced", max_iter=10_000),
        )
        model.fit(patients.loc[development_idx, features], labels[development_idx])
        predictions[test_idx] = model.predict(patients.loc[test_idx, features])
        fold_probabilities = model.predict_proba(patients.loc[test_idx, features])
        model_classes = model.named_steps["logisticregression"].classes_
        for j, label in enumerate(model_classes):
            probabilities[test_idx, np.where(classes == label)[0][0]] = fold_probabilities[:, j]
        selected_c.append(float(best_c))
    return predictions.astype(str), probabilities, selected_c


def metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    classes = np.sort(np.unique(labels))
    one_hot = label_binarize(labels, classes=classes)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_auroc": float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patients = build_patient_table(args.metadata, args.annotation)
    folds = load_folds(args.folds)
    expected = {sample for members in folds.values() for sample in members}
    observed = set(patients["sample_id"])
    if expected != observed:
        raise ValueError(f"Fold/patient mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    patients = patients.set_index("sample_id").loc[sorted(expected)].reset_index()
    labels = patients["disease"].astype(str).to_numpy()

    rows = []
    prediction_rows = []
    rng = np.random.default_rng(args.seed)
    for name, features in FEATURE_SETS.items():
        pred, prob, selected_c = fit_predict_nested(patients, folds, features, labels)
        observed_metrics = metrics(labels, pred, prob)
        null_f1 = np.empty(args.permutations, dtype=float)
        for permutation in range(args.permutations):
            permuted = rng.permutation(labels)
            perm_pred, perm_prob, _ = fit_predict_nested(patients, folds, features, permuted)
            null_f1[permutation] = metrics(permuted, perm_pred, perm_prob)["macro_f1"]
        if args.permutations:
            p_value = (1 + np.sum(null_f1 >= observed_metrics["macro_f1"])) / (args.permutations + 1)
            null_mean = float(null_f1.mean())
            null_q025 = float(np.quantile(null_f1, 0.025))
            null_q975 = float(np.quantile(null_f1, 0.975))
        else:
            p_value = null_mean = null_q025 = null_q975 = float("nan")
        rows.append({
            "feature_set": name,
            "n_features": len(features),
            **observed_metrics,
            "permutation_p_macro_f1": float(p_value),
            "null_macro_f1_mean": null_mean,
            "null_macro_f1_q025": null_q025,
            "null_macro_f1_q975": null_q975,
            "selected_c_by_fold": ";".join(map(str, selected_c)),
        })
        for i, patient in patients.iterrows():
            prediction_rows.append({
                "feature_set": name,
                "sample_id": patient["sample_id"],
                "disease": labels[i],
                "prediction": pred[i],
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "technical_covariate_summary.csv", index=False)
    prediction_table = pd.DataFrame(prediction_rows)
    prediction_table.to_csv(args.output_dir / "technical_covariate_predictions.csv", index=False)
    patients.to_csv(args.output_dir / "patient_qc_features.csv", index=False)

    if args.composition_predictions:
        composition = pd.read_parquet(args.composition_predictions)
        composition = composition[
            (composition["feature_set"].astype(str) == "pred_soft_subtype")
            & (composition["transform"].astype(str) == "raw")
        ][["patient_id", "true_disease", "predicted_disease"]]
        qc = prediction_table[
            prediction_table["feature_set"] == "All recorded QC proxies"
        ][["sample_id", "disease", "prediction"]]
        paired = composition.merge(
            qc, left_on="patient_id", right_on="sample_id", validate="one_to_one"
        )
        if len(paired) != len(patients) or not (
            paired["true_disease"].astype(str) == paired["disease"].astype(str)
        ).all():
            raise ValueError("Composition and QC predictions do not align patient-wise")
        classes = sorted(paired["true_disease"].astype(str).unique())
        y_true = paired["true_disease"].astype(str).to_numpy()
        composition_pred = paired["predicted_disease"].astype(str).to_numpy()
        qc_pred = paired["prediction"].astype(str).to_numpy()
        rng_bootstrap = np.random.default_rng(args.seed)
        differences = np.empty(args.bootstrap_replicates, dtype=float)
        for replicate in range(args.bootstrap_replicates):
            index = rng_bootstrap.integers(0, len(paired), len(paired))
            sampled_y = y_true[index]
            differences[replicate] = f1_score(
                sampled_y,
                composition_pred[index],
                labels=classes,
                average="macro",
                zero_division=0,
            ) - f1_score(
                sampled_y,
                qc_pred[index],
                labels=classes,
                average="macro",
                zero_division=0,
            )
        observed_difference = f1_score(
            y_true, composition_pred, labels=classes, average="macro", zero_division=0
        ) - f1_score(
            y_true, qc_pred, labels=classes, average="macro", zero_division=0
        )
        pd.DataFrame([{
            "comparison": "predicted_soft_subtype_minus_all_recorded_qc",
            "metric": "macro_f1",
            "difference": observed_difference,
            "ci_low": np.quantile(differences, 0.025),
            "ci_high": np.quantile(differences, 0.975),
            "bootstrap_replicates": args.bootstrap_replicates,
        }]).to_csv(args.output_dir / "composition_vs_qc_bootstrap.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
