#!/usr/bin/env python3
"""Evaluate disease prediction from gold and OOF-predicted cell compositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer


REPRESENTATIONS = [
    ("Gold subtype composition", "fine_true", "fine_true", "gold annotation"),
    (
        "OOF predicted subtype composition",
        "fine_pred",
        "fine_true",
        "patient-disjoint cell-level OOF prediction",
    ),
    ("Gold lineage composition", "coarse_true", "coarse_true", "gold annotation"),
    (
        "OOF predicted lineage composition",
        "coarse_pred",
        "coarse_true",
        "patient-disjoint cell-level OOF prediction",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument(
        "--fold-manifest",
        type=Path,
        default=Path("benchmark/splits/patient_folds.json"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("analysis/generated/composition_bridge_results.csv"),
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=Path("analysis/generated/composition_bridge_oof_predictions.csv"),
    )
    parser.add_argument("--pseudocount", type=float, default=1e-4)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_and_validate(
    oof_path: Path, manifest_path: Path
) -> tuple[pd.DataFrame, dict[str, list[str]], pd.Series]:
    frame = pd.read_csv(oof_path)
    required = {
        "sample_id",
        "disease",
        "fold",
        "fine_true",
        "fine_pred",
        "coarse_true",
        "coarse_pred",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OOF columns: {sorted(missing)}")

    with manifest_path.open() as handle:
        folds = json.load(handle)["folds"]
    manifest_patients = [patient for fold in folds.values() for patient in fold]
    if len(manifest_patients) != len(set(manifest_patients)):
        raise ValueError("Fold manifest contains duplicate patients")

    patient_meta = frame[["sample_id", "disease", "fold"]].drop_duplicates()
    if patient_meta["sample_id"].duplicated().any():
        raise ValueError("A patient has multiple disease labels or OOF folds")
    if set(patient_meta["sample_id"]) != set(manifest_patients):
        raise ValueError("OOF patients do not match the fixed fold manifest")
    expected_fold = {
        patient: int(fold) for fold, patients in folds.items() for patient in patients
    }
    observed_fold = patient_meta.set_index("sample_id")["fold"].astype(int).to_dict()
    if observed_fold != expected_fold:
        raise ValueError("OOF fold assignments do not match the fixed manifest")

    disease = (
        patient_meta.set_index("sample_id")["disease"].astype(str).sort_index()
    )
    return frame, folds, disease


def clr_transform(proportions: np.ndarray, pseudocount: float) -> np.ndarray:
    adjusted = proportions + pseudocount
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    logged = np.log(adjusted)
    return logged - logged.mean(axis=1, keepdims=True)


def composition_features(
    frame: pd.DataFrame,
    label_column: str,
    vocabulary_column: str,
    patient_index: pd.Index,
    pseudocount: float,
) -> tuple[np.ndarray, int]:
    vocabulary = sorted(frame[vocabulary_column].astype(str).unique())
    counts = pd.crosstab(frame["sample_id"], frame[label_column].astype(str))
    counts = counts.reindex(index=patient_index, columns=vocabulary, fill_value=0)
    proportions = counts.to_numpy(dtype=float)
    proportions /= proportions.sum(axis=1, keepdims=True)
    return clr_transform(proportions, pseudocount), len(vocabulary)


def evaluate(
    features: np.ndarray,
    representation: str,
    n_features: int,
    folds: dict[str, list[str]],
    disease: pd.Series,
    regularization_c: float,
    seed: int,
    cell_label_source: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    classes = sorted(disease.unique())
    all_true: list[str] = []
    all_pred: list[str] = []
    all_probabilities: list[np.ndarray] = []
    prediction_rows: list[dict[str, object]] = []

    for fold in sorted(folds, key=int):
        test_patients = folds[fold]
        train_patients = [p for p in disease.index if p not in test_patients]
        train_index = disease.index.get_indexer(train_patients)
        test_index = disease.index.get_indexer(test_patients)
        model = LogisticRegression(
            max_iter=1000,
            C=regularization_c,
            random_state=seed,
        )
        model.fit(features[train_index], disease.loc[train_patients].to_numpy())
        predictions = model.predict(features[test_index])
        fold_probabilities = model.predict_proba(features[test_index])
        probabilities = np.zeros((len(test_patients), len(classes)))
        for source_index, class_name in enumerate(model.classes_):
            probabilities[:, classes.index(class_name)] = fold_probabilities[:, source_index]

        truths = disease.loc[test_patients].to_numpy()
        all_true.extend(truths.tolist())
        all_pred.extend(predictions.tolist())
        all_probabilities.append(probabilities)
        for row_index, patient in enumerate(test_patients):
            row: dict[str, object] = {
                "representation": representation,
                "sample_id": patient,
                "fold": int(fold),
                "disease_true": truths[row_index],
                "disease_pred": predictions[row_index],
            }
            row.update(
                {
                    f"prob_{class_name}": probabilities[row_index, class_index]
                    for class_index, class_name in enumerate(classes)
                }
            )
            prediction_rows.append(row)

    truth = np.asarray(all_true)
    prediction = np.asarray(all_pred)
    probability = np.concatenate(all_probabilities)
    truth_binary = LabelBinarizer().fit(classes).transform(truth)
    summary = {
        "representation": representation,
        "n_features": n_features,
        "patient_accuracy": accuracy_score(truth, prediction),
        "patient_macro_f1": f1_score(truth, prediction, average="macro"),
        "patient_macro_auroc": roc_auc_score(
            truth_binary, probability, average="macro", multi_class="ovr"
        ),
        "n_patients": len(truth),
        "cell_label_source": cell_label_source,
        "clr_pseudocount": None,
        "logistic_regression_c": regularization_c,
    }
    return summary, prediction_rows


def main() -> None:
    args = parse_args()
    frame, folds, disease = load_and_validate(args.oof, args.fold_manifest)
    summaries: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for representation, label_column, vocabulary_column, cell_label_source in REPRESENTATIONS:
        features, n_features = composition_features(
            frame,
            label_column,
            vocabulary_column,
            disease.index,
            args.pseudocount,
        )
        summary, rows = evaluate(
            features,
            representation,
            n_features,
            folds,
            disease,
            args.regularization_c,
            args.seed,
            cell_label_source,
        )
        summary["clr_pseudocount"] = args.pseudocount
        summaries.append(summary)
        predictions.extend(rows)

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.summary_output, index=False)
    pd.DataFrame(predictions).to_csv(args.prediction_output, index=False)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
