"""Patient-label permutation test on patient-level median APT profiles."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from apt_jepa.data.dataset import load_merged_dataframe
from apt_jepa.data.preprocessing import apply_standardizer, fit_standardizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-dir", default="cell_JEPA/outputs/dropcascade_kfold5_splits")
    parser.add_argument("--output-dir", default="cell_JEPA/outputs/patient_permutation_n1000")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged, x, feature_names = load_merged_dataframe(
        args.data_dir,
        f"{args.data_dir}/metadata.csv",
        f"{args.data_dir}/cell_annotation.csv",
        subtype_col_in_annotation="Celltypes_new",
        drop_missing_subtype=True,
        drop_unknown=True,
        coarse_mapping_config="cell_JEPA/apt_jepa/configs/coarse_lineage_explicit.yaml",
    )
    frame = pd.DataFrame(x, columns=feature_names)
    frame["sample_id"] = merged["sample_id"].astype(str).to_numpy()
    patient_x = frame.groupby("sample_id", observed=True)[feature_names].median().sort_index()
    patient_disease = (
        merged[["sample_id", "disease"]]
        .drop_duplicates()
        .assign(sample_id=lambda d: d["sample_id"].astype(str))
        .set_index("sample_id")["disease"]
        .reindex(patient_x.index)
    )
    encoder = LabelEncoder().fit(patient_disease)
    y_true = encoder.transform(patient_disease)
    patients = patient_x.index.to_numpy()

    folds = []
    for fold in range(5):
        split = pd.read_csv(Path(args.split_dir) / f"fold{fold}_split.csv")
        split["sample_id"] = split["sample_id"].astype(str)
        split = split.set_index("sample_id").reindex(patients)
        train_idx = np.flatnonzero(split["split"].isin(["train", "val"]).to_numpy())
        test_idx = np.flatnonzero((split["split"] == "test").to_numpy())
        mean, std = fit_standardizer(patient_x.iloc[train_idx].to_numpy())
        folds.append(
            (
                train_idx,
                test_idx,
                apply_standardizer(patient_x.iloc[train_idx].to_numpy(), mean, std),
                apply_standardizer(patient_x.iloc[test_idx].to_numpy(), mean, std),
            )
        )

    def evaluate(labels: np.ndarray) -> tuple[float, float]:
        prediction = np.empty_like(labels)
        for fold, (train_idx, test_idx, x_train, x_test) in enumerate(folds):
            model = LogisticRegression(max_iter=1000, C=0.1, random_state=args.seed + fold)
            model.fit(x_train, labels[train_idx])
            prediction[test_idx] = model.predict(x_test)
        return (
            float(accuracy_score(labels, prediction)),
            float(f1_score(labels, prediction, labels=np.arange(len(encoder.classes_)), average="macro")),
        )

    observed_accuracy, observed_macro_f1 = evaluate(y_true)
    rng = np.random.default_rng(args.seed)
    rows = []
    original_counts = np.bincount(y_true, minlength=len(encoder.classes_))
    for permutation in range(args.n_permutations):
        shuffled = rng.permutation(y_true)
        if not np.array_equal(np.bincount(shuffled, minlength=len(encoder.classes_)), original_counts):
            raise AssertionError("Patient-level class counts changed during permutation")
        accuracy, macro_f1 = evaluate(shuffled)
        rows.append(
            {
                "permutation": permutation,
                "seed": args.seed,
                "patient_accuracy": accuracy,
                "patient_macro_f1": macro_f1,
            }
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / f"patient_label_permutations_n{args.n_permutations}.csv", index=False)

    summary = {
        "statistical_unit": "patient",
        "representation": "293-dimensional patient median APT profile",
        "n_patients": int(len(patients)),
        "n_permutations": int(args.n_permutations),
        "fixed_original_patient_folds": True,
        "class_counts_preserved": True,
        "cell_level_label_shuffling": False,
        "disease_classes": encoder.classes_.tolist(),
        "disease_patient_counts": original_counts.tolist(),
        "observed_patient_accuracy": observed_accuracy,
        "observed_patient_macro_f1": observed_macro_f1,
        "accuracy_n_as_extreme": int((raw["patient_accuracy"] >= observed_accuracy).sum()),
        "accuracy_p_value_plus_one": float(
            (1 + (raw["patient_accuracy"] >= observed_accuracy).sum()) / (1 + args.n_permutations)
        ),
        "macro_f1_n_as_extreme": int((raw["patient_macro_f1"] >= observed_macro_f1).sum()),
        "macro_f1_p_value_plus_one": float(
            (1 + (raw["patient_macro_f1"] >= observed_macro_f1).sum()) / (1 + args.n_permutations)
        ),
        "null_accuracy_mean": float(raw["patient_accuracy"].mean()),
        "null_accuracy_95pct": raw["patient_accuracy"].quantile([0.025, 0.975]).tolist(),
        "null_macro_f1_mean": float(raw["patient_macro_f1"].mean()),
        "null_macro_f1_95pct": raw["patient_macro_f1"].quantile([0.025, 0.975]).tolist(),
    }
    with open(output_dir / "patient_label_permutation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
