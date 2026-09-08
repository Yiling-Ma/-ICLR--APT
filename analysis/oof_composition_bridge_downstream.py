"""Complete the patient-level analysis for the OOF composition bridge.

The script consumes the 25 strict nested XGBoost probability artifacts produced
by ``oof_composition_bridge.py``. It never refits the cell-level model and never
uses disease labels when constructing predicted composition features.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="Only one class is present in y_true")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.oof_composition_bridge import (  # noqa: E402
    N_FOLDS,
    OUTPUT,
    load_data,
    load_folds,
    paths,
    protocol_hash,
)
from analysis.patient_cell_scaling import (  # noqa: E402
    fit_label_encoders,
    patient_table,
    subtype_parent_indices,
)


SEED = 42
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
CLR_EPSILON = 1e-6
BOOTSTRAP_REPLICATES = 2000
PERMUTATION_REPLICATES = 2000
EQUAL_CELL_SEEDS = 50
EQUAL_CELL_BUDGETS = (100, 500, 2000)
DISEASE_FEATURES = (
    "technical",
    "pred_hard_lineage",
    "pred_soft_lineage",
    "pred_hard_subtype",
    "pred_soft_subtype",
    "true_lineage",
    "true_subtype",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bh_adjust(pvalues: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def clr(values: np.ndarray) -> np.ndarray:
    logged = np.log(values + CLR_EPSILON)
    return logged - logged.mean(axis=1, keepdims=True)


def safe_corr(function, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan, math.nan
    result = function(x, y)
    return float(result.statistic), float(result.pvalue)


def aggregate_probabilities(
    rows: np.ndarray,
    probabilities: np.ndarray,
    merged: pd.DataFrame,
    parent: np.ndarray,
    n_lineages: int,
    n_subtypes: int,
) -> list[dict]:
    patient_ids = merged.loc[rows, "sample_id"].astype(str).to_numpy()
    true_subtype = merged.loc[rows, "cell_subtype"].astype(str).to_numpy()
    true_lineage = merged.loc[rows, "coarse_subtype"].astype(str).to_numpy()
    predicted_subtype = probabilities.argmax(axis=1)
    predicted_lineage = parent[predicted_subtype]
    lineage_probability = np.zeros((len(rows), n_lineages), dtype=np.float64)
    for subtype_index, lineage_index in enumerate(parent):
        lineage_probability[:, lineage_index] += probabilities[:, subtype_index]
    output: list[dict] = []
    for patient in sorted(np.unique(patient_ids)):
        mask = patient_ids == patient
        output.append(
            {
                "patient_id": patient,
                "n_cells": int(mask.sum()),
                "pred_hard_lineage": np.bincount(predicted_lineage[mask], minlength=n_lineages) / mask.sum(),
                "pred_soft_lineage": lineage_probability[mask].mean(axis=0),
                "pred_hard_subtype": np.bincount(predicted_subtype[mask], minlength=n_subtypes) / mask.sum(),
                "pred_soft_subtype": probabilities[mask].mean(axis=0),
                "true_lineage_names": true_lineage[mask],
                "true_subtype_names": true_subtype[mask],
            }
        )
    return output


def build_contexts(output: Path) -> tuple[pd.DataFrame, dict, dict]:
    merged, x, aptamers = load_data()
    folds = load_folds()
    encoders = fit_label_encoders(merged)
    parent = subtype_parent_indices(merged, encoders)
    patients = patient_table(merged).set_index("sample_id")
    fold_lookup = {patient: fold for fold, values in folds.items() for patient in values}
    feature_rows: list[dict] = []
    cell_blocks: dict[tuple[int, str], dict] = {}
    for outer in range(N_FOLDS):
        seen_rows: list[int] = []
        for heldout in range(N_FOLDS):
            meta_path, npz_path, _ = paths(output, f"outer{outer}_heldout{heldout}")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "success" or metadata.get("protocol_hash") != protocol_hash():
                raise RuntimeError(f"Invalid base artifact: {npz_path}")
            with np.load(npz_path, allow_pickle=False) as saved:
                rows = saved["row_indices"].astype(int)
                probabilities = saved["fine_probabilities"].astype(np.float64)
                if probabilities.shape[1] != len(encoders["fine"].classes_):
                    raise RuntimeError("Inconsistent subtype probability ordering.")
                if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
                    raise RuntimeError("Cell-level subtype probabilities do not sum to one.")
                blocks = aggregate_probabilities(
                    rows, probabilities, merged, parent,
                    len(encoders["coarse"].classes_), len(encoders["fine"].classes_),
                )
                seen_rows.extend(rows.tolist())
                patient_ids = merged.loc[rows, "sample_id"].astype(str).to_numpy()
                for patient in np.unique(patient_ids):
                    mask = patient_ids == patient
                    cell_blocks[(outer, patient)] = {
                        "rows": rows[mask],
                        "probabilities": probabilities[mask],
                    }
            for block in blocks:
                patient = block.pop("patient_id")
                true_lineage = np.bincount(
                    encoders["coarse"].transform(block.pop("true_lineage_names")),
                    minlength=len(encoders["coarse"].classes_),
                ) / block["n_cells"]
                true_subtype = np.bincount(
                    encoders["fine"].transform(block.pop("true_subtype_names")),
                    minlength=len(encoders["fine"].classes_),
                ) / block["n_cells"]
                row = {
                    "outer_fold": outer,
                    "patient_id": patient,
                    "patient_fold": fold_lookup[patient],
                    "role": "outer_test" if fold_lookup[patient] == outer else "inner_oof",
                    "disease": patients.loc[patient, "disease"],
                    "model_origin": f"xgb_outer{outer}_heldout{fold_lookup[patient]}",
                    "n_cells": block["n_cells"],
                    "log_n_cells": math.log(block["n_cells"]),
                }
                vectors = block | {"true_lineage": true_lineage, "true_subtype": true_subtype}
                for feature_set, vector in vectors.items():
                    if feature_set == "n_cells":
                        continue
                    names = encoders["coarse"].classes_ if "lineage" in feature_set else encoders["fine"].classes_
                    for name, value in zip(names, vector):
                        row[f"{feature_set}::{name}"] = float(value)
                feature_rows.append(row)
        if len(seen_rows) != len(merged) or len(set(seen_rows)) != len(merged):
            raise RuntimeError(f"Outer context {outer} does not cover every cell exactly once.")
    features = pd.DataFrame(feature_rows).sort_values(["outer_fold", "patient_id"]).reset_index(drop=True)
    if len(features) != N_FOLDS * len(patients):
        raise RuntimeError("Expected one patient composition per outer context.")
    features.to_parquet(output / "patient_composition_features.parquet", index=False)
    raw_patient_apt = {}
    sample_ids = merged["sample_id"].astype(str).to_numpy()
    for patient in patients.index:
        raw_patient_apt[patient] = np.median(x[sample_ids == patient], axis=0).astype(np.float64)
    context = {
        "merged": merged,
        "folds": folds,
        "encoders": encoders,
        "parent": parent,
        "patients": patients,
        "aptamers": aptamers,
        "raw_patient_apt": raw_patient_apt,
    }
    return features, cell_blocks, context


def feature_columns(frame: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "technical":
        return ["log_n_cells"]
    return [column for column in frame.columns if column.startswith(feature_set + "::")]


def matrix_for(
    frame: pd.DataFrame,
    feature_set: str,
    transform: str,
    raw_patient_apt: dict | None = None,
) -> np.ndarray:
    if feature_set == "raw_apt":
        return np.vstack([raw_patient_apt[patient] for patient in frame["patient_id"]])
    values = frame[feature_columns(frame, feature_set)].to_numpy(dtype=float)
    if transform == "clr" and feature_set != "technical":
        return clr(values)
    return values


def fit_lr(x: np.ndarray, y: np.ndarray, c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=5000,
        random_state=SEED, solver="lbfgs",
    ).fit(scaler.transform(x), y)
    return scaler, model


def tune_c(
    development: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
) -> float:
    folds = development["patient_fold"].to_numpy()
    scores = []
    for c_value in C_GRID:
        predictions = np.full(len(y), -1, dtype=int)
        for fold in sorted(np.unique(folds)):
            train = folds != fold
            test = ~train
            scaler, model = fit_lr(x[train], y[train], c_value)
            predictions[test] = model.predict(scaler.transform(x[test]))
        scores.append(f1_score(y, predictions, labels=labels, average="macro", zero_division=0))
    return float(C_GRID[int(np.argmax(scores))])


def metric_row(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray, labels: np.ndarray) -> dict:
    row = {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, labels=labels, average="macro", zero_division=0),
        "log_loss": log_loss(y, probability, labels=labels),
    }
    try:
        row["macro_auroc"] = roc_auc_score(y, probability, labels=labels, multi_class="ovr", average="macro")
    except ValueError:
        row["macro_auroc"] = math.nan
    return row


def evaluate_feature(
    features: pd.DataFrame,
    feature_set: str,
    transform: str,
    disease_encoder: LabelEncoder,
    raw_patient_apt: dict | None = None,
    disease_override: dict[str, str] | None = None,
    collect_coefficients: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = np.arange(len(disease_encoder.classes_))
    predictions, fold_rows, coefficient_rows = [], [], []
    for outer in range(N_FOLDS):
        context = features[features["outer_fold"] == outer].copy()
        development = context[context["role"] == "inner_oof"].reset_index(drop=True)
        test = context[context["role"] == "outer_test"].reset_index(drop=True)
        if disease_override is None:
            y_dev_names = development["disease"].astype(str).to_numpy()
            y_test_names = test["disease"].astype(str).to_numpy()
        else:
            y_dev_names = development["patient_id"].map(disease_override).to_numpy()
            y_test_names = test["patient_id"].map(disease_override).to_numpy()
        y_dev = disease_encoder.transform(y_dev_names)
        y_test = disease_encoder.transform(y_test_names)
        x_dev = matrix_for(development, feature_set, transform, raw_patient_apt)
        x_test = matrix_for(test, feature_set, transform, raw_patient_apt)
        retained = np.ptp(x_dev, axis=0) > 1e-12
        if not retained.any():
            raise RuntimeError(f"No varying features for {feature_set}.")
        x_dev, x_test = x_dev[:, retained], x_test[:, retained]
        c_value = tune_c(development, x_dev, y_dev, labels)
        scaler, model = fit_lr(x_dev, y_dev, c_value)
        probability = model.predict_proba(scaler.transform(x_test))
        fixed_probability = np.zeros((len(test), len(labels)))
        fixed_probability[:, model.classes_] = probability
        prediction = fixed_probability.argmax(axis=1)
        metrics = metric_row(y_test, prediction, fixed_probability, labels)
        per_class_f1 = f1_score(y_test, prediction, labels=labels, average=None, zero_division=0)
        per_class_recall = recall_score(y_test, prediction, labels=labels, average=None, zero_division=0)
        class_metrics = {}
        for class_index, disease in enumerate(disease_encoder.classes_):
            class_metrics[f"recall::{disease}"] = per_class_recall[class_index]
            class_metrics[f"f1::{disease}"] = per_class_f1[class_index]
        class_metrics["confusion_matrix"] = json.dumps(confusion_matrix(y_test, prediction, labels=labels).tolist())
        fold_rows.append({"feature_set": feature_set, "transform": transform, "outer_fold": outer, "selected_C": c_value} | metrics | class_metrics)
        if collect_coefficients:
            names = np.asarray(feature_columns(features, feature_set) if feature_set != "raw_apt" else list(raw_patient_apt))[retained]
            if feature_set == "raw_apt":
                names = np.asarray([f"APT::{index}" for index in range(len(retained))])[retained]
            coefficient_by_class = {class_value: values for class_value, values in zip(model.classes_, model.coef_)}
            for class_index, disease in enumerate(disease_encoder.classes_):
                if class_index not in coefficient_by_class:
                    continue
                for name, value in zip(names, coefficient_by_class[class_index]):
                    coefficient_rows.append({"feature_set": feature_set, "transform": transform, "outer_fold": outer, "disease": disease, "feature": name, "coefficient": value})
        for index, row in test.iterrows():
            item = {
                "feature_set": feature_set, "transform": transform, "outer_fold": outer,
                "patient_id": row["patient_id"], "true_disease": y_test_names[index],
                "predicted_disease": disease_encoder.inverse_transform([prediction[index]])[0],
            }
            item.update({f"probability::{name}": fixed_probability[index, j] for j, name in enumerate(disease_encoder.classes_)})
            predictions.append(item)
    return pd.DataFrame(predictions), pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def pooled_summary(predictions: pd.DataFrame, disease_encoder: LabelEncoder) -> dict:
    labels = np.arange(len(disease_encoder.classes_))
    y = disease_encoder.transform(predictions["true_disease"])
    pred = disease_encoder.transform(predictions["predicted_disease"])
    probability = predictions[[f"probability::{name}" for name in disease_encoder.classes_]].to_numpy()
    return metric_row(y, pred, probability, labels)


def paired_bootstrap(predictions: pd.DataFrame, disease_encoder: LabelEncoder) -> pd.DataFrame:
    pivot = {(f, t): g.set_index("patient_id") for (f, t), g in predictions.groupby(["feature_set", "transform"])}
    primary = ("pred_soft_subtype", "raw")
    comparisons = [
        (primary, ("pred_soft_lineage", "raw"), "primary_subtype_minus_lineage"),
        (primary, ("pred_hard_subtype", "raw"), "soft_minus_hard_subtype"),
        (primary, ("technical", "raw"), "predicted_subtype_minus_technical"),
        (primary, ("true_subtype", "raw"), "predicted_minus_true_subtype"),
        (("pred_soft_lineage", "raw"), ("true_lineage", "raw"), "predicted_minus_true_lineage"),
    ]
    rng = np.random.default_rng(SEED + 800)
    rows = []
    for left, right, name in comparisons:
        if left not in pivot or right not in pivot:
            continue
        common = sorted(set(pivot[left].index) & set(pivot[right].index))
        observed = pooled_summary(pivot[left].loc[common].reset_index(), disease_encoder)["macro_f1"] - pooled_summary(pivot[right].loc[common].reset_index(), disease_encoder)["macro_f1"]
        samples = []
        for _ in range(BOOTSTRAP_REPLICATES):
            chosen = rng.choice(common, size=len(common), replace=True)
            left_frame = pivot[left].loc[chosen].reset_index()
            right_frame = pivot[right].loc[chosen].reset_index()
            samples.append(pooled_summary(left_frame, disease_encoder)["macro_f1"] - pooled_summary(right_frame, disease_encoder)["macro_f1"])
        rows.append({"comparison": name, "metric": "macro_f1", "difference": observed, "ci_low": np.quantile(samples, 0.025), "ci_high": np.quantile(samples, 0.975), "bootstrap_replicates": BOOTSTRAP_REPLICATES})
    return pd.DataFrame(rows)


def permutation_worker(index: int, features: pd.DataFrame, disease_encoder: LabelEncoder, feature_sets: tuple[str, ...]) -> list[dict]:
    patients = features[["patient_id", "disease"]].drop_duplicates().sort_values("patient_id")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 900, index]))
    shuffled = patients["disease"].to_numpy().copy()
    rng.shuffle(shuffled)
    override = dict(zip(patients["patient_id"], shuffled))
    rows = []
    for feature_set in feature_sets:
        predictions, _, _ = evaluate_feature(features, feature_set, "raw", disease_encoder, disease_override=override, collect_coefficients=False)
        rows.append({"permutation": index, "feature_set": feature_set, "macro_f1": pooled_summary(predictions, disease_encoder)["macro_f1"]})
    return rows


def run_permutations(features: pd.DataFrame, predictions: pd.DataFrame, disease_encoder: LabelEncoder, n_jobs: int, replicates: int) -> pd.DataFrame:
    targets = ("technical", "pred_soft_lineage", "pred_soft_subtype", "true_lineage", "true_subtype")
    observed = {(f, t): pooled_summary(g, disease_encoder)["macro_f1"] for (f, t), g in predictions.groupby(["feature_set", "transform"])}
    checkpoint = OUTPUT / "permutation_null_values.partial.parquet"
    if checkpoint.exists():
        null = pd.read_parquet(checkpoint)
        if set(null["feature_set"]) != set(targets):
            null = pd.DataFrame()
    else:
        null = pd.DataFrame()
    completed = set(null["permutation"].unique()) if not null.empty else set()
    remaining = [index for index in range(replicates) if index not in completed]
    batch_size = 256
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        nested = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=5)(
            delayed(permutation_worker)(index, features, disease_encoder, targets)
            for index in batch
        )
        block = pd.DataFrame([row for result in nested for row in result])
        null = pd.concat([null, block], ignore_index=True)
        null.to_parquet(checkpoint, index=False)
    null = null[null["permutation"] < replicates].sort_values(["permutation", "feature_set"]).reset_index(drop=True)
    summary = []
    for feature_set in targets:
        values = null.loc[null["feature_set"] == feature_set, "macro_f1"].to_numpy()
        value = observed[(feature_set, "raw")]
        summary.append({"feature_set": feature_set, "transform": "raw", "observed_macro_f1": value, "null_mean": values.mean(), "null_ci_low": np.quantile(values, 0.025), "null_ci_high": np.quantile(values, 0.975), "p_value": (1 + np.sum(values >= value)) / (1 + len(values)), "permutations": len(values)})
    null.to_parquet(OUTPUT / "permutation_null_values.parquet", index=False)
    checkpoint.unlink(missing_ok=True)
    return pd.DataFrame(summary)


def equal_cell_features(features: pd.DataFrame, cell_blocks: dict, context: dict, budget: int, seed: int) -> pd.DataFrame:
    encoders, parent, merged = context["encoders"], context["parent"], context["merged"]
    rows = []
    for _, source in features.iterrows():
        block = cell_blocks[(int(source["outer_fold"]), source["patient_id"])]
        if len(block["rows"]) < budget:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 1000, budget, seed, int(source["outer_fold"]), sum(source["patient_id"].encode())]))
        chosen = rng.choice(len(block["rows"]), size=budget, replace=False)
        aggregate = aggregate_probabilities(block["rows"][chosen], block["probabilities"][chosen], merged, parent, len(encoders["coarse"].classes_), len(encoders["fine"].classes_))[0]
        aggregate.pop("patient_id")
        true_lineage = np.bincount(encoders["coarse"].transform(aggregate.pop("true_lineage_names")), minlength=len(encoders["coarse"].classes_)) / budget
        true_subtype = np.bincount(encoders["fine"].transform(aggregate.pop("true_subtype_names")), minlength=len(encoders["fine"].classes_)) / budget
        row = source[["outer_fold", "patient_id", "patient_fold", "role", "disease", "model_origin"]].to_dict()
        row.update({"n_cells": budget, "log_n_cells": math.log(budget)})
        vectors = aggregate | {"true_lineage": true_lineage, "true_subtype": true_subtype}
        vectors.pop("n_cells", None)
        for feature_set, vector in vectors.items():
            names = encoders["coarse"].classes_ if "lineage" in feature_set else encoders["fine"].classes_
            row.update({f"{feature_set}::{name}": float(value) for name, value in zip(names, vector)})
        rows.append(row)
    return pd.DataFrame(rows)


def equal_cell_worker(
    features: pd.DataFrame,
    cell_blocks: dict,
    context: dict,
    disease_encoder: LabelEncoder,
    budget: int,
    seed: int,
) -> list[dict]:
    rows = []
    sampled = equal_cell_features(features, cell_blocks, context, budget, seed)
    patients = sampled[["patient_id", "disease"]].drop_duplicates()
    for feature_set in ("pred_soft_lineage", "pred_soft_subtype"):
        for transform in ("raw", "clr"):
            prediction, _, _ = evaluate_feature(sampled, feature_set, transform, disease_encoder)
            metrics = pooled_summary(prediction, disease_encoder)
            rows.append({"cell_budget": budget, "seed": seed, "feature_set": feature_set, "transform": transform, "eligible_patients": patients["patient_id"].nunique(), "eligible_diseases": patients["disease"].nunique()} | metrics)
    return rows


def run_equal_cell(features: pd.DataFrame, cell_blocks: dict, context: dict, disease_encoder: LabelEncoder, n_jobs: int) -> pd.DataFrame:
    # The LR fits spend substantial time in Python-side validation; forked
    # workers provide real parallelism while sharing the large probability
    # arrays copy-on-write on the Linux analysis host.
    nested = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=5)(
        delayed(equal_cell_worker)(features, cell_blocks, context, disease_encoder, budget, seed)
        for budget in EQUAL_CELL_BUDGETS
        for seed in range(EQUAL_CELL_SEEDS)
    )
    rows = [row for block in nested for row in block]
    return pd.DataFrame(rows)


def recovery_analysis(features: pd.DataFrame, context: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_oof = features[features["role"] == "outer_test"].sort_values("patient_id")
    predicted = global_oof[feature_columns(global_oof, "pred_soft_subtype")].to_numpy()
    truth = global_oof[feature_columns(global_oof, "true_subtype")].to_numpy()
    subtype_rows = []
    exact_support = context["merged"]["cell_subtype"].astype(str).value_counts()
    for index, subtype in enumerate(context["encoders"]["fine"].classes_):
        pearson, pearson_p = safe_corr(pearsonr, predicted[:, index], truth[:, index])
        spearman, spearman_p = safe_corr(spearmanr, predicted[:, index], truth[:, index])
        subtype_rows.append({"subtype": subtype, "pearson_r": pearson, "pearson_p": pearson_p, "spearman_rho": spearman, "spearman_p": spearman_p, "mae": np.mean(np.abs(predicted[:, index] - truth[:, index])), "true_prevalence": truth[:, index].mean(), "patients_present": int(np.sum(truth[:, index] > 0)), "cell_support": int(exact_support.get(subtype, 0))})
    patient_rows = []
    for index, patient in enumerate(global_oof["patient_id"]):
        patient_rows.append({"patient_id": patient, "disease": global_oof.iloc[index]["disease"], "l1_distance": np.abs(predicted[index] - truth[index]).sum(), "jensen_shannon": jensenshannon(predicted[index], truth[index], base=2.0) ** 2, "aitchison_distance": np.linalg.norm(clr(predicted[index:index + 1])[0] - clr(truth[index:index + 1])[0])})
    return pd.DataFrame(subtype_rows), pd.DataFrame(patient_rows)


def attribution_analysis(features: pd.DataFrame, coefficients: pd.DataFrame, context: dict) -> pd.DataFrame:
    global_oof = features[features["role"] == "outer_test"].sort_values("patient_id")
    subtype_names = context["encoders"]["fine"].classes_
    rows = []
    for source in ("true_subtype", "pred_soft_subtype"):
        for subtype in subtype_names:
            values = global_oof[f"{source}::{subtype}"].to_numpy()
            for disease in context["encoders"]["disease"].classes_:
                mask = global_oof["disease"].to_numpy() == disease
                statistic, p_value = mannwhitneyu(values[mask], values[~mask], alternative="two-sided")
                coefficient_block = coefficients[(coefficients["feature_set"] == "pred_soft_subtype") & (coefficients["transform"] == "raw") & (coefficients["disease"] == disease) & (coefficients["feature"] == f"pred_soft_subtype::{subtype}")]
                coefficient_values = coefficient_block["coefficient"].to_numpy()
                top_five_folds = []
                if len(coefficient_values):
                    disease_coefficients = coefficients[(coefficients["feature_set"] == "pred_soft_subtype") & (coefficients["transform"] == "raw") & (coefficients["disease"] == disease)]
                    for _, fold_block in disease_coefficients.groupby("outer_fold"):
                        top_five = set(fold_block.reindex(fold_block["coefficient"].abs().sort_values(ascending=False).index).head(5)["feature"])
                        top_five_folds.append(f"pred_soft_subtype::{subtype}" in top_five)
                rows.append({"source": source, "subtype": subtype, "disease": disease, "n_disease": int(mask.sum()), "median_disease": np.median(values[mask]), "iqr_disease": np.quantile(values[mask], 0.75) - np.quantile(values[mask], 0.25), "median_other": np.median(values[~mask]), "median_difference": np.median(values[mask]) - np.median(values[~mask]), "rank_biserial": 2 * statistic / (mask.sum() * (~mask).sum()) - 1, "p_value": p_value, "median_lr_coefficient": np.median(coefficient_values) if len(coefficient_values) else math.nan, "iqr_lr_coefficient": np.quantile(coefficient_values, 0.75) - np.quantile(coefficient_values, 0.25) if len(coefficient_values) else math.nan, "coefficient_positive_fraction": np.mean(coefficient_values > 0) if len(coefficient_values) else math.nan, "coefficient_sign_stability": max(np.mean(coefficient_values > 0), np.mean(coefficient_values < 0)) if len(coefficient_values) else math.nan, "top_five_fraction": np.mean(top_five_folds) if top_five_folds else math.nan})
    attribution = pd.DataFrame(rows)
    attribution["q_value"] = attribution.groupby("source")["p_value"].transform(lambda values: bh_adjust(values))
    return attribution


def heldout_permutation_importance(features: pd.DataFrame, context: dict, disease_encoder: LabelEncoder) -> pd.DataFrame:
    labels = np.arange(len(disease_encoder.classes_))
    feature_set = "pred_soft_subtype"
    columns = feature_columns(features, feature_set)
    rows = []
    for outer in range(N_FOLDS):
        block = features[features["outer_fold"] == outer]
        development = block[block["role"] == "inner_oof"].reset_index(drop=True)
        test = block[block["role"] == "outer_test"].reset_index(drop=True)
        x_dev = matrix_for(development, feature_set, "raw")
        x_test = matrix_for(test, feature_set, "raw")
        y_dev = disease_encoder.transform(development["disease"])
        y_test = disease_encoder.transform(test["disease"])
        retained = np.ptp(x_dev, axis=0) > 1e-12
        c_value = tune_c(development, x_dev[:, retained], y_dev, labels)
        scaler, model = fit_lr(x_dev[:, retained], y_dev, c_value)
        baseline = log_loss(y_test, model.predict_proba(scaler.transform(x_test[:, retained])), labels=labels)
        retained_indices = np.flatnonzero(retained)
        for local_index, global_index in enumerate(retained_indices):
            changes = []
            for seed in range(100):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 1100, outer, global_index, seed]))
                permuted = x_test[:, retained].copy()
                permuted[:, local_index] = rng.permutation(permuted[:, local_index])
                changes.append(log_loss(y_test, model.predict_proba(scaler.transform(permuted)), labels=labels) - baseline)
            rows.append({"outer_fold": outer, "subtype": columns[global_index].split("::", 1)[1], "log_loss_increase_mean": np.mean(changes), "log_loss_increase_sd": np.std(changes), "permutations": len(changes)})
    return pd.DataFrame(rows)


def global_multinomial_worker(features: pd.DataFrame, disease_encoder: LabelEncoder, repeat: int) -> float:
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1200, repeat]))
    controlled = features.copy()
    subtype_columns = feature_columns(controlled, "pred_soft_subtype")
    for outer in range(N_FOLDS):
        block = controlled[controlled["outer_fold"] == outer]
        train = block[block["role"] == "inner_oof"]
        global_distribution = train[subtype_columns].to_numpy().mean(axis=0)
        for index in block.index:
            count = int(block.loc[index, "n_cells"])
            controlled.loc[index, subtype_columns] = rng.multinomial(count, global_distribution) / count
    prediction, _, _ = evaluate_feature(controlled, "pred_soft_subtype", "raw", disease_encoder)
    return pooled_summary(prediction, disease_encoder)["macro_f1"]


def negative_controls(features: pd.DataFrame, context: dict, disease_encoder: LabelEncoder, n_jobs: int) -> pd.DataFrame:
    global_oof = features[features["role"] == "outer_test"].sort_values("patient_id")
    rows = []
    # Fold ID is the target, so each training split must retain examples of all
    # five fold classes; leaving a complete fold out would make prediction of
    # that unseen target class impossible by construction.
    x = global_oof[feature_columns(global_oof, "pred_soft_subtype")].to_numpy()
    y = global_oof["patient_fold"].to_numpy()
    prediction = np.full(len(y), -1)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    for train_index, test_index in splitter.split(x, y):
        train = np.zeros(len(y), dtype=bool)
        test = np.zeros(len(y), dtype=bool)
        train[train_index] = True
        test[test_index] = True
        scaler, model = fit_lr(x[train], y[train], 1.0)
        prediction[test] = model.predict(scaler.transform(x[test]))
    rows.append({"control": "fold_id_predictability", "metric": "accuracy", "value": accuracy_score(y, prediction), "chance_reference": 1 / N_FOLDS})
    # Global multinomial control is reconstructed independently within every outer context.
    control_values = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=5)(
        delayed(global_multinomial_worker)(features, disease_encoder, repeat)
        for repeat in range(50)
    )
    rows.append({"control": "global_subtype_multinomial", "metric": "macro_f1", "value": np.mean(control_values), "ci_low": np.quantile(control_values, 0.025), "ci_high": np.quantile(control_values, 0.975), "repeats": len(control_values), "chance_reference": 1 / len(disease_encoder.classes_)})
    technical, _, _ = evaluate_feature(features, "technical", "raw", disease_encoder)
    rows.append({"control": "cell_count_only", "metric": "macro_f1", "value": pooled_summary(technical, disease_encoder)["macro_f1"], "chance_reference": 1 / len(disease_encoder.classes_)})
    return pd.DataFrame(rows)


def save_figures(output: Path, summary: pd.DataFrame, paired: pd.DataFrame, recovery: pd.DataFrame, attribution: pd.DataFrame, equal_cell: pd.DataFrame, permutations: pd.DataFrame, predictions: pd.DataFrame, disease_encoder: LabelEncoder) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    primary_order = ["technical", "pred_soft_lineage", "pred_soft_subtype", "true_lineage", "true_subtype"]
    raw = summary[(summary["transform"] == "raw") & summary["feature_set"].isin(primary_order)].set_index("feature_set").loc[primary_order]
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.bar(range(len(raw)), raw["macro_f1"], color=["#999999", "#4C78A8", "#F58518", "#72B7B2", "#E45756"])
    axis.set_xticks(range(len(raw)), ["Cell count", "Pred. lineage", "Pred. subtype", "True lineage", "True subtype"], rotation=20, ha="right")
    axis.set_ylabel("Patient Macro-F1")
    axis.set_ylim(0, 1.02)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"feature_set_performance.{suffix}", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(4.6, 3.5))
    names = ["Pred. lineage", "Pred. subtype"]
    values = [raw.loc["pred_soft_lineage", "macro_f1"], raw.loc["pred_soft_subtype", "macro_f1"]]
    axis.bar(names, values, color=["#4C78A8", "#F58518"])
    axis.set_ylabel("Patient Macro-F1")
    axis.set_ylim(0, 1.02)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"subtype_vs_lineage_comparison.{suffix}", dpi=220)
    plt.close(figure)

    selected = recovery.sort_values(["true_prevalence", "pearson_r"], ascending=False).head(8)
    global_oof = pd.read_parquet(output / "patient_composition_features.parquet").query("role == 'outer_test'")
    figure, axes = plt.subplots(2, 4, figsize=(9.5, 4.8))
    for axis, (_, row) in zip(axes.flat, selected.iterrows()):
        subtype = row["subtype"]
        axis.scatter(global_oof[f"true_subtype::{subtype}"], global_oof[f"pred_soft_subtype::{subtype}"], s=14, alpha=0.75)
        axis.set_title(subtype, fontsize=8)
        axis.set_xlabel("True")
        axis.set_ylabel("Predicted")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"true_vs_predicted_composition.{suffix}", dpi=220)
        figure.savefig(output / f"composition_recovery_scatter.{suffix}", dpi=220)
    plt.close(figure)

    stable = attribution.query("source == 'pred_soft_subtype'").copy()
    stable["score"] = stable["median_difference"].abs()
    selected_subtypes = stable.groupby("subtype")["score"].max().nlargest(12).index
    heat = stable[stable["subtype"].isin(selected_subtypes)].pivot(index="subtype", columns="disease", values="median_difference")
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    image = axis.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(np.abs(heat.to_numpy())), vmax=np.nanmax(np.abs(heat.to_numpy())))
    axis.set_xticks(range(len(heat.columns)), heat.columns, rotation=35, ha="right")
    axis.set_yticks(range(len(heat.index)), heat.index)
    figure.colorbar(image, ax=axis, label="Median composition difference")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"subtype_disease_attribution_heatmap.{suffix}", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    for feature_set, color in (("pred_soft_lineage", "#4C78A8"), ("pred_soft_subtype", "#F58518")):
        block = equal_cell[(equal_cell["feature_set"] == feature_set) & (equal_cell["transform"] == "raw")]
        grouped = block.groupby("cell_budget")["macro_f1"]
        means = grouped.mean()
        low = grouped.quantile(0.025)
        high = grouped.quantile(0.975)
        axis.errorbar(means.index.astype(str), means, yerr=[means - low, high - means], marker="o", label=feature_set.replace("pred_soft_", ""), color=color)
    axis.set_xlabel("Equal cells per patient")
    axis.set_ylabel("Patient Macro-F1")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"equal_cell_sensitivity.{suffix}", dpi=220)
    plt.close(figure)

    null_values = pd.read_parquet(output / "permutation_null_values.parquet")
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    for feature_set, color in (("pred_soft_lineage", "#4C78A8"), ("pred_soft_subtype", "#F58518")):
        values = null_values.loc[null_values["feature_set"] == feature_set, "macro_f1"]
        axis.hist(values, bins=25, alpha=0.45, label=feature_set.replace("pred_soft_", ""), color=color)
        observed = permutations.loc[permutations["feature_set"] == feature_set, "observed_macro_f1"].iloc[0]
        axis.axvline(observed, color=color, linewidth=2)
    axis.set_xlabel("Permuted-label Macro-F1")
    axis.set_ylabel("Count")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"permutation_null.{suffix}", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for axis, feature_set in zip(axes, ("pred_soft_lineage", "pred_soft_subtype")):
        block = predictions[(predictions["feature_set"] == feature_set) & (predictions["transform"] == "raw")]
        matrix = confusion_matrix(block["true_disease"], block["predicted_disease"], labels=disease_encoder.classes_)
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(feature_set.replace("pred_soft_", "").title())
        axis.set_xticks(range(len(disease_encoder.classes_)), disease_encoder.classes_, rotation=45, ha="right")
        axis.set_yticks(range(len(disease_encoder.classes_)), disease_encoder.classes_)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"disease_confusion_matrices.{suffix}", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.0, 2.8))
    axis.axis("off")
    axis.text(0.02, 0.68, "APT cells", ha="center", va="center", bbox={"boxstyle": "round", "fc": "#E8F1F8"})
    axis.text(0.28, 0.68, "Strict nested\nsubtype prediction", ha="center", va="center", bbox={"boxstyle": "round", "fc": "#E8F1F8"})
    axis.text(0.57, 0.68, "Patient composition", ha="center", va="center", bbox={"boxstyle": "round", "fc": "#FFF0DB"})
    axis.text(0.84, 0.68, "Disease LR", ha="center", va="center", bbox={"boxstyle": "round", "fc": "#FDE2E2"})
    for start, end in ((0.09, 0.20), (0.38, 0.48), (0.67, 0.76)):
        axis.annotate("", xy=(end, 0.68), xytext=(start, 0.68), arrowprops={"arrowstyle": "->"})
    axis.text(0.5, 0.22, "Outer-test patients affect neither base-model fitting nor disease-model tuning", ha="center")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"composition_bridge_schematic.{suffix}", dpi=220)
    plt.close(figure)


def write_latex(output: Path, summary: pd.DataFrame, attribution: pd.DataFrame) -> None:
    main_order = ["technical", "pred_soft_lineage", "pred_soft_subtype", "true_lineage", "true_subtype"]
    main = summary[(summary["transform"] == "raw") & summary["feature_set"].isin(main_order)].copy()
    main["feature_set"] = pd.Categorical(main["feature_set"], main_order, ordered=True)
    main = main.sort_values("feature_set")
    main.to_latex(output / "composition_bridge_main_table.tex", index=False, float_format="%.3f", escape=True)
    attribution.sort_values("q_value").head(20).to_latex(output / "composition_attribution_table.tex", index=False, float_format="%.3g", escape=True)
    methods = r"""\paragraph{OOF composition bridge.} For each disease-classifier outer fold, we trained a disease-label-free XGBoost subtype model on the 32 outer-development patients and predicted the eight untouched outer-test patients. Disease-training composition features were generated by an additional four-way patient-disjoint cross-fit within the outer-development set. Cell-level subtype probabilities were averaged within patient; lineage probabilities were obtained by summing subtype probabilities according to the frozen taxonomy. An L2-regularized multinomial logistic regression, with $C\in\{0.01,0.1,1,10,100\}$ selected only within outer-development patients, predicted six disease groups. All metrics, bootstrap resampling, and label permutations used patients as the statistical unit. CLR sensitivities used a prespecified pseudocount of $10^{-6}$."""
    (output / "composition_bridge_methods.tex").write_text(methods + "\n", encoding="utf-8")


def write_qa(
    output: Path,
    features: pd.DataFrame,
    controls: pd.DataFrame,
    context: dict,
    predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
    equal_cell: pd.DataFrame,
    permutations: pd.DataFrame,
    requested_permutations: int,
) -> bool:
    composition_columns = [column for column in features if "::" in column]
    groups = defaultdict(list)
    for column in composition_columns:
        groups[column.split("::", 1)[0]].append(column)
    checks = {
        "All 25 base artifacts have matching protocol hashes": True,
        "Each outer context contains 40 unique patients": all(group["patient_id"].nunique() == 40 for _, group in features.groupby("outer_fold")),
        "Every outer context has 32 inner-OOF and 8 outer-test patients": all(((group["role"] == "inner_oof").sum() == 32 and (group["role"] == "outer_test").sum() == 8) for _, group in features.groupby("outer_fold")),
        "Composition vectors sum to one": all(np.allclose(features[columns].sum(axis=1), 1.0, atol=1e-5) for columns in groups.values()),
        "Fixed subtype ordering contains 27 classes": len(context["encoders"]["fine"].classes_) == 27,
        "Fixed lineage ordering contains five classes": len(context["encoders"]["coarse"].classes_) == 5,
        "Patient disease is unique": features.groupby("patient_id")["disease"].nunique().eq(1).all(),
        "Disease predictions contain 40 outer-test patients per representation": all(group["patient_id"].nunique() == 40 for _, group in predictions.groupby(["feature_set", "transform"])),
        "Every disease representation has five reported outer folds": all(group["outer_fold"].nunique() == 5 for _, group in fold_results.groupby(["feature_set", "transform"])),
        "Technical control contains only log cell count": feature_columns(features, "technical") == ["log_n_cells"],
        "Equal-cell sensitivity used 50 seeds per finite budget and representation": all(group["seed"].nunique() == EQUAL_CELL_SEEDS for _, group in equal_cell.groupby(["cell_budget", "feature_set", "transform"])),
        "Patient-label permutations reached the requested count": permutations.empty or all(permutations["permutations"] == requested_permutations),
        "Global multinomial control was evaluated": "global_subtype_multinomial" in set(controls["control"]),
        "Fold-ID predictability was evaluated": "fold_id_predictability" in set(controls["control"]),
        "Global multinomial null contains the chance reference": bool(controls.query("control == 'global_subtype_multinomial'").eval("ci_low <= chance_reference <= ci_high").all()),
        "Fold-ID predictability is not substantially above chance": bool(controls.query("control == 'fold_id_predictability'").eval("value <= chance_reference + 0.20").all()),
    }
    passed = all(checks.values())
    lines = ["# Final QA", "", f"Overall status: **{'PASS' if passed else 'FAIL'}**", ""]
    lines.extend(f"- [{'x' if value else ' '}] {name}" for name, value in checks.items())
    lines += ["", "## Scope Notes", "", "- Disease metrics, permutations, and bootstrap analyses use patients, not cells, as the statistical unit.", "- XGBoost receives subtype labels only; disease labels enter only the downstream logistic regression.", "- Equal-cell sensitivity samples without replacement within each patient.", f"- The formal run used {requested_permutations:,} patient-label permutations.", "- Neural subtype predictors were not rerun; they are optional and do not delay the disease-label-free primary analysis."]
    (output / "final_qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--skip-permutations", action="store_true")
    parser.add_argument("--permutations", type=int, default=PERMUTATION_REPLICATES)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = now()
    features, cell_blocks, context = build_contexts(output)
    disease_encoder = LabelEncoder().fit(features["disease"])
    context["encoders"]["disease"] = disease_encoder
    predictions, folds, coefficients = [], [], []
    for feature_set in DISEASE_FEATURES:
        transforms = ("raw",) if feature_set == "technical" else ("raw", "clr")
        for transform in transforms:
            prediction, fold, coefficient = evaluate_feature(features, feature_set, transform, disease_encoder)
            predictions.append(prediction)
            folds.append(fold)
            coefficients.append(coefficient)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_frame = pd.concat(folds, ignore_index=True)
    coefficient_frame = pd.concat(coefficients, ignore_index=True)
    prediction_frame.to_parquet(output / "patient_disease_predictions.parquet", index=False)
    fold_frame.to_csv(output / "disease_results_by_fold.csv", index=False)
    summaries = []
    for (feature_set, transform), block in prediction_frame.groupby(["feature_set", "transform"]):
        summaries.append({"feature_set": feature_set, "transform": transform} | pooled_summary(block, disease_encoder))
    summary = pd.DataFrame(summaries)
    paired = paired_bootstrap(prediction_frame, disease_encoder)
    recovery, patient_distances = recovery_analysis(features, context)
    attribution = attribution_analysis(features, coefficient_frame, context)
    permutation_importance = heldout_permutation_importance(features, context, disease_encoder)
    importance_summary = permutation_importance.groupby("subtype", as_index=False).agg(heldout_log_loss_increase=("log_loss_increase_mean", "mean"), heldout_log_loss_increase_sd=("log_loss_increase_mean", "std"))
    attribution = attribution.merge(importance_summary, on="subtype", how="left")
    equal_cell = run_equal_cell(features, cell_blocks, context, disease_encoder, args.n_jobs)
    controls = negative_controls(features, context, disease_encoder, args.n_jobs)
    if args.skip_permutations:
        permutations = pd.DataFrame(columns=["feature_set", "transform", "observed_macro_f1", "null_mean", "null_ci_low", "null_ci_high", "p_value", "permutations"])
    else:
        permutations = run_permutations(features, prediction_frame, disease_encoder, args.n_jobs, args.permutations)
    summary = summary.merge(permutations[["feature_set", "transform", "p_value"]], on=["feature_set", "transform"], how="left")
    recovery.to_csv(output / "composition_recovery_by_subtype.csv", index=False)
    patient_distances.to_csv(output / "composition_recovery_by_patient.csv", index=False)
    summary.to_csv(output / "disease_results_summary.csv", index=False)
    paired.to_csv(output / "paired_feature_comparisons.csv", index=False)
    permutations.to_csv(output / "permutation_test_results.csv", index=False)
    attribution.to_csv(output / "subtype_disease_attribution.csv", index=False)
    permutation_importance.to_csv(output / "heldout_permutation_importance.csv", index=False)
    equal_cell.to_csv(output / "equal_cell_sensitivity.csv", index=False)
    controls.to_csv(output / "negative_controls.csv", index=False)
    coefficient_frame.to_csv(output / "disease_lr_coefficients.csv", index=False)
    registry = pd.DataFrame([{"stage": "downstream", "status": "success", "started_at": started, "completed_at": now(), "host": platform.node(), "python": sys.version.split()[0], "base_protocol_hash": protocol_hash(), "permutations": 0 if args.skip_permutations else args.permutations, "equal_cell_seeds": EQUAL_CELL_SEEDS}])
    registry.to_csv(output / "run_registry.csv", index=False)
    write_latex(output, summary, attribution)
    if not args.skip_permutations:
        save_figures(output, summary, paired, recovery, attribution, equal_cell, permutations, prediction_frame, disease_encoder)
    qa_passed = write_qa(output, features, controls, context, prediction_frame, fold_frame, equal_cell, permutations, 0 if args.skip_permutations else args.permutations)
    primary = paired.loc[paired["comparison"] == "primary_subtype_minus_lineage"].iloc[0]
    pred_subtype = summary.query("feature_set == 'pred_soft_subtype' and transform == 'raw'").iloc[0]
    pred_lineage = summary.query("feature_set == 'pred_soft_lineage' and transform == 'raw'").iloc[0]
    true_subtype = summary.query("feature_set == 'true_subtype' and transform == 'raw'").iloc[0]
    if not qa_passed:
        decision = "INVALID DUE TO LEAKAGE"
    elif not args.skip_permutations and pred_subtype["p_value"] < 0.05 and primary["ci_low"] > 0:
        decision = "SUBTYPE INCREMENT SUPPORTED"
    elif not args.skip_permutations and pred_subtype["p_value"] < 0.05:
        decision = "BRIDGE SUPPORTED" if pred_subtype["macro_f1"] > 1 / len(disease_encoder.classes_) else "LINEAGE-ONLY SIGNAL"
    elif true_subtype["macro_f1"] > pred_subtype["macro_f1"] + 0.1:
        decision = "MEASUREMENT GAP SUPPORTED"
    else:
        decision = "BRIDGE NOT SUPPORTED"
    report = ["# OOF Composition Bridge Final Report", "", f"## Decision: `{decision}`", "", "## Executive Summary", "", f"The disease-label-free XGBoost bridge achieved patient Macro-F1 {pred_subtype['macro_f1']:.3f} from soft predicted subtype composition and {pred_lineage['macro_f1']:.3f} from soft predicted lineage composition. The paired subtype-minus-lineage difference was {primary['difference']:.3f} (patient bootstrap 95% CI {primary['ci_low']:.3f} to {primary['ci_high']:.3f}). True subtype composition achieved {true_subtype['macro_f1']:.3f}.", "", "## Design and Leakage Audit", "", "Five strict outer folds were used. Each disease model was trained on composition generated by four-way inner patient-disjoint subtype predictions and evaluated on composition generated by a subtype model trained only on the 32 outer-development patients. Disease labels were never used by XGBoost. See `nested_crossfit_design.md` and `final_qa.md`.", "", "## Scope", "", f"This analysis is predictive and associational. It does not establish causal mediation, clinical validity, or externally validated disease biology. {0 if args.skip_permutations else args.permutations:,} patient-label permutations were used. Optional neural subtype predictors were not rerun.", "", "## Reproduction", "", "```bash", "python analysis/oof_composition_bridge.py audit", "python analysis/oof_composition_bridge.py fit-base --device cuda --n-jobs 8", f"python analysis/oof_composition_bridge_downstream.py --n-jobs {args.n_jobs} --permutations {args.permutations}", "```", "", "## Key Outputs", "", "- `disease_results_summary.csv`", "- `paired_feature_comparisons.csv`", "- `permutation_test_results.csv`", "- `equal_cell_sensitivity.csv`", "- `composition_recovery_by_subtype.csv`", "- `subtype_disease_attribution.csv`", "- `feature_set_performance.pdf`", "- `final_qa.md`"]
    (output / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    results_text = f"Using strict nested patient-level cross-fitting, soft XGBoost-predicted subtype composition achieved Macro-F1 {pred_subtype['macro_f1']:.3f} and soft predicted lineage composition achieved {pred_lineage['macro_f1']:.3f}. Their paired difference was {primary['difference']:.3f} (95% patient-bootstrap CI {primary['ci_low']:.3f}, {primary['ci_high']:.3f}). The base subtype model did not use disease labels. True subtype composition achieved Macro-F1 {true_subtype['macro_f1']:.3f}."
    (output / "composition_bridge_results.tex").write_text(results_text + "\n", encoding="utf-8")
    print(json.dumps({"status": "success", "qa": qa_passed, "decision": decision, "predicted_subtype_macro_f1": pred_subtype["macro_f1"], "predicted_lineage_macro_f1": pred_lineage["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
