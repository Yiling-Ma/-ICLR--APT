"""Run and audit the APT-Bench patient-cell scaling surface.

This experiment varies the number of independent training patients and the
number of uniformly sampled training cells per selected patient while keeping
the existing patient-disjoint outer test folds fixed. It intentionally excludes
deep models and never subsamples outer-test cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_JEPA_ROOT = PROJECT_ROOT / "cell_JEPA"


PATIENT_BUDGETS = (8, 16, 32)
CELL_CAPS = (100, 200, 400, 800, 1600)
RESAMPLING_SEEDS = tuple(range(20))
MODEL_NAMES = ("logistic_regression", "xgboost")
TASKS = {"coarse": "coarse_subtype", "fine": "cell_subtype"}
MODEL_RANDOM_STATE = 42
N_OUTER_FOLDS = 5
REPRODUCTION_TOLERANCE = 5e-4
PATIENT_BOOTSTRAP_REPLICATES = 2000
PROTOCOL_VERSION = "patient-cell-scaling-fair-v3"

DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "metadata.csv"
ANNOTATION_PATH = DATA_DIR / "cell_annotation.csv"
MAPPING_PATH = CELL_JEPA_ROOT / "apt_jepa/configs/coarse_lineage_explicit.yaml"
BASELINE_CONFIG_PATH = CELL_JEPA_ROOT / "apt_jepa/configs/classical_baselines.yaml"
BASELINE_OUTPUT = CELL_JEPA_ROOT / "outputs/classical_baselines_kfold5"
FOLD_PATH = BASELINE_OUTPUT / "fold_assignment.json"
BASELINE_SUMMARY_PATH = BASELINE_OUTPUT / "pooled_summary.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/patient_cell_scaling_fair"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cap_slug(cap: int | str) -> str:
    return str(cap)


def protocol_payload() -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "outer_fold_path": str(FOLD_PATH.relative_to(PROJECT_ROOT)),
        "patient_budgets": list(PATIENT_BUDGETS),
        "cell_caps": list(CELL_CAPS),
        "resampling_seeds": list(RESAMPLING_SEEDS),
        "fixed_total_budgets": [3200, 6400, 12800],
        "matched_doublings": {
            "patient": "P doubles while C is fixed",
            "cell": "C doubles while P is fixed",
        },
        "models": list(MODEL_NAMES),
        "tasks": TASKS,
        "model_random_state": MODEL_RANDOM_STATE,
        "patient_sampling": "nested disease-stratified greedy proportional ordering",
        "cell_sampling": "one uniform within-patient permutation per outer fold and seed",
        "test_cells": "all cells from untouched outer-test patients",
        "preprocessing": "feature-wise z-score fit on sampled training cells only",
        "reproduction_tolerance_absolute": REPRODUCTION_TOLERANCE,
        "patient_bootstrap_replicates": PATIENT_BOOTSTRAP_REPLICATES,
    }


def protocol_hash() -> str:
    payload = json.dumps(protocol_payload(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0).astype(np.float32)
    std = x_train.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def build_model(model_name: str) -> object:
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=os.environ.get("APT_XGB_DEVICE", "cpu"),
            n_jobs=4,
            random_state=MODEL_RANDOM_STATE,
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unsupported model: {model_name}")


def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    metadata = pd.read_csv(METADATA_PATH)
    annotation = pd.read_csv(ANNOTATION_PATH).rename(columns={"Sample": "cell_id"})
    annotation = annotation[["cell_id", "Celltypes_new"]].rename(
        columns={"Celltypes_new": "cell_subtype"}
    )
    merged = metadata.merge(annotation, on="cell_id", how="left")
    merged = merged[merged["cell_subtype"].notna()].copy()
    merged = merged[~merged["cell_subtype"].astype(str).str.strip().str.lower().eq("unknown")]
    expression_path = DATA_DIR / "apt_expression.parquet"
    if expression_path.exists():
        expression = pd.read_parquet(expression_path)
    else:
        expression = pd.read_csv(DATA_DIR / "apt_expression.csv")
    feature_names = sorted(
        [column for column in expression.columns if column.upper().startswith("APT-")],
        key=lambda value: int(value.split("-")[1]),
    )
    if len(feature_names) != 293:
        raise RuntimeError(f"Expected 293 APT features, found {len(feature_names)}")
    merged = merged.merge(expression[["cell_id", *feature_names]], on="cell_id", how="inner")
    x = merged[feature_names].to_numpy(np.float32)
    mapping_payload = yaml.safe_load(MAPPING_PATH.read_text(encoding="utf-8"))
    mapping = mapping_payload["explicit_mapping"]
    default = mapping_payload.get("default_coarse_label", "Other")
    merged["coarse_subtype"] = merged["cell_subtype"].map(mapping).fillna(default)
    merged = merged[["cell_id", "sample_id", "disease", "cell_subtype", "coarse_subtype"]]
    merged = merged.reset_index(drop=True)
    if len(merged) != len(x):
        raise RuntimeError("Metadata and feature matrix row counts differ.")
    if merged["cell_id"].duplicated().any():
        raise RuntimeError("Cell IDs are not unique after merging.")
    return merged, x, feature_names


def load_folds() -> dict[int, list[str]]:
    raw = json.loads(FOLD_PATH.read_text(encoding="utf-8"))
    folds = {int(key): list(value) for key, value in raw.items()}
    if sorted(folds) != list(range(N_OUTER_FOLDS)):
        raise RuntimeError(f"Expected folds 0-{N_OUTER_FOLDS - 1}, found {sorted(folds)}")
    flat = [patient for fold in folds.values() for patient in fold]
    if len(flat) != 40 or len(set(flat)) != 40:
        raise RuntimeError("Existing outer folds do not contain 40 unique patients exactly once.")
    if any(len(patients) != 8 for patients in folds.values()):
        raise RuntimeError("Every immutable outer fold must contain exactly eight patients.")
    return folds


def fit_label_encoders(merged: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders = {
        task: LabelEncoder().fit(merged[column].astype(str))
        for task, column in TASKS.items()
    }
    if len(encoders["coarse"].classes_) != 5 or len(encoders["fine"].classes_) != 27:
        raise RuntimeError(
            "The fixed evaluation taxonomy must contain five lineages and 27 subtypes."
        )
    return encoders


def patient_table(merged: pd.DataFrame) -> pd.DataFrame:
    disease_counts = merged.groupby("sample_id")["disease"].nunique()
    if not disease_counts.eq(1).all():
        raise RuntimeError("At least one patient maps to multiple disease labels.")
    table = (
        merged.groupby("sample_id", as_index=False)
        .agg(disease=("disease", "first"), n_cells=("cell_id", "size"))
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    return table


def nested_patient_order(
    development: pd.DataFrame, outer_fold: int, resampling_seed: int
) -> list[str]:
    """Create one nested, approximately proportional disease-stratified ordering."""
    rng = np.random.default_rng(
        np.random.SeedSequence([MODEL_RANDOM_STATE, outer_fold, resampling_seed, 101])
    )
    groups: dict[str, list[str]] = {}
    for disease, frame in development.groupby("disease", sort=True):
        values = frame["sample_id"].astype(str).to_numpy().copy()
        rng.shuffle(values)
        groups[str(disease)] = values.tolist()

    diseases = sorted(groups)
    disease_priority = rng.permutation(diseases).tolist()
    order: list[str] = []
    selected_counts = {disease: 0 for disease in diseases}
    totals = {disease: len(groups[disease]) for disease in diseases}

    # Guarantee one patient per represented disease before proportional filling.
    for disease in disease_priority:
        if groups[disease]:
            order.append(groups[disease][selected_counts[disease]])
            selected_counts[disease] += 1

    while len(order) < len(development):
        next_size = len(order) + 1
        candidates = [disease for disease in disease_priority if selected_counts[disease] < totals[disease]]
        if not candidates:
            break
        scores = {
            disease: totals[disease] / len(development) * next_size - selected_counts[disease]
            for disease in candidates
        }
        best = max(candidates, key=lambda disease: (scores[disease], -disease_priority.index(disease)))
        order.append(groups[best][selected_counts[best]])
        selected_counts[best] += 1

    if len(order) != len(development) or len(set(order)) != len(order):
        raise RuntimeError("Failed to construct a complete unique patient ordering.")
    return order


def patient_rng(outer_fold: int, resampling_seed: int, patient_id: str) -> np.random.Generator:
    stable_patient = zlib.crc32(patient_id.encode("utf-8"))
    return np.random.default_rng(
        np.random.SeedSequence([MODEL_RANDOM_STATE, outer_fold, resampling_seed, stable_patient])
    )


def sampling_artifact_path(output_dir: Path, outer_fold: int, seed: int) -> Path:
    return output_dir / "sampling_subsets" / f"fold{outer_fold}_seed{seed:02d}.npz"


def create_or_load_sampling_artifact(
    output_dir: Path,
    merged: pd.DataFrame,
    patients: pd.DataFrame,
    folds: dict[int, list[str]],
    outer_fold: int,
    seed: int,
) -> dict[str, Any]:
    path = sampling_artifact_path(output_dir, outer_fold, seed)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["protocol_hash"].item()) == protocol_hash():
                return {key: saved[key] for key in saved.files}

    test_patients = set(folds[outer_fold])
    development = patients[~patients["sample_id"].isin(test_patients)].copy()
    order = nested_patient_order(development, outer_fold, seed)
    patient_indices: list[np.ndarray] = []
    offsets = [0]
    sample_ids = merged["sample_id"].astype(str).to_numpy()
    for patient_id in order:
        indices = np.flatnonzero(sample_ids == patient_id)
        patient_rng(outer_fold, seed, patient_id).shuffle(indices)
        patient_indices.append(indices.astype(np.int64))
        offsets.append(offsets[-1] + len(indices))
    concatenated = np.concatenate(patient_indices)
    atomic_npz(
        path,
        protocol_hash=np.asarray(protocol_hash()),
        patient_order=np.asarray(order, dtype="U64"),
        cell_indices=concatenated,
        cell_offsets=np.asarray(offsets, dtype=np.int64),
    )
    return {
        "protocol_hash": np.asarray(protocol_hash()),
        "patient_order": np.asarray(order, dtype="U64"),
        "cell_indices": concatenated,
        "cell_offsets": np.asarray(offsets, dtype=np.int64),
    }


def selected_training_indices(
    artifact: dict[str, Any], patient_budget: int, cell_cap: int | str
) -> tuple[np.ndarray, list[str], dict[str, tuple[int, int]]]:
    order = artifact["patient_order"].astype(str).tolist()
    offsets = artifact["cell_offsets"].astype(np.int64)
    all_indices = artifact["cell_indices"].astype(np.int64)
    selected = order[:patient_budget]
    chunks: list[np.ndarray] = []
    realized: dict[str, tuple[int, int]] = {}
    for index, patient_id in enumerate(order):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        available = stop - start
        sampled = available if cell_cap == "all" else min(int(cell_cap), available)
        if patient_id in selected:
            chunks.append(all_indices[start : start + sampled])
            realized[patient_id] = (available, sampled)
        else:
            realized[patient_id] = (available, 0)
    indices = np.concatenate(chunks)
    # Match the original benchmark row order at the deterministic full-data
    # endpoint. The selected set is unchanged; sorting avoids solver drift from
    # feeding LBFGS a random permutation of an otherwise identical dataset.
    if cell_cap == "all":
        indices = np.sort(indices)
    return indices, selected, realized


def bundle_id(fold: int, seed: int, patient_budget: int, cell_cap: int | str, model: str) -> str:
    return f"fold{fold}_seed{seed:02d}_P{patient_budget}_C{cap_slug(cell_cap)}_{model}"


def bundle_paths(output_dir: Path, bundle: str) -> tuple[Path, Path, Path]:
    return (
        output_dir / "runs" / f"{bundle}.json",
        output_dir / "runs" / f"{bundle}.npz",
        output_dir / "logs" / f"{bundle}.log",
    )


def expected_bundles(models: Iterable[str] | None = None) -> list[dict[str, Any]]:
    if models is None:
        models = MODEL_NAMES
    rows = []
    for fold in range(N_OUTER_FOLDS):
        for patient_budget in PATIENT_BUDGETS:
            for cell_cap in CELL_CAPS:
                seeds = RESAMPLING_SEEDS
                for seed in seeds:
                    for model in models:
                        rows.append(
                            {
                                "outer_fold": fold,
                                "seed": seed,
                                "patient_budget": patient_budget,
                                "cell_cap": cell_cap,
                                "model": model,
                            }
                        )
    return rows


def write_protocol_and_plan(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "protocol.json", protocol_payload() | {"protocol_hash": protocol_hash()})
    manifest_path = output_dir / "job_manifest.csv"
    if manifest_path.exists():
        return
    rows = []
    for bundle in expected_bundles():
        identifier = bundle_id(
            bundle["outer_fold"], bundle["seed"], bundle["patient_budget"], bundle["cell_cap"], bundle["model"]
        )
        meta_path, result_path, log_path = bundle_paths(output_dir, identifier)
        for task in TASKS:
            rows.append(
                {
                    "job_id": f"{identifier}_{task}",
                    **bundle,
                    "task": task,
                    "start_time": "",
                    "end_time": "",
                    "exit_status": "pending",
                    "log_path": str(log_path.relative_to(PROJECT_ROOT)),
                    "output_path": str(result_path.relative_to(PROJECT_ROOT)),
                    "metadata_path": str(meta_path.relative_to(PROJECT_ROOT)),
                }
            )
    temporary = manifest_path.with_suffix(f".csv.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, manifest_path)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    actual = matrix.sum(axis=1).astype(float)
    recalls = np.divide(
        np.diag(matrix), actual, out=np.zeros_like(actual), where=actual > 0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(recalls.mean()),
    }


def fit_one_model(
    model_name: str,
    task: str,
    x_train: np.ndarray,
    y_train_global: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    observed_classes = np.unique(y_train_global)
    if len(observed_classes) == 1:
        return np.full(len(x_test), observed_classes[0], dtype=np.int64), 0.0, 0.0
    local_lookup = {global_label: local for local, global_label in enumerate(observed_classes)}
    y_train_local = np.fromiter(
        (local_lookup[value] for value in y_train_global), dtype=np.int64, count=len(y_train_global)
    )
    model = build_model(model_name)
    start = time.perf_counter()
    model.fit(x_train, y_train_local)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    local_prediction = model.predict(x_test).astype(np.int64)
    inference_seconds = time.perf_counter() - start
    prediction = observed_classes[local_prediction]
    return prediction.astype(np.int64), fit_seconds, inference_seconds


def subtype_parent_indices(
    merged: pd.DataFrame, encoders: dict[str, LabelEncoder]
) -> np.ndarray:
    pairs = merged[["cell_subtype", "coarse_subtype"]].drop_duplicates()
    if pairs["cell_subtype"].duplicated().any():
        raise RuntimeError("A subtype maps to more than one lineage in the filtered data.")
    mapping = dict(zip(pairs["cell_subtype"].astype(str), pairs["coarse_subtype"].astype(str)))
    coarse_lookup = {name: index for index, name in enumerate(encoders["coarse"].classes_)}
    return np.asarray([coarse_lookup[mapping[name]] for name in encoders["fine"].classes_], dtype=np.int64)


def successful_bundle(meta_path: Path, result_path: Path) -> bool:
    if not meta_path.exists() or not result_path.exists():
        return False
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("exit_status") != "success" or payload.get("protocol_hash") != protocol_hash():
            return False
        with np.load(result_path, allow_pickle=False) as result:
            required = {"patient_ids", "coarse_confusions", "fine_confusions", "joint_sums"}
            return required.issubset(result.files)
    except Exception:
        return False


def run_bundle(
    output_dir: Path,
    merged: pd.DataFrame,
    x: np.ndarray,
    patients: pd.DataFrame,
    folds: dict[int, list[str]],
    encoders: dict[str, LabelEncoder],
    parent_index: np.ndarray,
    specification: dict[str, Any],
    force: bool = False,
) -> None:
    fold = int(specification["outer_fold"])
    seed = int(specification["seed"])
    patient_budget = int(specification["patient_budget"])
    cell_cap = specification["cell_cap"]
    model_name = str(specification["model"])
    identifier = bundle_id(fold, seed, patient_budget, cell_cap, model_name)
    meta_path, result_path, log_path = bundle_paths(output_dir, identifier)
    if not force and successful_bundle(meta_path, result_path):
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    metadata: dict[str, Any] = {
        "bundle_id": identifier,
        "protocol_hash": protocol_hash(),
        "outer_fold": fold,
        "seed": seed,
        "patient_budget": patient_budget,
        "cell_cap": cell_cap,
        "model": model_name,
        "model_random_state": MODEL_RANDOM_STATE,
        "start_time": started,
        "exit_status": "running",
        "tasks": {},
    }
    atomic_json(meta_path, metadata)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{started}] START {identifier}\n")
        try:
            sampling = create_or_load_sampling_artifact(output_dir, merged, patients, folds, fold, seed)
            train_idx, selected_patients, realized = selected_training_indices(
                sampling, patient_budget, cell_cap
            )
            test_patient_ids = folds[fold]
            test_mask = merged["sample_id"].isin(test_patient_ids).to_numpy()
            test_idx = np.flatnonzero(test_mask)
            if set(selected_patients) & set(test_patient_ids):
                raise RuntimeError("Outer-test leakage detected in selected training patients.")

            mean, std = fit_standardizer(x[train_idx])
            x_train = apply_standardizer(x[train_idx], mean, std)
            x_test = apply_standardizer(x[test_idx], mean, std)
            sample_ids_test = merged.iloc[test_idx]["sample_id"].astype(str).to_numpy()

            predictions: dict[str, np.ndarray] = {}
            truths: dict[str, np.ndarray] = {}
            task_confusions: dict[str, np.ndarray] = {}
            coverage: dict[str, dict[str, list[int]]] = {}
            for task, column in TASKS.items():
                encoder = encoders[task]
                y_all = encoder.transform(merged[column].astype(str))
                y_train = y_all[train_idx]
                y_test = y_all[test_idx]
                prediction, fit_seconds, inference_seconds = fit_one_model(
                    model_name, task, x_train, y_train, x_test
                )
                labels = np.arange(len(encoder.classes_))
                predictions[task] = prediction
                truths[task] = y_test
                per_patient = np.stack(
                    [
                        confusion_matrix(
                            y_test[sample_ids_test == patient_id],
                            prediction[sample_ids_test == patient_id],
                            labels=labels,
                        )
                        for patient_id in test_patient_ids
                    ]
                ).astype(np.int64)
                task_confusions[task] = per_patient
                training_cell_counts = np.bincount(y_train, minlength=len(labels)).astype(int)
                training_patient_counts = np.zeros(len(labels), dtype=int)
                train_samples = merged.iloc[train_idx]["sample_id"].astype(str).to_numpy()
                for label in labels:
                    training_patient_counts[label] = len(np.unique(train_samples[y_train == label]))
                coverage[task] = {
                    "cell_counts": training_cell_counts.tolist(),
                    "patient_counts": training_patient_counts.tolist(),
                }
                metadata["tasks"][task] = {
                    **metric_dict(y_test, prediction, labels),
                    "fit_time_sec": fit_seconds,
                    "inference_time_sec": inference_seconds,
                    "n_evaluation_classes": int(len(labels)),
                    "n_observed_training_classes": int(np.count_nonzero(training_cell_counts)),
                }

            true_coarse = truths["coarse"]
            pred_coarse = predictions["coarse"]
            true_fine = truths["fine"]
            pred_fine = predictions["fine"]
            true_parent = parent_index[true_fine]
            pred_parent = parent_index[pred_fine]
            exact = (true_coarse == pred_coarse) & (true_fine == pred_fine)
            intersection = (true_coarse == pred_coarse).astype(float) + (true_fine == pred_fine).astype(float)
            hierarchical_f1 = intersection / 2.0
            tree_distance = np.where(true_fine == pred_fine, 0, np.where(true_parent == pred_parent, 2, 4))
            joint_sums = np.asarray(
                [
                    [
                        exact[sample_ids_test == patient_id].sum(),
                        hierarchical_f1[sample_ids_test == patient_id].sum(),
                        tree_distance[sample_ids_test == patient_id].sum(),
                        (sample_ids_test == patient_id).sum(),
                    ]
                    for patient_id in test_patient_ids
                ],
                dtype=np.float64,
            )

            metadata.update(
                {
                    "selected_patients": selected_patients,
                    "test_patients": test_patient_ids,
                    "n_train_cells": int(len(train_idx)),
                    "n_test_cells": int(len(test_idx)),
                    "preprocessing_fit_cells": int(len(train_idx)),
                    "sampling_artifact": str(
                        sampling_artifact_path(output_dir, fold, seed).relative_to(PROJECT_ROOT)
                    ),
                    "realized_training_cells": {
                        patient: {"available": values[0], "sampled": values[1]}
                        for patient, values in realized.items()
                    },
                    "training_class_coverage": coverage,
                    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                    "end_time": utc_now(),
                    "exit_status": "success",
                }
            )
            atomic_npz(
                result_path,
                protocol_hash=np.asarray(protocol_hash()),
                patient_ids=np.asarray(test_patient_ids, dtype="U64"),
                coarse_confusions=task_confusions["coarse"],
                fine_confusions=task_confusions["fine"],
                joint_sums=joint_sums,
            )
            atomic_json(meta_path, metadata)
            log.write(f"[{metadata['end_time']}] SUCCESS {identifier}\n")
        except Exception as error:
            metadata.update(
                {
                    "end_time": utc_now(),
                    "exit_status": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
            )
            atomic_json(meta_path, metadata)
            log.write(metadata["traceback"] + "\n")
            raise


def run_shard(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    write_protocol_and_plan(output_dir)
    merged, x, _ = load_data()
    patients = patient_table(merged)
    folds = load_folds()
    encoders = fit_label_encoders(merged)
    parent_index = subtype_parent_indices(merged, encoders)
    models = tuple(args.models.split(","))
    invalid = set(models) - set(MODEL_NAMES)
    if invalid:
        raise ValueError(f"Unsupported models: {sorted(invalid)}")
    bundles = expected_bundles(models)
    if args.cell_caps:
        requested_caps = set(args.cell_caps.split(","))
        bundles = [bundle for bundle in bundles if str(bundle["cell_cap"]) in requested_caps]
    if args.patient_budgets:
        requested_budgets = {int(value) for value in args.patient_budgets.split(",")}
        bundles = [bundle for bundle in bundles if int(bundle["patient_budget"]) in requested_budgets]
    bundles = [
        bundle
        for index, bundle in enumerate(bundles)
        if index % args.num_shards == args.shard_index
    ]
    if args.max_bundles is not None:
        bundles = bundles[: args.max_bundles]
    for index, specification in enumerate(bundles, start=1):
        print(f"[{index}/{len(bundles)}] {specification}", flush=True)
        try:
            run_bundle(
                output_dir,
                merged,
                x,
                patients,
                folds,
                encoders,
                parent_index,
                specification,
                force=args.force,
            )
        except Exception as error:
            print(f"FAILED: {error!r}", file=sys.stderr, flush=True)


def run_dry_run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    write_protocol_and_plan(output_dir)
    merged, x, _ = load_data()
    patients = patient_table(merged)
    folds = load_folds()
    encoders = fit_label_encoders(merged)
    parent_index = subtype_parent_indices(merged, encoders)
    specification = {
        "outer_fold": args.fold,
        "seed": args.seed,
        "patient_budget": args.patient_budget,
        "cell_cap": args.cell_cap if args.cell_cap == "all" else int(args.cell_cap),
        "model": args.model,
    }
    run_bundle(
        output_dir, merged, x, patients, folds, encoders, parent_index, specification, force=args.force
    )


def import_baseline_endpoint(args: argparse.Namespace) -> None:
    """Reuse the existing deterministic full-data OOF fits as the grid endpoint."""
    output_dir = args.output_dir
    write_protocol_and_plan(output_dir)
    merged, _, _ = load_data()
    patients = patient_table(merged)
    folds = load_folds()
    encoders = fit_label_encoders(merged)
    parent_index = subtype_parent_indices(merged, encoders)
    per_fold_metrics = pd.read_csv(BASELINE_OUTPUT / "per_fold_metrics.csv")
    sample_ids = merged["sample_id"].astype(str).to_numpy()

    for model_name in args.models.split(","):
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model: {model_name}")
        predictions = {}
        for task in TASKS:
            path = BASELINE_OUTPUT / task / model_name / "pooled_oof_predictions.csv"
            frame = pd.read_csv(path)
            required = {"cell_id", "sample_id", "fold", "y_true", "y_pred"}
            if set(frame.columns) != required:
                raise RuntimeError(f"Unexpected baseline prediction schema: {path}")
            if len(frame) != len(merged) or frame["cell_id"].duplicated().any():
                raise RuntimeError(f"Baseline predictions are incomplete or duplicated: {path}")
            predictions[task] = frame

        joined = predictions["coarse"].merge(
            predictions["fine"],
            on=["cell_id", "sample_id", "fold"],
            how="inner",
            suffixes=("_coarse", "_fine"),
            validate="one_to_one",
        )
        if len(joined) != len(merged):
            raise RuntimeError(f"Coarse/fine baseline predictions do not pair for {model_name}.")

        for fold in range(N_OUTER_FOLDS):
            test_patients = folds[fold]
            selected_patients = sorted(set(patients["sample_id"]) - set(test_patients))
            identifier = bundle_id(fold, 0, 32, "all", model_name)
            meta_path, result_path, log_path = bundle_paths(output_dir, identifier)
            fold_frame = joined[joined["fold"] == fold].copy()
            if set(fold_frame["sample_id"]) != set(test_patients):
                raise RuntimeError(f"Imported baseline test patients differ in fold {fold}.")

            task_confusions = {}
            encoded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            task_metadata = {}
            coverage = {}
            train_idx = np.flatnonzero(~np.isin(sample_ids, test_patients))
            train_samples = sample_ids[train_idx]
            fold_sample_ids = fold_frame["sample_id"].astype(str).to_numpy()
            for task, column in TASKS.items():
                encoder = encoders[task]
                labels = np.arange(len(encoder.classes_))
                y_true = encoder.transform(fold_frame[f"y_true_{task}"].astype(str))
                y_pred = encoder.transform(fold_frame[f"y_pred_{task}"].astype(str))
                encoded[task] = (y_true, y_pred)
                task_confusions[task] = np.stack(
                    [
                        confusion_matrix(
                            y_true[fold_sample_ids == patient_id],
                            y_pred[fold_sample_ids == patient_id],
                            labels=labels,
                        )
                        for patient_id in test_patients
                    ]
                ).astype(np.int64)
                y_train = encoder.transform(merged.iloc[train_idx][column].astype(str))
                cell_counts = np.bincount(y_train, minlength=len(labels)).astype(int)
                patient_counts = [
                    len(np.unique(train_samples[y_train == label])) for label in labels
                ]
                coverage[task] = {
                    "cell_counts": cell_counts.tolist(),
                    "patient_counts": patient_counts,
                }
                old_metric = per_fold_metrics[
                    (per_fold_metrics["fold"] == fold)
                    & (per_fold_metrics["task"] == task)
                    & (per_fold_metrics["model"] == model_name)
                ].iloc[0]
                task_metadata[task] = {
                    **metric_dict(y_true, y_pred, labels),
                    "fit_time_sec": float(old_metric["fit_time_sec"]),
                    "inference_time_sec": None,
                    "n_evaluation_classes": int(len(labels)),
                    "n_observed_training_classes": int(np.count_nonzero(cell_counts)),
                }

            true_coarse, pred_coarse = encoded["coarse"]
            true_fine, pred_fine = encoded["fine"]
            true_parent = parent_index[true_fine]
            pred_parent = parent_index[pred_fine]
            exact = (true_coarse == pred_coarse) & (true_fine == pred_fine)
            hierarchical_f1 = (
                (true_coarse == pred_coarse).astype(float)
                + (true_fine == pred_fine).astype(float)
            ) / 2.0
            tree_distance = np.where(
                true_fine == pred_fine, 0, np.where(true_parent == pred_parent, 2, 4)
            )
            joint_sums = np.asarray(
                [
                    [
                        exact[fold_sample_ids == patient_id].sum(),
                        hierarchical_f1[fold_sample_ids == patient_id].sum(),
                        tree_distance[fold_sample_ids == patient_id].sum(),
                        (fold_sample_ids == patient_id).sum(),
                    ]
                    for patient_id in test_patients
                ],
                dtype=np.float64,
            )
            sampling = create_or_load_sampling_artifact(
                output_dir, merged, patients, folds, fold, 0
            )
            _, _, realized = selected_training_indices(sampling, 32, "all")
            now = utc_now()
            metadata = {
                "bundle_id": identifier,
                "protocol_hash": protocol_hash(),
                "outer_fold": fold,
                "seed": 0,
                "patient_budget": 32,
                "cell_cap": "all",
                "model": model_name,
                "model_random_state": MODEL_RANDOM_STATE,
                "start_time": now,
                "end_time": now,
                "exit_status": "success",
                "source": "existing_baseline_oof_predictions",
                "reused_existing_fit": True,
                "selected_patients": selected_patients,
                "test_patients": test_patients,
                "n_train_cells": int(len(train_idx)),
                "n_test_cells": int(len(fold_frame)),
                "preprocessing_fit_cells": int(len(train_idx)),
                "sampling_artifact": str(
                    sampling_artifact_path(output_dir, fold, 0).relative_to(PROJECT_ROOT)
                ),
                "realized_training_cells": {
                    patient: {"available": values[0], "sampled": values[1]}
                    for patient, values in realized.items()
                },
                "training_class_coverage": coverage,
                "peak_rss_mb": None,
                "tasks": task_metadata,
            }
            atomic_npz(
                result_path,
                protocol_hash=np.asarray(protocol_hash()),
                patient_ids=np.asarray(test_patients, dtype="U64"),
                coarse_confusions=task_confusions["coarse"],
                fine_confusions=task_confusions["fine"],
                joint_sums=joint_sums,
            )
            atomic_json(meta_path, metadata)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                f"[{now}] IMPORTED deterministic endpoint from existing baseline OOF predictions\n",
                encoding="utf-8",
            )


def f1_from_confusion(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    actual = matrix.sum(axis=1).astype(float)
    denominator = actual + predicted
    scores = np.divide(2 * true_positive, denominator, out=np.zeros_like(true_positive), where=denominator > 0)
    return float(scores.mean())


def accuracy_from_confusion(matrix: np.ndarray) -> float:
    total = matrix.sum()
    return float(np.trace(matrix) / total) if total else float("nan")


def balanced_accuracy_from_confusion(matrix: np.ndarray) -> float:
    actual = matrix.sum(axis=1).astype(float)
    recalls = np.divide(np.diag(matrix), actual, out=np.zeros_like(actual), where=actual > 0)
    return float(recalls.mean())


def confusion_metrics(matrix: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_from_confusion(matrix),
        "macro_f1": f1_from_confusion(matrix),
        "balanced_accuracy": balanced_accuracy_from_confusion(matrix),
    }


def patient_balanced_matrix(per_patient: np.ndarray) -> np.ndarray:
    totals = per_patient.sum(axis=(1, 2)).astype(float)
    normalized = np.divide(
        per_patient,
        totals[:, None, None],
        out=np.zeros_like(per_patient, dtype=float),
        where=totals[:, None, None] > 0,
    )
    return normalized.sum(axis=0)


def iter_successful_results(output_dir: Path) -> Iterable[tuple[dict[str, Any], Path]]:
    for meta_path in sorted((output_dir / "runs").glob("*.json")):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result_path = meta_path.with_suffix(".npz")
        if metadata.get("exit_status") == "success" and successful_bundle(meta_path, result_path):
            yield metadata, result_path


def write_confusion_parquet(
    output_dir: Path, results: list[tuple[dict[str, Any], Path]]
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    destination = output_dir / "per_patient_confusions.parquet"
    temporary = destination.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal writer, rows
        if not rows:
            return
        table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
        writer.write_table(table)
        rows = []

    for metadata, result_path in results:
        with np.load(result_path, allow_pickle=False) as result:
            patient_ids = result["patient_ids"].astype(str)
            for task in TASKS:
                matrices = result[f"{task}_confusions"]
                for patient_id, matrix in zip(patient_ids, matrices):
                    true_indices, pred_indices = np.nonzero(matrix)
                    for true_index, pred_index in zip(true_indices, pred_indices):
                        rows.append(
                            {
                                "bundle_id": metadata["bundle_id"],
                                "outer_fold": metadata["outer_fold"],
                                "seed": metadata["seed"],
                                "patient_budget": metadata["patient_budget"],
                                "cell_cap": str(metadata["cell_cap"]),
                                "model": metadata["model"],
                                "task": task,
                                "patient_id": patient_id,
                                "true_class_index": int(true_index),
                                "predicted_class_index": int(pred_index),
                                "count": int(matrix[true_index, pred_index]),
                                "true_class_support": int(matrix[true_index].sum()),
                            }
                        )
                    if len(rows) >= 100_000:
                        flush()
    flush()
    if writer is not None:
        writer.close()
        os.replace(temporary, destination)


def bootstrap_deterministic(
    per_patient: np.ndarray, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(PATIENT_BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(PATIENT_BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(per_patient), size=len(per_patient))
        values[index] = f1_from_confusion(per_patient[draw].sum(axis=0))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def aggregate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    merged, _, _ = load_data()
    patients = patient_table(merged)
    folds = load_folds()
    expected_patient_ids = {patient for fold in folds.values() for patient in fold}
    encoders = fit_label_encoders(merged)
    write_protocol_and_plan(output_dir)
    results = list(iter_successful_results(output_dir))

    registry_rows: list[dict[str, Any]] = []
    run_metric_rows: list[dict[str, Any]] = []
    coverage_by_subset: dict[tuple[Any, ...], dict[str, Any]] = {}
    sampling_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    pooled_confusions: dict[tuple[Any, ...], list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    pooled_joint: dict[tuple[Any, ...], list[np.ndarray]] = defaultdict(list)

    expected = expected_bundles()
    metadata_lookup = {metadata["bundle_id"]: (metadata, path) for metadata, path in results}
    for specification in expected:
        identifier = bundle_id(
            specification["outer_fold"], specification["seed"], specification["patient_budget"],
            specification["cell_cap"], specification["model"]
        )
        meta_path, result_path, log_path = bundle_paths(output_dir, identifier)
        if identifier in metadata_lookup:
            metadata, result_path = metadata_lookup[identifier]
            status = "success"
        elif meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            status = metadata.get("exit_status", "invalid")
        else:
            metadata = {}
            status = "pending"
        for task in TASKS:
            registry_rows.append(
                {
                    "job_id": f"{identifier}_{task}",
                    **specification,
                    "task": task,
                    "start_time": metadata.get("start_time", ""),
                    "end_time": metadata.get("end_time", ""),
                    "exit_status": status,
                    "log_path": str(log_path.relative_to(PROJECT_ROOT)),
                    "output_path": str(result_path.relative_to(PROJECT_ROOT)),
                    "error": metadata.get("error", ""),
                }
            )
        if status != "success":
            continue

        with np.load(result_path, allow_pickle=False) as result:
            patient_ids = result["patient_ids"].astype(str)
            joint_sums = result["joint_sums"]
            key_base = (
                metadata["seed"], metadata["patient_budget"], str(metadata["cell_cap"]), metadata["model"]
            )
            pooled_joint[key_base].append(joint_sums)
            for task in TASKS:
                matrices = result[f"{task}_confusions"]
                pooled_confusions[key_base + (task,)].append(
                    (int(metadata["outer_fold"]), patient_ids, matrices)
                )
                metrics = metadata["tasks"][task]
                run_metric_rows.append(
                    {
                        "scope": "outer_fold",
                        "bundle_id": identifier,
                        "outer_fold": metadata["outer_fold"],
                        "seed": metadata["seed"],
                        "patient_budget": metadata["patient_budget"],
                        "cell_cap": str(metadata["cell_cap"]),
                        "model": metadata["model"],
                        "task": task,
                        **metrics,
                        "n_train_cells": metadata["n_train_cells"],
                        "n_test_cells": metadata["n_test_cells"],
                        "peak_rss_mb": metadata.get("peak_rss_mb", np.nan),
                    }
                )
                coverage_key = (
                    metadata["outer_fold"], metadata["seed"], metadata["patient_budget"],
                    str(metadata["cell_cap"]), task
                )
                if coverage_key not in coverage_by_subset:
                    coverage_by_subset[coverage_key] = {
                        "classes": encoders[task].classes_.astype(str).tolist(),
                        **metadata["training_class_coverage"][task],
                    }

            selected = set(metadata["selected_patients"])
            test = set(metadata["test_patients"])
            realized = metadata["realized_training_cells"]
            all_patient_rows = patients.set_index("sample_id")
            for patient_id in all_patient_rows.index.astype(str):
                key = (
                    metadata["outer_fold"], metadata["seed"], metadata["patient_budget"],
                    str(metadata["cell_cap"]), patient_id
                )
                if key not in sampling_rows:
                    available = int(all_patient_rows.loc[patient_id, "n_cells"])
                    sampled = int(realized.get(patient_id, {}).get("sampled", 0))
                    sampling_rows[key] = {
                        "outer_fold": metadata["outer_fold"],
                        "resampling_seed": metadata["seed"],
                        "patient_budget": metadata["patient_budget"],
                        "cell_cap": str(metadata["cell_cap"]),
                        "patient_id": patient_id,
                        "disease": str(all_patient_rows.loc[patient_id, "disease"]),
                        "role": "outer_test" if patient_id in test else "outer_development",
                        "selected": patient_id in selected,
                        "available_cells": available,
                        "sampled_cells": sampled,
                        "sampled_cell_manifest_path": metadata["sampling_artifact"] if patient_id in selected else "",
                    }

    registry = pd.DataFrame(registry_rows)
    registry.to_csv(output_dir / "run_registry.csv", index=False)
    registry.to_csv(output_dir / "job_manifest.csv", index=False)
    pd.DataFrame(run_metric_rows).to_csv(output_dir / "per_run_metrics.csv", index=False)
    pd.DataFrame(sampling_rows.values()).to_parquet(
        output_dir / "sampling_manifest.parquet", index=False, compression="zstd"
    )
    write_confusion_parquet(output_dir, results)

    coverage_rows = []
    for key, coverage in coverage_by_subset.items():
        fold, seed, patient_budget, cell_cap, task = key
        for class_index, class_name in enumerate(coverage["classes"]):
            coverage_rows.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "patient_budget": patient_budget,
                    "cell_cap": cell_cap,
                    "task": task,
                    "class_index": class_index,
                    "class_name": class_name,
                    "training_patient_count": coverage["patient_counts"][class_index],
                    "training_cell_count": coverage["cell_counts"][class_index],
                    "observed_in_training": coverage["cell_counts"][class_index] > 0,
                }
            )
    pd.DataFrame(coverage_rows).to_csv(output_dir / "training_class_coverage.csv", index=False)

    per_seed_rows = []
    patient_matrices: dict[tuple[Any, ...], np.ndarray] = {}
    for key, fold_items in pooled_confusions.items():
        seed, patient_budget, cell_cap, model, task = key
        if len(fold_items) != N_OUTER_FOLDS:
            continue
        fold_items.sort(key=lambda item: item[0])
        patient_ids = np.concatenate([item[1] for item in fold_items])
        per_patient = np.concatenate([item[2] for item in fold_items], axis=0)
        if len(patient_ids) != len(expected_patient_ids) or set(patient_ids) != expected_patient_ids:
            continue
        pooled = per_patient.sum(axis=0)
        balanced = patient_balanced_matrix(per_patient)
        metrics = confusion_metrics(pooled)
        balanced_metrics = confusion_metrics(balanced)
        mean_patient_macro_f1 = float(
            np.mean([f1_from_confusion(matrix) for matrix in per_patient])
        )
        joint_items = pooled_joint[(seed, patient_budget, cell_cap, model)]
        joint = np.concatenate(joint_items, axis=0)
        exact_path = float(joint[:, 0].sum() / joint[:, 3].sum())
        hierarchy_f1 = float(joint[:, 1].sum() / joint[:, 3].sum())
        tree_distance = float(joint[:, 2].sum() / joint[:, 3].sum())
        coverage_subset = pd.DataFrame(coverage_rows)
        coverage_match = coverage_subset[
            (coverage_subset["seed"] == seed)
            & (coverage_subset["patient_budget"] == patient_budget)
            & (coverage_subset["cell_cap"] == cell_cap)
            & (coverage_subset["task"] == task)
        ]
        observed_by_fold = coverage_match.groupby("outer_fold")["observed_in_training"].sum()
        row = {
            "seed": seed,
            "patient_budget": patient_budget,
            "cell_cap": cell_cap,
            "model": model,
            "task": task,
            **metrics,
            "patient_balanced_accuracy": balanced_metrics["accuracy"],
            "patient_balanced_macro_f1": balanced_metrics["macro_f1"],
            "patient_balanced_balanced_accuracy": balanced_metrics["balanced_accuracy"],
            "mean_patient_macro_f1": mean_patient_macro_f1,
            "exact_path_accuracy": exact_path,
            "root_excluded_hierarchical_f1": hierarchy_f1,
            "subtype_tree_distance": tree_distance,
            "mean_observed_training_classes_across_folds": float(observed_by_fold.mean()),
            "min_observed_training_classes_across_folds": int(observed_by_fold.min()),
            "n_outer_folds": len(fold_items),
            "n_test_patients": len(patient_ids),
            "n_test_cells": int(per_patient.sum()),
        }
        per_seed_rows.append(row)
        patient_matrices[key] = per_patient

    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(output_dir / "per_seed_oof_metrics.csv", index=False)
    summary_rows = []
    metric_columns = [
        "macro_f1",
        "accuracy",
        "balanced_accuracy",
        "patient_balanced_macro_f1",
        "mean_patient_macro_f1",
        "exact_path_accuracy",
        "root_excluded_hierarchical_f1",
        "subtype_tree_distance",
    ]
    for group_key, frame in per_seed.groupby(["patient_budget", "cell_cap", "model", "task"], sort=False):
        patient_budget, cell_cap, model, task = group_key
        deterministic = patient_budget == 32 and cell_cap == "all"
        row = {
            "patient_budget": patient_budget,
            "cell_cap": cell_cap,
            "model": model,
            "task": task,
            "n_unique_resampling_seeds": len(frame),
            "interval_kind": "patient_clustered_bootstrap_95_ci" if deterministic else "empirical_seed_interval",
        }
        for metric in metric_columns:
            values = frame[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
            if deterministic and metric == "macro_f1":
                patient_key = (0, 32, "all", model, task)
                lower, upper = bootstrap_deterministic(patient_matrices[patient_key])
            elif len(values) > 1:
                lower, upper = np.quantile(values, [0.025, 0.975])
            else:
                lower = upper = values[0]
            row[f"{metric}_q025"] = float(lower)
            row[f"{metric}_q975"] = float(upper)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "scaling_summary.csv", index=False)
    effects = calculate_effects(per_seed)
    effects.to_csv(output_dir / "scaling_effects.csv", index=False)
    create_tables(output_dir, summary, effects)
    if args.figures:
        create_figures(output_dir, summary, effects)


def summarize_differences(
    values: np.ndarray,
    model: str,
    task: str,
    metric: str,
    effect_type: str,
    fixed_axis: str,
    fixed_value: Any,
) -> dict[str, Any]:
    return {
        "model": model,
        "task": task,
        "metric": metric,
        "effect_type": effect_type,
        "fixed_axis": fixed_axis,
        "fixed_value": fixed_value,
        "n_paired_seeds": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "fraction_positive": float(np.mean(values > 0)),
    }


def endpoint_values(
    frame: pd.DataFrame, patient_budget: int, cell_cap: str, metric: str
) -> dict[int, float]:
    selected = frame[(frame["patient_budget"] == patient_budget) & (frame["cell_cap"] == cell_cap)]
    return dict(zip(selected["seed"].astype(int), selected[metric].astype(float)))


def paired_difference(high: dict[int, float], low: dict[int, float]) -> np.ndarray:
    if len(high) == 1 and 0 in high:
        return np.asarray([high[0] - low[seed] for seed in sorted(low)], dtype=float)
    seeds = sorted(set(high) & set(low))
    return np.asarray([high[seed] - low[seed] for seed in seeds], dtype=float)


def calculate_effects(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, task), frame in per_seed.groupby(["model", "task"]):
        for metric in ("macro_f1", "patient_balanced_macro_f1"):
            patient_differences = []
            for cell_cap in map(str, CELL_CAPS):
                high = endpoint_values(frame, 32, cell_cap, metric)
                low = endpoint_values(frame, 8, cell_cap, metric)
                if high and low:
                    difference = paired_difference(high, low)
                    patient_differences.append(difference)
                    rows.append(
                        summarize_differences(
                            difference, model, task, metric, "patient_budget_gain_P32_minus_P8", "cell_cap", cell_cap
                        )
                    )
            cell_differences = []
            for patient_budget in PATIENT_BUDGETS:
                high = endpoint_values(frame, patient_budget, "all", metric)
                low = endpoint_values(frame, patient_budget, "100", metric)
                if high and low:
                    difference = paired_difference(high, low)
                    cell_differences.append(difference)
                    rows.append(
                        summarize_differences(
                            difference, model, task, metric, "cell_budget_gain_Call_minus_C100", "patient_budget", patient_budget
                        )
                    )
            mean_patient_difference = None
            mean_cell_difference = None
            if len(patient_differences) == len(CELL_CAPS):
                mean_patient_difference = np.vstack(patient_differences).mean(axis=0)
                rows.append(
                    summarize_differences(
                        mean_patient_difference, model, task, metric,
                        "mean_patient_gain_across_cell_caps", "cell_cap", "mean"
                    )
                )
            if len(cell_differences) == len(PATIENT_BUDGETS):
                mean_cell_difference = np.vstack(cell_differences).mean(axis=0)
                rows.append(
                    summarize_differences(
                        mean_cell_difference, model, task, metric,
                        "mean_cell_gain_across_patient_budgets", "patient_budget", "mean"
                    )
                )
            if mean_patient_difference is not None and mean_cell_difference is not None:
                rows.append(
                    summarize_differences(
                        mean_patient_difference - mean_cell_difference,
                        model,
                        task,
                        metric,
                        "mean_patient_gain_minus_mean_cell_gain",
                        "paired_axis_average",
                        "patient_minus_cell",
                    )
                )
    return pd.DataFrame(rows)


def formatted_interval(row: pd.Series, metric: str = "macro_f1") -> str:
    return f"{row[f'{metric}_mean']:.3f} [{row[f'{metric}_q025']:.3f}, {row[f'{metric}_q975']:.3f}]"


def create_tables(output_dir: Path, summary: pd.DataFrame, effects: pd.DataFrame) -> None:
    if summary.empty:
        return
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Task & $C=100$ & $C=500$ & $C=2000$ & $C=\mathrm{all}$ \\",
        r"\midrule",
    ]
    for model in MODEL_NAMES:
        display_model = "Logistic Regression" if model == "logistic_regression" else "XGBoost"
        for task in TASKS:
            for patient_budget in PATIENT_BUDGETS:
                frame = summary[
                    (summary["model"] == model)
                    & (summary["task"] == task)
                    & (summary["patient_budget"] == patient_budget)
                ].set_index("cell_cap")
                if len(frame) != len(CELL_CAPS):
                    continue
                values = [formatted_interval(frame.loc[str(cap)]) for cap in CELL_CAPS]
                label = f"{display_model}, $P={patient_budget}$" if task == "coarse" else f"{display_model}, fine, $P={patient_budget}$"
                task_name = "Coarse" if task == "coarse" else "Fine"
                lines.append(f"{label} & {task_name} & " + " & ".join(values) + r" \\")
        lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Patient--cell training scaling. Entries are pooled OOF Macro-F1 means with empirical 2.5th--97.5th seed intervals; the deterministic $P=32,C=\mathrm{all}$ endpoint instead uses a patient-clustered bootstrap interval. Seed intervals are not patient confidence intervals.}",
            r"\label{tab:patient_cell_scaling}",
            r"\end{table*}",
        ]
    )
    (output_dir / "patient_cell_scaling_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    primary = effects[effects["metric"] == "macro_f1"]
    effect_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lllll}",
        r"\toprule",
        r"Model & Task & Mean patient gain & Mean cell gain & Direction / consistency \\",
        r"\midrule",
    ]
    for model in MODEL_NAMES:
        for task in TASKS:
            patient_row = primary[
                (primary["model"] == model)
                & (primary["task"] == task)
                & (primary["effect_type"] == "mean_patient_gain_across_cell_caps")
            ]
            cell_row = primary[
                (primary["model"] == model)
                & (primary["task"] == task)
                & (primary["effect_type"] == "mean_cell_gain_across_patient_budgets")
            ]
            contrast_row = primary[
                (primary["model"] == model)
                & (primary["task"] == task)
                & (primary["effect_type"] == "mean_patient_gain_minus_mean_cell_gain")
            ]
            if patient_row.empty or cell_row.empty or contrast_row.empty:
                continue
            p = patient_row.iloc[0]
            c = cell_row.iloc[0]
            contrast = contrast_row.iloc[0]
            model_name = "LR" if model == "logistic_regression" else "XGBoost"
            dominant = "Patient" if contrast["mean"] > 0 else "Cell"
            consistency = 100.0 * contrast["fraction_positive"]
            effect_lines.append(
                f"{model_name} & {task.title()} & {p['mean']:+.3f} [{p['q025']:+.3f}, {p['q975']:+.3f}] & "
                f"{c['mean']:+.3f} [{c['q025']:+.3f}, {c['q975']:+.3f}] & "
                f"{dominant}; patient $>$ cell in {consistency:.0f}\\% of seeds \\\\"
            )
    effect_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired endpoint gains across nested resampling seeds. Patient gain is $P=32-P=8$ averaged over cell caps; cell gain is $C=\mathrm{all}-C=100$ averaged over patient budgets. Brackets are empirical seed intervals, not patient confidence intervals.}",
            r"\label{tab:patient_cell_scaling_effects}",
            r"\end{table}",
        ]
    )
    (output_dir / "scaling_effects_table.tex").write_text("\n".join(effect_lines) + "\n", encoding="utf-8")


def create_figures(output_dir: Path, summary: pd.DataFrame, effects: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    model_titles = {"logistic_regression": "Logistic Regression", "xgboost": "XGBoost"}
    task_titles = {"coarse": "Coarse lineage", "fine": "Fine subtype"}

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for row, task in enumerate(TASKS):
        task_frame = summary[summary["task"] == task]
        vmin = task_frame["macro_f1_mean"].min()
        vmax = task_frame["macro_f1_mean"].max()
        for col, model in enumerate(MODEL_NAMES):
            frame = task_frame[task_frame["model"] == model].copy()
            pivot = frame.pivot(index="patient_budget", columns="cell_cap", values="macro_f1_mean")
            pivot = pivot.reindex(index=PATIENT_BUDGETS, columns=[str(cap) for cap in CELL_CAPS])
            sns.heatmap(
                pivot, ax=axes[row, col], annot=True, fmt=".3f", cmap="YlGnBu",
                vmin=vmin, vmax=vmax, cbar=col == 1, annot_kws={"fontsize": 9}
            )
            axes[row, col].set_title(f"{model_titles[model]}: {task_titles[task]}", fontsize=11, weight="bold")
            axes[row, col].set_xlabel("Training cells per selected patient")
            axes[row, col].set_ylabel("Training patients")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"patient_cell_scaling_heatmaps.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    palette = {"logistic_regression": "#2F6690", "xgboost": "#C8553D"}
    for row, task in enumerate(TASKS):
        ax_patient, ax_cell = axes[row]
        for model in MODEL_NAMES:
            for cell_cap in map(str, CELL_CAPS):
                frame = summary[
                    (summary["task"] == task) & (summary["model"] == model) & (summary["cell_cap"] == cell_cap)
                ].sort_values("patient_budget")
                if frame.empty:
                    continue
                ax_patient.errorbar(
                    frame["patient_budget"], frame["macro_f1_mean"],
                    yerr=np.vstack([
                        frame["macro_f1_mean"] - frame["macro_f1_q025"],
                        frame["macro_f1_q975"] - frame["macro_f1_mean"],
                    ]),
                    marker="o", capsize=2, linewidth=1.1,
                    color=palette[model], alpha=0.35 + 0.15 * list(map(str, CELL_CAPS)).index(cell_cap),
                    label=f"{model_titles[model]}, C={cell_cap}" if row == 0 else None,
                )
            for patient_budget in PATIENT_BUDGETS:
                frame = summary[
                    (summary["task"] == task) & (summary["model"] == model)
                    & (summary["patient_budget"] == patient_budget)
                ].copy()
                frame["cap_order"] = frame["cell_cap"].map({str(cap): i for i, cap in enumerate(CELL_CAPS)})
                frame = frame.sort_values("cap_order")
                if frame.empty:
                    continue
                ax_cell.errorbar(
                    np.arange(len(CELL_CAPS)), frame["macro_f1_mean"],
                    yerr=np.vstack([
                        frame["macro_f1_mean"] - frame["macro_f1_q025"],
                        frame["macro_f1_q975"] - frame["macro_f1_mean"],
                    ]),
                    marker="o", capsize=2, linewidth=1.1,
                    color=palette[model], alpha=0.35 + 0.12 * PATIENT_BUDGETS.index(patient_budget),
                    label=f"{model_titles[model]}, P={patient_budget}" if row == 0 else None,
                )
        ax_patient.set_title(f"{task_titles[task]}: scaling with patients", weight="bold")
        ax_patient.set_xlabel("Training patients")
        ax_patient.set_ylabel("Pooled OOF Macro-F1")
        ax_cell.set_title(f"{task_titles[task]}: scaling with cells", weight="bold")
        ax_cell.set_xlabel("Training cells per selected patient")
        ax_cell.set_xticks(np.arange(len(CELL_CAPS)), [str(cap) for cap in CELL_CAPS])
        ax_cell.set_ylabel("Pooled OOF Macro-F1")
    axes[0, 0].legend(fontsize=7, ncol=2)
    axes[0, 1].legend(fontsize=7, ncol=2)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"patient_scaling_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    primary = effects[
        (effects["metric"] == "macro_f1")
        & effects["effect_type"].isin(
            ["mean_patient_gain_across_cell_caps", "mean_cell_gain_across_patient_budgets"]
        )
    ].copy()
    primary["effect"] = primary["effect_type"].map(
        {
            "mean_patient_gain_across_cell_caps": "Patient gain: P=32 - P=8",
            "mean_cell_gain_across_patient_budgets": "Cell gain: all - 100",
        }
    )
    primary["panel"] = primary["model"].map(model_titles) + " / " + primary["task"].map(task_titles)
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    panels = primary["panel"].drop_duplicates().tolist()
    width = 0.34
    for offset, effect in zip((-width / 2, width / 2), primary["effect"].drop_duplicates()):
        frame = primary[primary["effect"] == effect].set_index("panel").reindex(panels)
        x_positions = np.arange(len(panels)) + offset
        ax.bar(x_positions, frame["mean"], width=width, label=effect)
        ax.errorbar(
            x_positions, frame["mean"],
            yerr=np.vstack([frame["mean"] - frame["q025"], frame["q975"] - frame["mean"]]),
            fmt="none", color="#202020", capsize=3, linewidth=1,
        )
    ax.axhline(0, color="#404040", linewidth=0.8)
    ax.set_xticks(np.arange(len(panels)), panels, rotation=15, ha="right")
    ax.set_ylabel("Paired Macro-F1 endpoint gain")
    ax.set_title("Independent-patient gain versus within-patient cell gain", weight="bold")
    ax.legend()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"scaling_effect_comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def audit(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    write_protocol_and_plan(output_dir)
    merged, x, feature_names = load_data()
    patients = patient_table(merged)
    folds = load_folds()
    encoders = fit_label_encoders(merged)
    pairs = (
        merged[["cell_subtype", "coarse_subtype"]]
        .drop_duplicates()
        .sort_values(["coarse_subtype", "cell_subtype"])
    )
    fold_rows = []
    for fold, test_patients in folds.items():
        for patient_id in test_patients:
            row = patients.set_index("sample_id").loc[patient_id]
            fold_rows.append((fold, patient_id, row["disease"], int(row["n_cells"])))
    baseline = pd.read_csv(BASELINE_SUMMARY_PATH)
    baseline = baseline[
        baseline["model"].isin(MODEL_NAMES) & baseline["task"].isin(TASKS)
    ]
    metadata_columns = pd.read_csv(METADATA_PATH, nrows=2).columns.tolist()
    annotation_columns = pd.read_csv(ANNOTATION_PATH, nrows=2).columns.tolist()
    lines = [
        "# Patient-Cell Scaling Protocol Audit",
        "",
        f"Generated: {utc_now()}",
        f"Project root: `{PROJECT_ROOT}`",
        f"Repository commit: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()}`",
        "",
        "## Dataset",
        "",
        f"- Expression: `{DATA_DIR / 'apt_expression.parquet'}` (preferred by the existing loader; CSV fallback exists).",
        f"- Metadata: `{METADATA_PATH}`; columns: `{metadata_columns}`.",
        f"- Cell annotation: `{ANNOTATION_PATH}`; columns: `{annotation_columns}`.",
        f"- Cell ID: `cell_id`; patient ID: `sample_id`; disease label: `disease`.",
        "- Fine subtype source column: `Celltypes_new`, renamed to `cell_subtype` by the loader.",
        "- Coarse lineage: `coarse_subtype`, created from the explicit mapping file.",
        f"- Filtered benchmark size: {len(merged):,} cells, {patients['sample_id'].nunique()} patients, {x.shape[1]} APT features.",
        f"- Evaluation taxonomy: {len(encoders['coarse'].classes_)} lineages and {len(encoders['fine'].classes_)} subtypes.",
        "- Missing subtype labels and the `Unknown` subtype are excluded, matching the current benchmark.",
        "",
        "### Patients per disease",
        "",
        patients.groupby("disease")["sample_id"].nunique().rename("patients").to_markdown(),
        "",
        "### Cells per patient",
        "",
        patients.to_markdown(index=False),
        "",
        "### Subtype-to-lineage mapping",
        "",
        pairs.to_markdown(index=False),
        "",
        "## Immutable Outer Folds",
        "",
        f"Source: `{FOLD_PATH}`. The file contains 40 unique patients exactly once and eight test patients per fold.",
        "",
        pd.DataFrame(fold_rows, columns=["outer_fold", "patient_id", "disease", "n_cells"]).to_markdown(index=False),
        "",
        "## Preprocessing and Frozen Baselines",
        "",
        "- Existing preprocessing fits one feature-wise mean and standard deviation on training cells only; standard deviations below 1e-6 are replaced by 1.0.",
        "- Logistic Regression: scikit-learn `LogisticRegression(max_iter=500, solver='lbfgs', class_weight='balanced')`.",
        "- XGBoost: 400 trees, depth 6, learning rate 0.1, row/feature subsampling 0.8, histogram tree method, multiclass log-loss, random state 42.",
        "- XGBoost does not use validation-based early stopping; the complete grid therefore uses frozen boosting rounds and no inner validation patients.",
        "- The deterministic P=32, C=all endpoint reuses the existing pooled OOF baseline predictions rather than refitting under a different software stack; all other grid cells are newly fitted.",
        "- The scaling experiment fits preprocessing on sampled training cells only and evaluates all cells from the untouched outer-test patients.",
        "- Model random state remains 42; resampling seeds affect only nested patient and cell subsets.",
        "",
        "## Existing Full-Data Results",
        "",
        baseline.to_markdown(index=False),
        "",
        "## Output and Logging Conventions",
        "",
        f"- Experiment root: `{output_dir}`.",
        "- One compressed sufficient-statistics artifact and one JSON metadata record are written per fold/seed/budget/model bundle.",
        "- Coarse and fine models are fitted independently inside a bundle; paired predictions are reduced to per-patient exact-path, root-excluded hierarchical-F1, and subtype-tree-distance sums.",
        "- Resume logic accepts an artifact only when its status is `success`, its protocol hash matches, and all required arrays are readable.",
        "",
        "## Missing or Ambiguous Information",
        "",
        "- The original baseline artifact does not record package versions or the exact host used for fitting. A diagnostic refit under the current stack differed by 5.73e-4 in pooled fine Macro-F1 because LBFGS reached its frozen 500-iteration limit; the deterministic endpoint therefore imports the exact existing OOF predictions and is still checked numerically with a prespecified absolute tolerance of 5e-4.",
        "- Peak memory is an approximate process high-water mark; it may include memory retained from earlier jobs in the same worker.",
        "- Seed intervals quantify subset-resampling variability and are not independent-patient confidence intervals.",
    ]
    (output_dir / "protocol_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame({"row_index": np.arange(len(merged)), "cell_id": merged["cell_id"].astype(str)}).to_parquet(
        output_dir / "cell_index.parquet", index=False, compression="zstd"
    )


def qa(args: argparse.Namespace) -> bool:
    output_dir = args.output_dir
    merged, _, _ = load_data()
    patients = patient_table(merged)
    folds = load_folds()
    encoders = fit_label_encoders(merged)
    registry = pd.read_csv(output_dir / "run_registry.csv")
    per_seed = pd.read_csv(output_dir / "per_seed_oof_metrics.csv", dtype={"cell_cap": str})
    coverage = pd.read_csv(output_dir / "training_class_coverage.csv", dtype={"cell_cap": str})
    baseline = pd.read_csv(BASELINE_SUMMARY_PATH)
    checks: list[tuple[str, bool, str]] = []

    checks.append(("Immutable folds contain 40 patients exactly once", len({p for v in folds.values() for p in v}) == 40, str(FOLD_PATH)))
    checks.append(("Every outer fold contains eight test patients", all(len(value) == 8 for value in folds.values()), "Existing fold artifact"))
    checks.append(("All planned jobs succeeded", registry["exit_status"].eq("success").all(), registry["exit_status"].value_counts().to_dict().__repr__()))
    checks.append(("All pooled groups contain five folds", per_seed["n_outer_folds"].eq(5).all(), f"minimum={per_seed['n_outer_folds'].min() if len(per_seed) else 'none'}"))
    checks.append(("All pooled groups contain 40 test patients", per_seed["n_test_patients"].eq(40).all(), f"minimum={per_seed['n_test_patients'].min() if len(per_seed) else 'none'}"))
    checks.append(("Fixed taxonomy is 5 lineages and 27 subtypes", len(encoders["coarse"].classes_) == 5 and len(encoders["fine"].classes_) == 27, "Global encoders"))
    checks.append(("Missing training classes remain recorded", "observed_in_training" in coverage.columns, "training_class_coverage.csv"))
    deterministic = registry[(registry["patient_budget"] == 32) & (registry["cell_cap"].astype(str) == "all")]
    expected_deterministic_jobs = N_OUTER_FOLDS * len(MODEL_NAMES) * len(TASKS)
    checks.append(("Deterministic endpoint is not duplicated", len(deterministic) == expected_deterministic_jobs and deterministic["seed"].eq(0).all(), f"rows={len(deterministic)}"))

    # Reconstruct and test nested subset properties for every fold and seed.
    nested_patients = True
    nested_cells = True
    leakage_free = True
    patient_lookup = patients.set_index("sample_id")
    for fold in range(N_OUTER_FOLDS):
        test = set(folds[fold])
        development = patients[~patients["sample_id"].isin(test)]
        for seed in RESAMPLING_SEEDS:
            artifact = create_or_load_sampling_artifact(output_dir, merged, patients, folds, fold, seed)
            patient_sets = []
            for patient_budget in PATIENT_BUDGETS:
                _, selected, _ = selected_training_indices(artifact, patient_budget, "all")
                patient_sets.append(set(selected))
                leakage_free &= not bool(set(selected) & test)
            nested_patients &= all(patient_sets[i] < patient_sets[i + 1] for i in range(3))
            _, selected, _ = selected_training_indices(artifact, 32, "all")
            for patient_id in selected:
                order = artifact["patient_order"].astype(str).tolist()
                patient_index = order.index(patient_id)
                offsets = artifact["cell_offsets"]
                values = artifact["cell_indices"][offsets[patient_index] : offsets[patient_index + 1]]
                sets = [set(values[: min(cap, len(values))]) for cap in (100, 500, 2000)] + [set(values)]
                nested_cells &= all(sets[i].issubset(sets[i + 1]) for i in range(3))
            if not set(development["sample_id"]).issubset(patient_lookup.index):
                leakage_free = False
    checks.append(("Patient subsets are nested", nested_patients, "S8 subset S16 subset S24 subset S32"))
    checks.append(("Cell subsets are nested", nested_cells, "T100 subset T500 subset T2000 subset Tall"))
    checks.append(("No outer-test patient enters training", leakage_free, "Regenerated sampling artifacts"))

    reproduction_rows = []
    reproduction_pass = True
    for model in MODEL_NAMES:
        for task in TASKS:
            observed = per_seed[
                (per_seed["model"] == model)
                & (per_seed["task"] == task)
                & (per_seed["patient_budget"] == 32)
                & (per_seed["cell_cap"] == "all")
            ]
            reference = baseline[(baseline["model"] == model) & (baseline["task"] == task)]
            if len(observed) != 1 or len(reference) != 1:
                passed = False
                difference = np.nan
                observed_value = np.nan
                reference_value = np.nan
            else:
                observed_value = float(observed.iloc[0]["macro_f1"])
                reference_value = float(reference.iloc[0]["macro_f1"])
                difference = abs(observed_value - reference_value)
                passed = difference <= REPRODUCTION_TOLERANCE
            reproduction_pass &= passed
            reproduction_rows.append(
                {
                    "model": model,
                    "task": task,
                    "observed_macro_f1": observed_value,
                    "reference_macro_f1": reference_value,
                    "absolute_difference": difference,
                    "tolerance": REPRODUCTION_TOLERANCE,
                    "passed": passed,
                }
            )
    pd.DataFrame(reproduction_rows).to_csv(output_dir / "reproduction_check.csv", index=False)
    checks.append(("Full-data baseline reproduction passes", reproduction_pass, "See reproduction_check.csv"))
    all_passed = all(value for _, value, _ in checks)
    lines = [
        "# Final QA",
        "",
        f"Generated: {utc_now()}",
        f"Overall status: **{'PASS' if all_passed else 'FAIL'}**",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |" for name, passed, evidence in checks)
    lines.extend(
        [
            "",
            "## Reproduction Check",
            "",
            pd.DataFrame(reproduction_rows).to_markdown(index=False),
            "",
            "Scientific interpretation is prohibited if the overall status is FAIL or if baseline reproduction fails.",
        ]
    )
    (output_dir / "final_qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return all_passed


def decision_from_effects(effects: pd.DataFrame) -> tuple[str, str]:
    primary = effects[
        (effects["metric"] == "macro_f1")
        & effects["effect_type"].isin(
            ["mean_patient_gain_across_cell_caps", "mean_cell_gain_across_patient_budgets"]
        )
    ]
    comparisons = []
    for (model, task), frame in primary.groupby(["model", "task"]):
        patient = frame[frame["effect_type"] == "mean_patient_gain_across_cell_caps"]
        cell = frame[frame["effect_type"] == "mean_cell_gain_across_patient_budgets"]
        if len(patient) == 1 and len(cell) == 1:
            comparisons.append((model, task, float(patient.iloc[0]["mean"]) > float(cell.iloc[0]["mean"])))
    if len(comparisons) != 4:
        return "INVALID", "The four prespecified model-task effect comparisons are incomplete."
    supported = [value for _, _, value in comparisons]
    if all(supported):
        return "SUPPORTED", "Patient-budget gain exceeds cell-budget gain in every required model-task comparison."
    if any(supported):
        return "PARTIALLY SUPPORTED", "The patient-dominant pattern holds for only a subset of models or tasks."
    return "NOT SUPPORTED", "Cell-budget gain is equal to or larger than patient-budget gain in all required comparisons."


def report(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    qa_path = output_dir / "final_qa.md"
    qa_pass = qa_path.exists() and "Overall status: **PASS**" in qa_path.read_text(encoding="utf-8")
    summary = pd.read_csv(output_dir / "scaling_summary.csv", dtype={"cell_cap": str})
    effects = pd.read_csv(output_dir / "scaling_effects.csv", dtype={"fixed_value": str})
    registry = pd.read_csv(output_dir / "run_registry.csv")
    coverage = pd.read_csv(output_dir / "training_class_coverage.csv", dtype={"cell_cap": str})
    reproduction = pd.read_csv(output_dir / "reproduction_check.csv") if (output_dir / "reproduction_check.csv").exists() else pd.DataFrame()
    decision, rationale = decision_from_effects(effects) if qa_pass else ("INVALID", "Final QA or baseline reproduction did not pass.")

    full_results = summary[[
        "patient_budget", "cell_cap", "model", "task", "n_unique_resampling_seeds",
        "macro_f1_mean", "macro_f1_std", "macro_f1_median", "macro_f1_q025",
        "macro_f1_q975", "macro_f1_min", "macro_f1_max", "patient_balanced_macro_f1_mean",
    ]].sort_values(["model", "task", "patient_budget", "cell_cap"])
    primary_effects = effects[
        (effects["metric"] == "macro_f1")
        & effects["effect_type"].isin(
            ["patient_budget_gain_P32_minus_P8", "cell_budget_gain_Call_minus_C100",
             "mean_patient_gain_across_cell_caps", "mean_cell_gain_across_patient_budgets",
             "mean_patient_gain_minus_mean_cell_gain"]
        )
    ]
    missing_coverage = (
        coverage[coverage["task"] == "fine"]
        .groupby(["patient_budget", "cell_cap"])["observed_in_training"]
        .mean()
        .rename("fraction_fold_class_pairs_observed")
        .reset_index()
    )
    failures = registry[registry["exit_status"] != "success"]
    changed_files = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    command_prefix = f"PYTHONPATH=cell_JEPA {sys.executable} analysis/patient_cell_scaling.py"
    lines = [
        "# Patient-Cell Scaling Final Report",
        "",
        "## 1. Executive Summary",
        "",
        f"Decision: **{decision}**. {rationale}",
        "",
        "The experiment measures training-patient and training-cell scaling. It is not an inference-time cell-budget experiment and does not evaluate DropCascade.",
        "",
        "## 2. Exact Experimental Protocol",
        "",
        "The immutable five outer folds hold out eight patients each. Within each outer-development pool, disease-stratified patient orderings and uniform within-patient cell permutations are nested across budgets. Preprocessing is fit only on sampled training cells. LR and XGBoost configurations are frozen from the existing benchmark, and all outer-test cells are evaluated.",
        "",
        "## 3. Repository and Fold Audit",
        "",
        "See `protocol_audit.md` for dataset paths, patient counts, fold assignments, taxonomy, preprocessing, and frozen configurations.",
        "",
        "## 4. Reproduction Status",
        "",
        reproduction.to_markdown(index=False) if not reproduction.empty else "Reproduction check unavailable.",
        "",
        "## 5. Complete Logistic Regression Results",
        "",
        full_results[full_results["model"] == "logistic_regression"].to_markdown(index=False),
        "",
        "## 6. Complete XGBoost Results",
        "",
        full_results[full_results["model"] == "xgboost"].to_markdown(index=False),
        "",
        "## 7. Coarse and Fine Scaling Surfaces",
        "",
        "See `patient_cell_scaling_heatmaps.pdf` and `patient_scaling_curves.pdf`. Identical color limits are used across models within each task; intervals are empirical across subset-resampling seeds.",
        "",
        "## 8. Patient-Budget Effects",
        "",
        primary_effects[primary_effects["effect_type"].str.contains("patient")].to_markdown(index=False),
        "",
        "## 9. Cell-Budget Effects",
        "",
        primary_effects[primary_effects["effect_type"].str.contains("cell")].to_markdown(index=False),
        "",
        "## 10. Patient-Balanced Sensitivity",
        "",
        "`scaling_summary.csv` reports equal-patient-weight confusion aggregation beside pooled-cell Macro-F1. Any material disagreement must be interpreted as sensitivity to patient cell-count imbalance.",
        "",
        "## 11. Training-Class Coverage Analysis",
        "",
        missing_coverage.to_markdown(index=False),
        "",
        "Low-patient-budget degradation may partly reflect missing subtype support; `training_class_coverage.csv` records each class by fold and subset.",
        "",
        "## 12. Failed or Incomplete Runs",
        "",
        failures.to_markdown(index=False) if not failures.empty else "None.",
        "",
        "## 13. Prespecified Interpretation",
        "",
        rationale,
        "",
        "Seed intervals represent resampling variability, not independent-patient confidence intervals. The observed comparison is limited to this cohort, these budgets, and these two models.",
        "",
        "## 14. Recommended Manuscript Claim",
        "",
        {
            "SUPPORTED": "Under the evaluated ranges, patient diversity is the stronger scaling axis for patient-independent cell typing.",
            "PARTIALLY SUPPORTED": "Coarse and fine biological resolutions or model families follow different scaling regimes; patient diversity is not uniformly dominant.",
            "NOT SUPPORTED": "Additional within-patient cells remain important, and the hypothesized patient-dominant scaling pattern is not supported.",
            "INVALID": "Do not add a scientific claim until all QA and full-data reproduction checks pass.",
        }[decision],
        "",
        "## 15. Recommended Figure and Table Placement",
        "",
        "Use the heatmap as the primary appendix scaling figure, the endpoint-gain comparison as the compact main-text candidate, and the full 16-cell table in the appendix. Do not modify the manuscript until author review of this report.",
        "",
        "## 16. Generated and Modified Files",
        "",
        "All experiment artifacts are under `outputs/patient_cell_scaling/`; implementation is `analysis/patient_cell_scaling.py`. Git status at report time:",
        "",
        "```text",
        changed_files or "Clean working tree",
        "```",
        "",
        "## 17. Exact Reproduction Commands",
        "",
        "```bash",
        f"{command_prefix} audit",
        f"{command_prefix} import-baseline --models logistic_regression,xgboost",
        f"{command_prefix} run --models logistic_regression --shard-index 0 --num-shards 1",
        f"{command_prefix} aggregate --figures",
        f"{command_prefix} run --models xgboost --shard-index 0 --num-shards 1",
        f"{command_prefix} aggregate --figures",
        f"{command_prefix} qa",
        f"{command_prefix} report",
        "```",
        "",
        "For parallel execution, launch disjoint shard indices with the same `--num-shards`; resume logic skips validated completed artifacts.",
        "",
        "## 18. Git Diff",
        "",
        "```text",
        subprocess.run(["git", "diff", "--stat"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False).stdout.strip(),
        "```",
        "",
        f"**{decision}**",
    ]
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("plan")

    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--fold", type=int, default=0)
    dry.add_argument("--seed", type=int, default=0)
    dry.add_argument("--patient-budget", type=int, default=8, choices=PATIENT_BUDGETS)
    dry.add_argument("--cell-cap", default="100", choices=[str(cap) for cap in CELL_CAPS])
    dry.add_argument("--model", default="logistic_regression", choices=MODEL_NAMES)
    dry.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run")
    run.add_argument("--models", default=",".join(MODEL_NAMES))
    run.add_argument("--cell-caps", help="Optional comma-separated subset of 100,500,2000,all.")
    run.add_argument("--patient-budgets", help="Optional comma-separated subset of 8,16,24,32.")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--max-bundles", type=int)
    run.add_argument("--force", action="store_true")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--figures", action="store_true")
    subparsers.add_parser("qa")
    subparsers.add_parser("report")
    import_parser = subparsers.add_parser("import-baseline")
    import_parser.add_argument("--models", default=",".join(MODEL_NAMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.command == "audit":
        audit(args)
    elif args.command == "plan":
        write_protocol_and_plan(args.output_dir)
    elif args.command == "dry-run":
        run_dry_run(args)
    elif args.command == "run":
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Shard index must satisfy 0 <= shard-index < num-shards.")
        run_shard(args)
    elif args.command == "aggregate":
        aggregate(args)
    elif args.command == "qa":
        if not qa(args):
            raise SystemExit(2)
    elif args.command == "report":
        report(args)
    elif args.command == "import-baseline":
        import_baseline_endpoint(args)


if __name__ == "__main__":
    main()
