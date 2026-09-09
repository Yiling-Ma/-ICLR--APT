"""Matched paired-RNA oracle diagnostic for the APT granularity gap."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import patient_cell_scaling as core  # noqa: E402


OUTPUT = ROOT / "outputs/paired_rna_oracle"
RNA_MATRIX = ROOT / "data/transcriptome_hvg2000_lognorm.mtx"
RNA_CELLS = ROOT / "data/transcriptome_hvg2000_cells.txt"
RNA_GENES = ROOT / "data/transcriptome_hvg2000_genes.txt"
ALIGNED_RNA = OUTPUT / "prepared/rna_aligned.npz"
ALIGNED_IDS = OUTPUT / "prepared/cell_ids.npy"
MODALITIES = ("apt", "rna", "apt_rna")
TASKS = ("coarse", "fine")
BOOTSTRAPS = 2000
RANDOM_SEED = 20270909
VERSION = "paired-rna-oracle-v2"
TRAIN_CELLS_PER_PATIENT = 2000


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare(force: bool) -> None:
    metadata, _, _ = core.load_data()
    cell_ids = metadata.cell_id.astype(str).to_numpy(dtype="U128")
    if ALIGNED_RNA.exists() and ALIGNED_IDS.exists() and not force:
        try:
            cached = np.load(ALIGNED_IDS, allow_pickle=False).astype(str)
            if np.array_equal(cached, cell_ids):
                audit_path = OUTPUT / "alignment_audit.json"
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                audit["version"] = VERSION
                atomic_json(audit_path, audit)
                return
        except ValueError:
            # Replace legacy object-dtype caches without ever enabling pickle.
            pass
    source_ids = np.asarray(RNA_CELLS.read_text(encoding="utf-8").splitlines(), dtype=str)
    if len(np.unique(source_ids)) != len(source_ids):
        raise RuntimeError("Paired RNA cell identifiers are not unique.")
    positions = pd.Index(source_ids).get_indexer(cell_ids)
    if np.any(positions < 0):
        raise RuntimeError(f"Missing paired RNA for {int(np.sum(positions < 0))} APT cells.")
    matrix = mmread(RNA_MATRIX).tocsr().astype(np.float32)
    if matrix.shape[0] != len(source_ids):
        raise RuntimeError("RNA matrix and cell identifier file differ.")
    aligned = matrix[positions]
    ALIGNED_RNA.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(ALIGNED_RNA, aligned, compressed=False)
    np.save(ALIGNED_IDS, cell_ids, allow_pickle=False)
    atomic_json(OUTPUT / "alignment_audit.json", {
        "status": "PASS",
        "version": VERSION,
        "apt_cells": len(cell_ids),
        "rna_source_cells": len(source_ids),
        "aligned_cells": aligned.shape[0],
        "rna_features": aligned.shape[1],
        "cell_ids_exactly_aligned": True,
        "rna_gene_file_entries": len(RNA_GENES.read_text(encoding="utf-8").splitlines()),
    })


def patient_class_weights(patient_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(patient_ids, return_inverse=True, return_counts=True)
    patient_weights = 1.0 / counts[inverse]
    patient_weights /= patient_weights.mean()
    class_weights = compute_sample_weight(class_weight="balanced", y=labels)
    weights = patient_weights * class_weights
    return (weights / weights.mean()).astype(np.float64)


def matched_training_indices(sample_ids: np.ndarray, test_mask: np.ndarray, fold: int) -> np.ndarray:
    chosen: list[np.ndarray] = []
    for patient in sorted(set(sample_ids[~test_mask])):
        candidates = np.flatnonzero((sample_ids == patient) & ~test_mask)
        rng = np.random.default_rng(
            np.random.SeedSequence([RANDOM_SEED, fold, zlib.crc32(patient.encode("utf-8"))])
        )
        if len(candidates) > TRAIN_CELLS_PER_PATIENT:
            candidates = rng.choice(candidates, size=TRAIN_CELLS_PER_PATIENT, replace=False)
        chosen.append(np.sort(candidates))
    return np.sort(np.concatenate(chosen))


def scaled_inputs(
    apt: np.ndarray,
    rna: sparse.csr_matrix,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, tuple[Any, Any]]:
    apt_scaler = StandardScaler().fit(apt[train_idx])
    apt_train = apt_scaler.transform(apt[train_idx]).astype(np.float32)
    apt_test = apt_scaler.transform(apt[test_idx]).astype(np.float32)
    rna_scaler = StandardScaler(with_mean=False).fit(rna[train_idx])
    rna_train = rna_scaler.transform(rna[train_idx]).astype(np.float32).tocsr()
    rna_test = rna_scaler.transform(rna[test_idx]).astype(np.float32).tocsr()
    return {
        "apt": (apt_train, apt_test),
        "rna": (rna_train, rna_test),
        "apt_rna": (
            sparse.hstack([sparse.csr_matrix(apt_train), rna_train], format="csr"),
            sparse.hstack([sparse.csr_matrix(apt_test), rna_test], format="csr"),
        ),
    }


def run_fold(fold: int, force: bool) -> None:
    output_npz = OUTPUT / "runs" / f"fold{fold}.npz"
    output_json = OUTPUT / "runs" / f"fold{fold}.json"
    if output_npz.exists() and output_json.exists() and not force:
        return
    prepare(False)
    metadata, apt, _ = core.load_data()
    cached_ids = np.load(ALIGNED_IDS, allow_pickle=False).astype(str)
    if not np.array_equal(cached_ids, metadata.cell_id.astype(str).to_numpy()):
        raise RuntimeError("Cached RNA alignment no longer matches the APT analysis cells.")
    rna = sparse.load_npz(ALIGNED_RNA).tocsr()
    folds = core.load_folds()
    encoders = core.fit_label_encoders(metadata)
    sample_ids = metadata.sample_id.astype(str).to_numpy()
    test_patients = folds[fold]
    test_mask = np.isin(sample_ids, test_patients)
    train_idx = matched_training_indices(sample_ids, test_mask, fold)
    test_idx = np.flatnonzero(test_mask)
    inputs = scaled_inputs(apt, rna, train_idx, test_idx)
    arrays: dict[str, np.ndarray] = {"patient_ids": np.asarray(test_patients, dtype="U64")}
    registry: list[dict[str, Any]] = []
    for task in TASKS:
        labels = encoders[task].transform(metadata[core.TASKS[task]].astype(str))
        weights = patient_class_weights(sample_ids[train_idx], labels[train_idx])
        n_classes = len(encoders[task].classes_)
        for modality in MODALITIES:
            x_train, x_test = inputs[modality]
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-5,
                max_iter=30,
                tol=1e-3,
                average=True,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
            started = time.perf_counter()
            model.fit(x_train, labels[train_idx], sample_weight=weights)
            fit_seconds = time.perf_counter() - started
            prediction = model.predict(x_test).astype(np.int64)
            confusions = np.stack([
                confusion_matrix(
                    labels[test_idx][sample_ids[test_idx] == patient],
                    prediction[sample_ids[test_idx] == patient],
                    labels=np.arange(n_classes),
                )
                for patient in test_patients
            ]).astype(np.int64)
            arrays[f"{modality}_{task}"] = confusions
            registry.append({
                "fold": fold,
                "task": task,
                "modality": modality,
                "fit_seconds": fit_seconds,
                "iterations": int(model.n_iter_),
                "n_train_cells": len(train_idx),
                "n_test_cells": len(test_idx),
                "n_features": int(x_train.shape[1]),
            })
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_npz.with_name(output_npz.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output_npz)
    atomic_json(output_json, {
        "status": "success",
        "version": VERSION,
        "fold": fold,
        "test_patients": test_patients,
        "train_test_patient_overlap": 0,
        "runs": registry,
    })


def aggregate() -> None:
    matrices: dict[tuple[str, str], list[np.ndarray]] = {
        (modality, task): [] for modality in MODALITIES for task in TASKS
    }
    patients: list[str] = []
    for fold in range(core.N_OUTER_FOLDS):
        metadata_path = OUTPUT / "runs" / f"fold{fold}.json"
        result_path = OUTPUT / "runs" / f"fold{fold}.npz"
        if not metadata_path.exists() or not result_path.exists():
            raise RuntimeError(f"Fold {fold} is incomplete.")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if payload.get("status") != "success":
            raise RuntimeError(f"Fold {fold} did not succeed.")
        with np.load(result_path, allow_pickle=False) as saved:
            patients.extend(saved["patient_ids"].astype(str).tolist())
            for key in matrices:
                matrices[key].append(saved[f"{key[0]}_{key[1]}"])
    if len(patients) != 40 or len(set(patients)) != 40:
        raise RuntimeError("OOF patient coverage is not exactly 40 unique patients.")
    pooled = {key: np.concatenate(value, axis=0) for key, value in matrices.items()}
    metric_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_scores: dict[tuple[str, str], np.ndarray] = {}
    for key, matrix in pooled.items():
        balanced = core.patient_balanced_matrix(matrix)
        draws = np.empty(BOOTSTRAPS, dtype=float)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(matrix), size=len(matrix))
            draws[replicate] = core.f1_from_confusion(core.patient_balanced_matrix(matrix[indices]))
        bootstrap_scores[key] = draws
        metric_rows.append({
            "modality": key[0],
            "task": key[1],
            "subject_balanced_macro_f1": core.f1_from_confusion(balanced),
            "pooled_cell_macro_f1": core.f1_from_confusion(matrix.sum(axis=0)),
            "patient_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
            "patient_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
        })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "oracle_metrics.csv", index=False)
    contrasts: list[dict[str, Any]] = []
    contrast_specs = (("rna", "apt"), ("apt_rna", "rna"), ("apt_rna", "apt"))
    for task in TASKS:
        for high, low in contrast_specs:
            # Recompute with paired patient draws rather than differencing independent intervals.
            high_matrix, low_matrix = pooled[(high, task)], pooled[(low, task)]
            values = np.empty(BOOTSTRAPS, dtype=float)
            paired_rng = np.random.default_rng(
                np.random.SeedSequence([RANDOM_SEED, len(task), len(high), len(low)])
            )
            for replicate in range(BOOTSTRAPS):
                indices = paired_rng.integers(0, len(high_matrix), size=len(high_matrix))
                values[replicate] = (
                    core.f1_from_confusion(core.patient_balanced_matrix(high_matrix[indices]))
                    - core.f1_from_confusion(core.patient_balanced_matrix(low_matrix[indices]))
                )
            contrasts.append({
                "task": task,
                "contrast": f"{high}_minus_{low}",
                "delta": float(
                    core.f1_from_confusion(core.patient_balanced_matrix(high_matrix))
                    - core.f1_from_confusion(core.patient_balanced_matrix(low_matrix))
                ),
                "patient_bootstrap_ci_low": float(np.quantile(values, 0.025)),
                "patient_bootstrap_ci_high": float(np.quantile(values, 0.975)),
                "probability_positive": float(np.mean(values > 0)),
            })
    pd.DataFrame(contrasts).to_csv(OUTPUT / "oracle_contrasts.csv", index=False)
    atomic_json(OUTPUT / "qa.json", {
        "status": "PASS",
        "version": VERSION,
        "completed_folds": core.N_OUTER_FOLDS,
        "unique_oof_patients": len(set(patients)),
        "modalities": list(MODALITIES),
        "tasks": list(TASKS),
        "patient_bootstrap_replicates": BOOTSTRAPS,
        "training_cell_budget": f"up to {TRAIN_CELLS_PER_PATIENT} identically sampled cells per development patient",
        "scope": "paired-RNA internal reconstructability oracle, not independent label validation",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--force", action="store_true")
    run_parser = subparsers.add_parser("run-fold")
    run_parser.add_argument("--fold", type=int, required=True, choices=range(core.N_OUTER_FOLDS))
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.force)
    elif args.command == "run-fold":
        run_fold(args.fold, args.force)
    else:
        aggregate()


if __name__ == "__main__":
    main()
