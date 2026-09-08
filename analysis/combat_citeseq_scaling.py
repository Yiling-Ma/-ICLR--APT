"""Run the COMBAT CITE-seq donor-cell scaling replication.

This adapter reuses the audited APT scaling engine while replacing only the
dataset, taxonomy, donor folds, and protocol metadata. Prepared matrices are
created by ``prepare_combat_citeseq.py`` and may live on node-local storage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import patient_cell_scaling as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODALITY = os.environ.get("COMBAT_MODALITY", "rna").lower()
if MODALITY not in {"rna", "adt"}:
    raise ValueError("COMBAT_MODALITY must be 'rna' or 'adt'.")
PREPARED_DIR = Path(
    os.environ.get("COMBAT_PREPARED_DIR", PROJECT_ROOT / "data/external/combat_prepared")
).resolve()
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "outputs/combat_citeseq_scaling"
    if MODALITY == "rna"
    else "outputs/combat_citeseq_scaling_adt"
)
PROTOCOL_VERSION = f"combat-citeseq-{MODALITY}-donor-cell-scaling-v1"


def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    metadata = pd.read_parquet(PREPARED_DIR / "metadata.parquet")
    matrix = np.load(PREPARED_DIR / "expression.npy", mmap_mode="r")
    features = pd.read_csv(PREPARED_DIR / "features.csv")["feature_name"].astype(str).tolist()
    audit = json.loads((PREPARED_DIR / "dataset_audit.json").read_text(encoding="utf-8"))
    if len(metadata) != len(matrix):
        raise RuntimeError("COMBAT metadata and expression rows differ.")
    if matrix.shape[1] != int(audit["features"]) or len(features) != int(audit["features"]):
        raise RuntimeError("COMBAT matrix does not match its frozen feature audit.")
    return metadata, matrix, features


def load_folds() -> dict[int, list[str]]:
    payload = json.loads((PREPARED_DIR / "folds.json").read_text(encoding="utf-8"))
    folds = {int(key): list(value) for key, value in payload.items()}
    if sorted(folds) != list(range(core.N_OUTER_FOLDS)):
        raise RuntimeError("COMBAT folds must be indexed 0 through 4.")
    flat = [patient for fold in folds.values() for patient in fold]
    if len(flat) != len(set(flat)):
        raise RuntimeError("A COMBAT donor occurs in multiple outer folds.")
    return folds


def fit_label_encoders(metadata: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders = {
        task: LabelEncoder().fit(metadata[column].astype(str))
        for task, column in core.TASKS.items()
    }
    if len(encoders["coarse"].classes_) != 5:
        raise RuntimeError("COMBAT coarse taxonomy must contain five lineages.")
    if len(encoders["fine"].classes_) < 20:
        raise RuntimeError("COMBAT fine taxonomy was unexpectedly truncated.")
    return encoders


def protocol_payload() -> dict[str, object]:
    audit = json.loads((PREPARED_DIR / "dataset_audit.json").read_text(encoding="utf-8"))
    return {
        "version": PROTOCOL_VERSION,
        "dataset": f"COMBAT CITE-seq {MODALITY.upper()} channel",
        "source_url": audit["source_url"],
        "source_sha256": audit["source_sha256"],
        "evaluation_donors": audit["evaluation_donors"],
        "evaluation_cells": audit["evaluation_cells"],
        "calibration_donors": audit["calibration_donors"],
        "features": audit["features"],
        "feature_selection": (
            "variance on ten disjoint calibration donors"
            if MODALITY == "rna"
            else "all 192 author-released antibody-derived tag features"
        ),
        "patient_budgets": list(core.PATIENT_BUDGETS),
        "cell_caps": list(core.CELL_CAPS),
        "resampling_seeds": list(core.RESAMPLING_SEEDS),
        "fixed_total_budgets": [3200, 6400, 12800],
        "models": list(core.MODEL_NAMES),
        "tasks": core.TASKS,
        "model_random_state": core.MODEL_RANDOM_STATE,
        "donor_sampling": "nested source-stratified ordering within outer development",
        "cell_sampling": "one uniform within-donor permutation per outer fold and seed",
        "test_cells": "all valid cells from untouched outer-test donors",
        "preprocessing": "feature-wise z-score fit on sampled training cells only",
        "patient_bootstrap_replicates": core.PATIENT_BOOTSTRAP_REPLICATES,
    }


def build_model(model_name: str) -> object:
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
    if model_name == "xgboost":
        device = os.environ.get("COMBAT_XGB_DEVICE", "cpu")
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            n_jobs=int(os.environ.get("COMBAT_XGB_N_JOBS", "2")),
            random_state=core.MODEL_RANDOM_STATE,
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unsupported model: {model_name}")


def configure_core() -> None:
    core.PROTOCOL_VERSION = PROTOCOL_VERSION
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    if MODALITY == "adt":
        core.MODEL_NAMES = ("logistic_regression",)
    core.load_data = load_data
    core.load_folds = load_folds
    core.fit_label_encoders = fit_label_encoders
    core.protocol_payload = protocol_payload
    core.build_model = build_model


def audit() -> None:
    metadata, matrix, features = load_data()
    folds = load_folds()
    encoders = fit_label_encoders(metadata)
    folded = {patient for values in folds.values() for patient in values}
    observed = set(metadata["sample_id"].astype(str))
    payload = {
        "status": "PASS" if folded == observed else "FAIL",
        "prepared_dir": str(PREPARED_DIR),
        "cells": len(metadata),
        "donors": len(observed),
        "features": len(features),
        "matrix_shape": list(matrix.shape),
        "fold_sizes": {str(key): len(value) for key, value in folds.items()},
        "coarse_classes": encoders["coarse"].classes_.tolist(),
        "fine_classes": encoders["fine"].classes_.tolist(),
        "minimum_cells_per_donor": int(metadata.groupby("sample_id").size().min()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit")
    subparsers.add_parser("plan")
    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--fold", type=int, default=0)
    dry.add_argument("--seed", type=int, default=0)
    dry.add_argument("--patient-budget", type=int, default=8, choices=core.PATIENT_BUDGETS)
    dry.add_argument("--cell-cap", default="100", choices=[str(value) for value in core.CELL_CAPS])
    dry.add_argument("--model", default="logistic_regression", choices=core.MODEL_NAMES)
    dry.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--models", default=",".join(core.MODEL_NAMES))
    run.add_argument("--cell-caps")
    run.add_argument("--patient-budgets")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--max-bundles", type=int)
    run.add_argument("--force", action="store_true")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_core()
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.command == "audit":
        audit()
    elif args.command == "plan":
        core.write_protocol_and_plan(args.output_dir)
    elif args.command == "dry-run":
        core.run_dry_run(args)
    elif args.command == "run":
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Shard index must satisfy 0 <= index < num_shards.")
        core.run_shard(args)
    elif args.command == "aggregate":
        core.aggregate(args)


if __name__ == "__main__":
    main()
