"""Run the OneK1K pool- and donor-disjoint scaling replication."""

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
PREPARED_DIR = Path(
    os.environ.get("ONEK1K_PREPARED_DIR", PROJECT_ROOT / "data/external/onek1k_prepared")
).resolve()
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/onek1k_scaling"
PROTOCOL_VERSION = "onek1k-rna-pool-donor-scaling-v1"
PATIENT_BUDGETS = (8, 16, 32)
CELL_CAPS = (100, 200, 400, 800)
MODEL_NAMES = ("logistic_regression", "xgboost")


def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    metadata = pd.read_parquet(PREPARED_DIR / "metadata.parquet")
    matrix = np.load(PREPARED_DIR / "expression.npy", mmap_mode="r")
    features = pd.read_csv(PREPARED_DIR / "features.csv")["feature_name"].astype(str).tolist()
    audit = json.loads((PREPARED_DIR / "dataset_audit.json").read_text(encoding="utf-8"))
    if len(metadata) != len(matrix) or matrix.shape[1] != len(features):
        raise RuntimeError("OneK1K prepared matrix and metadata differ.")
    if len(features) != int(audit["features"]):
        raise RuntimeError("OneK1K feature audit does not match the matrix.")
    return metadata, matrix, features


def load_folds() -> dict[int, list[str]]:
    payload = json.loads((PREPARED_DIR / "folds.json").read_text(encoding="utf-8"))
    folds = {int(key): list(value) for key, value in payload.items()}
    if sorted(folds) != list(range(core.N_OUTER_FOLDS)):
        raise RuntimeError("OneK1K folds must be indexed 0 through 4.")
    flat = [donor for values in folds.values() for donor in values]
    if len(flat) != len(set(flat)):
        raise RuntimeError("A OneK1K donor occurs in multiple folds.")
    return folds


def fit_label_encoders(metadata: pd.DataFrame) -> dict[str, LabelEncoder]:
    encoders = {
        task: LabelEncoder().fit(metadata[column].astype(str))
        for task, column in core.TASKS.items()
    }
    if len(encoders["coarse"].classes_) != 5 or len(encoders["fine"].classes_) < 20:
        raise RuntimeError("OneK1K taxonomy was unexpectedly truncated.")
    return encoders


def protocol_payload() -> dict[str, object]:
    audit = json.loads((PREPARED_DIR / "dataset_audit.json").read_text(encoding="utf-8"))
    return {
        "version": PROTOCOL_VERSION,
        "dataset": "OneK1K healthy PBMC scRNA-seq",
        "source_url": audit["source_url"],
        "source_sha256": audit["source_sha256"],
        "evaluation_donors": audit["evaluation_donors"],
        "evaluation_cells": audit["evaluation_cells"],
        "calibration_pool": audit["calibration_pool"],
        "calibration_donors": audit["calibration_donors"],
        "features": audit["features"],
        "feature_selection": "variance on ten donors from one excluded calibration pool",
        "patient_budgets": list(PATIENT_BUDGETS),
        "cell_caps": list(CELL_CAPS),
        "resampling_seeds": list(core.RESAMPLING_SEEDS),
        "fixed_total_budgets": [3200, 6400],
        "models": list(MODEL_NAMES),
        "tasks": core.TASKS,
        "model_random_state": core.MODEL_RANDOM_STATE,
        "outer_splits": "five pool- and donor-disjoint folds",
        "donor_sampling": "nested sex-stratified ordering within outer development",
        "cell_sampling": "one uniform within-donor permutation per outer fold and seed",
        "test_cells": "all valid cells from untouched outer-test donors and pools",
        "preprocessing": "feature-wise z-score fit on sampled training cells only",
        "patient_bootstrap_replicates": core.PATIENT_BOOTSTRAP_REPLICATES,
    }


def build_model(model_name: str) -> object:
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=int(os.environ.get("ONEK1K_XGB_N_JOBS", "1")),
            random_state=core.MODEL_RANDOM_STATE,
            eval_metric="mlogloss",
        )
    raise ValueError(f"Unsupported model: {model_name}")


def configure_core() -> None:
    core.PROTOCOL_VERSION = PROTOCOL_VERSION
    core.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    core.PATIENT_BUDGETS = PATIENT_BUDGETS
    core.CELL_CAPS = CELL_CAPS
    core.MODEL_NAMES = MODEL_NAMES
    core.load_data = load_data
    core.load_folds = load_folds
    core.fit_label_encoders = fit_label_encoders
    core.protocol_payload = protocol_payload
    core.build_model = build_model


def audit() -> None:
    metadata, matrix, features = load_data()
    folds = load_folds()
    encoders = fit_label_encoders(metadata)
    folded = {donor for values in folds.values() for donor in values}
    observed = set(metadata["sample_id"].astype(str))
    fold_pools = [set(metadata[metadata["sample_id"].isin(values)]["pool"].astype(str)) for values in folds.values()]
    pool_overlap = sum(len(a & b) for i, a in enumerate(fold_pools) for b in fold_pools[i + 1 :])
    payload = {
        "status": "PASS" if folded == observed and pool_overlap == 0 else "FAIL",
        "cells": len(metadata),
        "donors": len(observed),
        "features": len(features),
        "matrix_shape": list(matrix.shape),
        "fold_sizes": {str(key): len(value) for key, value in folds.items()},
        "coarse_classes": encoders["coarse"].classes_.tolist(),
        "fine_classes": encoders["fine"].classes_.tolist(),
        "minimum_cells_per_donor": int(metadata.groupby("sample_id").size().min()),
        "cross_fold_pool_overlap": pool_overlap,
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
    dry.add_argument("--patient-budget", type=int, default=8, choices=PATIENT_BUDGETS)
    dry.add_argument("--cell-cap", default="100", choices=[str(value) for value in CELL_CAPS])
    dry.add_argument("--model", default="logistic_regression", choices=MODEL_NAMES)
    dry.add_argument("--force", action="store_true")
    run = subparsers.add_parser("run")
    run.add_argument("--models", default=",".join(MODEL_NAMES))
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
        core.run_shard(args)
    elif args.command == "aggregate":
        core.aggregate(args)


if __name__ == "__main__":
    main()
