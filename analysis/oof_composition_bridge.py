"""Strictly nested OOF subtype predictions for the composition bridge audit.

This module implements the base-model layer only. Downstream disease models
must consume inner-OOF features for outer-training patients and outer-model
features for outer-test patients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CELL_JEPA_ROOT = PROJECT_ROOT / "cell_JEPA"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CELL_JEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(CELL_JEPA_ROOT))

from apt_jepa.data.preprocessing import apply_standardizer, fit_standardizer  # noqa: E402
from analysis.patient_cell_scaling import (  # noqa: E402
    BASELINE_CONFIG_PATH,
    FOLD_PATH,
    MAPPING_PATH,
    TASKS,
    fit_label_encoders,
    load_data,
    load_folds,
    patient_table,
    subtype_parent_indices,
)


OUTPUT = PROJECT_ROOT / "outputs/oof_composition_bridge"
SEED = 42
N_FOLDS = 5
PROTOCOL_VERSION = "oof-composition-bridge-base-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def protocol() -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "outer_folds": str(FOLD_PATH.relative_to(PROJECT_ROOT)),
        "base_model": "XGBoost fine-subtype classifier",
        "base_model_uses_disease_labels": False,
        "lineage_probabilities": "sum fixed subtype probabilities by frozen subtype-to-lineage mapping",
        "outer_contexts": list(range(N_FOLDS)),
        "inner_folds": "the four immutable folds remaining after removing the outer-test fold",
        "preprocessing": "feature z-score fit on base-model training patients only",
        "random_state": SEED,
        "xgboost": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
        },
    }


def protocol_hash() -> str:
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()[:16]


def jobs() -> list[dict]:
    return [
        {
            "outer_fold": outer,
            "heldout_fold": heldout,
            "role": "outer-test" if heldout == outer else "inner-oof",
            "job_id": f"outer{outer}_heldout{heldout}",
        }
        for outer in range(N_FOLDS)
        for heldout in range(N_FOLDS)
    ]


def paths(output: Path, job_id: str) -> tuple[Path, Path, Path]:
    return (
        output / "base_predictions" / f"{job_id}.json",
        output / "base_predictions" / f"{job_id}.npz",
        output / "logs" / f"{job_id}.log",
    )


def audit(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    merged, x, features = load_data()
    folds = load_folds()
    encoders = fit_label_encoders(merged)
    parent = subtype_parent_indices(merged, encoders)
    patients = patient_table(merged)
    fold_lookup = {patient: fold for fold, values in folds.items() for patient in values}
    patients["outer_fold"] = patients["sample_id"].map(fold_lookup)
    mapping = pd.DataFrame(
        {
            "fine_subtype": encoders["fine"].classes_,
            "coarse_lineage": encoders["coarse"].classes_[parent],
        }
    )
    mapping.to_csv(output / "subtype_to_lineage_mapping.csv", index=False)
    patients.to_csv(output / "patient_fold_manifest.csv", index=False)
    manifest = []
    for job in jobs():
        meta, result, log = paths(output, job["job_id"])
        manifest.append(
            job
            | {
                "status": "success" if meta.exists() and result.exists() else "pending",
                "metadata_path": str(meta.relative_to(PROJECT_ROOT)),
                "output_path": str(result.relative_to(PROJECT_ROOT)),
                "log_path": str(log.relative_to(PROJECT_ROOT)),
            }
        )
    pd.DataFrame(manifest).to_csv(output / "job_manifest.csv", index=False)
    atomic_json(output / "protocol.json", protocol() | {"protocol_hash": protocol_hash()})
    disease_counts = patients["disease"].value_counts().sort_index()
    cell_stats = patients["n_cells"].describe()
    lines = [
        "# OOF Composition Bridge Protocol Audit",
        "",
        f"- Protocol: `{PROTOCOL_VERSION}` (`{protocol_hash()}`)",
        f"- Project root: `{PROJECT_ROOT}`",
        f"- Data matrix: `{x.shape[0]:,}` cells x `{x.shape[1]}` aptamers",
        f"- Retained benchmark patients: `{len(patients)}`",
        f"- Cell ID unique: `{not merged['cell_id'].duplicated().any()}`",
        f"- Patient ID column: `sample_id`; disease column: `disease`",
        f"- Coarse label column: `{TASKS['coarse']}`; fine label column: `{TASKS['fine']}`",
        f"- Fixed taxonomy: `{len(encoders['coarse'].classes_)}` lineages and `{len(encoders['fine'].classes_)}` subtypes",
        f"- Immutable fold file: `{FOLD_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Mapping config: `{MAPPING_PATH.relative_to(PROJECT_ROOT)}`",
        f"- Frozen baseline config: `{BASELINE_CONFIG_PATH.relative_to(PROJECT_ROOT)}`",
        "- Base-model disease supervision: `No`",
        "- Existing pooled OOF hard labels are patient-disjoint but insufficient for primary stacking; strict nested refitting is required.",
        "- Existing classical OOF artifacts contain hard labels only; probability refits are required.",
        "",
        "## Patients Per Disease",
        "",
        disease_counts.rename("patients").to_frame().to_markdown(),
        "",
        "## Cells Per Patient",
        "",
        cell_stats.rename("cells").to_frame().to_markdown(),
        "",
        "## Nested Design",
        "",
        "For outer fold f, fold f is untouched outer test. For each other immutable fold g, the base model is fit on the 24 patients outside f and g and predicts fold g. These four inner-OOF blocks form disease-training features. A fifth model is fit on all 32 outer-development patients and predicts fold f.",
        "",
        "## Ambiguities and Scope",
        "",
        "- No batch, acquisition-run, panel-version, or processing-date field has yet been verified; technical-only analysis defaults to log cell count unless audit finds such fields.",
        "- DropCascade uses disease-aware contrastive supervision and is secondary only.",
        "- This command does not edit the manuscript or overwrite prior benchmark outputs.",
    ]
    (output / "protocol_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    design = """# Strict Nested Cross-Fitting Design

For every outer fold `f`:

1. Keep the eight patients in `f` untouched.
2. For each remaining fixed fold `g`, train the fine-subtype XGBoost model on patients outside `f` and `g`, then predict all cells in `g`.
3. Concatenate the four predictions from step 2 to obtain inner-OOF subtype probabilities for all 32 outer-development patients.
4. Train one fine-subtype XGBoost model on all 32 outer-development patients and predict all cells in `f`.
5. Aggregate probabilities within patient. Train the disease model on step-3 patient features and evaluate on step-4 patient features.

Preprocessing is re-fit inside every base-model fit. Disease labels are never provided to the base subtype model. Soft lineage composition is obtained by summing the 27 fixed subtype probabilities according to the frozen subtype-to-lineage mapping.
"""
    (output / "nested_crossfit_design.md").write_text(design, encoding="utf-8")


def valid_result(meta_path: Path, result_path: Path) -> bool:
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "success" or meta.get("protocol_hash") != protocol_hash():
            return False
        with np.load(result_path, allow_pickle=False) as result:
            return result["fine_probabilities"].shape[1] == 27
    except Exception:
        return False


def fit_job(output: Path, job: dict, device: str, n_jobs: int, force: bool) -> None:
    meta_path, result_path, log_path = paths(output, job["job_id"])
    if not force and valid_result(meta_path, result_path):
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    metadata = job | {
        "protocol_hash": protocol_hash(),
        "status": "running",
        "start_time": started,
        "host": platform.node(),
        "device": device,
    }
    atomic_json(meta_path, metadata)
    try:
        from xgboost import XGBClassifier

        merged, x, _ = load_data()
        folds = load_folds()
        encoders = fit_label_encoders(merged)
        outer = set(folds[job["outer_fold"]])
        heldout = set(folds[job["heldout_fold"]])
        train_patients = set(merged["sample_id"].astype(str)) - outer
        if job["heldout_fold"] != job["outer_fold"]:
            train_patients -= heldout
        evaluation_patients = outer if job["heldout_fold"] == job["outer_fold"] else heldout
        if train_patients & evaluation_patients:
            raise RuntimeError("Base-model train/evaluation patient leakage.")
        sample_ids = merged["sample_id"].astype(str).to_numpy()
        train_idx = np.flatnonzero(np.isin(sample_ids, sorted(train_patients)))
        eval_idx = np.flatnonzero(np.isin(sample_ids, sorted(evaluation_patients)))
        mean, std = fit_standardizer(x[train_idx])
        x_train = apply_standardizer(x[train_idx], mean, std)
        x_eval = apply_standardizer(x[eval_idx], mean, std)
        y = encoders["fine"].transform(merged["cell_subtype"].astype(str))
        observed = np.unique(y[train_idx])
        local = {value: index for index, value in enumerate(observed)}
        y_local = np.asarray([local[value] for value in y[train_idx]], dtype=np.int64)
        model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            n_jobs=n_jobs,
            random_state=SEED,
            eval_metric="mlogloss",
        )
        fit_start = time.time()
        model.fit(x_train, y_local)
        local_probability = model.predict_proba(x_eval)
        probability = np.zeros((len(eval_idx), len(encoders["fine"].classes_)), dtype=np.float32)
        probability[:, observed] = local_probability.astype(np.float32)
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Expanded subtype probabilities do not sum to one.")
        atomic_npz(
            result_path,
            row_indices=eval_idx.astype(np.int64),
            fine_probabilities=probability,
            fine_true=y[eval_idx].astype(np.int16),
        )
        metadata.update(
            {
                "status": "success",
                "end_time": now(),
                "fit_time_sec": time.time() - fit_start,
                "train_patients": sorted(train_patients),
                "evaluation_patients": sorted(evaluation_patients),
                "n_train_patients": len(train_patients),
                "n_evaluation_patients": len(evaluation_patients),
                "n_train_cells": len(train_idx),
                "n_evaluation_cells": len(eval_idx),
                "observed_global_class_indices": observed.tolist(),
                "fixed_class_order": encoders["fine"].classes_.astype(str).tolist(),
            }
        )
        atomic_json(meta_path, metadata)
        log_path.write_text(f"[{started}] SUCCESS {job['job_id']}\n", encoding="utf-8")
    except Exception as error:
        metadata.update({"status": "failed", "end_time": now(), "error": repr(error), "traceback": traceback.format_exc()})
        atomic_json(meta_path, metadata)
        log_path.write_text(metadata["traceback"], encoding="utf-8")
        raise


def run(args: argparse.Namespace) -> None:
    selected = [job for index, job in enumerate(jobs()) if index % args.num_shards == args.shard_index]
    for index, job in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {job['job_id']}", flush=True)
        fit_job(args.output_dir, job, args.device, args.n_jobs, args.force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    fit = sub.add_parser("fit-base")
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--n-jobs", type=int, default=4)
    fit.add_argument("--shard-index", type=int, default=0)
    fit.add_argument("--num-shards", type=int, default=1)
    fit.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "audit":
        audit(args.output_dir)
    else:
        run(args)


if __name__ == "__main__":
    main()
