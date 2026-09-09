"""Class-matched fixed-total control for patient-cell scaling.

For every dataset, outer fold, resampling seed, task, and total-cell budget,
the P=8 development subset defines an integer per-class quota.  The identical
quota is then sampled from nested P=8, 16, and 32 subject sets.  This freezes
training label support and class proportions while varying the number of
subjects contributing cells to each class.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
import patient_cell_scaling as core  # noqa: E402


PATIENT_BUDGETS = (8, 16, 32)
TOTAL_BUDGETS = (3200, 6400)
SEEDS = tuple(range(20))
TASKS = core.TASKS
VERSION = "class-matched-fixed-total-v2"
INDEX_CACHE: dict[tuple[int, str], dict[tuple[int, str], np.ndarray]] = {}
TEST_MATRIX_CACHE: dict[tuple[int, int], np.ndarray] = {}


def configure_dataset(name: str) -> tuple[Any, tuple[str, ...]]:
    if name == "apt":
        return core, ("logistic_regression",)
    if name == "onek1k":
        adapter = importlib.import_module("onek1k_scaling")
        adapter.configure_core()
        return adapter, ("logistic_regression",)
    if name in {"combat_rna", "combat_adt"}:
        os.environ["COMBAT_MODALITY"] = name.split("_", 1)[1]
        adapter = importlib.import_module("combat_citeseq_scaling")
        adapter.configure_core()
        return adapter, ("logistic_regression",)
    raise ValueError(f"Unsupported dataset: {name}")


def protocol(dataset: str, models: tuple[str, ...]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "dataset": dataset,
        "patient_budgets": list(PATIENT_BUDGETS),
        "total_training_cell_budgets": list(TOTAL_BUDGETS),
        "seeds": list(SEEDS),
        "models": list(models),
        "tasks": TASKS,
        "quota_reference": "P=8 nested development subset, separately by fold/seed/task/T",
        "class_matching": "identical integer per-class cell quota at P=8,16,32",
        "within_class_sampling": "capacity-constrained round robin across eligible selected subjects",
        "test_data": "all cells from immutable untouched outer-test subjects",
        "primary_metric": "subject-balanced macro-F1 from pooled OOF subject-normalized confusions",
    }


def protocol_hash(dataset: str, models: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(protocol(dataset, models), sort_keys=True).encode()).hexdigest()[:16]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def allocate_quotas(reference_counts: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder quotas, retaining every class observed at P=8."""
    counts = reference_counts.astype(np.int64)
    observed = np.flatnonzero(counts > 0)
    if counts.sum() < total:
        raise ValueError(f"P=8 has only {counts.sum()} cells, below T={total}.")
    if total < len(observed):
        raise ValueError("Total budget cannot retain every observed class.")
    quota = np.zeros_like(counts)
    quota[observed] = 1
    remaining = total - len(observed)
    if remaining:
        weights = counts[observed] / counts[observed].sum()
        raw = weights * remaining
        base = np.floor(raw).astype(np.int64)
        quota[observed] += base
        remainder = remaining - int(base.sum())
        order = observed[np.argsort(-(raw - base), kind="stable")]
        quota[order[:remainder]] += 1
    # A proportional quota cannot exceed P=8 capacity in aggregate, but the
    # mandatory one-cell floor can. Repair deterministically if needed.
    while np.any(quota > counts):
        over = np.flatnonzero(quota > counts)
        excess = int((quota[over] - counts[over]).sum())
        quota[over] = counts[over]
        spare = np.flatnonzero(counts > quota)
        for label in spare[np.argsort(-(counts[spare] - quota[spare]), kind="stable")]:
            add = min(excess, int(counts[label] - quota[label]))
            quota[label] += add
            excess -= add
            if excess == 0:
                break
        if excess:
            raise RuntimeError("Unable to repair class quotas.")
    if quota.sum() != total:
        raise RuntimeError("Class quotas do not sum to the requested total.")
    return quota


def shuffled_class_queues(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    selected: list[str],
    fold: int,
    seed: int,
    task: str,
) -> dict[tuple[int, str], np.ndarray]:
    cache_key = (id(metadata), task)
    base = INDEX_CACHE.get(cache_key)
    if base is None:
        frame = pd.DataFrame(
            {
                "label": labels.astype(np.int64),
                "patient": metadata["sample_id"].astype(str).to_numpy(),
                "row": np.arange(len(metadata), dtype=np.int64),
            }
        )
        base = {
            (int(label), str(patient)): group["row"].to_numpy(np.int64)
            for (label, patient), group in frame.groupby(["label", "patient"], sort=False)
        }
        INDEX_CACHE[cache_key] = base
    queues: dict[tuple[int, str], np.ndarray] = {}
    for (label, patient), base_indices in base.items():
        if patient not in selected:
            continue
        indices = base_indices.copy()
        if not len(indices):
                continue
        stable = zlib.crc32(f"{task}:{patient}:{label}".encode())
        rng = np.random.default_rng(np.random.SeedSequence([core.MODEL_RANDOM_STATE, fold, seed, stable]))
        rng.shuffle(indices)
        queues[(int(label), patient)] = indices
    return queues


def sample_matched(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    selected: list[str],
    quotas: np.ndarray,
    fold: int,
    seed: int,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Meet each class quota while spreading it over subjects round-robin."""
    queues = shuffled_class_queues(metadata, labels, selected, fold, seed, task)
    chosen: list[int] = []
    donor_counts = np.zeros(len(quotas), dtype=np.int64)
    for label, quota in enumerate(quotas):
        if quota == 0:
            continue
        donors = [patient for patient in selected if (label, patient) in queues]
        if sum(len(queues[(label, patient)]) for patient in donors) < quota:
            raise RuntimeError(f"Insufficient capacity for class {label} at P={len(selected)}.")
        positions = {patient: 0 for patient in donors}
        active = donors.copy()
        taken = 0
        while taken < quota:
            next_active: list[str] = []
            for patient in active:
                queue = queues[(label, patient)]
                position = positions[patient]
                if position < len(queue) and taken < quota:
                    chosen.append(int(queue[position]))
                    positions[patient] += 1
                    taken += 1
                if positions[patient] < len(queue):
                    next_active.append(patient)
            if not next_active and taken < quota:
                raise RuntimeError(f"Round-robin exhausted class {label} early.")
            active = next_active
        donor_counts[label] = sum(position > 0 for position in positions.values())
    result = np.asarray(chosen, dtype=np.int64)
    if len(result) != int(quotas.sum()) or len(np.unique(result)) != len(result):
        raise RuntimeError("Matched sample has an incorrect size or duplicate cells.")
    return result, donor_counts


def specification_rows(models: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {"fold": fold, "seed": seed, "total": total, "patient_budget": patient_budget, "model": model, "task": task}
        for fold in range(core.N_OUTER_FOLDS)
        for seed in SEEDS
        for total in TOTAL_BUDGETS
        for patient_budget in PATIENT_BUDGETS
        for model in models
        for task in TASKS
    ]


def bundle_name(spec: dict[str, Any]) -> str:
    return "fold{fold}_seed{seed:02d}_T{total}_P{patient_budget}_{model}_{task}".format(**spec)


def run_one(
    output: Path,
    metadata: pd.DataFrame,
    matrix: np.ndarray,
    folds: dict[int, list[str]],
    encoders: dict[str, Any],
    patients: pd.DataFrame,
    models: tuple[str, ...],
    spec: dict[str, Any],
    force: bool,
) -> None:
    name = bundle_name(spec)
    meta_path = output / "runs" / f"{name}.json"
    result_path = output / "runs" / f"{name}.npz"
    phash = protocol_hash(output.name, models)
    if not force and meta_path.exists() and result_path.exists():
        existing = json.loads(meta_path.read_text())
        if existing.get("status") == "success" and existing.get("protocol_hash") == phash:
            return
    fold, seed, total, patient_budget = (int(spec[key]) for key in ("fold", "seed", "total", "patient_budget"))
    task, model_name = str(spec["task"]), str(spec["model"])
    test_patients = folds[fold]
    development = patients[~patients["sample_id"].isin(test_patients)].copy()
    order = core.nested_patient_order(development, fold, seed)
    selected = order[:patient_budget]
    reference = order[:8]
    labels = encoders[task].transform(metadata[TASKS[task]].astype(str))
    sample_ids = metadata["sample_id"].astype(str).to_numpy()
    reference_idx = np.flatnonzero(np.isin(sample_ids, reference))
    reference_counts = np.bincount(labels[reference_idx], minlength=len(encoders[task].classes_))
    quotas = allocate_quotas(reference_counts, total)
    train_idx, donor_counts = sample_matched(metadata, labels, selected, quotas, fold, seed, task)
    realized_counts = np.bincount(labels[train_idx], minlength=len(quotas))
    if not np.array_equal(realized_counts, quotas):
        raise RuntimeError("Realized class counts differ from frozen quotas.")
    test_idx = np.flatnonzero(np.isin(sample_ids, test_patients))
    mean, std = core.fit_standardizer(matrix[train_idx])
    x_train = core.apply_standardizer(matrix[train_idx], mean, std)
    test_cache_key = (id(matrix), fold)
    raw_test = TEST_MATRIX_CACHE.get(test_cache_key)
    if raw_test is None:
        raw_test = np.asarray(matrix[test_idx], dtype=np.float32)
        TEST_MATRIX_CACHE[test_cache_key] = raw_test
    x_test = core.apply_standardizer(raw_test, mean, std)
    started = time.perf_counter()
    prediction, fit_seconds, inference_seconds = core.fit_one_model(
        model_name, task, x_train, labels[train_idx], x_test
    )
    del x_train, x_test
    label_range = np.arange(len(quotas))
    test_sample_ids = sample_ids[test_idx]
    per_patient = np.stack([
        confusion_matrix(labels[test_idx][test_sample_ids == patient], prediction[test_sample_ids == patient], labels=label_range)
        for patient in test_patients
    ]).astype(np.int64)
    contributing = len(np.unique(sample_ids[train_idx]))
    selected_metadata = metadata[metadata["sample_id"].astype(str).isin(selected)]
    selected_covariates: dict[str, Any] = {}
    for column in ("disease", "source", "institute", "sex", "pool"):
        if column not in selected_metadata:
            continue
        patient_covariate = selected_metadata.groupby("sample_id")[column].first().astype(str)
        selected_covariates[column] = {
            "n_levels": int(patient_covariate.nunique()),
            "counts": patient_covariate.value_counts().sort_index().to_dict(),
        }
    payload = {
        **spec,
        "status": "success",
        "protocol_hash": phash,
        "selected_patients": selected,
        "test_patients": test_patients,
        "n_train_cells": int(len(train_idx)),
        "n_test_cells": int(len(test_idx)),
        "n_classes": int(len(quotas)),
        "observed_class_count": int(np.count_nonzero(quotas)),
        "unseen_at_training_class_count": int(np.count_nonzero(quotas == 0)),
        "actual_contributing_patients": contributing,
        "mean_class_donor_coverage": float(donor_counts[quotas > 0].mean()),
        "min_class_donor_coverage": int(donor_counts[quotas > 0].min()),
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter() - started,
        "class_quotas": quotas.tolist(),
        "class_donor_counts": donor_counts.tolist(),
        "class_names": encoders[task].classes_.astype(str).tolist(),
        "selected_subject_covariates": selected_covariates,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    expected_shape = (len(test_patients), len(label_range), len(label_range))
    if per_patient.shape != expected_shape or per_patient.dtype == object:
        raise RuntimeError(
            f"Invalid per-patient confusion tensor {per_patient.shape}/{per_patient.dtype}; "
            f"expected {expected_shape} with a numeric dtype."
        )
    temporary = result_path.with_name(result_path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, patient_ids=np.asarray(test_patients, dtype="U64"), confusions=per_patient)
    os.replace(temporary, result_path)
    atomic_json(meta_path, payload)


def run(args: argparse.Namespace, adapter: Any, models: tuple[str, ...]) -> None:
    metadata, matrix, _ = core.load_data()
    folds = core.load_folds()
    encoders = core.fit_label_encoders(metadata)
    patients = core.patient_table(metadata)
    rows = specification_rows(models)
    requested_models = set(args.models.split(",")) if args.models else set(models)
    rows = [row for row in rows if row["model"] in requested_models]
    if args.fold is not None:
        rows = [row for row in rows if row["fold"] == args.fold]
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
    if args.max_jobs is not None:
        rows = rows[: args.max_jobs]
    for index, spec in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {bundle_name(spec)}", flush=True)
        try:
            run_one(args.output, metadata, matrix, folds, encoders, patients, models, spec, args.force)
        except Exception as error:
            print(f"FAILED {bundle_name(spec)}: {error!r}", file=sys.stderr, flush=True)


def aggregate(args: argparse.Namespace, models: tuple[str, ...]) -> None:
    rows: list[dict[str, Any]] = []
    confusions: dict[tuple[Any, ...], list[tuple[int, np.ndarray]]] = {}
    quota_groups: dict[tuple[Any, ...], dict[int, tuple[int, ...]]] = {}
    for path in sorted((args.output / "runs").glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "success":
            continue
        key = (payload["seed"], payload["total"], payload["patient_budget"], payload["model"], payload["task"])
        quota_key = (payload["fold"], payload["seed"], payload["total"], payload["model"], payload["task"])
        quota_groups.setdefault(quota_key, {})[int(payload["patient_budget"])] = tuple(payload["class_quotas"])
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            confusions.setdefault(key, []).append((int(payload["fold"]), saved["confusions"]))
        rows.append({key: value for key, value in payload.items() if key not in {"selected_patients", "test_patients", "class_quotas", "class_donor_counts", "class_names", "selected_subject_covariates"}})
    run_frame = pd.DataFrame(rows)
    run_frame.to_csv(args.output / "run_registry.csv", index=False)
    seed_rows: list[dict[str, Any]] = []
    patient_matrices: dict[tuple[Any, ...], np.ndarray] = {}
    for key, items in confusions.items():
        if len(items) != core.N_OUTER_FOLDS:
            continue
        seed, total, patient_budget, model, task = key
        per_patient = np.concatenate([item[1] for item in sorted(items)], axis=0)
        patient_matrices[key] = per_patient
        balanced = core.patient_balanced_matrix(per_patient)
        matching = run_frame[
            (run_frame.seed == seed) & (run_frame.total == total) &
            (run_frame.patient_budget == patient_budget) & (run_frame.model == model) &
            (run_frame.task == task)
        ]
        seed_rows.append({
            "seed": seed, "total": total, "patient_budget": patient_budget, "model": model, "task": task,
            "patient_balanced_macro_f1": core.f1_from_confusion(balanced),
            "observed_class_count": float(matching.observed_class_count.mean()),
            "unseen_at_training_class_count": float(matching.unseen_at_training_class_count.mean()),
            "actual_contributing_patients": float(matching.actual_contributing_patients.mean()),
            "mean_class_donor_coverage": float(matching.mean_class_donor_coverage.mean()),
        })
    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(args.output / "per_seed_oof_metrics.csv", index=False)
    effects: list[dict[str, Any]] = []
    for (total, model, task), frame in per_seed.groupby(["total", "model", "task"]):
        wide = frame.pivot(index="seed", columns="patient_budget", values="patient_balanced_macro_f1").dropna()
        if not {8, 32}.issubset(wide.columns):
            continue
        delta = (wide[32] - wide[8]).to_numpy(float)
        coverage = frame.pivot(index="seed", columns="patient_budget", values="mean_class_donor_coverage").dropna()
        p8_matrices = np.mean(
            [patient_matrices[(seed, total, 8, model, task)] for seed in wide.index], axis=0
        )
        p32_matrices = np.mean(
            [patient_matrices[(seed, total, 32, model, task)] for seed in wide.index], axis=0
        )
        rng = np.random.default_rng(np.random.SeedSequence([2027, int(total), len(task)]))
        patient_bootstrap = np.empty(core.PATIENT_BOOTSTRAP_REPLICATES, dtype=float)
        for replicate in range(core.PATIENT_BOOTSTRAP_REPLICATES):
            draw = rng.integers(0, len(p8_matrices), size=len(p8_matrices))
            patient_bootstrap[replicate] = (
                core.f1_from_confusion(core.patient_balanced_matrix(p32_matrices[draw]))
                - core.f1_from_confusion(core.patient_balanced_matrix(p8_matrices[draw]))
            )
        effects.append({
            "total": total, "model": model, "task": task, "n_seeds": len(delta),
            "p8_mean": float(wide[8].mean()), "p32_mean": float(wide[32].mean()),
            "delta_p32_minus_p8_mean": float(delta.mean()),
            "delta_ci_low": float(np.quantile(delta, 0.025)), "delta_ci_high": float(np.quantile(delta, 0.975)),
            "patient_bootstrap_ci_low": float(np.quantile(patient_bootstrap, 0.025)),
            "patient_bootstrap_ci_high": float(np.quantile(patient_bootstrap, 0.975)),
            "positive_seed_fraction": float(np.mean(delta > 0)),
            "mean_class_donor_coverage_p8": float(coverage[8].mean()),
            "mean_class_donor_coverage_p32": float(coverage[32].mean()),
        })
    effect_frame = pd.DataFrame(effects)
    effect_frame.to_csv(args.output / "class_matched_effects.csv", index=False)
    expected = len(specification_rows(models))
    quotas_exact = bool(quota_groups) and all(
        set(group) == set(PATIENT_BUDGETS) and len(set(group.values())) == 1
        for group in quota_groups.values()
    )
    qa = {
        "status": "PASS" if len(run_frame) == expected and len(effect_frame) == len(TOTAL_BUDGETS) * len(models) * len(TASKS) and quotas_exact else "INCOMPLETE",
        "expected_jobs": expected,
        "completed_jobs": len(run_frame),
        "effect_rows": len(effect_frame),
        "total_cell_budget_exact": bool(len(run_frame) and run_frame.n_train_cells.eq(run_frame.total).all()),
        "class_quota_vectors_exactly_matched_across_p": quotas_exact,
        "all_nominal_patients_contribute": bool(len(run_frame) and run_frame.actual_contributing_patients.eq(run_frame.patient_budget).all()),
    }
    atomic_json(args.output / "qa.json", qa)
    print(json.dumps(qa, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("apt", "onek1k", "combat_rna", "combat_adt"), required=True)
    parser.add_argument("--output", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--models")
    run_parser.add_argument("--fold", type=int, choices=range(core.N_OUTER_FOLDS))
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--shard-index", type=int, default=0)
    run_parser.add_argument("--max-jobs", type=int)
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("aggregate")
    args = parser.parse_args()
    adapter, models = configure_dataset(args.dataset)
    args.output = (args.output or PROJECT_ROOT / "outputs" / "class_matched_scaling" / args.dataset).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "protocol.json", protocol(args.dataset, models) | {"protocol_hash": protocol_hash(args.dataset, models)})
    if args.command == "run":
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("Invalid shard specification.")
        run(args, adapter, models)
    else:
        aggregate(args, models)


if __name__ == "__main__":
    main()
