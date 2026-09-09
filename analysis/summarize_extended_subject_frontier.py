"""Combine P=32 baseline runs with class-matched external subject frontiers."""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import patient_cell_scaling as core


REPLICATES = 2000
RANDOM_SEED = 20270909


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_runs(root: Path, allowed_budgets: set[int], max_seeds: int) -> dict[tuple[Any, ...], np.ndarray]:
    folds: dict[tuple[Any, ...], list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for path in sorted((root / "runs").glob("*.json")):
        payload = json.loads(path.read_text())
        patient_budget = int(payload.get("patient_budget", -1))
        seed = int(payload.get("seed", -1))
        if payload.get("status") != "success" or patient_budget not in allowed_budgets or seed >= max_seeds:
            continue
        key = (seed, int(payload["total"]), patient_budget, str(payload["model"]), str(payload["task"]))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            folds.setdefault(key, []).append(
                (int(payload["fold"]), saved["patient_ids"].astype(str), saved["confusions"])
            )
    matrices: dict[tuple[Any, ...], np.ndarray] = {}
    for key, items in folds.items():
        if len(items) != core.N_OUTER_FOLDS:
            continue
        ordered = sorted(items)
        patient_ids = np.concatenate([item[1] for item in ordered])
        if len(np.unique(patient_ids)) != len(patient_ids):
            raise RuntimeError(f"Repeated held-out subject in {key}.")
        matrices[key] = np.concatenate([item[2] for item in ordered], axis=0)
    return matrices


def paired_joint_interval(
    low: dict[tuple[Any, ...], np.ndarray],
    high: dict[tuple[Any, ...], np.ndarray],
    total: int,
    low_budget: int,
    high_budget: int,
    model: str,
    task: str,
    seeds: list[int],
    dataset: str,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(
        np.random.SeedSequence([
            RANDOM_SEED,
            total,
            low_budget,
            high_budget,
            zlib.crc32(f"{dataset}:{model}:{task}".encode()),
        ])
    )
    values = np.empty(REPLICATES, dtype=float)
    for replicate in range(REPLICATES):
        seed = int(rng.choice(seeds))
        low_matrix = low[(seed, total, low_budget, model, task)]
        high_matrix = high[(seed, total, high_budget, model, task)]
        if low_matrix.shape != high_matrix.shape:
            raise RuntimeError(f"Held-out tensor mismatch for seed {seed}, T={total}, {task}.")
        draw = rng.integers(0, len(low_matrix), size=len(low_matrix))
        values[replicate] = (
            core.f1_from_confusion(core.patient_balanced_matrix(high_matrix[draw]))
            - core.f1_from_confusion(core.patient_balanced_matrix(low_matrix[draw]))
        )
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values > 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    baseline = load_runs(args.baseline, {32}, args.max_seeds)
    extension = load_runs(args.extension, {64, 128}, args.max_seeds)
    combined = baseline | extension
    budgets = sorted({int(key[2]) for key in combined})
    transitions = [(left, right) for left, right in zip(budgets, budgets[1:])]
    if len(budgets) > 2:
        transitions.append((budgets[0], budgets[-1]))

    metric_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int, str, str], list[float]] = {}
    for (seed, total, budget, model, task), matrix in combined.items():
        score = core.f1_from_confusion(core.patient_balanced_matrix(matrix))
        grouped.setdefault((total, budget, model, task), []).append(score)
        metric_rows.append({
            "dataset": args.dataset,
            "seed": seed,
            "total": total,
            "patient_budget": budget,
            "model": model,
            "task": task,
            "patient_balanced_macro_f1": score,
        })
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(args.output / "per_seed_frontier_metrics.csv", index=False)
    summary_rows = [
        {
            "dataset": args.dataset,
            "total": total,
            "patient_budget": budget,
            "model": model,
            "task": task,
            "n_seeds": len(scores),
            "mean": float(np.mean(scores)),
            "seed_ci_low": float(np.quantile(scores, 0.025)),
            "seed_ci_high": float(np.quantile(scores, 0.975)),
        }
        for (total, budget, model, task), scores in sorted(grouped.items())
    ]
    pd.DataFrame(summary_rows).to_csv(args.output / "frontier_metrics.csv", index=False)

    effect_rows: list[dict[str, Any]] = []
    totals = sorted(metric_frame.total.unique())
    models = sorted(metric_frame.model.unique())
    tasks = sorted(metric_frame.task.unique())
    for total in totals:
        for model in models:
            for task in tasks:
                for low_budget, high_budget in transitions:
                    seeds = sorted(set(
                        metric_frame[
                            (metric_frame.total == total)
                            & (metric_frame.model == model)
                            & (metric_frame.task == task)
                            & (metric_frame.patient_budget == low_budget)
                        ].seed
                    ) & set(
                        metric_frame[
                            (metric_frame.total == total)
                            & (metric_frame.model == model)
                            & (metric_frame.task == task)
                            & (metric_frame.patient_budget == high_budget)
                        ].seed
                    ))
                    if not seeds:
                        continue
                    deltas = np.asarray([
                        core.f1_from_confusion(core.patient_balanced_matrix(combined[(seed, total, high_budget, model, task)]))
                        - core.f1_from_confusion(core.patient_balanced_matrix(combined[(seed, total, low_budget, model, task)]))
                        for seed in seeds
                    ])
                    ci_low, ci_high, probability = paired_joint_interval(
                        combined, combined, total, low_budget, high_budget, model, task, seeds, args.dataset
                    )
                    effect_rows.append({
                        "dataset": args.dataset,
                        "total": total,
                        "from_patient_budget": low_budget,
                        "to_patient_budget": high_budget,
                        "model": model,
                        "task": task,
                        "n_seeds": len(seeds),
                        "delta_mean": float(deltas.mean()),
                        "seed_ci_low": float(np.quantile(deltas, 0.025)),
                        "seed_ci_high": float(np.quantile(deltas, 0.975)),
                        "joint_ci_low": ci_low,
                        "joint_ci_high": ci_high,
                        "joint_probability_positive": probability,
                    })
    effects = pd.DataFrame(effect_rows)
    effects.to_csv(args.output / "frontier_effects.csv", index=False)

    expected_budgets = {32, 64, 128} if args.dataset == "onek1k" else {32, 64}
    expected_metric_rows = len(expected_budgets) * len(totals) * len(models) * len(tasks) * args.max_seeds
    qa = {
        "status": "PASS" if set(budgets) == expected_budgets and len(metric_frame) == expected_metric_rows else "INCOMPLETE",
        "dataset": args.dataset,
        "patient_budgets": budgets,
        "completed_seed_metrics": len(metric_frame),
        "expected_seed_metrics": expected_metric_rows,
        "effect_rows": len(effects),
        "joint_resampling_replicates": REPLICATES,
    }
    atomic_json(args.output / "qa.json", qa)
    atomic_json(args.output / "uncertainty_protocol.json", {
        "replicates": REPLICATES,
        "random_seed": RANDOM_SEED,
        "procedure": [
            "sample one matched training-subset seed uniformly",
            "bootstrap held-out subjects with one paired draw for both subject budgets",
            "compute the subject-balanced macro-F1 difference",
        ],
    })
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
