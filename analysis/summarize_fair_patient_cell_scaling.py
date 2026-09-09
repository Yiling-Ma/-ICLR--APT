"""Summarize the fair APT-Bench patient-cell scaling design.

This script consumes the sufficient statistics produced by
``patient_cell_scaling.py``. It never refits a model and does not read labels
from outer-test patients during model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PATIENT_BUDGETS = (8, 16, 32)
CELL_CAPS = (100, 200, 400, 800, 1600)
TOTAL_BUDGETS = (3200, 6400, 12800)
METRICS = ("macro_f1", "patient_balanced_macro_f1", "mean_patient_macro_f1")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20270908


def f1_from_confusion(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    actual = matrix.sum(axis=1).astype(float)
    denominator = actual + predicted
    values = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(values.mean())


def score_patient_matrices(values: np.ndarray, metric: str) -> float:
    if metric == "macro_f1":
        return f1_from_confusion(values.sum(axis=0))
    if metric == "patient_balanced_macro_f1":
        totals = values.sum(axis=(1, 2)).astype(float)
        normalized = np.divide(
            values,
            totals[:, None, None],
            out=np.zeros_like(values, dtype=float),
            where=totals[:, None, None] > 0,
        )
        return f1_from_confusion(normalized.sum(axis=0))
    if metric == "mean_patient_macro_f1":
        return float(np.mean([f1_from_confusion(matrix) for matrix in values]))
    raise ValueError(f"Unsupported metric: {metric}")


def f1_from_confusion_batch(matrices: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(matrices, axis1=1, axis2=2)
    predicted = matrices.sum(axis=1)
    actual = matrices.sum(axis=2)
    denominator = actual + predicted
    values = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator > 0,
    )
    return values.mean(axis=1)


def bootstrap_scores(
    values: np.ndarray, metric: str, draws: np.ndarray, chunk_size: int = 5
) -> np.ndarray:
    if metric == "mean_patient_macro_f1":
        per_patient = np.asarray([f1_from_confusion(matrix) for matrix in values])
        return per_patient[draws].mean(axis=1)
    if metric == "patient_balanced_macro_f1":
        totals = values.sum(axis=(1, 2)).astype(float)
        values = np.divide(
            values,
            totals[:, None, None],
            out=np.zeros_like(values, dtype=float),
            where=totals[:, None, None] > 0,
        )
    scores = np.empty(len(draws), dtype=float)
    for start in range(0, len(draws), chunk_size):
        stop = min(start + chunk_size, len(draws))
        matrices = values[draws[start:stop]].sum(axis=1)
        scores[start:stop] = f1_from_confusion_batch(matrices)
    return scores


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n_subset_seeds": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
        "fraction_positive": float(np.mean(values > 0)),
    }


def paired_seed_values(
    frame: pd.DataFrame,
    low: tuple[int, int],
    high: tuple[int, int],
    metric: str,
) -> np.ndarray:
    key = ["seed"]
    low_frame = frame[
        (frame["patient_budget"] == low[0]) & (frame["cell_cap"] == str(low[1]))
    ][key + [metric]].rename(columns={metric: "low"})
    high_frame = frame[
        (frame["patient_budget"] == high[0]) & (frame["cell_cap"] == str(high[1]))
    ][key + [metric]].rename(columns={metric: "high"})
    paired = low_frame.merge(high_frame, on=key, validate="one_to_one")
    if len(paired) != 20:
        raise RuntimeError(f"Expected 20 paired seeds for {low} -> {high}, found {len(paired)}")
    return (paired["high"] - paired["low"]).to_numpy(float)


def fixed_total_rows(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, task), frame in per_seed.groupby(["model", "task"], sort=True):
        for total in TOTAL_BUDGETS:
            for patients in PATIENT_BUDGETS:
                cells = total // patients
                selected = frame[
                    (frame["patient_budget"] == patients)
                    & (frame["cell_cap"] == str(cells))
                ]
                if len(selected) != 20:
                    raise RuntimeError(
                        f"Incomplete fixed-total cell: {model}/{task}/T{total}/P{patients}/C{cells}"
                    )
                for metric in METRICS:
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "metric": metric,
                            "total_training_cells": total,
                            "patient_budget": patients,
                            "cells_per_patient": cells,
                            **summarize(selected[metric].to_numpy(float)),
                        }
                    )
    return pd.DataFrame(rows)


def effect_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for total in TOTAL_BUDGETS:
        for low, high in ((8, 16), (16, 32)):
            specs.append(
                {
                    "design": "fixed_total_patient_doubling",
                    "fixed_axis": "total_training_cells",
                    "fixed_value": total,
                    "low": (low, total // low),
                    "high": (high, total // high),
                }
            )
        specs.append(
            {
                "design": "fixed_total_patient_8_to_32",
                "fixed_axis": "total_training_cells",
                "fixed_value": total,
                "low": (8, total // 8),
                "high": (32, total // 32),
            }
        )
    for cells in CELL_CAPS:
        for low, high in ((8, 16), (16, 32)):
            specs.append(
                {
                    "design": "matched_patient_doubling",
                    "fixed_axis": "cells_per_patient",
                    "fixed_value": cells,
                    "low": (low, cells),
                    "high": (high, cells),
                }
            )
    for patients in PATIENT_BUDGETS:
        for low, high in zip(CELL_CAPS[:-1], CELL_CAPS[1:]):
            specs.append(
                {
                    "design": "matched_cell_doubling",
                    "fixed_axis": "patient_budget",
                    "fixed_value": patients,
                    "low": (patients, low),
                    "high": (patients, high),
                }
            )
    return specs


def load_patient_matrices(path: Path) -> dict[tuple[object, ...], dict[str, np.ndarray]]:
    frame = pd.read_parquet(path)
    frame["cell_cap"] = frame["cell_cap"].astype(str)
    result: dict[tuple[object, ...], dict[str, np.ndarray]] = {}
    group_columns = ["model", "task", "patient_budget", "cell_cap", "patient_id"]
    for key, group in frame.groupby(group_columns, sort=False, observed=True):
        n_classes = int(max(group["true_class_index"].max(), group["predicted_class_index"].max()) + 1)
        matrix = np.zeros((n_classes, n_classes), dtype=float)
        np.add.at(
            matrix,
            (
                group["true_class_index"].to_numpy(int),
                group["predicted_class_index"].to_numpy(int),
            ),
            group["count"].to_numpy(float) / 20.0,
        )
        result.setdefault(key[:4], {})[str(key[4])] = matrix
    return result


def align_matrices(
    matrices: dict[tuple[object, ...], dict[str, np.ndarray]],
    model: str,
    task: str,
    condition: tuple[int, int],
) -> tuple[list[str], np.ndarray]:
    values = matrices[(model, task, condition[0], str(condition[1]))]
    patients = sorted(values)
    size = max(matrix.shape[0] for matrix in values.values())
    stacked = np.zeros((len(patients), size, size), dtype=float)
    for index, patient in enumerate(patients):
        matrix = values[patient]
        stacked[index, : matrix.shape[0], : matrix.shape[1]] = matrix
    return patients, stacked


def paired_patient_bootstrap(
    matrices: dict[tuple[object, ...], dict[str, np.ndarray]],
    model: str,
    task: str,
    low: tuple[int, int],
    high: tuple[int, int],
    metric: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    low_patients, low_values = align_matrices(matrices, model, task, low)
    high_patients, high_values = align_matrices(matrices, model, task, high)
    if low_patients != high_patients:
        raise RuntimeError("Patient-clustered bootstrap requires identical OOF patients.")
    n_patients = len(low_patients)
    draws = rng.integers(0, n_patients, size=(BOOTSTRAP_REPLICATES, n_patients))
    differences = bootstrap_scores(high_values, metric, draws) - bootstrap_scores(
        low_values, metric, draws
    )
    return tuple(np.quantile(differences, [0.025, 0.975]).astype(float))


def add_fixed_patient_intervals(
    fixed: pd.DataFrame,
    matrices: dict[tuple[object, ...], dict[str, np.ndarray]],
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED - 1)
    fixed = fixed.copy()
    fixed["patient_clustered_bootstrap_q025"] = np.nan
    fixed["patient_clustered_bootstrap_q975"] = np.nan
    for index, row in fixed.iterrows():
        _, values = align_matrices(
            matrices,
            str(row["model"]),
            str(row["task"]),
            (int(row["patient_budget"]), int(row["cells_per_patient"])),
        )
        draws = rng.integers(
            0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
        )
        bootstrap = bootstrap_scores(values, str(row["metric"]), draws)
        q025, q975 = np.quantile(bootstrap, [0.025, 0.975])
        fixed.loc[index, "patient_clustered_bootstrap_q025"] = q025
        fixed.loc[index, "patient_clustered_bootstrap_q975"] = q975
    return fixed


def effect_rows(
    per_seed: pd.DataFrame,
    matrices: dict[tuple[object, ...], dict[str, np.ndarray]],
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for (model, task), frame in per_seed.groupby(["model", "task"], sort=True):
        for spec in effect_specs():
            low = spec["low"]
            high = spec["high"]
            assert isinstance(low, tuple) and isinstance(high, tuple)
            for metric in METRICS:
                patient_q025, patient_q975 = paired_patient_bootstrap(
                    matrices, model, task, low, high, metric, rng
                )
                values = paired_seed_values(frame, low, high, metric)
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "metric": metric,
                        **{key: value for key, value in spec.items() if key not in {"low", "high"}},
                        "low_patient_budget": low[0],
                        "low_cells_per_patient": low[1],
                        "high_patient_budget": high[0],
                        "high_cells_per_patient": high[1],
                        **summarize(values),
                        "patient_clustered_bootstrap_q025": patient_q025,
                        "patient_clustered_bootstrap_q975": patient_q975,
                    }
                )
    return pd.DataFrame(rows)


def response_surface(per_seed: pd.DataFrame, per_run: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows: list[dict[str, object]] = []
    for (model, task), pooled_frame in per_seed.groupby(["model", "task"], sort=True):
        fold_frame = per_run[
            (per_run["model"] == model) & (per_run["task"] == task)
        ].copy()
        for metric in METRICS:
            # The primary pooled-cell response surface uses outer-fold fixed
            # effects. Mean within-patient F1 is available only after pooling
            # the per-patient sufficient statistics, so it uses the pooled
            # seed-level surface as a sensitivity analysis.
            if metric == "macro_f1":
                frame = fold_frame
                fold_dummies = pd.get_dummies(
                    frame["outer_fold"].astype(int), prefix="fold", drop_first=True, dtype=float
                ).to_numpy()
            else:
                frame = pooled_frame.copy()
                fold_dummies = np.empty((len(frame), 0), dtype=float)
            frame["log2_patients"] = np.log2(frame["patient_budget"].to_numpy(float) / 8.0)
            frame["log2_cells"] = np.log2(frame["cell_cap"].astype(float).to_numpy() / 100.0)
            design = np.column_stack(
                [
                    np.ones(len(frame)),
                    frame["log2_patients"],
                    frame["log2_cells"],
                    frame["log2_patients"] * frame["log2_cells"],
                    fold_dummies,
                ]
            )
            outcome = frame[metric].to_numpy(float)
            estimate = np.linalg.lstsq(design, outcome, rcond=None)[0]
            bootstrap = np.empty((BOOTSTRAP_REPLICATES, design.shape[1]), dtype=float)
            seeds = np.sort(frame["seed"].unique())
            for index in range(BOOTSTRAP_REPLICATES):
                sampled = rng.choice(seeds, size=len(seeds), replace=True)
                selected_indices = np.concatenate(
                    [np.flatnonzero(frame["seed"].to_numpy() == seed) for seed in sampled]
                )
                bootstrap[index] = np.linalg.lstsq(
                    design[selected_indices], outcome[selected_indices], rcond=None
                )[0]
            for term_index, term in enumerate(
                ("intercept", "log2_patient_budget", "log2_cells_per_patient", "interaction")
            ):
                q025, q975 = np.quantile(bootstrap[:, term_index], [0.025, 0.975])
                predictor_sd = float(np.std(design[:, term_index], ddof=1)) if term_index else np.nan
                outcome_sd = float(np.std(outcome, ddof=1))
                scale = predictor_sd / outcome_sd if term_index and outcome_sd > 0 else np.nan
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "metric": metric,
                        "term": term,
                        "estimate": float(estimate[term_index]),
                        "cluster_seed_bootstrap_q025": float(q025),
                        "cluster_seed_bootstrap_q975": float(q975),
                        "standardized_estimate": float(estimate[term_index] * scale) if term_index else np.nan,
                        "standardized_bootstrap_q025": float(q025 * scale) if term_index else np.nan,
                        "standardized_bootstrap_q975": float(q975 * scale) if term_index else np.nan,
                        "n_subset_seeds": len(seeds),
                        "outer_fold_adjustment": "fixed effects" if metric == "macro_f1" else "not applicable after pooled patient summary",
                        "interpretation": "descriptive response surface; not causal",
                    }
                )
    return pd.DataFrame(rows)


def create_figure(fixed: pd.DataFrame, effects: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    primary = fixed[fixed["metric"] == "macro_f1"]
    models = [
        model
        for model in ("logistic_regression", "xgboost")
        if model in set(primary["model"])
    ]
    fig, axes = plt.subplots(
        2,
        len(models),
        figsize=(4.8 * len(models), 6.8),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    colors = {3200: "#315B7D", 6400: "#C05A3B", 12800: "#4F7A51"}
    for row, task in enumerate(("coarse", "fine")):
        for column, model in enumerate(models):
            ax = axes[row, column]
            subset = primary[(primary["task"] == task) & (primary["model"] == model)]
            for total in TOTAL_BUDGETS:
                values = subset[subset["total_training_cells"] == total].sort_values("patient_budget")
                ax.errorbar(
                    values["patient_budget"],
                    values["mean"],
                    yerr=np.vstack([values["mean"] - values["q025"], values["q975"] - values["mean"]]),
                    marker="o",
                    capsize=3,
                    color=colors[total],
                    label=f"T={total:,}",
                )
            ax.set_title(f"{'LR' if model == 'logistic_regression' else 'XGBoost'} / {task}")
            ax.set_xlabel("Training patients (fixed total cells)")
            ax.set_ylabel("Pooled OOF Macro-F1")
            ax.set_xticks(PATIENT_BUDGETS)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"fixed_total_scaling.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected = effects[
        (effects["metric"] == "macro_f1")
        & effects["design"].isin(["matched_patient_doubling", "matched_cell_doubling"])
    ]
    summary = (
        selected.groupby(["model", "task", "design"], observed=True)["mean"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    summary.to_csv(output_dir / "matched_doubling_average_effects.csv", index=False)


def write_table(fixed: pd.DataFrame, output_dir: Path) -> None:
    primary = fixed[fixed["metric"] == "macro_f1"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcrrr}",
        r"\toprule",
        r"Model & Task & Total cells & $P=8$ & $P=16$ & $P=32$ \\",
        r"\midrule",
    ]
    for model in ("logistic_regression", "xgboost"):
        if model not in set(primary["model"]):
            continue
        for task in ("coarse", "fine"):
            for total in TOTAL_BUDGETS:
                frame = primary[
                    (primary["model"] == model)
                    & (primary["task"] == task)
                    & (primary["total_training_cells"] == total)
                ].set_index("patient_budget")
                values = [
                    f"{frame.loc[p, 'mean']:.3f} [{frame.loc[p, 'q025']:.3f}, {frame.loc[p, 'q975']:.3f}]"
                    for p in PATIENT_BUDGETS
                ]
                lines.append(
                    f"{'LR' if model == 'logistic_regression' else 'XGBoost'} & {task.title()} & "
                    f"{total:,} & " + " & ".join(values) + r" \\"
                )
        lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Fixed-total training-cell scaling. Each row holds the total number of sampled training cells constant while reallocating them across independent patients. Entries are pooled patient-disjoint OOF Macro-F1 means with empirical 95\% intervals over 20 matched subset seeds. All outer-test cells are evaluated.}",
            r"\label{tab:fixed_total_scaling}",
            r"\end{table}",
        ]
    )
    (output_dir / "fixed_total_scaling_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def qa(per_seed: pd.DataFrame, fixed: pd.DataFrame, effects: pd.DataFrame) -> dict[str, object]:
    expected_groups = (
        len(PATIENT_BUDGETS)
        * len(CELL_CAPS)
        * per_seed["model"].nunique()
        * per_seed["task"].nunique()
    )
    checks = {
        "all_grid_groups_present": per_seed.groupby(["patient_budget", "cell_cap", "model", "task"]).ngroups == expected_groups,
        "twenty_subset_seeds_per_group": bool(
            per_seed.groupby(["patient_budget", "cell_cap", "model", "task"]).size().eq(20).all()
        ),
        "fixed_total_cells_exact": bool(
            (fixed["patient_budget"] * fixed["cells_per_patient"] == fixed["total_training_cells"]).all()
        ),
        "all_effects_have_twenty_pairs": bool(effects["n_subset_seeds"].eq(20).all()),
        "no_nonfinite_primary_values": bool(
            np.isfinite(per_seed[list(METRICS)].to_numpy(float)).all()
        ),
    }
    return {"overall_status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main() -> None:
    global PATIENT_BUDGETS, CELL_CAPS, TOTAL_BUDGETS, BOOTSTRAP_REPLICATES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/patient_cell_scaling_fair"))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        PATIENT_BUDGETS = tuple(int(value) for value in protocol["patient_budgets"])
        CELL_CAPS = tuple(int(value) for value in protocol["cell_caps"])
        TOTAL_BUDGETS = tuple(int(value) for value in protocol["fixed_total_budgets"])
        BOOTSTRAP_REPLICATES = int(protocol.get("patient_bootstrap_replicates", BOOTSTRAP_REPLICATES))
    per_seed = pd.read_csv(output_dir / "per_seed_oof_metrics.csv", dtype={"cell_cap": str})
    per_run = pd.read_csv(output_dir / "per_run_metrics.csv", dtype={"cell_cap": str})
    matrices = load_patient_matrices(output_dir / "per_patient_confusions.parquet")
    fixed = add_fixed_patient_intervals(fixed_total_rows(per_seed), matrices)
    effects = effect_rows(per_seed, matrices)
    surface = response_surface(per_seed, per_run)
    fixed.to_csv(output_dir / "fixed_total_results.csv", index=False)
    effects.to_csv(output_dir / "matched_doubling_effects.csv", index=False)
    surface.to_csv(output_dir / "response_surface_coefficients.csv", index=False)
    write_table(fixed, output_dir)
    create_figure(fixed, effects, output_dir)
    status = qa(per_seed, fixed, effects)
    (output_dir / "fair_scaling_qa.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if status["overall_status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
