#!/usr/bin/env python3
"""Rebuild HBM tables, paired intervals, and figures from saved predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hbm_core import FINAL_SEEDS, atomic_json

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_full_budget_mlp as unified


ORDER = ("plain", "parent", "parent_cons", "kd", "parent_kd", "full")
DISPLAY = {
    "plain": "Plain MLP",
    "parent": "+ parent-mass",
    "parent_cons": "+ parent-mass + consistency",
    "kd": "+ KD",
    "parent_kd": "+ parent-mass + KD",
    "full": "Full HBM",
}


def load_condition(root: Path, condition: str):
    seeds = []
    reference_ids = None
    for seed in FINAL_SEEDS:
        blocks = []
        for fold in range(5):
            path = root / condition / f"seed{seed}" / f"fold{fold}" / "predictions.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            # Early runs stored sample IDs from pandas as object arrays. These are
            # trusted, locally generated artifacts; later runs write fixed-width
            # Unicode arrays, but keep backward-compatible loading here.
            blocks.append(np.load(path, allow_pickle=True))
        cell_ids = np.concatenate([block["cell_ids"].astype(str) for block in blocks])
        if len(cell_ids) != 361792 or len(set(cell_ids)) != 361792:
            raise AssertionError(f"{condition}/{seed}: invalid pooled OOF cell coverage")
        if reference_ids is None:
            reference_ids = cell_ids
        elif not np.array_equal(reference_ids, cell_ids):
            raise AssertionError(f"{condition}: cell order differs across seeds")
        patients = np.concatenate([block["patients"].astype(str) for block in blocks])
        if len(patients) != 40 or len(set(patients)) != 40:
            raise AssertionError(f"{condition}/{seed}: invalid patient coverage")
        keys = blocks[0]["diagnostic_keys"].astype(str).tolist()
        counts = np.concatenate([block["diagnostic_counts"] for block in blocks])
        seeds.append({
            "seed": seed,
            "patients": patients,
            "fine_cm": np.concatenate([block["fine_cm"] for block in blocks]),
            "coarse_cm": np.concatenate([block["coarse_cm"] for block in blocks]),
            "oracle_cm": np.concatenate([block["oracle_cm"] for block in blocks]),
            "counts": {key: counts[:, index].astype(float) for index, key in enumerate(keys)},
        })
        for block in blocks:
            block.close()
    return seeds, reference_ids


def safe_ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def point_metrics(seeds):
    rows = []
    for item in seeds:
        counts = item["counts"]
        fine = unified.score(item["fine_cm"])
        oracle = unified.score(item["oracle_cm"])
        rows.append({
            "fine_sb_macro_f1": fine,
            "coarse_sb_macro_f1": unified.score(item["coarse_cm"]),
            "cross_lineage_fraction_of_errors": safe_ratio(counts["cross_lineage_error"], counts["fine_error"]).mean(),
            "within_lineage_fraction_of_errors": safe_ratio(counts["within_lineage_error"], counts["fine_error"]).mean(),
            "cross_lineage_error_rate": safe_ratio(counts["cross_lineage_error"], counts["n"]).mean(),
            "fine_outside_parent_given_coarse_correct": safe_ratio(
                counts["fine_outside_parent_and_coarse_correct"], counts["coarse_correct"]
            ).mean(),
            "oracle_fine_sb_macro_f1": oracle,
            "hierarchy_gap": oracle - fine,
            "correct_rate": safe_ratio(counts["fine_correct"], counts["n"]).mean(),
            "within_lineage_error_rate": safe_ratio(counts["within_lineage_error"], counts["n"]).mean(),
        })
    return pd.DataFrame(rows)


def sampled_metric(item, patient_draw, metric):
    if metric == "fine_sb_macro_f1":
        return unified.score(item["fine_cm"][patient_draw])
    if metric == "coarse_sb_macro_f1":
        return unified.score(item["coarse_cm"][patient_draw])
    if metric == "oracle_fine_sb_macro_f1":
        return unified.score(item["oracle_cm"][patient_draw])
    if metric == "hierarchy_gap":
        return unified.score(item["oracle_cm"][patient_draw]) - unified.score(item["fine_cm"][patient_draw])
    counts = item["counts"]
    mapping = {
        "cross_lineage_fraction_of_errors": ("cross_lineage_error", "fine_error"),
        "within_lineage_fraction_of_errors": ("within_lineage_error", "fine_error"),
        "cross_lineage_error_rate": ("cross_lineage_error", "n"),
        "fine_outside_parent_given_coarse_correct": ("fine_outside_parent_and_coarse_correct", "coarse_correct"),
        "correct_rate": ("fine_correct", "n"),
        "within_lineage_error_rate": ("within_lineage_error", "n"),
    }
    numerator, denominator = mapping[metric]
    return safe_ratio(counts[numerator][patient_draw], counts[denominator][patient_draw]).mean()


def bootstrap(all_models, metrics, n_boot, seed):
    rng = np.random.default_rng(seed)
    model_draws = {name: {metric: np.empty(n_boot) for metric in metrics} for name in all_models}
    seed_draws = rng.integers(0, len(FINAL_SEEDS), size=(n_boot, len(FINAL_SEEDS)))
    patient_draws = rng.integers(0, 40, size=(n_boot, 40))
    for draw_index in range(n_boot):
        for name, items in all_models.items():
            for metric in metrics:
                model_draws[name][metric][draw_index] = np.mean([
                    sampled_metric(items[seed_index], patient_draws[draw_index], metric)
                    for seed_index in seed_draws[draw_index]
                ])
    return model_draws


def write_markdown(frame: pd.DataFrame, path: Path):
    columns = [
        ("Model", "model"), ("Fine SB-F1", "fine_sb_macro_f1"),
        ("Coarse SB-F1", "coarse_sb_macro_f1"),
        ("Cross-lineage error", "cross_lineage_error_rate"),
        ("Fine-outside-parent given coarse-correct", "fine_outside_parent_given_coarse_correct"),
        ("Oracle F1", "oracle_fine_sb_macro_f1"), ("Hierarchy gap", "hierarchy_gap"),
    ]
    lines = ["| " + " | ".join(label for label, _ in columns) + " |",
             "|" + "|".join(["---"] + ["---:" for _ in columns[1:]]) + "|"]
    for _, row in frame.iterrows():
        values = [str(row[key]) if key == "model" else f"{row[key]:.3f}" for _, key in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_latex(frame: pd.DataFrame, path: Path):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & Fine SB-F1 & Coarse SB-F1 & Cross-lineage & Outside$\mid$coarse correct & Oracle F1 & Gap \\",
             r"\midrule"]
    for _, row in frame.iterrows():
        name = str(row["model"]).replace("+", r"$+$")
        values = [row[key] for key in ("fine_sb_macro_f1", "coarse_sb_macro_f1",
                  "cross_lineage_error_rate", "fine_outside_parent_given_coarse_correct",
                  "oracle_fine_sb_macro_f1", "hierarchy_gap")]
        lines.append(name + " & " + " & ".join(f"{value:.3f}" for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def figures(frame: pd.DataFrame, output: Path):
    names = frame["model"].tolist()
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    width = 0.36
    ax.bar(x - width / 2, frame["fine_sb_macro_f1"], width, label="Deployable APT-only")
    ax.bar(x + width / 2, frame["oracle_fine_sb_macro_f1"], width, label="True-lineage oracle diagnostic")
    ax.set_ylabel("Subject-balanced Macro-F1")
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.set_ylim(0, max(frame["oracle_fine_sb_macro_f1"].max() * 1.15, 0.45))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"figure_a_bottleneck_mitigation.{suffix}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    bottom = np.zeros(len(frame))
    for column, label, color in (
        ("correct_rate", "Correct", "#3B8F6B"),
        ("cross_lineage_error_rate", "Cross-lineage error", "#C95A49"),
        ("within_lineage_error_rate", "Within-lineage error", "#D9A441"),
    ):
        ax.bar(x, frame[column], bottom=bottom, label=label, color=color)
        bottom += frame[column].to_numpy()
    ax.set_ylabel("Patient-balanced fraction of cells")
    ax.set_xticks(x, names, rotation=28, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"figure_b_error_structure.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_models, cell_ids = {}, None
    for condition in ORDER:
        items, ids = load_condition(args.input, condition)
        if cell_ids is None:
            cell_ids = ids
        elif not np.array_equal(cell_ids, ids):
            raise AssertionError(f"{condition}: OOF cell order differs from Plain MLP")
        all_models[condition] = items
    metric_names = list(point_metrics(all_models["plain"]).columns)
    draws = bootstrap(all_models, metric_names, args.n_boot, args.seed)
    rows = []
    seed_rows = []
    for condition in ORDER:
        points = point_metrics(all_models[condition])
        for seed_index, seed in enumerate(FINAL_SEEDS):
            seed_rows.append({"condition": condition, "seed": seed, **points.iloc[seed_index].to_dict()})
        row = {"condition": condition, "model": DISPLAY[condition]}
        for metric in metric_names:
            row[metric] = float(points[metric].mean())
            row[f"{metric}_ci_low"] = float(np.quantile(draws[condition][metric], 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(draws[condition][metric], 0.975))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "ablation_results.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(args.output / "per_seed_results.csv", index=False)
    comparisons = []
    for condition in ORDER[1:]:
        for metric in ("fine_sb_macro_f1", "cross_lineage_error_rate", "hierarchy_gap"):
            difference = draws[condition][metric] - draws["plain"][metric]
            comparisons.append({
                "comparison": f"{condition}-plain", "metric": metric,
                "estimate": float(result.set_index("condition").loc[condition, metric]
                                  - result.set_index("condition").loc["plain", metric]),
                "ci95_low": float(np.quantile(difference, 0.025)),
                "ci95_high": float(np.quantile(difference, 0.975)),
            })
    pd.DataFrame(comparisons).to_csv(args.output / "paired_comparisons.csv", index=False)
    write_markdown(result, args.output / "ablation_table.md")
    write_latex(result, args.output / "ablation_table.tex")
    figures(result, args.output)
    atomic_json(args.output / "completion.json", {
        "status": "PASS", "conditions": list(ORDER), "seeds": list(FINAL_SEEDS),
        "outer_folds": 5, "cells": len(cell_ids), "patients": 40,
        "bootstrap_replicates": args.n_boot,
        "uncertainty": "paired seed-slot and held-out-patient bootstrap",
        "tables_generated_from_predictions": True,
    })
    print(result[["model", *metric_names]].to_string(index=False))


if __name__ == "__main__":
    main()
