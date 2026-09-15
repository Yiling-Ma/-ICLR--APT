"""Fail-closed aggregation, paired bootstrap, tables, and figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CONTROLS, PRIMARY, SEEDS, WIDTHS, atomic_json, patient_confusions, score


def expected_conditions(width):
    return PRIMARY + (CONTROLS if width == 10000 else ())


def load_all(output):
    arrays = {}; canonical = {}
    for width in WIDTHS:
        for condition in expected_conditions(width):
            seed_blocks = []
            for seed in SEEDS:
                folds = []
                for fold in range(5):
                    path = output / "runs" / f"hvg{width}" / condition / f"f{fold}" / f"s{seed}.npz"
                    if not path.exists(): raise FileNotFoundError(path)
                    with np.load(path, allow_pickle=False) as item:
                        data = {key: item[key].copy() for key in item.files}
                    key = (width, seed, fold)
                    observed = (data["cell_ids"], data["patient_ids"], data["fine_truth"], data["coarse_truth"])
                    if key not in canonical: canonical[key] = observed
                    else:
                        for left, right in zip(canonical[key], observed): np.testing.assert_array_equal(left, right)
                    folds.append(data)
                cells = np.concatenate([item["cell_ids"] for item in folds])
                patients = np.concatenate([item["patient_ids"] for item in folds])
                if len(cells) != 361792 or len(set(cells)) != 361792 or len(set(patients)) != 40:
                    raise RuntimeError("Incomplete or duplicated OOF coverage")
                fine_cm = []; coarse_cm = []
                for item in folds:
                    _, fcm = patient_confusions(item["fine_truth"], item["fine_prob"].argmax(1), item["patient_ids"], 27)
                    _, ccm = patient_confusions(item["coarse_truth"], item["coarse_prob"].argmax(1), item["patient_ids"], 5)
                    fine_cm.append(fcm); coarse_cm.append(ccm)
                seed_blocks.append({"fine": np.concatenate(fine_cm), "coarse": np.concatenate(coarse_cm)})
            arrays[(width, condition)] = {task: np.stack([block[task] for block in seed_blocks]) for task in ("fine", "coarse")}
    return arrays


def bootstrap(arrays, n_boot, seed):
    rng = np.random.default_rng(seed)
    patient_draws = rng.integers(0, 40, size=(n_boot, 40))
    seed_draws = rng.integers(0, 3, size=(n_boot, 3))
    estimates = {}; draws = {}; rows = []
    for key, tasks in arrays.items():
        width, condition = key
        for task, matrices in tasks.items():
            values = [score(matrix) for matrix in matrices]
            sample = np.empty(n_boot)
            for replicate, (patients, slots) in enumerate(zip(patient_draws, seed_draws)):
                sample[replicate] = np.mean([score(matrices[slot, patients]) for slot in slots])
            estimates[(width, condition, task)] = float(np.mean(values))
            draws[(width, condition, task)] = sample
            rows.append({"rna_width": width, "budget": "full", "condition": condition,
                         "task": task, "estimate": np.mean(values),
                         "ci_low": np.quantile(sample, .025), "ci_high": np.quantile(sample, .975),
                         "seed_values": json.dumps(values)})
    contrasts = []
    comparisons = [(5000, "rna_apt", "rna"), (10000, "rna_apt", "rna")]
    comparisons += [(10000, "rna_apt", condition) for condition in CONTROLS]
    for width, high, low in comparisons:
        for task in ("fine", "coarse"):
            delta = draws[(width, high, task)] - draws[(width, low, task)]
            contrasts.append({"rna_width": width, "task": task,
                              "comparison": f"{high}-minus-{low}",
                              "estimate": estimates[(width, high, task)] - estimates[(width, low, task)],
                              "ci_low": np.quantile(delta, .025), "ci_high": np.quantile(delta, .975)})
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def load_capped(capped_root):
    capped = []
    missing = []
    for width in (2000, 5000):
        path = capped_root / f"remaining_modality_hvg{width}_v1" / "summary.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        for condition in ("rna", "rna_apt"):
            row = frame[(frame.task == "fine") & (frame.model == "mlp") & (frame.comparison == condition)].iloc[0]
            capped.append({"rna_width": width, "budget": "capped", "condition": condition,
                           "task": "fine", "estimate": row.estimate})
            coarse = frame[(frame.task == "coarse") & (frame.model == "mlp") & (frame.comparison == condition)]
            if not coarse.empty:
                capped.append({"rna_width": width, "budget": "capped", "condition": condition,
                               "task": "coarse", "estimate": coarse.iloc[0].estimate})
        delta = frame[(frame.task == "fine") & (frame.model == "mlp") &
                      (frame.comparison == "rna_apt-minus-rna")]
        if not delta.empty:
            row = delta.iloc[0]
            capped.append({"rna_width": width, "budget": "capped", "condition": "rna_apt-minus-rna",
                           "task": "fine", "estimate": row.estimate,
                           "ci_low": getattr(row, "low", np.nan), "ci_high": getattr(row, "high", np.nan)})
    columns = ["rna_width", "budget", "condition", "task", "estimate", "ci_low", "ci_high"]
    return pd.DataFrame(capped, columns=columns), missing


def tables(report, scores, contrasts, capped):
    combined = pd.concat([pd.DataFrame(capped), scores], ignore_index=True, sort=False)
    combined.to_csv(report / "comparison_table.csv", index=False)
    contrasts.to_csv(report / "paired_contrasts.csv", index=False)
    controls = contrasts[(contrasts.rna_width == 10000) & (contrasts.task == "fine")]
    controls.to_csv(report / "control_table.csv", index=False)

    lines = ["| RNA features | Budget | Condition | Fine SB-F1 | Coarse SB-F1 | Delta vs RNA | 95% CI |",
             "|---|---|---|---:|---:|---:|---:|"]
    for _, row in combined[combined.task == "fine"].iterrows():
        coarse = combined[(combined.rna_width == row.rna_width) & (combined.budget == row.budget) &
                          (combined.condition == row.condition) & (combined.task == "coarse")]
        delta = contrasts[(contrasts.rna_width == row.rna_width) & (contrasts.task == "fine") &
                          (contrasts.comparison == "rna_apt-minus-rna")] if row.condition == "rna_apt" else pd.DataFrame()
        if delta.empty and row.condition == "rna_apt":
            delta = combined[(combined.rna_width == row.rna_width) & (combined.budget == row.budget) &
                             (combined.condition == "rna_apt-minus-rna") & (combined.task == "fine")]
        d = "" if delta.empty else f"{delta.iloc[0].estimate:+.4f}"
        ci = "" if delta.empty else f"[{delta.iloc[0].ci_low:.4f}, {delta.iloc[0].ci_high:.4f}]"
        c = "" if coarse.empty else f"{coarse.iloc[0].estimate:.3f}"
        lines.append(f"| {int(row.rna_width):,} HVG | {row.budget} | {row.condition} | {row.estimate:.3f} | {c} | {d} | {ci} |")
    (report / "comparison_table.md").write_text("\n".join(lines) + "\n")


def figure(report, scores, contrasts, capped):
    labels = []
    rna = []; apt = []
    for width in (2000, 5000):
        query = capped[(capped.rna_width == width) & (capped.task == "fine")]
        if query.empty:
            continue
        labels.append(f"{width // 1000}k\ncapped")
        rna.append(float(query[query.condition == "rna"].estimate.iloc[0]))
        apt.append(float(query[query.condition == "rna_apt"].estimate.iloc[0]))
    for width in WIDTHS:
        query = scores[(scores.rna_width == width) & (scores.task == "fine")]
        labels.append(f"{width // 1000}k\nfull")
        rna.append(float(query[query.condition == "rna"].estimate.iloc[0]))
        apt.append(float(query[query.condition == "rna_apt"].estimate.iloc[0]))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(labels)); width = .35
    axes[0].bar(x - width / 2, rna, width, label="RNA-only")
    axes[0].bar(x + width / 2, apt, width, label="RNA+APT")
    axes[0].set_xticks(x, labels); axes[0].set_ylabel("Fine SB-Macro-F1"); axes[0].set_ylim(0, 0.75); axes[0].legend(frameon=False)
    full = contrasts[(contrasts.task == "fine") & (contrasts.comparison == "rna_apt-minus-rna")].sort_values("rna_width")
    historical = capped[(capped.task == "fine") & (capped.condition == "rna_apt-minus-rna")].sort_values("rna_width")
    delta = np.r_[historical.estimate.to_numpy(), full.estimate.to_numpy()]
    low = np.r_[historical.ci_low.to_numpy(), full.ci_low.to_numpy()]
    high = np.r_[historical.ci_high.to_numpy(), full.ci_high.to_numpy()]
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].errorbar(x, delta, yerr=[delta-low, high-delta], fmt="o", capsize=3)
    axes[1].set_xticks(x, labels); axes[1].set_ylabel("Delta Fine SB-Macro-F1")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(report / f"full_budget_rna_apt.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capped-root", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    arrays = load_all(args.input)
    scores, contrasts = bootstrap(arrays, args.n_boot, 20260915)
    scores.to_csv(args.output / "scores.csv", index=False)
    capped, missing_capped = load_capped(args.capped_root)
    tables(args.output, scores, contrasts, capped)
    figure(args.output, scores, contrasts, capped)
    atomic_json(args.output / "completion.json", {"status": "PASS", "bootstrap": args.n_boot,
                "conditions": len(arrays), "seeds": list(SEEDS), "patients": 40, "cells": 361792,
                "missing_capped_summaries": missing_capped})


if __name__ == "__main__": main()
