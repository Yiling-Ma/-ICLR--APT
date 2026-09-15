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


def tables(report, scores, contrasts, capped_root):
    capped = []
    for width in (2000, 5000):
        frame = pd.read_csv(capped_root / f"remaining_modality_hvg{width}_v1" / "summary.csv")
        for condition in ("rna", "rna_apt"):
            row = frame[(frame.task == "fine") & (frame.model == "mlp") & (frame.comparison == condition)].iloc[0]
            capped.append({"rna_width": width, "budget": "capped", "condition": condition,
                           "task": "fine", "estimate": row.estimate})
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
        d = "" if delta.empty else f"{delta.iloc[0].estimate:+.4f}"
        ci = "" if delta.empty else f"[{delta.iloc[0].ci_low:.4f}, {delta.iloc[0].ci_high:.4f}]"
        c = "" if coarse.empty else f"{coarse.iloc[0].estimate:.3f}"
        lines.append(f"| {int(row.rna_width):,} HVG | {row.budget} | {row.condition} | {row.estimate:.3f} | {c} | {d} | {ci} |")
    (report / "comparison_table.md").write_text("\n".join(lines) + "\n")


def figure(report, scores, contrasts, capped_root):
    labels = ["2k\ncapped", "5k\ncapped", "5k\nfull", "10k\nfull"]
    rna = []; apt = []
    for width in (2000, 5000):
        frame = pd.read_csv(capped_root / f"remaining_modality_hvg{width}_v1" / "summary.csv")
        query = frame[(frame.task == "fine") & (frame.model == "mlp")]
        rna.append(float(query[query.comparison == "rna"].estimate.iloc[0]))
        apt.append(float(query[query.comparison == "rna_apt"].estimate.iloc[0]))
    for width in WIDTHS:
        query = scores[(scores.rna_width == width) & (scores.task == "fine")]
        rna.append(float(query[query.condition == "rna"].estimate.iloc[0]))
        apt.append(float(query[query.condition == "rna_apt"].estimate.iloc[0]))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(4); width = .35
    axes[0].bar(x-width/2, rna, width, label="RNA-only")
    axes[0].bar(x+width/2, apt, width, label="RNA+APT")
    axes[0].set_xticks(x, labels); axes[0].set_ylabel("Fine SB-Macro-F1"); axes[0].set_ylim(0, 0.75); axes[0].legend(frameon=False)
    full = contrasts[(contrasts.task == "fine") & (contrasts.comparison == "rna_apt-minus-rna")].sort_values("rna_width")
    historical = []
    for width_value in (2000, 5000):
        frame = pd.read_csv(capped_root / f"remaining_modality_hvg{width_value}_v1" / "summary.csv")
        historical.append(frame[(frame.task == "fine") & (frame.model == "mlp") &
                                (frame.comparison == "rna_apt-minus-rna")].iloc[0])
    delta = np.r_[[row.estimate for row in historical], full.estimate]
    low = np.r_[[row.low for row in historical], full.ci_low]
    high = np.r_[[row.high for row in historical], full.ci_high]
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
    tables(args.output, scores, contrasts, args.capped_root)
    figure(args.output, scores, contrasts, args.capped_root)
    atomic_json(args.output / "completion.json", {"status": "PASS", "bootstrap": args.n_boot,
                "conditions": len(arrays), "seeds": list(SEEDS), "patients": 40, "cells": 361792})


if __name__ == "__main__": main()
