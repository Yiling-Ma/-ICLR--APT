"""Create the cross-cohort fixed-total patient-cell scaling summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOTAL_BUDGETS = (3200, 6400, 12800)
PRIMARY_METRIC = "patient_balanced_macro_f1"
DATASETS = {
    "apt": ("APT", "Aptamer", Path("outputs/patient_cell_scaling_fair")),
    "combat_rna": ("COMBAT", "RNA", Path("outputs/combat_citeseq_scaling")),
    "combat_adt": ("COMBAT", "ADT", Path("outputs/combat_citeseq_scaling_adt")),
    "onek1k": ("OneK1K", "RNA", Path("outputs/onek1k_scaling")),
}
DATASET_ORDER = {key: index for index, key in enumerate(DATASETS)}

EXPECTED_TOTALS = {
    "apt": TOTAL_BUDGETS,
    "combat_rna": TOTAL_BUDGETS,
    "combat_adt": TOTAL_BUDGETS,
    "onek1k": (3200, 6400),
}


def load_effects(root: Path) -> pd.DataFrame:
    frames = []
    for key, (dataset, modality, default_dir) in DATASETS.items():
        path = root / default_dir / "matched_doubling_effects.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "dataset_key", key)
        frame.insert(1, "dataset", dataset)
        frame.insert(2, "modality", modality)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No completed scaling summaries were found.")
    return pd.concat(frames, ignore_index=True)


def select_fixed_total(effects: pd.DataFrame) -> pd.DataFrame:
    selected = effects[
        (effects["design"] == "fixed_total_patient_8_to_32")
        & (effects["metric"] == PRIMARY_METRIC)
    ].copy()
    selected["model_label"] = selected["model"].map(
        {"logistic_regression": "LR", "xgboost": "XGBoost"}
    )
    selected["task_label"] = selected["task"].str.title()
    selected = selected.rename(columns={"fixed_value": "total_training_cells"})
    selected["dataset_order"] = selected["dataset_key"].map(DATASET_ORDER)
    return selected.sort_values(
        ["dataset_order", "model", "task", "total_training_cells"]
    ).drop(columns="dataset_order")


def write_table(frame: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"Dataset & Modality & Model/task & $T=3{,}200$ & $T=6{,}400$ & $T=12{,}800$ \\",
        r"\midrule",
    ]
    grouped = frame.groupby(
        ["dataset_key", "dataset", "modality", "model_label", "task_label"],
        sort=False,
    )
    for (_, dataset, modality, model, task), values in grouped:
        values = values.set_index("total_training_cells")
        cells = []
        for total in TOTAL_BUDGETS:
            if total not in values.index:
                cells.append(r"--")
                continue
            row = values.loc[total]
            cells.append(
                f"{row['mean']:+.3f} "
                f"[{row['patient_clustered_bootstrap_q025']:+.3f},"
                f" {row['patient_clustered_bootstrap_q975']:+.3f}]"
            )
        lines.append(
            f"{dataset} & {modality} & {model}/{task} & "
            + " & ".join(cells)
            + r" \\" 
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\caption{Unmatched diagnostic: change in subject-balanced pooled, subject-disjoint OOF Macro-F1 when the same total training-cell budget is reallocated from 8 to 32 independent subjects without freezing the marginal training label distribution. Each held-out subject confusion matrix is normalized to unit mass before pooling. Brackets are 95\% paired subject-bootstrap intervals over the fixed outer-test subjects. Positive values favor broader subject coverage but can also reflect changes in observed classes, per-class counts, and class proportions. COMBAT ADT is a prespecified modality sensitivity using LR only. OneK1K uses the two common exact budgets supported by at least 800 cells from nearly all donors.}",
            r"\label{tab:cross_cohort_scaling}",
            r"\end{table}",
        ]
    )
    (output_dir / "cross_cohort_scaling_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def create_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    labels = []
    groups = []
    for key, values in frame.groupby(
        ["dataset", "modality", "model_label", "task_label"], sort=False
    ):
        labels.append(" / ".join(key))
        groups.append(values.set_index("total_training_cells"))
    offsets = np.linspace(-0.22, 0.22, len(TOTAL_BUDGETS))
    colors = ("#315B7D", "#C05A3B", "#4F7A51")
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7.6, max(3.6, 0.47 * len(groups) + 1.5)))
    for offset, color, total in zip(offsets, colors, TOTAL_BUDGETS):
        available = np.array([total in group.index for group in groups])
        available_groups = [group for group in groups if total in group.index]
        means = np.array([group.loc[total, "mean"] for group in available_groups])
        lows = np.array([group.loc[total, "patient_clustered_bootstrap_q025"] for group in available_groups])
        highs = np.array([group.loc[total, "patient_clustered_bootstrap_q975"] for group in available_groups])
        ax.errorbar(
            means,
            y[available] + offset,
            xerr=np.vstack((means - lows, highs - means)),
            fmt="o",
            color=color,
            markersize=5.5,
            elinewidth=1.5,
            capsize=4,
            capthick=1.3,
            label=f"T={total:,}",
        )
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ subject-balanced pooled Macro-F1: 32 vs. 8 training subjects", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        frameon=False,
        fontsize=9,
        markerscale=1.1,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"cross_cohort_scaling.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cross_cohort_scaling")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = select_fixed_total(load_effects(root))
    frame.to_csv(output_dir / "cross_cohort_fixed_total_effects.csv", index=False)
    write_table(frame, output_dir)
    create_figure(frame, output_dir)
    qa = {
        "overall_status": "PASS",
        "primary_metric": PRIMARY_METRIC,
        "datasets_present": sorted(frame["dataset_key"].unique().tolist()),
        "rows": len(frame),
        "all_effects_finite": bool(np.isfinite(frame["mean"]).all()),
        "all_expected_total_budgets_present": all(
            set(values["total_training_cells"]) == set(EXPECTED_TOTALS[key])
            for (key, _, _), values in frame.groupby(["dataset_key", "model", "task"])
        ),
    }
    if not qa["all_effects_finite"] or not qa["all_expected_total_budgets_present"]:
        qa["overall_status"] = "FAIL"
    (output_dir / "qa.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if qa["overall_status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
