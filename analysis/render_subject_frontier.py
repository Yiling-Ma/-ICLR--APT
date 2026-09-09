"""Render the extended class-matched external subject frontier for the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/class_matched_subject_frontier"
DATASETS = {"combat_rna": "COMBAT RNA", "onek1k": "OneK1K RNA"}
COLORS = {3200: "#176B87", 6400: "#D95F3D"}


def load(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(OUTPUT / name / "frontier_metrics.csv")
    effects = pd.read_csv(OUTPUT / name / "frontier_effects.csv")
    metrics["dataset_label"] = DATASETS[name]
    effects["dataset_label"] = DATASETS[name]
    return metrics, effects


def render_figure(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.25), sharex="row")
    for row, dataset in enumerate(DATASETS.values()):
        for column, task in enumerate(("coarse", "fine")):
            ax = axes[row, column]
            subset = metrics[(metrics.dataset_label == dataset) & (metrics.task == task)]
            for total in sorted(subset.total.unique()):
                line = subset[subset.total == total].sort_values("patient_budget")
                ax.errorbar(
                    line.patient_budget,
                    line["mean"],
                    yerr=[line["mean"] - line.seed_ci_low, line.seed_ci_high - line["mean"]],
                    marker="o",
                    markersize=3.8,
                    linewidth=1.25,
                    capsize=2.2,
                    color=COLORS[int(total)],
                    label=f"T={int(total):,}",
                )
            ax.set_title(f"{dataset}: {task.capitalize()}", fontsize=9)
            ax.grid(color="#D9D9D9", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7.5)
            if column == 0:
                ax.set_ylabel("Subject-balanced pooled Macro-F1", fontsize=8)
            if row == 1:
                ax.set_xlabel("Training subjects", fontsize=8)
            ax.set_xticks(sorted(subset.patient_budget.unique()))
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=2, loc="best")
    fig.tight_layout(pad=0.6)
    fig.savefig(ROOT / "figures/extended_subject_frontier.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures/extended_subject_frontier.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def effect_text(row: pd.Series) -> str:
    return (
        f"{row.delta_mean:+.4f} "
        f"[{row.joint_ci_low:+.4f}, {row.joint_ci_high:+.4f}]"
    )


def render_table(effects: pd.DataFrame) -> None:
    adjacent = effects[
        ((effects.from_patient_budget == 32) & (effects.to_patient_budget == 64))
        | ((effects.from_patient_budget == 64) & (effects.to_patient_budget == 128))
    ].copy()
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrcc}",
        r"\toprule",
        r"Dataset & Task & $T$ & Subjects & $\Delta$ SB-pooled M-F1 & $\Pr(\Delta>0)$ \\",
        r"\midrule",
    ]
    previous = None
    for _, row in adjacent.sort_values(
        ["dataset", "from_patient_budget", "task", "total"]
    ).iterrows():
        if previous is not None and row.dataset != previous:
            lines.append(r"\midrule")
        lines.append(
            f"{row.dataset_label} & {str(row.task).capitalize()} & {int(row.total):,} & "
            f"{int(row.from_patient_budget)}$\\rightarrow${int(row.to_patient_budget)} & "
            f"{effect_text(row)} & {row.joint_probability_positive:.3f} \\\\"
        )
        previous = row.dataset
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Extended class-matched subject frontier for Logistic Regression. Each comparison holds the total training-cell budget, observed class set, and exact per-class cell quotas fixed while increasing the number of contributing training subjects. Values are paired mean changes; brackets are joint 95\% stability intervals from 2{,}000 replicates that sample a matched training-subset seed and then bootstrap held-out subjects. The final column is the empirical fraction of positive replicates, not a posterior probability. These ranges are not confidence intervals for seed-averaged effects.}",
        r"\label{tab:extended_subject_frontier}",
        r"\end{table}",
    ]
    (ROOT / "tables/extended_subject_frontier.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    frames = [load(name) for name in DATASETS]
    metrics = pd.concat([frame[0] for frame in frames], ignore_index=True)
    effects = pd.concat([frame[1] for frame in frames], ignore_index=True)
    render_figure(metrics)
    render_table(effects)


if __name__ == "__main__":
    main()
