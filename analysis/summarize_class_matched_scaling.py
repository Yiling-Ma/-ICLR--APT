"""Summarize class-coverage diagnostics and class-matched scaling controls."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/class_matched_scaling"
DATASETS = {
    "apt": ("APT", ROOT / "outputs/patient_cell_scaling_fair"),
    "combat_rna": ("COMBAT RNA", ROOT / "outputs/combat_citeseq_scaling"),
    "onek1k": ("OneK1K RNA", ROOT / "outputs/onek1k_scaling"),
}
FIXED_POINTS = {
    3200: {8: "400", 32: "100"},
    6400: {8: "800", 32: "200"},
}


def original_coverage() -> pd.DataFrame:
    rows = []
    for key, (label, directory) in DATASETS.items():
        coverage = pd.read_csv(directory / "training_class_coverage.csv", dtype={"cell_cap": str})
        for total, points in FIXED_POINTS.items():
            for patient_budget, cell_cap in points.items():
                subset = coverage[
                    (coverage.task == "fine")
                    & (coverage.patient_budget == patient_budget)
                    & (coverage.cell_cap == cell_cap)
                ]
                grouped = subset.groupby(["outer_fold", "seed"]).agg(
                    observed_classes=("observed_in_training", "sum"),
                    total_classes=("class_index", "count"),
                    mean_donors_per_class=("training_patient_count", "mean"),
                )
                rows.append(
                    {
                        "dataset_key": key,
                        "dataset": label,
                        "total": total,
                        "patient_budget": patient_budget,
                        "observed_classes_mean": grouped.observed_classes.mean(),
                        "unseen_classes_mean": (grouped.total_classes - grouped.observed_classes).mean(),
                        "mean_donors_per_class": grouped.mean_donors_per_class.mean(),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "original_fine_class_coverage.csv", index=False)
    return result


def matched_effects() -> pd.DataFrame:
    rows = []
    for key, (label, _) in DATASETS.items():
        path = OUTPUT / key / "class_matched_effects.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", label)
        frame.insert(0, "dataset_key", key)
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUTPUT / "class_matched_effects_all.csv", index=False)
    return result


def format_effect(row: pd.Series) -> str:
    return (
        f"{row.delta_p32_minus_p8_mean:+.4f} "
        f"[{row.patient_bootstrap_ci_low:+.4f}, {row.patient_bootstrap_ci_high:+.4f}]"
    )


def write_table(effects: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrcc}",
        r"\toprule",
        "Dataset & Task & $T$ & Classes (mean/total) & Donors/class ($P$: 8$\\rightarrow$32) & $\\Delta$ subject-balanced M-F1 \\\\",
        r"\midrule",
    ]
    for _, row in effects.sort_values(["dataset_key", "task", "total"]).iterrows():
        donor_text = f"{row.mean_class_donor_coverage_p8:.1f}$\\rightarrow${row.mean_class_donor_coverage_p32:.1f}"
        lines.append(
            f"{row.dataset} & {str(row.task).capitalize()} & {int(row.total):,} & "
            f"{row.matched_class_count:.1f}/{int(row.total_class_count)} & {donor_text} & {format_effect(row)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Class-matched fixed-total control using class-weighted multinomial Logistic Regression. Within each outer-fold/seed/task pair, the $P=8$ subset freezes the observed classes and exact per-class cell quotas; the same quotas are used at $P=16$ and $P=32$. Brackets are 95\% paired subject-bootstrap intervals after averaging the 20 seed-level sufficient statistics; empirical seed intervals are released separately. Donors/class reports the mean number of subjects contributing cells to an observed class.}",
        r"\label{tab:class_matched_scaling}",
        r"\end{table}",
    ]
    (ROOT / "tables/class_matched_scaling.tex").write_text("\n".join(lines) + "\n")


def write_figure(effects: pd.DataFrame) -> None:
    order = [
        ("APT", "coarse"),
        ("APT", "fine"),
        ("COMBAT RNA", "coarse"),
        ("COMBAT RNA", "fine"),
        ("OneK1K RNA", "coarse"),
        ("OneK1K RNA", "fine"),
    ]
    labels = [f"{dataset} / {task.capitalize()}" for dataset, task in order]
    colors = {3200: "#176B87", 6400: "#D95F3D"}
    offsets = {3200: -0.12, 6400: 0.12}

    fig, ax = plt.subplots(figsize=(6.6, 3.55))
    for total in sorted(colors):
        subset = effects[effects.total == total].set_index(["dataset", "task"])
        means = [subset.loc[key, "delta_p32_minus_p8_mean"] for key in order]
        lows = [subset.loc[key, "patient_bootstrap_ci_low"] for key in order]
        highs = [subset.loc[key, "patient_bootstrap_ci_high"] for key in order]
        y = [index + offsets[total] for index in range(len(order))]
        ax.errorbar(
            means,
            y,
            xerr=[[mean - low for mean, low in zip(means, lows)],
                  [high - mean for mean, high in zip(means, highs)]],
            fmt="o",
            color=colors[total],
            markersize=4.5,
            capsize=2.5,
            linewidth=1.15,
            label=f"T={total:,}",
        )

    ax.axvline(0, color="#2C2C2C", linewidth=0.8)
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ subject-balanced Macro-F1: 32 vs. 8 subjects")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout(pad=0.35)
    fig.savefig(ROOT / "figures/class_matched_scaling.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures/class_matched_scaling.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def add_matched_class_counts(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in DATASETS:
        registry = pd.read_csv(OUTPUT / key / "run_registry.csv")
        counts = registry.groupby(["total", "task"]).agg(
            matched_class_count=("observed_class_count", "mean"),
            total_class_count=("n_classes", "first"),
        ).reset_index()
        counts.insert(0, "dataset_key", key)
        rows.append(counts)
    return effects.merge(pd.concat(rows), on=["dataset_key", "total", "task"], validate="many_to_one")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_coverage()
    effects = add_matched_class_counts(matched_effects())
    effects.to_csv(OUTPUT / "class_matched_effects_all.csv", index=False)
    write_table(effects)
    write_figure(effects)


if __name__ == "__main__":
    main()
