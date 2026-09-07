"""Create patient-balanced lineage summaries for the top stable PSAS aptamers."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
PSAS_DIR = Path("cell_JEPA/outputs/psas_pipeline")
FIGURE_DIR = Path("figures")
DISEASE_ORDER = ["CRC", "GC", "HCC", "PC", "BTC", "Normal"]
LINEAGE_ORDER = ["T/NK", "B", "Myeloid"]
COLORS = {
    "CRC": "#315D6B",
    "GC": "#4D7C8A",
    "HCC": "#C56A3D",
    "PC": "#D59B45",
    "BTC": "#739B72",
    "Normal": "#8A8175",
}


def patient_lineage_table() -> tuple[pd.DataFrame, list[str]]:
    from apt_jepa.data.dataset import load_merged_dataframe

    merged, x, feature_names = load_merged_dataframe(
        str(DATA_DIR),
        str(DATA_DIR / "metadata.csv"),
        str(DATA_DIR / "cell_annotation.csv"),
        subtype_col_in_annotation="Celltypes_new",
        drop_missing_subtype=True,
        drop_unknown=True,
        coarse_mapping_config="cell_JEPA/apt_jepa/configs/coarse_lineage_explicit.yaml",
    )
    stability = pd.read_csv(PSAS_DIR / "aptamer_stability.csv").sort_values("mean_rank")
    aptamers = stability["aptamer"].head(3).tolist()
    feature_index = {name: i for i, name in enumerate(feature_names)}
    selected = np.asarray([feature_index[name] for name in aptamers])
    values = x[:, selected].astype(np.float64)

    lineage = merged["coarse_subtype"].replace({"T": "T/NK", "NK": "T/NK"})
    keep = lineage.isin(LINEAGE_ORDER).to_numpy()
    meta = merged.loc[keep, ["sample_id", "disease"]].reset_index(drop=True)
    meta["lineage"] = lineage.loc[keep].to_numpy()
    raw = values[keep]

    # Remove each patient's feature-wise median before lineage aggregation.
    centered = values.copy()
    for sample_id in merged["sample_id"].unique():
        mask = merged["sample_id"].to_numpy() == sample_id
        centered[mask] -= np.median(values[mask], axis=0, keepdims=True)
    centered = centered[keep]

    rows = []
    for signal, matrix in [("Raw", raw), ("Patient-centered", centered)]:
        frame = meta.copy()
        frame[aptamers] = matrix
        means = frame.groupby(["sample_id", "disease", "lineage"], observed=True)[aptamers].mean()
        counts = frame.groupby(["sample_id", "disease", "lineage"], observed=True).size()
        long = means.reset_index().melt(
            id_vars=["sample_id", "disease", "lineage"],
            value_vars=aptamers,
            var_name="aptamer",
            value_name="patient_lineage_mean",
        )
        long["signal"] = signal
        long = long.merge(counts.rename("n_cells").reset_index())
        rows.append(long)
    result = pd.concat(rows, ignore_index=True)
    return result, aptamers


def draw_panel(data: pd.DataFrame, aptamers: list[str], signal: str, output: Path) -> None:
    import matplotlib.pyplot as plt

    subset = data[data["signal"] == signal]
    fig, axes = plt.subplots(3, 3, figsize=(14.2, 7.6), sharex=False)
    rng = np.random.default_rng(42)

    for row, lineage in enumerate(LINEAGE_ORDER):
        for col, aptamer in enumerate(aptamers):
            ax = axes[row, col]
            panel = subset[(subset["lineage"] == lineage) & (subset["aptamer"] == aptamer)]
            values_by_disease = [
                panel.loc[panel["disease"] == disease, "patient_lineage_mean"].to_numpy()
                for disease in DISEASE_ORDER
            ]
            boxes = ax.boxplot(
                values_by_disease,
                positions=np.arange(len(DISEASE_ORDER)),
                widths=0.58,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#1D2426", "linewidth": 1.2},
                whiskerprops={"color": "#596164", "linewidth": 0.8},
                capprops={"color": "#596164", "linewidth": 0.8},
                boxprops={"edgecolor": "#596164", "linewidth": 0.8},
            )
            for disease, box in zip(DISEASE_ORDER, boxes["boxes"]):
                box.set_facecolor(COLORS[disease])
                box.set_alpha(0.22)
            for pos, (disease, vals) in enumerate(zip(DISEASE_ORDER, values_by_disease)):
                jitter = rng.uniform(-0.14, 0.14, len(vals))
                ax.scatter(
                    pos + jitter,
                    vals,
                    s=28,
                    color=COLORS[disease],
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.9,
                    zorder=3,
                )

            # Corresponding raw and centered panels use the same scale.
            both = data[(data["lineage"] == lineage) & (data["aptamer"] == aptamer)]
            low, high = both["patient_lineage_mean"].min(), both["patient_lineage_mean"].max()
            pad = max((high - low) * 0.09, 0.05)
            ax.set_ylim(low - pad, high + pad)
            counts = [len(vals) for vals in values_by_disease]
            labels = [f"{d}\n$n$={n}" for d, n in zip(DISEASE_ORDER, counts)]
            ax.set_xticks(np.arange(len(DISEASE_ORDER)), labels, fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            ax.grid(axis="y", color="#D9DEDE", linewidth=0.6, alpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(aptamer, fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{lineage}\npatient-lineage mean", fontsize=11)
            else:
                ax.set_ylabel("")

    title = "Raw APT signal" if signal == "Raw" else "Patient-centered APT signal"
    fig.suptitle(
        f"{title}: each point is one patient",
        x=0.5,
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.0, w_pad=0.8)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Plot an existing patient-level table instead of loading cell-level data.",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Write patient-level tables without importing Matplotlib.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PSAS_DIR.mkdir(parents=True, exist_ok=True)
    if args.input_csv:
        data = pd.read_csv(args.input_csv)
        aptamers = data["aptamer"].drop_duplicates().tolist()
    else:
        data, aptamers = patient_lineage_table()
    data.to_csv(PSAS_DIR / "patient_lineage_attribution.csv", index=False)
    summary = (
        data.groupby(["signal", "lineage", "aptamer", "disease"], observed=True)
        .agg(
            n_patients=("sample_id", "nunique"),
            median=("patient_lineage_mean", "median"),
            q1=("patient_lineage_mean", lambda s: s.quantile(0.25)),
            q3=("patient_lineage_mean", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    summary.to_csv(PSAS_DIR / "patient_lineage_attribution_summary.csv", index=False)
    if not args.table_only:
        draw_panel(data, aptamers, "Raw", FIGURE_DIR / "patient_lineage_attribution_raw.png")
        draw_panel(
            data,
            aptamers,
            "Patient-centered",
            FIGURE_DIR / "patient_lineage_attribution_centered.png",
        )
    print(f"Top stable PSAS aptamers: {', '.join(aptamers)}")
    print(data.groupby("disease")["sample_id"].nunique().reindex(DISEASE_ORDER))


if __name__ == "__main__":
    main()
