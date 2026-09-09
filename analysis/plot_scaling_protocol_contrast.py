"""Compare completed legacy protocols; do not impute pending sequential stages."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/scaling_protocol_contrast"
SOURCES = [ROOT / "outputs/cross_cohort_scaling/cross_cohort_fixed_total_effects.csv",
           ROOT / "outputs/class_matched_scaling/class_matched_effects_all.csv"]
ORDER = [(d, t) for d in ("apt", "combat_rna", "onek1k") for t in ("coarse", "fine")]
MODELS = {"logistic_regression": ("LR", "#176B87", "o", -.13),
          "xgboost": ("XGBoost", "#C75B36", "s", .13)}


def main():
    unmatched, matched = [pd.read_csv(p) for p in SOURCES]
    unmatched = unmatched[(unmatched.total_training_cells == 6400)
                          & unmatched.dataset_key.isin([d for d, _ in ORDER])]
    matched = matched[matched.total == 6400]
    keys = ["dataset_key", "task", "model"]
    a = unmatched.set_index(keys)
    b = matched.set_index(keys)
    expected = {(d, t, m) for d, t in ORDER for m in MODELS}
    assert set(a.index) == set(b.index) == expected
    assert not a.index.duplicated().any() and not b.index.duplicated().any()
    assert set(a.metric) == {"patient_balanced_macro_f1"}
    rows = []
    for d, t in ORDER:
        for m in MODELS:
            u, v = a.loc[(d, t, m)], b.loc[(d, t, m)]
            rows.append(dict(dataset=d, task=t, model=m, unmatched=u["mean"],
                unmatched_low=u.q025, unmatched_high=u.q975, unmatched_seeds=int(u.n_subset_seeds),
                matched=v.delta_p32_minus_p8_mean, matched_low=v.delta_ci_low,
                matched_high=v.delta_ci_high, matched_seeds=int(v.n_seeds),
                descriptive_change=v.delta_p32_minus_p8_mean-u["mean"],
                sign_change=bool(u["mean"]*v.delta_p32_minus_p8_mean < 0)))
    data = pd.DataFrame(rows)
    assert np.isfinite(data.select_dtypes(include="number")).all().all()
    OUTPUT.mkdir(exist_ok=True)
    data.to_csv(OUTPUT / "plotted_effects.csv", index=False)
    limits = data[["unmatched_low", "unmatched_high", "matched_low", "matched_high"]].to_numpy()
    bound = np.ceil(np.abs(limits).max()*100)/100
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), sharex=True, sharey=True)
    for ax, stage, title in zip(axes, ("unmatched", "matched"),
                                ("a  Unmatched", "b  Class-matched + donor allocation")):
        for i in range(6):
            if i % 2:
                ax.axhspan(i-.5, i+.5, color="#F0F3F5", zorder=0)
        for m, (label, color, marker, offset) in MODELS.items():
            subset = data[data.model == m]
            means = subset[stage].to_numpy()
            lo, hi = subset[stage+"_low"].to_numpy(), subset[stage+"_high"].to_numpy()
            # Draw endpoints directly: empirical quantiles need not bracket the mean.
            y = np.arange(6)+offset
            ax.hlines(y, lo, hi, color=color, lw=1.6)
            ax.scatter(means, y, c=color, marker=marker, s=27, label=label, zorder=3)
        ax.axvline(0, color="#333333", lw=.9, ls="--")
        ax.set_xlim(-bound*1.13, bound*1.13)
        ax.set_xticks([-bound, 0, bound])
        ax.set_title(title, loc="left", fontsize=10, pad=12, fontweight="bold")
        ax.grid(axis="x", color="#DFE3E5", lw=.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(6), [f"{d} / {t}" for d in ("APT", "COMBAT RNA", "OneK1K RNA")
                                 for t in ("Coarse", "Fine")])
    axes[0].set_ylim(5.55, -.55)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.61, 1.0),
               frameon=False, fontsize=9, ncol=2)
    fig.supxlabel("Change in subject-balanced pooled Macro-F1: 32 minus 8 subjects", fontsize=10, y=.025)
    fig.subplots_adjust(left=.235, right=.98, top=.81, bottom=.18, wspace=.18)
    fig.savefig(ROOT / "figures/scaling_protocol_contrast.pdf")
    fig.savefig(ROOT / "figures/scaling_protocol_contrast.png", dpi=220)
    plt.close(fig)
    audit = dict(rows=len(data), total=6400, subject_budgets=[8, 32],
        interval="Empirical 2.5th--97.5th percentiles over training-subset seeds; not mean CIs",
        interpretation="Descriptive cross-protocol comparison, not a paired attenuation test; matched includes donor round robin",
        pending_stage="Real-data support-only results not complete; excluded rather than imputed",
        hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},
        sign_changes=data.loc[data.sign_change, ["dataset", "task", "model"]].to_dict("records"))
    (OUTPUT / "manifest.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
