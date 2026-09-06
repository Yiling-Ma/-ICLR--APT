"""Pool matched FT-Transformer predictions and compute five-fold OOF metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("cell_JEPA/outputs/matched_ft_transformer")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_rows = []
    for variant in ("flat_ce", "hce"):
        variant_dir = args.root / variant
        mapping = json.loads((variant_dir / "fold0" / "label_mapping.json").read_text())
        frames = []
        for fold in range(5):
            frame = pd.read_csv(variant_dir / f"fold{fold}" / "test_predictions.csv")
            frame["fold"] = fold
            frame["fine_true"] = frame["fine_true_id"].map(dict(enumerate(mapping["subtypes"])))
            frame["fine_pred"] = frame["fine_pred_id"].map(dict(enumerate(mapping["subtypes"])))
            frame["coarse_true"] = frame["coarse_true_id"].map(dict(enumerate(mapping["lineages"])))
            frame["coarse_pred"] = frame["coarse_pred_id"].map(dict(enumerate(mapping["lineages"])))
            frames.append(frame)
        pooled = pd.concat(frames, ignore_index=True)
        if pooled["sample_id"].nunique() != 40:
            raise ValueError(f"{variant}: expected 40 OOF patients")
        patient_fold_counts = pooled.groupby("sample_id")["fold"].nunique()
        if not (patient_fold_counts == 1).all():
            raise ValueError(f"{variant}: a patient appears in multiple test folds")
        pooled.to_csv(variant_dir / "pooled_oof_predictions.csv", index=False)
        row = {
            "model": "Flat FT-Transformer" if variant == "flat_ce" else "FT-Transformer + HCE",
            "variant": variant,
            "n_cells": len(pooled),
            "n_patients": pooled["sample_id"].nunique(),
            "fine_macro_f1": f1_score(pooled["fine_true"], pooled["fine_pred"], average="macro", zero_division=0),
            "fine_accuracy": accuracy_score(pooled["fine_true"], pooled["fine_pred"]),
            "coarse_macro_f1": f1_score(pooled["coarse_true"], pooled["coarse_pred"], average="macro", zero_division=0),
            "coarse_accuracy": accuracy_score(pooled["coarse_true"], pooled["coarse_pred"]),
        }
        summary_rows.append(row)
        (variant_dir / "pooled_summary.json").write_text(json.dumps(row, indent=2) + "\n")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.root / "pooled_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
