#!/usr/bin/env python3
"""Build the historical-model comparison from committed result CSVs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unified_value(frame: pd.DataFrame, model: str, decoding: str) -> float:
    row = frame[(frame.model == model) & (frame.decoding == decoding)]
    if len(row) != 1:
        raise AssertionError(f"Expected one unified row for {model}/{decoding}, got {len(row)}")
    return float(row.iloc[0]["mean"])


def conditional_value(frame: pd.DataFrame, condition: str) -> float:
    row = frame[(frame.model == "mlp") & (frame.scope == -1) & (frame.condition == condition)]
    if len(row) != 1:
        raise AssertionError(f"Expected one conditional row for {condition}, got {len(row)}")
    return float(row.iloc[0]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--unified-scores", type=Path, required=True)
    parser.add_argument("--conditional-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ablation = pd.read_csv(args.ablation)
    unified = pd.read_csv(args.unified_scores)
    conditional = pd.read_csv(args.conditional_scores)
    rows = []
    for model, label in (("mlp", "Plain MLP"), ("flat", "Flat Transformer"),
                         ("hce", "HCE Transformer"), ("cascade", "Soft-cascade")):
        rows.append({"model": label,
                     "fine_sb_macro_f1": unified_value(unified, model, "D0"),
                     "coarse_sb_macro_f1": unified_value(unified, model, "coarse"),
                     "oracle_fine_sb_macro_f1": unified_value(unified, model, "D4"),
                     "artifact": "committed unified_v2"})
    rows.append({"model": "Conditional subtype MLP",
                 "fine_sb_macro_f1": conditional_value(conditional, "conditional_ordinary"),
                 "coarse_sb_macro_f1": np.nan,
                 "oracle_fine_sb_macro_f1": conditional_value(conditional, "conditional_oracle"),
                 "artifact": "committed hierarchy_signal"})
    full = ablation[ablation.condition == "full"]
    if len(full) != 1:
        raise AssertionError("Ablation table must contain exactly one Full HBM row")
    rows.append({"model": "Full HBM (new reproduction)",
                 "fine_sb_macro_f1": float(full.iloc[0].fine_sb_macro_f1),
                 "coarse_sb_macro_f1": float(full.iloc[0].coarse_sb_macro_f1),
                 "oracle_fine_sb_macro_f1": float(full.iloc[0].oracle_fine_sb_macro_f1),
                 "artifact": "new saved HBM predictions"})
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "historical_comparison.csv", index=False)
    lines = ["| Model | Fine SB-F1 | Coarse SB-F1 | Oracle SB-F1 | Artifact |",
             "|---|---:|---:|---:|---|"]
    for _, row in result.iterrows():
        coarse = "—" if pd.isna(row.coarse_sb_macro_f1) else f"{row.coarse_sb_macro_f1:.3f}"
        lines.append(f"| {row.model} | {row.fine_sb_macro_f1:.3f} | {coarse} | "
                     f"{row.oracle_fine_sb_macro_f1:.3f} | {row.artifact} |")
    (args.output / "historical_comparison.md").write_text("\n".join(lines) + "\n")
    (args.output / "historical_provenance.json").write_text(json.dumps({
        "unified_scores": str(args.unified_scores),
        "unified_scores_sha256": digest(args.unified_scores),
        "conditional_scores": str(args.conditional_scores),
        "conditional_scores_sha256": digest(args.conditional_scores),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
