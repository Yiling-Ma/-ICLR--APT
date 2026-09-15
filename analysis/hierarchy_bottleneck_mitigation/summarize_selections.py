#!/usr/bin/env python3
"""Summarize fold-specific validation selections without reading test scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CONDITIONS = ("plain", "parent", "parent_cons", "teacher", "kd", "parent_kd", "full")


def flatten(condition: str, fold: int, value: dict, trial_index: int | None = None) -> dict:
    weights = value.get("weights", {})
    row = {
        "condition": condition, "fold": fold, "lr": value.get("lr"),
        "weight_decay": value.get("decay"), "selected_epoch": value.get("epoch"),
        "validation_fine_sb_macro_f1": value.get("score"),
        "temperature": value.get("temperature", 1.0),
        "lambda_coarse": weights.get("coarse"), "lambda_parent": weights.get("parent", 0.0),
        "lambda_consistency": weights.get("cons", 0.0), "lambda_kd": weights.get("kd", 0.0),
    }
    if trial_index is not None:
        row["trial_index"] = trial_index
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected_rows, trial_rows = [], []
    for condition in CONDITIONS:
        for fold in range(5):
            path = args.input / condition / "seed17" / f"fold{fold}" / "selection.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            selected_rows.append(flatten(condition, fold, payload["selected"]))
            for trial_index, trial in enumerate(payload["trials"]):
                trial_rows.append(flatten(condition, fold, trial, trial_index))
    pd.DataFrame(selected_rows).to_csv(args.output / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(args.output / "validation_search_trials.csv", index=False)


if __name__ == "__main__":
    main()
