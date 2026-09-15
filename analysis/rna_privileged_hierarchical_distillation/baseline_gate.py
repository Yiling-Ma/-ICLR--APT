"""Fail-closed Plain MLP reproduction gate before any RPH outer refit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_full_budget_mlp as unified  # noqa: E402

SEEDS = (17, 29, 43)
REFERENCE = 0.13601048390624015
TOLERANCE = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    per_seed = []
    canonical_patients = None
    for seed in SEEDS:
        matrices = []
        patients = []
        cells = []
        for fold in range(5):
            path = args.output / "plain" / f"seed{seed}" / f"fold{fold}" / "predictions.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as item:
                matrices.append(item["fine_cm"])
                patients.append(item["patients"].astype(str))
                cells.append(item["cell_ids"].astype(str))
        patient_order = np.concatenate(patients)
        cell_order = np.concatenate(cells)
        if len(patient_order) != len(set(patient_order)) or len(patient_order) != 40:
            raise AssertionError("Plain OOF patients are incomplete or duplicated")
        if len(cell_order) != len(set(cell_order)) or len(cell_order) != 361792:
            raise AssertionError("Plain OOF cells are incomplete or duplicated")
        if canonical_patients is None:
            canonical_patients = patient_order
        else:
            np.testing.assert_array_equal(canonical_patients, patient_order)
        per_seed.append(unified.score(np.concatenate(matrices)))
    estimate = float(np.mean(per_seed))
    passed = abs(estimate - REFERENCE) <= TOLERANCE
    payload = {
        "status": "PASS" if passed else "FAIL",
        "estimate": estimate,
        "seed_values": per_seed,
        "reference": REFERENCE,
        "absolute_difference": abs(estimate - REFERENCE),
        "tolerance": TOLERANCE,
        "action": "RPH outer refits may proceed" if passed else "STOP before RPH outer refits",
    }
    destination = args.output / "baseline_gate.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
