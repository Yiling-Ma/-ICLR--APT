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
import patient_cell_scaling as benchmark  # noqa: E402

SEEDS = (17, 29, 43)
REFERENCE = 0.13601048390624015
TOLERANCE = 0.01


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta, _, _ = benchmark.load_data()
    folds = benchmark.load_folds()
    ids = meta.sample_id.astype(str).to_numpy()
    per_seed = []
    for seed in SEEDS:
        matrices = []
        observed_cells = 0
        for fold in range(5):
            path = args.output / "plain" / f"seed{seed}" / f"fold{fold}" / "predictions.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as item:
                matrix = item["fine_cm"]
            if matrix.shape != (8, 27, 27):
                raise AssertionError(f"Unexpected patient confusion shape: {matrix.shape}")
            expected = int(np.isin(ids, folds[fold]).sum())
            if int(matrix.sum()) != expected:
                raise AssertionError(f"Fold {fold} confusion support does not match metadata")
            sidecar = json.loads((path.parent / "metrics.json").read_text())
            if sidecar["fold"] != fold or sidecar["seed"] != seed or sidecar["n_test_cells"] != expected:
                raise AssertionError("Prediction sidecar does not match requested fold/seed")
            matrices.append(matrix)
            observed_cells += expected
        if observed_cells != 361792:
            raise AssertionError("Plain OOF cell coverage is incomplete")
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
