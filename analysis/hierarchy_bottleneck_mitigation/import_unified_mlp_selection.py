#!/usr/bin/env python3
"""Import frozen, committed unified_v2 MLP validation selections.

This avoids rerunning an already validated historical grid while preserving a
machine-verifiable provenance link and the complete validation histories.
"""
import argparse
import hashlib
import json
from pathlib import Path

from hbm_core import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Path to analysis/unified_v2/results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for fold in range(5):
        source = args.source / f"mlp_f{fold}" / "joint_selection.json"
        raw = json.loads(source.read_text())
        chosen = raw["selected"]
        selected = {
            "lr": float(chosen["config"]["lr"]),
            "decay": float(chosen["config"]["decay"]),
            "weights": {"coarse": 0.5},
            "score": float(chosen["score"]),
            "epoch": int(chosen["epochs"]),
            "history": chosen["history"],
        }
        payload = {
            "selected": selected,
            "trials": raw["trials"],
            "seed": 17,
            "fold": fold,
            "selection_uses": "frozen committed unified_v2 validation selection",
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        destination = args.output / "plain" / "seed17" / f"fold{fold}" / "selection.json"
        atomic_json(destination, payload)
        print("IMPORTED", destination, selected)


if __name__ == "__main__":
    main()
