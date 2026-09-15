#!/usr/bin/env python3
"""Rebuild all RPH tables, paired intervals and figures from predictions."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent / "hierarchy_bottleneck_mitigation" / "aggregate.py"
sys.path.insert(0, str(source.parent))
spec = importlib.util.spec_from_file_location("hbm_saved_prediction_aggregate", source)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

module.ORDER = (
    "plain", "route", "within", "align", "route_align", "route_within",
    "full", "full_shuffled", "strong_lineage",
)
module.DISPLAY = {
    "plain": "Plain APT MLP",
    "route": "+ routing KD",
    "within": "+ within-lineage KD",
    "align": "+ representation alignment",
    "route_align": "+ routing KD + alignment",
    "route_within": "+ routing + within-lineage KD",
    "full": "Full RPH-Distill",
    "full_shuffled": "Full RPH (shuffled teacher)",
    "strong_lineage": "Strong lineage-label control",
}

if __name__ == "__main__":
    module.main()
