#!/usr/bin/env python3
"""Build the paper-facing HBM table from saved aggregate results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ORDER = ["plain", "parent", "parent_cons", "kd", "parent_kd", "full"]
LABELS = {
    "plain": "Plain MLP",
    "parent": "$+$ parent mass",
    "parent_cons": "$+$ parent mass $+$ consistency",
    "kd": "$+$ distillation",
    "parent_kd": "$+$ parent mass $+$ distillation",
    "full": "Full HBM",
}
COLUMNS = [
    "fine_sb_macro_f1",
    "coarse_sb_macro_f1",
    "cross_lineage_error_rate",
    "fine_outside_parent_given_coarse_correct",
    "oracle_fine_sb_macro_f1",
    "hierarchy_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.results.open(newline="") as handle:
        by_condition = {row["condition"]: row for row in csv.DictReader(handle)}
    if set(ORDER) - set(by_condition):
        missing = ", ".join(sorted(set(ORDER) - set(by_condition)))
        raise ValueError(f"missing required HBM conditions: {missing}")

    rows = []
    for condition in ORDER:
        source = by_condition[condition]
        values = [f"{float(source[column]):.3f}" for column in COLUMNS]
        if condition == "kd":
            values[0] = rf"\textbf{{{values[0]}}}"
        rows.append(f"{LABELS[condition]} & " + " & ".join(values) + r" \\")

    body = "\n".join(rows)
    table = rf"""\begin{{table}}[t]
\centering\small
\setlength{{\tabcolsep}}{{3.2pt}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
Training objective & Fine & Coarse & Cross error & Outside $\mid$ coarse correct & Oracle & Gap \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\caption{{Hierarchy-bottleneck mitigation with the unified MLP backbone.
Fine, coarse and oracle columns are subject-balanced Macro-F1; cross error is
the patient-balanced fraction of all predictions that are both wrong and
cross-lineage. Outside $\mid$ coarse correct is the fraction of cells whose
fine prediction leaves the true lineage among cells with a correct auxiliary
coarse prediction. Gap is oracle minus ordinary fine F1. Entries average three
matched seed slots over the same five patient folds. Oracle decoding supplies
the true lineage and is diagnostic, not deployable. Bold identifies the highest
deployable fine point estimate, not statistical superiority.}}
\label{{tab:hbm_ablation}}
\end{{table}}
"""
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.write_text(table)


if __name__ == "__main__":
    main()
