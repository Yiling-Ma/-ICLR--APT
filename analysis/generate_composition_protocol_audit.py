#!/usr/bin/env python3
"""Generate a non-causal audit table for legacy versus strict composition pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LEGACY_TO_STRICT = {
    "Gold subtype composition": "true_subtype",
    "OOF predicted subtype composition": "pred_soft_subtype",
    "Gold lineage composition": "true_lineage",
    "OOF predicted lineage composition": "pred_soft_lineage",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        type=Path,
        default=Path("analysis/generated/composition_bridge_results.csv"),
    )
    parser.add_argument(
        "--strict",
        type=Path,
        default=Path("outputs/oof_composition_bridge/disease_results_summary.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/oof_composition_bridge")
    )
    args = parser.parse_args()

    legacy = pd.read_csv(args.legacy).set_index("representation")
    strict = pd.read_csv(args.strict)
    strict = strict[strict["transform"].eq("raw")].set_index("feature_set")
    rows = []
    for legacy_name, strict_name in LEGACY_TO_STRICT.items():
        rows.append(
            {
                "representation": legacy_name,
                "legacy_pooled_oof_macro_f1": float(
                    legacy.loc[legacy_name, "patient_macro_f1"]
                ),
                "strict_nested_macro_f1": float(strict.loc[strict_name, "macro_f1"]),
                "legacy_feature_construction": (
                    "gold composition"
                    if legacy_name.startswith("Gold")
                    else "hard labels from global pooled OOF predictions"
                ),
                "strict_feature_construction": (
                    "gold composition"
                    if legacy_name.startswith("Gold")
                    else "soft probabilities from inner/outer nested cross-fitting"
                ),
                "direct_effect_interpretation_valid": False,
            }
        )
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "legacy_vs_nested_audit.csv", index=False)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Representation & Legacy pooled OOF & Strict nested \\",
        r"\midrule",
    ]
    for row in rows:
        name = str(row["representation"]).replace("OOF predicted", "Predicted")
        lines.append(
            f"{name} & {row['legacy_pooled_oof_macro_f1']:.3f} & "
            f"{row['strict_nested_macro_f1']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Protocol audit of patient macro-F1. The legacy predicted rows use hard labels from one global pooled-OOF prediction set, CLR features, and fixed LR regularization; the strict rows use nested soft probabilities, raw proportions, and development-only LR selection. Because both feature construction and downstream fitting changed, the numerical differences are not estimates of a leakage effect. Only strict nested results support current claims.}",
            r"\label{tab:composition_protocol_audit}",
            r"\end{table}",
        ]
    )
    (args.output_dir / "legacy_vs_nested_audit.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
