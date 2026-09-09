"""Render the paired-RNA oracle table from released result artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/paired_rna_oracle"
LABELS = {"apt": "APT only", "rna": "Paired RNA only", "apt_rna": "APT + paired RNA"}


def estimate(row: pd.Series) -> str:
    return (
        f"{row.subject_balanced_macro_f1:.3f} "
        f"[{row.patient_bootstrap_ci_low:.3f}, {row.patient_bootstrap_ci_high:.3f}]"
    )


def main() -> None:
    frame = pd.read_csv(OUTPUT / "oracle_metrics.csv").set_index(["modality", "task"])
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Input & Coarse Macro-F1 & Fine Macro-F1 & Granularity gap \\",
        r"\midrule",
    ]
    for modality in LABELS:
        coarse = frame.loc[(modality, "coarse")]
        fine = frame.loc[(modality, "fine")]
        gap = coarse.subject_balanced_macro_f1 - fine.subject_balanced_macro_f1
        lines.append(
            f"{LABELS[modality]} & {estimate(coarse)} & {estimate(fine)} & {gap:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Paired-RNA internal reconstructability oracle under the five fixed patient-disjoint folds. All inputs use the same patient- and class-weighted linear probe, up to 2{,}000 identically sampled training cells per development patient, training-fold standardization, and all held-out cells. Scores are subject-balanced pooled Macro-F1; brackets are patient-bootstrap 95\% intervals, and the granularity gap is coarse minus fine Macro-F1. This diagnostic uses the transcriptomic modality from which the annotations were constructed and a fixed 2{,}000-HVG representation selected during data development, so it is neither independent label validation nor a leakage-free benchmark baseline.}",
        r"\label{tab:paired_rna_oracle}",
        r"\end{table}",
    ]
    (ROOT / "tables/paired_rna_oracle.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
