"""Render the historical CW table from audited, unrounded predictions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render():
    audit = json.loads((ROOT / 'analysis/generated/historical_split_audit.json').read_text())
    assert audit['route'] == 'A' and audit['new_training_runs'] == 0
    rows = []
    assert len(audit['contrasts']) == 4
    for item in audit['contrasts']:
        assert abs(item['difference'] - (item['cell'] - item['patient'])) < 1e-14
        model = 'XGBoost' if item['model'] == 'XGBoost' else r'\ourmethod{}'
        task = 'Coarse lineage' if item['task'] == 'coarse' else 'Fine subtype'
        rows.append(f"{task} & {model} & {item['patient']:.3f} & {item['cell']:.3f} & "
                    f"${item['difference']:+.3f}$ " + r'\\')
    table = r"""\begin{table}[!htbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\begin{tabular}{llccc}
\toprule
\textbf{Task} & \textbf{Model} & \textbf{Patient-disjoint} & \textbf{Cell split} & \textbf{Observed difference} \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Historical descriptive protocol comparison. Both score columns are
cell-weighted pooled Macro-F1 (CW), not the subject-balanced metric of
Table~\ref{tab:main}. Observed difference is cell split minus patient-disjoint,
computed from unrounded saved predictions before display rounding.
Training, validation and test allocations were not matched; the patient-disjoint
column pools five outer folds, whereas cell split evaluates one subset of cells
from already-seen patients. Protocol-difference uncertainty was not estimated.
These related model/resolution contrasts are not independent replications or
causal estimates of leakage.}
\label{tab:protocol_sensitivity}
\end{table}
"""
    (ROOT / 'tables/protocol_sensitivity.tex').write_text(table)


if __name__ == '__main__':
    render()
