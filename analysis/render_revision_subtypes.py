"""Keep every subtype while removing redundant columns, not shrinking type."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root/'analysis/generated/subtype_per_class.csv').open()))
assert len(rows) == 27 and sum(int(r['cell_count']) for r in rows) == 361792
lines = [r'\begingroup\small', r'\setlength{\tabcolsep}{4pt}',
         r'\renewcommand{\arraystretch}{1.10}', r'\begin{longtable}{llrrll}',
         r'\caption{Complete subtype support and cell-weighted per-class OOF F1 (not SB-F1), with pointwise 95\% patient-bootstrap intervals. Counts and patient coverage refer to the same 361,792 cells.}\label{tab:subtype_per_class} \\',
         r'\toprule', r'Subtype & Lineage & Cells & Patients & Soft-cascade & XGBoost \\',
         r'\midrule\endfirsthead',r'\toprule',
         r'Subtype & Lineage & Cells & Patients & Soft-cascade & XGBoost \\',
         r'\midrule\endhead',r'\bottomrule\endfoot']
for r in rows:
    scores = [f'{float(r[m+"_f1"]):.3f} [{float(r[m+"_ci_lower"]):.3f}, {float(r[m+"_ci_upper"]):.3f}]'
              for m in ['dropcascade','xgboost']]
    lines.append(' & '.join([r['subtype'],r['lineage'],f'{int(r["cell_count"]):,}',r['patient_count'],*scores])+r' \\')
lines += [r'\end{longtable}',r'\endgroup']
(root/'tables/subtype_per_class.tex').write_text('\n'.join(lines)+'\n')
print('PASS: retained all 27 classes and 361792 cells')
