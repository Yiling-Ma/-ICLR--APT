"""Publish every ground-truth scenario and both bias and sampling-error metrics."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
directory=root/"outputs/scaling_ground_truth_validation"
data=json.loads((directory/"summary.json").read_text())
names=dict(known_null="Known null",mild_donor="Mild donor shift",strong_donor="Strong donor shift",
           correlated_cells="Correlated cells",weak_subtypes="Weak subtypes",batch_shift="Batch shift")
rows=[]
tex=[r"\begin{table}[htbp]",r"\centering\small",r"\begin{tabular}{lrrrr}",r"\toprule",
     r"Setting & Target (MCSE) & A: mean / RMSE & B: mean / RMSE & C: mean / RMSE \\",r"\midrule"]
for r in data["results"]:
    s=r["summary"]
    target="0 (exact)" if s["target_kind"]=="exact" else f"{s['target']:.4f} ({s['target_mcse']:.4f})"
    cells=[names[r["scenario"]["name"]],target]
    for c in "ABC":
        e=s["estimators"][c]
        cells.append(f"{e['mean']:+.4f} / {e['rmse']:.4f}")
        rows.append(dict(scenario=r["scenario"]["name"],condition=c,target=s["target"],target_mcse=s["target_mcse"],
                         estimate=e["mean"],bias=e["bias"],bias_mcse=e["bias_mcse"],
                         bias_lower=e["bias_interval"][0],bias_upper=e["bias_interval"][1],rmse=e["rmse"]))
    tex.append(" & ".join(cells)+r" \\")
tex += [r"\bottomrule",r"\end{tabular}",
 r"\caption{Known-null and independent Monte Carlo validation of the overlap-quota target. A, B, and C denote unmatched, support-matched, and exact-quota sampling. Non-null targets use 512 independent reference cohorts; estimator means and RMSE use 100 separate evaluation cohorts. MCSE denotes reference Monte Carlo standard error. The target is the expected effect of the C allocation policy, not unrestricted subject recruitment. All six settings are reported; RMSE is measured relative to the exact null or estimated reference target.}",
 r"\label{tab:scaling_ground_truth}",r"\end{table}"]
(root/"tables/scaling_ground_truth.tex").write_text("\n".join(tex)+"\n")
with (directory/"estimator_audit.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sources=[root/"analysis"/s for s in ("validate_scaling_ground_truth.py","sequential_scaling_controls.py","class_matched_scaling.py","audit_granularity_chance.py")]
(directory/"source_manifest.json").write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+"\n")
