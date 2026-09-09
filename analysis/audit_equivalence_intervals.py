"""Post-hoc practical-equivalence sensitivity using released joint intervals.

No TOST p-values are inferred from interval endpoints. Legacy intervals mix
training-subset variability and test resampling; containment is descriptive.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARGINS = (.005, .01, .02)


def flags(lo, hi, margin):
    if lo > hi:
        raise ValueError("Reversed interval")
    return dict(contains_zero=lo <= 0 <= hi,
                contained=lo > -margin and hi < margin,
                spans_both_meaningful_directions=lo <= -margin and hi >= margin)


def main():
    paths = [ROOT/"outputs/class_matched_scaling/class_matched_effects_all.csv"]
    paths += [ROOT/f"outputs/class_matched_subject_frontier/{d}/frontier_effects.csv"
              for d in ("combat_rna","onek1k")]
    rows = []
    for path in paths:
        for r in csv.DictReader(path.open()):
            primary = "dataset_key" in r
            lo = float(r["joint_resampling_ci_low" if primary else "joint_ci_low"])
            hi = float(r["joint_resampling_ci_high" if primary else "joint_ci_high"])
            row = dict(source=str(path.relative_to(ROOT)),
                dataset=r["dataset_key"] if primary else r["dataset"], task=r["task"],
                model=r["model"], total=int(r["total"]),
                contrast="8-32" if primary else r["from_patient_budget"]+"-"+r["to_patient_budget"],
                family="primary" if primary else "frontier", lower=lo, upper=hi,
                estimate=float(r["delta_p32_minus_p8_mean" if primary else "delta_mean"]))
            for margin in MARGINS:
                row[f"contained_{margin}"] = flags(lo,hi,margin)["contained"]
            row.update({k:v for k,v in flags(lo,hi,.01).items() if k != "contained"})
            rows.append(row)
    out = ROOT/"outputs/equivalence_sensitivity"
    out.mkdir(parents=True,exist_ok=True)
    with (out/"interval_audit.csv").open("w") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = []
    for family in ("primary","frontier"):
        for dataset in ("apt","combat_rna","onek1k"):
            for task in ("coarse","fine"):
                group = [r for r in rows if r["family"]==family and r["dataset"]==dataset
                         and r["task"]==task and (family=="frontier" or r["total"]==6400)
                         and r["contrast"] != "32-128"]
                if not group:
                    continue
                groups.append(dict(family=family,dataset=dataset,task=task,n=len(group),
                    counts=[sum(r[f"contained_{m}"] for r in group) for m in MARGINS],
                    spans_both=sum(r["spans_both_meaningful_directions"] for r in group)))
    result = dict(margins=MARGINS,groups=groups,rows=rows,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope="Post-hoc pointwise joint-interval containment; not preregistered, not formal TOST, not a simultaneous equivalence claim.")
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    tex = [r"\begin{table}[htbp]",r"\centering\small",r"\begin{tabular}{lllrrrrr}",r"\toprule",
           r"Comparison & Dataset & Task & $n$ & $\pm .005$ & $\pm .010$ & $\pm .020$ & Both signs$^*$ \\",r"\midrule"]
    names = dict(apt="APT",combat_rna="COMBAT",onek1k="OneK1K")
    for g in groups:
        tex.append(" & ".join(["8--32" if g["family"]=="primary" else "Extended LR",
            names[g["dataset"]],g["task"],str(g["n"]),*map(str,g["counts"]),str(g["spans_both"])])+r" \\")
    tex += [r"\bottomrule",r"\end{tabular}",
        r"\caption{Post-hoc practical-equivalence sensitivity: number of released joint 95\% intervals strictly inside each stated margin. The 8--32 rows use $T=6{,}400$ and both LR and XGBoost; extended rows use LR at both totals for adjacent subject-budget comparisons. $^*$Intervals reaching both $-0.01$ and $+0.01$, indicating insufficient precision to exclude meaningful effects in either direction. Counts are pointwise diagnostics, not formal TOST decisions or evidence that an entire scaling frontier is equivalent.}",
        r"\label{tab:equivalence_sensitivity}",r"\end{table}"]
    (ROOT/"tables/equivalence_sensitivity.tex").write_text("\n".join(tex)+"\n")
    print(json.dumps(groups,indent=2))


if __name__ == "__main__":
    main()
