"""Render completed, audited sequential controls without changing inference."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/sequential_scaling"
MODELS = {"logistic_regression": "LR", "xgboost": "XGBoost", "mlp": "MLP"}
audit = json.loads((DATA / "completion_audit.json").read_text())
assert audit["status"] == "PASS" and audit["fits"] == 7200


def cell(value):
    return (r"\makecell{" + f'{value["estimate"]:+.4f}' + r"\\{["
            + f'{value["lower"]:+.4f}, {value["upper"]:+.4f}' + "]}}")


blocks = []
for dataset, title in [("apt", "APT"), ("combat_rna", "COMBAT RNA"), ("onek1k", "OneK1K")]:
    for filename in ("summary", "protocol"):
        path = DATA / dataset / f"{filename}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["datasets"][dataset][f"{filename}_sha256"]
    assert json.loads((DATA / dataset / "completion.json").read_text()) == {"complete": True, "runs": 2400}
    rows = json.loads((DATA / dataset / "summary.json").read_text())["results"]
    assert len(rows) == 6
    blocks += [r"\clearpage", r"\subsection{" + title + "}",
               r"\begin{table}[htbp]\centering\small",
               r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.2}",
               r"\caption{" + title + r": paired $P=32$ minus $P=8$ effects at $T=6{,}400$. Each entry gives the mean effect and its pointwise joint 95\% bootstrap interval.}",
               r"\begin{tabular}{llrrrr}\toprule",
               r"Task & Model & A: pooled & B: support & C: quotas & D: donors \\\midrule"]
    for row in rows:
        blocks.append(" & ".join([row["task"].capitalize(), MODELS[row["model"]]]
                                  + [cell(row["conditions"][c]) for c in "ABCD"]) + r" \\")
    blocks += [r"\bottomrule\end{tabular}\end{table}",
               r"\begin{table}[htbp]\centering\small",
               r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.2}",
               r"\caption{" + title + r": paired sequential differences between allocation effects, with pointwise joint 95\% intervals. These are descriptive policy contrasts, not causal mediation estimates.}",
               r"\begin{tabular}{llrrr}\toprule",
               r"Task & Model & $\Delta_A-\Delta_B$ & $\Delta_B-\Delta_C$ & $\Delta_C-\Delta_D$ \\\midrule"]
    for row in rows:
        blocks.append(" & ".join([row["task"].capitalize(), MODELS[row["model"]]]
                                  + [cell(row["contrasts"][c]) for c in ["A-B", "B-C", "C-D"]]) + r" \\")
    blocks += [r"\bottomrule\end{tabular}\end{table}",
               "Intervals condition on the fixed cohort, folds, and fitted models. Near-zero bounds may round to signed zero; full precision is retained in the released JSON summaries."]
(ROOT / "tables/sequential_completed.tex").write_text("\n".join(blocks) + "\n")
print("PASS: source hashes, completion flags, and 18 summary rows; rendered six tables.")
