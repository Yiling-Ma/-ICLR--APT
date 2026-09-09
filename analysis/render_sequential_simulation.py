"""Render all completed synthetic settings, without selecting favorable seeds."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root/"outputs/sequential_scaling_simulation/results.json").read_text())
assert len(data["rows"]) == 8 * data["seeds"]
assert all(r["effects"]["A"] == r["effects"]["B"] for r in data["rows"])
lines = [r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabular}{rrrccc}",
         r"\toprule", r"Offset & Mixture & Separation & A: Unmatched & C: Quota & D: Quota + donor \\", r"\midrule"]
for r in data["summary"]:
    cells = []
    for c in "ACD":
        s = r["conditions"][c]
        lo,hi = s["empirical_seed_interval"]
        cells.append(r"\shortstack{"+f"{s['mean']:+.3f}"+r"\\{}"+f"[{lo:+.3f},{hi:+.3f}]"+"}")
    lines.append(f"{r['offset']:.1f} & {r['mixture']:.1f} & {r['separation']:.1f} & "+" & ".join(cells)+r" \\")
lines += [r"\bottomrule",r"\end{tabular}",
 r"\caption{Synthetic mechanism check: subject-balanced Macro-F1 difference between 32 and 8 training subjects at fixed total 1,600 cells. Entries are means across 20 independent seeds with empirical 2.5th--97.5th seed percentiles, not confidence intervals of the mean. Condition B equals A in every run because support is already complete and the support floor does not alter selection. All eight parameter settings are reported.}",
 r"\label{tab:sequential_simulation}",r"\end{table}"]
(root/"tables/sequential_simulation.tex").write_text("\n".join(lines)+"\n")
