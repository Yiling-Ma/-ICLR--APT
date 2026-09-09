"""Audit resolution scores using subject-balanced confusion matrices.

The chance reference breaks true/predicted association while retaining both
pooled marginals. It is MacroF1 of an independence confusion matrix, not an
exact finite-sample expected permutation F1 or a cardinality-matched task.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/granularity_chance_audit"
N_BOOT = 2000
SEED = 42


def f1(cm):
    den = cm.sum(axis=-1) + cm.sum(axis=-2)
    return np.divide(2 * np.diagonal(cm, axis1=-2, axis2=-1), den,
                     out=np.zeros_like(den), where=den > 0).mean(axis=-1)


def scores(cm):
    cm = cm / cm.sum(axis=(-2, -1), keepdims=True)
    p, q = cm.sum(axis=-1), cm.sum(axis=-2)
    null = p[..., :, None] * q[..., None, :]
    chance = f1(null)
    raw = f1(cm)
    return raw, chance, (raw - chance) / (1 - chance)


def main():
    files = sorted((ROOT / "outputs/paired_rna_oracle/runs").glob("fold*.npz"))
    if len(files) != 5:
        raise ValueError("Expected five completed folds")
    arrays = [dict(np.load(p, allow_pickle=False)) for p in files]
    patients = np.concatenate([a["patient_ids"] for a in arrays])
    if len(patients) != 40 or len(np.unique(patients)) != 40:
        raise ValueError("Expected 40 unique OOF patients")
    weights = np.random.default_rng(SEED).multinomial(40, np.full(40, 1 / 40), N_BOOT)
    names = {"apt": "APT", "rna": "RNA", "apt_rna": "APT + RNA"}
    rows, gaps = [], []
    for modality, name in names.items():
        boot, point = {}, {}
        for task, k in (("coarse", 5), ("fine", 27)):
            cm = np.concatenate([a[f"{modality}_{task}"] for a in arrays])
            if cm.shape != (40, k, k) or (cm < 0).any():
                raise ValueError("Invalid confusion matrices")
            masses = cm.sum(axis=(1, 2))
            if (masses <= 0).any():
                raise ValueError("Empty patient")
            sb = cm / masses[:, None, None]
            raw, chance, adjusted = scores(sb.mean(axis=0))
            boot[task] = scores(np.einsum("bp,pij->bij", weights, sb))[2]
            point[task] = adjusted
            p = sb.mean(axis=0).sum(axis=1)
            entropy = -(p[p > 0] * np.log(p[p > 0])).sum()
            lo, hi = np.quantile(boot[task], [0.025, 0.975])
            rows.append(dict(modality=modality, task=task, classes=k,
                             raw=float(raw), chance=float(chance), adjusted=float(adjusted),
                             lower=float(lo), upper=float(hi), entropy=float(entropy),
                             effective_classes=float(np.exp(entropy)),
                             min_prevalence=float(p.min()), max_prevalence=float(p.max())))
        delta = boot["coarse"] - boot["fine"]
        lo, hi = np.quantile(delta, [0.025, 0.975])
        gaps.append(dict(modality=modality, adjusted_gap=float(point["coarse"] - point["fine"]),
                         lower=float(lo), upper=float(hi)))
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(seed=SEED, n_boot=N_BOOT, patients=40, metrics=rows, gaps=gaps,
                  sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in files})
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    tex = [r"\begin{table}[htbp]", r"\centering\small",
           r"\begin{tabular}{llrrr}", r"\toprule",
           r"Input & Resolution & Raw F1 & Chance reference & Adjusted F1 [95\% interval] \\",
           r"\midrule"]
    for row in rows:
        tex.append(f"{names[row['modality']]} & {row['task']} & {row['raw']:.3f} & "
                   f"{row['chance']:.3f} & {row['adjusted']:.3f} "
                   f"[{row['lower']:.3f}, {row['upper']:.3f}] " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Marginal-chance adjustment for the matched linear-probe RNA diagnostic. Scores use subject-balanced pooled confusion matrices from the same 40 OOF patients. Each of 2,000 paired patient-bootstrap draws recomputes the observed score and both null marginals. This is an association-breaking reference, not a task-cardinality or annotation-quality control.}",
            r"\label{tab:granularity_chance}", r"\end{table}"]
    (ROOT / "tables/granularity_chance.tex").write_text("\n".join(tex) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
