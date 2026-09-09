"""Audit committed artifacts, not permission to release participant data."""
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/release_audit"


def main():
    folds = json.loads((ROOT / "benchmark/splits/patient_folds.json").read_text())["folds"]
    patients = [p for group in folds.values() for p in group]
    assert len(folds) == 5 and all(len(g) == 8 for g in folds.values())
    assert len(patients) == len(set(patients)) == 40
    with (ROOT / "analysis/generated/subtype_per_class.csv").open() as f:
        subtypes = list(csv.DictReader(f))
    assert len(subtypes) == len({r["subtype"] for r in subtypes}) == 27
    assert len({r["lineage"] for r in subtypes}) == 5
    assert sum(int(r["cell_count"]) for r in subtypes) == 361792
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "subtype_to_lineage.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["subtype", "lineage"])
        writer.writerows((r["subtype"], r["lineage"]) for r in subtypes)
    paths = ["benchmark/splits/patient_folds.json", "analysis/generated/subtype_per_class.csv",
             "analysis/generated/hierarchy_main_metrics.csv", "analysis/generated/hierarchy_path_metrics.csv",
             "outputs/cross_cohort_scaling/cross_cohort_fixed_total_effects.csv",
             "outputs/class_matched_scaling/class_matched_effects_all.csv",
             "outputs/scaling_ground_truth_validation/summary.json",
             "outputs/scaling_protocol_contrast/plotted_effects.csv",
             "analysis/plot_scaling_protocol_contrast.py", "analysis/validate_scaling_ground_truth.py",
             "analysis/audit_release_inventory.py", "benchmark/release_audit/subtype_to_lineage.csv"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    (OUT / "checksums.json").write_text(json.dumps(hashes, indent=2)+"\n")
    audit = dict(folds=5, patients=40, unique_subtypes=27, lineages=5, cells=361792,
                 mapping_check="Structural lookup consistency only; no independent biological validation",
                 scope="Selected committed artifacts, not a complete dataset release")
    (OUT / "inventory.json").write_text(json.dumps(audit, indent=2)+"\n")
    versions = {}
    for name in ["numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "xgboost"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    (OUT / "local_analysis_environment.json").write_text(json.dumps(dict(
        python=platform.python_version(), system=platform.system(), packages=versions,
        scope="Current local audit/render environment; not original training environment or lockfile"), indent=2)+"\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
