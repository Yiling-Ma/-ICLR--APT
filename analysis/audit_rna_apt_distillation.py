"""Audit completed distillation artifacts without selecting a winning method."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rna_apt_distillation as experiment


def audit(out):
    config = experiment.read_json(out / "protocol.json")
    meta = pd.read_csv(out / "cells.csv")
    y, coarse, parents = experiment.taxonomy(meta)
    records, alphas = [], []
    fit_count = 0
    for plan in config["plans"]:
        tr, te = np.asarray(plan["train"]), np.asarray(plan["test"])
        train_ids = set(meta.sample_id.iloc[tr])
        test_ids = set(meta.sample_id.iloc[te])
        assert train_ids.isdisjoint(test_ids)
        covered = []
        for job in plan["jobs"]:
            data = np.load(out / (job["name"] + "_targets.npz"))
            source, target = data["source"], data["target"]
            assert set(source) | set(target) == set(tr)
            assert set(meta.sample_id.iloc[source]).isdisjoint(
                set(meta.sample_id.iloc[target]) | test_ids)
            covered.extend(target.tolist())
        assert len(covered) == len(set(covered)) == len(tr)
        assert set(covered) == set(tr)
        for seed in experiment.SEEDS:
            for method in experiment.METHODS:
                name = f"f{plan['fold']}_{plan['stage']}_s{seed}_{method}.npz"
                data = np.load(out / name)
                assert str(data["protocol_hash"]) == config["protocol_hash"]
                assert np.array_equal(data["cells"], te)
                q = data["probabilities"]
                assert q.shape == (len(te), 27)
                assert np.isfinite(q).all() and (q >= 0).all()
                np.testing.assert_allclose(q.sum(1), 1, atol=1e-5)
                ids, cms = experiment.evaluate(q, y[te], coarse[te],
                    meta.sample_id.iloc[te].to_numpy(), parents)
                assert np.array_equal(ids.astype(str), data["patients"])
                for task in ("fine", "coarse"):
                    np.testing.assert_array_equal(cms[task], data[task])
                    records.append(dict(fold=plan["fold"], seed=seed,
                        method=method, task=task,
                        score=experiment.core.f1_from_confusion(
                            experiment.core.patient_balanced_matrix(cms[task]))))
                if seed == experiment.SEEDS[0] and method == "lineage":
                    alphas.append(dict(fold=plan["fold"],
                        **{f"lineage_{m}": float(a) for m, a in enumerate(data["alpha"])}))
                fit_count += 1
    pd.DataFrame(records).to_csv(out / "fold_seed_scores.csv", index=False)
    pd.DataFrame(alphas).to_csv(out / "lineage_alphas.csv", index=False)
    result = dict(status="PASS", fits_audited=fit_count,
                  protocol_hash=config["protocol_hash"],
                  checks=["upstream patient isolation and target coverage",
                          "evaluation indices and probability normalization",
                          "confusion matrices independently reconstructed"])
    experiment.save_json(out / "audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
