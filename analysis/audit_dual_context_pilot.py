"""Check actual episode membership, evaluation isolation, and predictions."""
import argparse
from pathlib import Path
import numpy as np
import dual_context_pilot as p


def audit(out):
    cfg = p.common.read_json(out/"protocol.json")
    data = dict(np.load(out/"data.npz"))
    assert str(data["protocol_hash"]) == cfg["protocol_hash"]
    assert set(data["train_patients"]).isdisjoint(data["val_patients"])
    assert set(data["train_cells"]).isdisjoint(data["val_cells"])
    outer = set(map(str, p.common.core.load_folds()[0]))
    assert outer.isdisjoint(set(data["train_patients"]) | set(data["val_patients"]))
    episodes = 0
    for seed in p.SEEDS:
        schedule = dict(np.load(out/f"schedule_{seed}.npz"))
        assert str(schedule["protocol_hash"]) == cfg["protocol_hash"]
        assert schedule["query"].shape == (p.STEPS, p.PATIENTS_PER_STEP, p.QUERIES)
        for query, first, second in zip(schedule["query"].reshape(-1, p.QUERIES),
                schedule["first"].reshape(-1, p.SUPPORT), schedule["second"].reshape(-1, p.SUPPORT)):
            assert len(np.unique(first)) == len(np.unique(second)) == p.SUPPORT
            assert len(np.unique(query)) == p.QUERIES
            assert set(query).isdisjoint(set(first) | set(second))
            assert len(np.unique(data["train_patients"][np.r_[query, first, second]])) == 1
            np.testing.assert_array_equal(np.unique(data["train_y"][first]), np.unique(data["train_y"][second]))
            episodes += 1
    selected = data["queries"]
    assert len(selected) == len(np.unique(selected))
    for j, patient in enumerate(data["patients"]):
        pool, first, second = data["pool"][j], data["support"][j], data["alternative"][j]
        assert len(np.unique(pool)) == 256
        assert len(np.unique(first)) == len(np.unique(second)) == p.SUPPORT
        assert set(first) <= set(pool) and set(second) <= set(pool)
        assert set(pool).isdisjoint(selected)
        assert np.all(data["val_patients"][pool] == patient)
        np.testing.assert_array_equal(np.unique(data["val_y"][first]), np.unique(data["val_y"][second]))
    fits = 0
    for method in p.METHODS:
        for seed in p.SEEDS:
            fit = dict(np.load(out/f"{method}_{seed}.npz"))
            assert str(fit["protocol_hash"]) == cfg["protocol_hash"]
            np.testing.assert_array_equal(fit["cells"], data["val_cells"][selected])
            assert np.isfinite(fit["training_losses"]).all()
            expected = 0 if method in ("raw", "context") else 8
            assert fit["diagnostics"].shape == (expected, 2)
            assert np.isfinite(fit["diagnostics"]).all()
            for condition in p.CONDITIONS:
                q = fit[condition+"_probabilities"]
                assert q.shape == (len(selected), 27) and np.isfinite(q).all() and (q >= 0).all()
                np.testing.assert_allclose(q.sum(1), 1., atol=1e-5)
                patients, cms = p.common.evaluate(q, data["val_y"][selected], data["val_coarse"][selected],
                    data["val_patients"][selected], data["parents"])
                np.testing.assert_array_equal(patients, fit["patients"])
                for task in ("fine", "coarse"):
                    np.testing.assert_array_equal(cms[task], fit[condition+"_"+task])
            fits += 1
    result = dict(status="PASS", fits_audited=fits, training_episodes_audited=episodes,
        protocol_hash=cfg["protocol_hash"], checks=["outer patients excluded", "queries excluded from supports",
        "same subtype support in composition views", "validation pool disjoint from every query",
        "all condition probabilities and confusion matrices independently reconstructed"])
    p.common.save_json(out/"audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
