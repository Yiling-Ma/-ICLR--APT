"""Independently verify development context membership and saved predictions."""
import argparse
from pathlib import Path
import numpy as np
import patient_context_pilot as p


def audit(out):
    cfg = p.common.read_json(out/"protocol.json")
    manifest = p.common.read_json(out/"context_manifest.json")
    splits = p.common.read_json(out/"upstream_splits.json")
    tr = set(cfg["train"])
    va = set(cfg["validation"])
    assert tr.isdisjoint(va)
    assert set(cfg["train_patients"]).isdisjoint(cfg["validation_patients"])
    covered = []
    mapping = {}
    for name in ("train", "validation"):
        data = dict(np.load(out/f"{name}.npz"))
        mapping.update(dict(zip(data["cells"].tolist(), data["patients"].tolist())))
        for patient in np.unique(data["patients"]):
            ix = np.flatnonzero(data["patients"] == patient)
            cells = data["cells"][ix]
            chosen = manifest[str(patient)]
            assert len(chosen) == len(set(chosen)) == min(p.BUDGET, len(ix))
            assert set(chosen) <= set(cells)
            context_ix = np.flatnonzero(np.isin(cells, chosen))
            x = data["x"][ix].astype(float)
            for j in range(len(ix)):
                others = context_ix[context_ix != j]
                expected = x[others].mean(0) if len(others) else np.zeros(x.shape[1])
                np.testing.assert_allclose(data["mean"][ix[j]], expected, atol=2e-5)
    for split in splits:
        source, target = set(split["source"]), set(split["target"])
        assert source | target == tr
        assert source.isdisjoint(target | va)
        assert {mapping[c] for c in source}.isdisjoint({mapping[c] for c in target | va})
        covered.extend(target)
    assert len(covered) == len(set(covered)) == len(tr)
    data = dict(np.load(out/"validation.npz"))
    fits = 0
    for method in p.METHODS:
        for loss in p.LOSSES:
            for seed in p.SEEDS:
                result = np.load(out/f"{method}_{loss}_{seed}.npz")
                assert str(result["protocol_hash"]) == cfg["protocol_hash"]
                np.testing.assert_array_equal(result["cells"], data["cells"])
                q = result["probabilities"]
                assert q.shape == (len(data["cells"]), 27)
                assert np.isfinite(q).all() and (q >= 0).all()
                np.testing.assert_allclose(q.sum(1), 1., atol=1e-5)
                ids, cms = p.common.evaluate(q, data["y"], data["coarse"], data["patients"], data["parents"])
                np.testing.assert_array_equal(ids, result["patients"])
                for task in ("fine", "coarse"):
                    np.testing.assert_array_equal(cms[task], result[task])
                fits += 1
    value = dict(status="PASS", fits_audited=fits, protocol_hash=cfg["protocol_hash"],
                 checks=["patient and cell partition isolation", "context budget and patient membership",
                         "every saved context mean excludes its own query", "probabilities and reconstructed confusion matrices"])
    p.common.save_json(out/"audit.json", value)
    print(value, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
