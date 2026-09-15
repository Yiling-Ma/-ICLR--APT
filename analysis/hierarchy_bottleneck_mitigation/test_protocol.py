"""Integration assertions against the real frozen APTBench protocol."""
import argparse
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from run import indices_for_fold, labels, load_protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log1p-cache", type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.log1p_cache)
    fine, coarse = labels(protocol)
    np.testing.assert_array_equal(protocol["parent"][fine], coarse)
    all_test_cells = []
    for fold in range(5):
        index = indices_for_fold(protocol, fold)
        assert not set(index["dev"]) & set(index["test"])
        assert set(index["train"]) | set(index["val"]) == set(index["dev"])
        scaler = StandardScaler().fit(protocol["x"][index["train"]])
        direct = scaler.transform(protocol["x"][index["val"][:100]])
        np.testing.assert_allclose(
            direct,
            (protocol["x"][index["val"][:100]] - scaler.mean_) / scaler.scale_,
            rtol=1e-6,
            atol=2e-7,
        )
        all_test_cells.extend(index["test"].tolist())
    assert len(all_test_cells) == 361792
    assert len(set(all_test_cells)) == 361792
    print("PASS real-data fold, ontology, scaler, and OOF coverage assertions")


if __name__ == "__main__":
    main()
