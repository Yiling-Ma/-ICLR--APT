import numpy as np
from scipy import sparse
from common import MatchedMLP, dense_batch, grouped_permutation


def test_permutations():
    patient = np.array(["a", "a", "a", "b", "b"])
    lineage = np.array([0, 0, 1, 0, 0])
    order, _ = grouped_permutation(patient, lineage, 7, False)
    np.testing.assert_array_equal(patient, patient[order])
    order, _ = grouped_permutation(patient, lineage, 8, True)
    np.testing.assert_array_equal(patient, patient[order])
    np.testing.assert_array_equal(lineage, lineage[order])


def test_dimension_control():
    rna = sparse.csr_matrix(np.ones((3, 5000), dtype=np.float32))
    extra = np.zeros((3, 293), dtype=np.float32)
    assert dense_batch(rna, extra, np.array([0, 2])).shape == (2, 5293)
    apt = MatchedMLP(5293); noise = MatchedMLP(5293); base = MatchedMLP(5000)
    assert sum(p.numel() for p in apt.parameters()) == sum(p.numel() for p in noise.parameters())
    assert sum(p.numel() for p in apt.parameters()) - sum(p.numel() for p in base.parameters()) == 293 * 256


if __name__ == "__main__":
    test_permutations(); test_dimension_control(); print("PASS")
