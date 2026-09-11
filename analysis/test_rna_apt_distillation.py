"""Numerical and statistical contract tests; no clinical claims."""
import unittest
import numpy as np
import rna_apt_distillation as d


class DistillationTests(unittest.TestCase):
    def setUp(self):
        self.parents = np.arange(27) % 5
        rng = np.random.default_rng(1)
        self.q = rng.dirichlet(np.ones(27), 80).astype(np.float32)
        self.r = rng.dirichlet(np.ones(27), 80).astype(np.float32)

    def test_preserves_parent_mass_and_normalization(self):
        for a in (np.zeros(5), np.ones(5), np.linspace(0, 1, 5)):
            mixed = d.mix_targets(self.q, self.r, self.parents, a)
            np.testing.assert_allclose(mixed.sum(1), 1, atol=2e-7)
            np.testing.assert_allclose(d.coarse_probs(mixed, self.parents),
                                       d.coarse_probs(self.q, self.parents), atol=2e-7)
        np.testing.assert_allclose(d.mix_targets(self.q, self.r, self.parents, np.ones(5)),
                                   self.q, atol=2e-7)

    def test_patient_weight_not_cell_weight(self):
        ids = np.array(["a"]*2 + ["b"]*8)
        w = d.patient_weights(ids)
        self.assertAlmostEqual(float(w[ids == "a"].sum()), float(w[ids == "b"].sum()))

    def test_identical_teachers_make_mixing_irrelevant(self):
        mixed = d.mix_targets(self.q, self.q, self.parents, np.linspace(0, 1, 5))
        np.testing.assert_allclose(mixed, self.q, atol=2e-7)

    def test_confusion_uses_all_ontology_classes(self):
        y = np.arange(80) % 27
        ids = np.array(["a"]*40 + ["b"]*40)
        patients, cm = d.evaluate(self.q, y, self.parents[y], ids, self.parents)
        self.assertEqual(cm["fine"].shape, (2, 27, 27))
        np.testing.assert_array_equal(cm["fine"].sum((1, 2)), [40, 40])


if __name__ == "__main__":
    unittest.main()
