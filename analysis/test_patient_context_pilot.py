import unittest
import numpy as np
import patient_context_pilot as p


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(60, dtype=np.float32).reshape(20, 3)/10
        self.q = np.eye(5, dtype=np.float32)[np.arange(20) % 5]
        self.ref = np.zeros((5, 3), np.float32)

    def test_query_does_not_enter_own_context(self):
        indices = np.arange(10)
        before = p.statistics(self.x, self.q, indices, self.ref)
        changed = self.x.copy()
        changed[0] += 100
        after = p.statistics(changed, self.q, indices, self.ref)
        for key in before:
            np.testing.assert_allclose(before[key][0], after[key][0], atol=2e-4)

    def test_uniform_reference_recovers_shared_shift_in_ideal_case(self):
        ref = np.arange(15, dtype=np.float32).reshape(5, 3)
        shift = np.array([2., -1., 3.], np.float32)
        x = ref[np.arange(20) % 5]+shift
        stats = p.statistics(x, self.q, np.arange(20), ref)
        np.testing.assert_allclose(stats["uniform"], np.tile(shift, (20, 1)), atol=1e-6)
        self.assertTrue(np.all(np.linalg.norm(stats["reliable"], axis=1) < np.linalg.norm(shift)))

    def test_singleton_correction_is_zero(self):
        stats = p.statistics(self.x[:1], self.q[:1], np.array([0]), self.ref)
        for key in ("mean", "std", "uniform", "reliable"):
            np.testing.assert_array_equal(stats[key], 0)

    def test_all_designs_have_same_nominal_dimension(self):
        stats = p.statistics(self.x, self.q, np.arange(10), self.ref)
        for method in p.METHODS:
            arr = p.design(self.x, stats, method)
            self.assertEqual(arr.shape, (20, 24))
            self.assertTrue(np.isfinite(arr).all())

    def test_training_references_weight_patients_equally(self):
        x = np.array([[0.], [0.], [0.], [10.]], np.float32)
        q = np.ones((4, 5), np.float32)/5
        ref = p.references(x, q, np.array(["a", "a", "a", "b"]))
        np.testing.assert_allclose(ref, 5., atol=1e-6)


if __name__ == "__main__":
    unittest.main()
