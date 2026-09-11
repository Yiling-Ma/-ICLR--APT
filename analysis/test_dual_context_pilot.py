import unittest
import numpy as np
import torch
import dual_context_pilot as p


class DualContextTests(unittest.TestCase):
    def test_composition_preserves_support_without_replacement(self):
        labels = np.repeat(np.arange(5), 60)
        original = np.arange(25, 225, 2)
        result = p.composition_view(np.arange(300), labels, original, np.random.default_rng(8))
        self.assertEqual(len(result), len(original))
        self.assertEqual(len(np.unique(result)), len(original))
        np.testing.assert_array_equal(np.unique(labels[result]), np.unique(labels[original]))

    def test_context_permutation_invariance(self):
        torch.manual_seed(9)
        net = p.ContextModel(7).eval()
        x = torch.randn(2, 15, 7)
        a, b = net.summarize(x), net.summarize(x[:, torch.randperm(15)])
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_shape_branch_is_translation_invariant(self):
        net = p.ContextModel(7).eval()
        x, shift = torch.randn(2, 15, 7), torch.randn(2, 1, 7)
        torch.testing.assert_close(net.summarize(x)[1], net.summarize(x+shift)[1], atol=1e-6, rtol=1e-5)

    def test_zero_offset_cannot_satisfy_equivariance(self):
        zero = torch.zeros(2, 7)
        comp, equiv = p.constraint_terms(zero, zero, zero, torch.ones_like(zero))
        self.assertEqual(float(comp), 0.)
        self.assertEqual(float(equiv), 1.)

    def test_correct_offset_increment_has_zero_penalty(self):
        a, shift = torch.randn(2, 7), torch.randn(2, 7)
        comp, equiv = p.constraint_terms(a, a, a+shift, shift)
        self.assertLess(float(comp), 1e-10)
        self.assertLess(float(equiv), 1e-10)

    def test_all_methods_output_finite_logits(self):
        net = p.ContextModel(7).eval()
        q, context = torch.randn(2, 3, 7), torch.randn(2, 15, 7)
        for method in p.METHODS:
            logits = net.classify(q, net.summarize(context), method)
            self.assertEqual(tuple(logits.shape), (2, 3, 27))
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
