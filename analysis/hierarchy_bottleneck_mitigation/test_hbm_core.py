import numpy as np
import torch

from hbm_core import (
    HBMStudent,
    aggregate_parent_log_probs,
    hierarchy_consistency_loss,
    parent_mass_loss,
    within_lineage_kd_loss,
)


PARENT = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 4], dtype=torch.long)


def _pad_parent():
    # Tests use the required 27-way/5-lineage shape.
    return torch.tensor(([0] * 5) + ([1] * 5) + ([2] * 5) + ([3] * 6) + ([4] * 6))


def test_parent_aggregation_is_normalized_and_matches_probability_sum():
    torch.manual_seed(2)
    logits = torch.randn(7, 27)
    parent = _pad_parent()
    log_parent = aggregate_parent_log_probs(logits, parent)
    assert torch.allclose(log_parent.exp().sum(1), torch.ones(7), atol=1e-6)
    fine = logits.softmax(1)
    for group in range(5):
        assert torch.allclose(log_parent[:, group].exp(), fine[:, parent == group].sum(1))


def test_losses_are_finite_and_both_heads_receive_consistency_gradients():
    torch.manual_seed(3)
    fine = torch.randn(8, 27, requires_grad=True)
    coarse = torch.randn(8, 5, requires_grad=True)
    parent = _pad_parent()
    truth = torch.arange(8) % 5
    loss = parent_mass_loss(fine, truth, parent) + hierarchy_consistency_loss(coarse, fine, parent)
    assert torch.isfinite(loss)
    loss.backward()
    assert fine.grad is not None and fine.grad.abs().sum() > 0
    assert coarse.grad is not None and coarse.grad.abs().sum() > 0


def test_within_lineage_kd_ignores_logits_outside_true_parent():
    torch.manual_seed(4)
    parent = _pad_parent()
    truth = torch.tensor([0, 4, 2])
    student = torch.randn(3, 27)
    teacher = torch.randn(3, 27)
    first = within_lineage_kd_loss(student, teacher, truth, parent, 2.0)
    changed = teacher.clone()
    changed[parent.unsqueeze(0) != truth.unsqueeze(1)] += 1000
    second = within_lineage_kd_loss(student, changed, truth, parent, 2.0)
    assert torch.allclose(first, second)


def test_student_inference_api_has_no_lineage_argument():
    model = HBMStudent()
    output = model(torch.randn(2, 293))
    assert output["fine_logits"].shape == (2, 27)
    assert output["coarse_logits"].shape == (2, 5)


def test_student_parameters_and_forward_match_unified_v2_mlp():
    class UnifiedReference(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(293, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            )
            self.fine = torch.nn.Linear(128, 27)
            self.coarse = torch.nn.Linear(128, 5)

        def forward(self, x):
            z = self.encoder(x)
            return self.fine(z), self.coarse(z)

    torch.manual_seed(17)
    reference = UnifiedReference().eval()
    torch.manual_seed(17)
    student = HBMStudent().eval()
    for expected, observed in zip(reference.parameters(), student.parameters()):
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
    x = torch.randn(13, 293)
    expected_fine, expected_coarse = reference(x)
    observed = student(x)
    torch.testing.assert_close(expected_fine, observed["fine_logits"], rtol=0, atol=0)
    torch.testing.assert_close(expected_coarse, observed["coarse_logits"], rtol=0, atol=0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
