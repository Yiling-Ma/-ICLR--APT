import inspect
import numpy as np
import torch

from common import RNATeacher, cosine_alignment, grouped_target_permutation, routing_kd_loss
from hbm_core import HBMStudent, within_lineage_kd_loss


def test_losses_and_gradients():
    torch.manual_seed(3)
    student = HBMStudent(); teacher = RNATeacher()
    apt = torch.randn(8, 293); rna = torch.randn(8, 5000)
    parent = torch.tensor([0] * 5 + [1] * 6 + [2] * 5 + [3] * 5 + [4] * 6)
    truth_lineage = torch.tensor([0, 1, 2, 3, 4, 0, 1, 4])
    so = student(apt); to = teacher(rna)
    loss = (routing_kd_loss(so["coarse_logits"], to["coarse_logits"], 2)
            + within_lineage_kd_loss(so["fine_logits"], to["fine_logits"], truth_lineage, parent, 2)
            + cosine_alignment(so["representation"], to["representation"]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None for value in student.parameters())
    assert all(value.grad is None for value in teacher.parameters())


def test_student_inference_has_no_rna_or_lineage():
    assert list(inspect.signature(HBMStudent.forward).parameters) == ["self", "x"]


def test_target_shuffle_boundaries():
    patient = np.array(["a", "a", "a", "b", "b", "b"])
    lineage = np.array([0, 0, 1, 0, 0, 1])
    order = grouped_target_permutation(patient, lineage, 11)
    np.testing.assert_array_equal(patient, patient[order])
    np.testing.assert_array_equal(lineage, lineage[order])


if __name__ == "__main__":
    test_losses_and_gradients(); test_student_inference_has_no_rna_or_lineage()
    test_target_shuffle_boundaries(); print("PASS")
