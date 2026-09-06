import torch

from apt_jepa.losses.hierarchical_cross_entropy import (
    build_two_level_reachability,
    hierarchical_nll,
    hierarchical_scores,
)


def test_reachability_and_hce_scores():
    reachability = build_two_level_reachability([0, 0, 1], num_coarse=2, num_fine=3)
    assert reachability.shape == (5, 5)
    assert reachability[0].tolist() == [1, 0, 1, 1, 0]
    logits = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0]])
    scores = hierarchical_scores(logits, reachability)
    probabilities = logits.softmax(dim=1)
    assert torch.allclose(scores[0, 0], probabilities[0, [0, 2, 3]].sum())
    assert torch.allclose(scores[0, 4], probabilities[0, 4])


def test_hce_losses_are_finite_and_differentiable():
    logits = torch.randn(5, 5, requires_grad=True)
    y_coarse = torch.tensor([0, 0, 1, 1, 0])
    y_fine = torch.tensor([0, 1, 2, 2, 0])
    reachability = build_two_level_reachability([0, 0, 1], 2, 3)
    fine_loss = hierarchical_nll(logits, 2 + y_fine, reachability)
    coarse_loss = hierarchical_nll(logits, y_coarse, reachability)
    loss = fine_loss + 0.5 * coarse_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_leaf_hce_reduces_to_leaf_probability():
    logits = torch.randn(4, 5)
    reachability = build_two_level_reachability([0, 0, 1], 2, 3)
    scores = hierarchical_scores(logits, reachability)
    probabilities = logits.softmax(dim=1)
    assert torch.allclose(scores[:, 2:], probabilities[:, 2:])
