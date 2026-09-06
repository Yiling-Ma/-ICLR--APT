"""Reachability-matrix hierarchical cross-entropy utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_two_level_reachability(
    subtype_to_lineage, num_coarse: int, num_fine: int
) -> torch.Tensor:
    """Return R[i,j]=1 when node j is node i or a descendant of node i."""
    mapping = torch.as_tensor(subtype_to_lineage, dtype=torch.long)
    if mapping.shape != (num_fine,):
        raise ValueError(f"Expected {num_fine} subtype mappings, got {mapping.shape}")
    if mapping.min().item() < 0 or mapping.max().item() >= num_coarse:
        raise ValueError("subtype_to_lineage contains an invalid lineage id")
    num_nodes = num_coarse + num_fine
    reachability = torch.eye(num_nodes, dtype=torch.float32)
    fine_ids = torch.arange(num_fine, dtype=torch.long) + num_coarse
    reachability[mapping, fine_ids] = 1.0
    return reachability


def hierarchical_scores(logits: torch.Tensor, reachability: torch.Tensor) -> torch.Tensor:
    """Aggregate each node's softmax mass with all descendant mass."""
    if logits.ndim != 2 or reachability.shape != (logits.shape[1], logits.shape[1]):
        raise ValueError("Logits and reachability dimensions do not match")
    probabilities = logits.softmax(dim=1)
    return probabilities @ reachability.to(logits).T


def hierarchical_nll(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reachability: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    scores = hierarchical_scores(logits, reachability)
    return F.nll_loss((scores + epsilon).log(), targets)
