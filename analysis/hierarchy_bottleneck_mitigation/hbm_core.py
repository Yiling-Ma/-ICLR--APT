"""Core models, hierarchy losses, metrics, and protocol checks for HBM-MLP.

This module is intentionally independent of the historical model classes.  The
student backbone matches ``analysis/run_full_budget_mlp.py`` exactly, while the
data/fold/metric implementation is imported by the runner from the audited
unified benchmark code.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FINAL_SEEDS = (17, 29, 43)
N_FEATURES = 293
N_LINEAGES = 5
N_SUBTYPES = 27


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_strings(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


class APTBackbone(nn.Module):
    """Exact hidden stack of the unified full-budget Plain MLP."""

    def __init__(self, input_dim: int = N_FEATURES, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HBMStudent(nn.Module):
    """APT-only student with fine and auxiliary coarse heads."""

    def __init__(self, input_dim: int = N_FEATURES):
        super().__init__()
        self.backbone = APTBackbone(input_dim)
        self.fine_head = nn.Linear(128, N_SUBTYPES)
        self.coarse_head = nn.Linear(128, N_LINEAGES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.backbone(x)
        return {
            "representation": representation,
            "fine_logits": self.fine_head(representation),
            "coarse_logits": self.coarse_head(representation),
        }


class OracleTeacher(nn.Module):
    """Fold-specific privileged teacher; true lineage is a training-only input."""

    def __init__(self, input_dim: int = N_FEATURES, lineage_embedding_dim: int = 16):
        super().__init__()
        self.backbone = APTBackbone(input_dim)
        self.lineage_embedding = nn.Embedding(N_LINEAGES, lineage_embedding_dim)
        self.fine_head = nn.Linear(128 + lineage_embedding_dim, N_SUBTYPES)

    def forward(self, x: torch.Tensor, true_lineage: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.backbone(x)
        conditioned = torch.cat([representation, self.lineage_embedding(true_lineage)], dim=1)
        return {"representation": representation, "fine_logits": self.fine_head(conditioned)}


def validate_parent_map(parent: np.ndarray | torch.Tensor) -> None:
    values = torch.as_tensor(parent, dtype=torch.long)
    if values.shape != (N_SUBTYPES,):
        raise ValueError(f"Expected parent map [{N_SUBTYPES}], got {tuple(values.shape)}")
    if values.min().item() != 0 or values.max().item() != N_LINEAGES - 1:
        raise ValueError("Parent IDs must span exactly 0..4")
    if torch.bincount(values, minlength=N_LINEAGES).min().item() == 0:
        raise ValueError("Every lineage must contain at least one subtype")


def aggregate_parent_log_probs(
    fine_logits: torch.Tensor, subtype_to_lineage: torch.Tensor
) -> torch.Tensor:
    """Stable log p(parent|x) implied directly by the 27-way fine head."""
    validate_parent_map(subtype_to_lineage)
    denominator = torch.logsumexp(fine_logits, dim=1, keepdim=True)
    columns = []
    for lineage in range(N_LINEAGES):
        mask = subtype_to_lineage == lineage
        columns.append(torch.logsumexp(fine_logits[:, mask], dim=1) - denominator[:, 0])
    result = torch.stack(columns, dim=1)
    # Aggregated groups partition all fine classes, so these are normalized.
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite aggregated parent log probabilities")
    return result


def parent_mass_loss(
    fine_logits: torch.Tensor,
    true_lineage: torch.Tensor,
    subtype_to_lineage: torch.Tensor,
) -> torch.Tensor:
    return F.nll_loss(
        aggregate_parent_log_probs(fine_logits, subtype_to_lineage), true_lineage
    )


def symmetric_kl_from_log_probs(log_q: torch.Tensor, log_a: torch.Tensor) -> torch.Tensor:
    """Symmetric KL with both arguments receiving gradients."""
    q, a = log_q.exp(), log_a.exp()
    kl_q_a = torch.sum(q * (log_q - log_a), dim=1)
    kl_a_q = torch.sum(a * (log_a - log_q), dim=1)
    return 0.5 * (kl_q_a + kl_a_q).mean()


def hierarchy_consistency_loss(
    coarse_logits: torch.Tensor,
    fine_logits: torch.Tensor,
    subtype_to_lineage: torch.Tensor,
) -> torch.Tensor:
    log_q = F.log_softmax(coarse_logits, dim=1)
    log_a = aggregate_parent_log_probs(fine_logits, subtype_to_lineage)
    return symmetric_kl_from_log_probs(log_q, log_a)


def mask_logits_to_true_lineage(
    logits: torch.Tensor,
    true_lineage: torch.Tensor,
    subtype_to_lineage: torch.Tensor,
) -> torch.Tensor:
    allowed = subtype_to_lineage.unsqueeze(0) == true_lineage.unsqueeze(1)
    return logits.masked_fill(~allowed, -torch.inf)


def within_lineage_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_lineage: torch.Tensor,
    subtype_to_lineage: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """T^2 KL(teacher || student), normalized among true-lineage children."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    masked_student = mask_logits_to_true_lineage(
        student_logits / temperature, true_lineage, subtype_to_lineage
    )
    masked_teacher = mask_logits_to_true_lineage(
        teacher_logits.detach() / temperature, true_lineage, subtype_to_lineage
    )
    student_log = F.log_softmax(masked_student, dim=1)
    teacher_log = F.log_softmax(masked_teacher, dim=1)
    teacher_prob = teacher_log.exp()
    # Avoid 0 * (inf - inf) on masked positions by explicitly zeroing them.
    allowed = subtype_to_lineage.unsqueeze(0) == true_lineage.unsqueeze(1)
    terms = torch.where(
        allowed, teacher_prob * (teacher_log - student_log), torch.zeros_like(student_log)
    )
    return temperature**2 * terms.sum(dim=1).mean()


def oracle_predictions(
    fine_probabilities: np.ndarray, truth_fine: np.ndarray, parent: np.ndarray
) -> np.ndarray:
    allowed = parent[None, :] == parent[truth_fine, None]
    return np.where(allowed, fine_probabilities, -np.inf).argmax(1)


def diagnostic_counts(
    truth_fine: np.ndarray,
    pred_fine: np.ndarray,
    truth_coarse: np.ndarray,
    pred_coarse: np.ndarray,
    parent: np.ndarray,
) -> dict[str, int]:
    predicted_parent = parent[pred_fine]
    fine_error = pred_fine != truth_fine
    cross = fine_error & (predicted_parent != truth_coarse)
    within = fine_error & ~cross
    coarse_correct = pred_coarse == truth_coarse
    return {
        "n": int(len(truth_fine)),
        "fine_correct": int((~fine_error).sum()),
        "fine_error": int(fine_error.sum()),
        "cross_lineage_error": int(cross.sum()),
        "within_lineage_error": int(within.sum()),
        "coarse_correct": int(coarse_correct.sum()),
        "fine_outside_parent_and_coarse_correct": int((cross & coarse_correct).sum()),
    }


def assert_patient_partition(
    train_patients: Sequence[str], val_patients: Sequence[str], test_patients: Sequence[str]
) -> None:
    train, val, test = map(lambda x: set(map(str, x)), (train_patients, val_patients, test_patients))
    if train & val or train & test or val & test:
        raise AssertionError("Patient leakage across train/validation/test")
    if len(train | val | test) != 40:
        raise AssertionError("Frozen partition must contain exactly 40 patients")
