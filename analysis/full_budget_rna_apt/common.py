"""Shared, frozen-protocol utilities for the full-budget paired experiment."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import patient_cell_scaling as core  # noqa: E402
import validate_paired_information as paired  # noqa: E402

PROTOCOL = Path(__file__).with_name("PROTOCOL.md")
WIDTHS = (5000, 10000)
SEEDS = (17, 29, 43)
SHUFFLE_SEEDS = {17: 73001, 29: 73002, 43: 73003}
PRIMARY = ("rna", "rna_apt")
CONTROLS = ("rna_noise", "patient_shuffle", "lineage_shuffle")
GRID = tuple(
    {"lr": lr, "weight_decay": wd}
    for lr in (3e-4, 1e-3)
    for wd in (1e-5, 1e-3)
)


def protocol_sha256() -> str:
    return hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def freeze_protocol(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "PROTOCOL.md"
    source = PROTOCOL.read_text()
    if destination.exists() and destination.read_text() != source:
        raise RuntimeError("Frozen output protocol differs from source protocol.")
    if not destination.exists():
        destination.write_text(source)
    atomic_json(output / "protocol_hash.json", {"sha256": protocol_sha256()})


def partitions(meta, fold: int) -> dict[str, np.ndarray]:
    folds = core.load_folds()
    ids = meta.sample_id.to_numpy(dtype=str)
    test = np.flatnonzero(np.isin(ids, folds[fold]))
    validation = np.flatnonzero(np.isin(ids, folds[(fold + 1) % 5]))
    inner = np.flatnonzero(~np.isin(ids, folds[fold] + folds[(fold + 1) % 5]))
    refit = np.flatnonzero(~np.isin(ids, folds[fold]))
    result = {"inner": inner, "validation": validation, "refit": refit, "test": test}
    patient_sets = {key: set(ids[value]) for key, value in result.items()}
    assert len(patient_sets["inner"]) == 24
    assert len(patient_sets["validation"]) == len(patient_sets["test"]) == 8
    assert len(patient_sets["refit"]) == 32
    assert not patient_sets["inner"] & patient_sets["validation"]
    assert not patient_sets["refit"] & patient_sets["test"]
    assert np.array_equal(np.sort(np.r_[inner, validation]), refit)
    return result


def grouped_permutation(patient: np.ndarray, lineage: np.ndarray, seed: int,
                        conditional: bool) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(patient))
    keys = np.asarray([f"{p}\x1f{g}" if conditional else p
                       for p, g in zip(patient.astype(str), lineage.astype(str))])
    singleton_groups = 0
    for key in np.unique(keys):
        positions = np.flatnonzero(keys == key)
        order[positions] = rng.permutation(positions)
        singleton_groups += int(len(positions) == 1)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(order)))
    np.testing.assert_array_equal(patient, patient[order])
    if conditional:
        np.testing.assert_array_equal(lineage, lineage[order])
    return order, {
        "conditional_on_lineage": conditional,
        "singleton_groups": singleton_groups,
        "fixed_fraction": float(np.mean(order == np.arange(len(order)))),
    }


def patient_confusions(truth: np.ndarray, prediction: np.ndarray,
                       patients: np.ndarray, classes: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.unique(patients.astype(str))
    matrices = np.stack([
        confusion_matrix(truth[patients == patient], prediction[patients == patient],
                         labels=np.arange(classes))
        for patient in order
    ])
    return order, matrices


def score(matrices: np.ndarray) -> float:
    return core.f1_from_confusion(core.patient_balanced_matrix(matrices))


class MatchedMLP(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        )
        self.fine = torch.nn.Linear(128, 27)
        self.coarse = torch.nn.Linear(128, 5)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(values)
        return self.fine(representation), self.coarse(representation)


def dense_batch(rna: sparse.csr_matrix, extra: np.ndarray | None,
                indices: np.ndarray) -> np.ndarray:
    values = rna[indices].toarray().astype(np.float32, copy=False)
    if extra is not None:
        values = np.concatenate([values, extra[indices]], axis=1)
    return values
