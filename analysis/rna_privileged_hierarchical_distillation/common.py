"""Models, losses, aligned data, and isolation checks for RPH-Distill."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
ROOT = ANALYSIS.parent
sys.path[:0] = [str(ANALYSIS), str(ANALYSIS / "hierarchy_bottleneck_mitigation")]
import patient_cell_scaling as benchmark  # noqa: E402
import run_full_budget_mlp as unified  # noqa: E402
from hbm_core import (  # noqa: E402
    FINAL_SEEDS, HBMStudent, diagnostic_counts, oracle_predictions, set_seed,
    sha256_strings, within_lineage_kd_loss,
)

PROTOCOL_PATH = HERE / "PROTOCOL.md"
CONDITIONS = ("route", "within", "align", "route_align", "route_within", "full",
              "full_shuffled", "strong_lineage")


def protocol_hash() -> str:
    return hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_frozen(output: Path) -> None:
    source = PROTOCOL_PATH.read_text()
    destination = output / "PROTOCOL.md"
    if not destination.exists() or destination.read_text() != source:
        raise RuntimeError("Output protocol is missing or differs from frozen source")


def load_benchmark(apt_cache: Path) -> dict:
    meta, _, features = benchmark.load_data()
    manifest = json.loads(apt_cache.with_suffix(".json").read_text())
    if manifest.get("cell_id_sha256") != sha256_strings(meta.cell_id.astype(str)):
        raise AssertionError("APT cache cell order mismatch")
    if manifest.get("shape") != [361792, 293] or manifest.get("dtype") != "float32":
        raise AssertionError("APT cache shape/dtype mismatch")
    apt = np.load(apt_cache, mmap_mode="r")
    enc = benchmark.fit_label_encoders(meta); folds = benchmark.load_folds()
    ids = meta.sample_id.astype(str).to_numpy()
    fine = enc["fine"].transform(meta.cell_subtype.astype(str)).astype(np.int64)
    coarse = enc["coarse"].transform(meta.coarse_subtype.astype(str)).astype(np.int64)
    mapping = meta[["cell_subtype", "coarse_subtype"]].drop_duplicates().set_index("cell_subtype")
    parent = enc["coarse"].transform(mapping.loc[enc["fine"].classes_, "coarse_subtype"]).astype(np.int64)
    np.testing.assert_array_equal(parent[fine], coarse)
    assert len(meta) == 361792 and len(set(ids)) == 40 and apt.shape == (361792, 293)
    return dict(meta=meta, apt=apt, features=features, enc=enc, folds=folds, ids=ids,
                fine=fine, coarse=coarse, parent=parent)


def fold_indices(data: dict, fold: int) -> dict[str, np.ndarray]:
    folds, ids = data["folds"], data["ids"]
    test = np.flatnonzero(np.isin(ids, folds[fold]))
    val = np.flatnonzero(np.isin(ids, folds[(fold + 1) % 5]))
    train = np.flatnonzero(~np.isin(ids, folds[fold] + folds[(fold + 1) % 5]))
    dev = np.flatnonzero(~np.isin(ids, folds[fold]))
    result = dict(train=train, val=val, dev=dev, test=test)
    patients = {key: set(ids[index]) for key, index in result.items()}
    assert len(patients["train"]) == 24 and len(patients["val"]) == len(patients["test"]) == 8
    assert len(patients["dev"]) == 32 and not patients["train"] & patients["val"]
    assert not patients["dev"] & patients["test"]
    np.testing.assert_array_equal(np.sort(np.r_[train, val]), dev)
    return result


def apt_partition(data: dict, index: dict, fit: str) -> tuple[StandardScaler, dict[str, np.ndarray]]:
    scaler = StandardScaler().fit(np.asarray(data["apt"][index[fit]]))
    names = (fit, "val") if fit == "train" else (fit, "test")
    transformed = {name: scaler.transform(np.asarray(data["apt"][index[name]])).astype(np.float32)
                   for name in names}
    return scaler, transformed


def rna_partition(rna_root: Path, index: dict, fold: int, stage: str):
    prefix = rna_root / f"f{fold}_{stage}"
    left, right = (("train", "val") if stage == "inner" else ("dev", "test"))
    stored_left = np.loadtxt(str(prefix) + "_train.txt", dtype=int)
    stored_right = np.loadtxt(str(prefix) + "_test.txt", dtype=int)
    np.testing.assert_array_equal(stored_left, index[left])
    np.testing.assert_array_equal(stored_right, index[right])
    genes = Path(str(prefix) + "_genes.txt").read_text().splitlines()
    if len(genes) != 5000 or len(set(genes)) != 5000:
        raise AssertionError("RNA teacher requires exactly 5,000 unique train-selected HVGs")
    train = sparse.load_npz(str(prefix) + "_train_scaled.npz").tocsr()
    evaluation = sparse.load_npz(str(prefix) + "_test_scaled.npz").tocsr()
    assert train.shape == (len(index[left]), 5000) and evaluation.shape == (len(index[right]), 5000)
    return {left: train, right: evaluation}, genes


class RNATeacher(torch.nn.Module):
    def __init__(self, input_dim: int = 5000):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        )
        self.fine_head = torch.nn.Linear(128, 27)
        self.coarse_head = torch.nn.Linear(128, 5)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.encoder(values)
        return {"representation": representation, "fine_logits": self.fine_head(representation),
                "coarse_logits": self.coarse_head(representation)}


def routing_kd_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    teacher_prob = F.softmax(teacher.detach() / temperature, dim=1)
    student_log = F.log_softmax(student / temperature, dim=1)
    teacher_log = F.log_softmax(teacher.detach() / temperature, dim=1)
    return temperature ** 2 * (teacher_prob * (teacher_log - student_log)).sum(1).mean()


def cosine_alignment(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return (1 - (F.normalize(student, dim=1) * F.normalize(teacher.detach(), dim=1)).sum(1)).mean()


def grouped_target_permutation(patient: np.ndarray, lineage: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed); order = np.arange(len(patient))
    keys = np.asarray([f"{p}\x1f{g}" for p, g in zip(patient.astype(str), lineage)])
    for key in np.unique(keys):
        positions = np.flatnonzero(keys == key); order[positions] = rng.permutation(positions)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(order)))
    np.testing.assert_array_equal(patient, patient[order])
    np.testing.assert_array_equal(lineage, lineage[order])
    return order


def dense_rna_batch(matrix: sparse.csr_matrix, indices: np.ndarray) -> np.ndarray:
    return matrix[indices].toarray().astype(np.float32, copy=False)


def validation_score(truth, prediction, patients, classes=27) -> float:
    _, matrix = unified.matrices(truth, prediction, patients, classes)
    return float(unified.score(matrix))
