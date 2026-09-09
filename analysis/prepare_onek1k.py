"""Prepare OneK1K for pool- and donor-disjoint scaling replication.

One entire multiplexing pool is reserved for unsupervised feature calibration.
Evaluation folds are disjoint in both donor and pool. The 800-cell eligibility
threshold preserves most donors while supporting the common 3,200- and
6,400-cell fixed-total comparisons exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy import sparse
from sklearn.model_selection import StratifiedGroupKFold


RANDOM_STATE = 20270908
MIN_EVALUATION_CELLS = 800
N_FEATURES = 293
N_OUTER_FOLDS = 5
MAX_CALIBRATION_DONORS = 10

FINE_COLUMN = "predicted.celltype.l2"
EXCLUDED_FINE = {"Doublet"}


def coarse_label(label: str) -> str:
    if label.startswith(("CD4", "CD8")) or label in {"Treg", "gdT", "MAIT", "dnT"}:
        return "T"
    if label.startswith("B ") or label == "Plasmablast":
        return "B"
    if label.startswith("NK") or label == "ILC":
        return "NK"
    if "Mono" in label or label.startswith(("cDC", "pDC")) or label == "ASDC":
        return "Myeloid"
    if label in {"HSPC", "Platelet", "Eryth"}:
        return "Other"
    raise ValueError(f"Unmapped OneK1K fine label: {label}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_value(values: pd.Series) -> str:
    unique = sorted(values.astype(str).unique())
    if len(unique) != 1:
        raise RuntimeError(f"Expected one donor-level value, found {unique}")
    return unique[0]


def donor_table(obs: pd.DataFrame, valid: np.ndarray) -> pd.DataFrame:
    table = (
        obs.loc[valid, ["donor_id", "pool_number", "sex"]]
        .groupby("donor_id", observed=True)
        .agg(
            pool=("pool_number", stable_value),
            sex=("sex", stable_value),
            n_valid_cells=("donor_id", "size"),
        )
        .reset_index()
    )
    return table.sort_values("donor_id").reset_index(drop=True)


def choose_calibration_pool(table: pd.DataFrame) -> tuple[str, list[str]]:
    candidates = []
    for pool, frame in table.groupby("pool", sort=True, observed=True):
        if len(frame) < MAX_CALIBRATION_DONORS or frame["sex"].nunique() < 2:
            continue
        imbalance = abs(frame["sex"].value_counts(normalize=True).get("female", 0.0) - 0.5)
        candidates.append((imbalance, str(pool), frame))
    if not candidates:
        raise RuntimeError("No OneK1K pool can supply the frozen calibration set.")
    _, pool, frame = min(candidates, key=lambda item: (item[0], item[1]))
    rng = np.random.default_rng(RANDOM_STATE)
    selected = []
    for _, group in frame.groupby("sex", sort=True, observed=True):
        values = group["donor_id"].astype(str).to_numpy().copy()
        rng.shuffle(values)
        selected.extend(values[: MAX_CALIBRATION_DONORS // 2])
    if len(selected) < MAX_CALIBRATION_DONORS:
        remaining = sorted(set(frame["donor_id"].astype(str)) - set(selected))
        selected.extend(remaining[: MAX_CALIBRATION_DONORS - len(selected)])
    return pool, sorted(selected)


def select_features(
    dataset: ad.AnnData,
    obs: pd.DataFrame,
    valid: np.ndarray,
    calibration_donors: list[str],
) -> tuple[np.ndarray, pd.DataFrame, int]:
    rng = np.random.default_rng(RANDOM_STATE)
    donor_values = obs["donor_id"].astype(str).to_numpy()
    sampled = []
    for donor in calibration_donors:
        indices = np.flatnonzero(valid & (donor_values == donor))
        sampled.append(np.sort(rng.choice(indices, size=min(1000, len(indices)), replace=False)))
    calibration_indices = np.sort(np.concatenate(sampled))

    names = dataset.var["feature_name"].astype(str)
    protein_coding = dataset.var["feature_type"].astype(str).eq("protein_coding")
    nontechnical = ~names.str.startswith(("MT-", "RPL", "RPS"))
    candidate_indices = np.flatnonzero((protein_coding & nontechnical).to_numpy())
    values = dataset.X[calibration_indices, :][:, candidate_indices]
    if sparse.issparse(values):
        mean = np.asarray(values.mean(axis=0)).ravel()
        second = np.asarray(values.power(2).mean(axis=0)).ravel()
    else:
        dense = np.asarray(values, dtype=np.float64)
        mean = dense.mean(axis=0)
        second = np.square(dense).mean(axis=0)
    variance = second - np.square(mean)
    order = np.argsort(-variance, kind="stable")[:N_FEATURES]
    selected = candidate_indices[order]
    features = dataset.var.iloc[selected].copy()
    features.insert(0, "feature_index", selected)
    features["calibration_variance"] = variance[order]
    features["calibration_cells"] = len(calibration_indices)
    return selected.astype(np.int64), features.reset_index(names="feature_id"), len(calibration_indices)


def create_folds(table: pd.DataFrame) -> dict[str, list[str]]:
    splitter = StratifiedGroupKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    folds = {}
    for fold, (_, test) in enumerate(
        splitter.split(table, y=table["sex"], groups=table["pool"])
    ):
        folds[str(fold)] = sorted(table.iloc[test]["donor_id"].astype(str))
    return folds


def extract_matrix(
    dataset: ad.AnnData,
    evaluation_indices: np.ndarray,
    feature_indices: np.ndarray,
    destination: Path,
) -> None:
    matrix = open_memmap(
        destination,
        mode="w+",
        dtype=np.float32,
        shape=(len(evaluation_indices), len(feature_indices)),
    )
    for start in range(0, len(evaluation_indices), 20_000):
        stop = min(start + 20_000, len(evaluation_indices))
        values = dataset.X[evaluation_indices[start:stop], :][:, feature_indices]
        if sparse.issparse(values):
            values = values.toarray()
        matrix[start:stop] = np.asarray(values, dtype=np.float32)
        matrix.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ad.read_h5ad(args.input, backed="r")
    obs = dataset.obs.copy()
    fine = obs[FINE_COLUMN].astype(str)
    valid = (~fine.isin(EXCLUDED_FINE)).to_numpy()
    donors = donor_table(obs, valid)
    calibration_pool, calibration_donors = choose_calibration_pool(donors)
    evaluation_donors = donors[
        donors["pool"].astype(str).ne(calibration_pool)
        & donors["n_valid_cells"].ge(MIN_EVALUATION_CELLS)
    ].copy()
    if len(evaluation_donors) < 5 * 32:
        raise RuntimeError("OneK1K has too few eligible donors for five outer folds.")

    feature_indices, features, calibration_cells = select_features(
        dataset, obs, valid, calibration_donors
    )
    donor_values = obs["donor_id"].astype(str)
    evaluation_mask = valid & donor_values.isin(evaluation_donors["donor_id"].astype(str)).to_numpy()
    evaluation_indices = np.flatnonzero(evaluation_mask)
    selected_obs = obs.iloc[evaluation_indices]
    metadata = pd.DataFrame(
        {
            "cell_id": selected_obs.index.astype(str),
            "sample_id": selected_obs["donor_id"].astype(str).to_numpy(),
            "disease": selected_obs["sex"].astype(str).to_numpy(),
            "sex": selected_obs["sex"].astype(str).to_numpy(),
            "pool": selected_obs["pool_number"].astype(str).to_numpy(),
            "cell_subtype": selected_obs[FINE_COLUMN].astype(str).to_numpy(),
        }
    )
    metadata["coarse_subtype"] = metadata["cell_subtype"].map(coarse_label)
    if metadata.isna().any().any() or metadata["cell_id"].duplicated().any():
        raise RuntimeError("Prepared OneK1K metadata are invalid.")
    pairs = metadata[["cell_subtype", "coarse_subtype"]].drop_duplicates()
    if pairs["cell_subtype"].duplicated().any():
        raise RuntimeError("A OneK1K fine type maps to multiple coarse lineages.")

    folds = create_folds(evaluation_donors)
    folded = [donor for values in folds.values() for donor in values]
    if len(folded) != len(set(folded)) or set(folded) != set(metadata["sample_id"]):
        raise RuntimeError("OneK1K fold assignment is incomplete or duplicated.")
    fold_pools = {
        key: sorted(evaluation_donors[evaluation_donors["donor_id"].isin(values)]["pool"].unique())
        for key, values in folds.items()
    }
    if sum(len(set(a) & set(b)) for i, a in fold_pools.items() for j, b in fold_pools.items() if i < j):
        raise RuntimeError("A OneK1K multiplexing pool occurs in multiple folds.")

    metadata.to_parquet(args.output_dir / "metadata.parquet", index=False)
    features.to_csv(args.output_dir / "features.csv", index=False)
    evaluation_donors.to_csv(args.output_dir / "donors.csv", index=False)
    (args.output_dir / "folds.json").write_text(
        json.dumps(folds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    extract_matrix(dataset, evaluation_indices, feature_indices, args.output_dir / "expression.npy")

    audit = {
        "status": "PASS",
        "source_url": "https://datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad",
        "source_sha256": sha256(args.input),
        "raw_cells": int(dataset.n_obs),
        "raw_donors": int(donors.shape[0]),
        "calibration_pool": calibration_pool,
        "calibration_donors": calibration_donors,
        "calibration_cells": calibration_cells,
        "evaluation_donors": int(metadata["sample_id"].nunique()),
        "evaluation_cells": int(len(metadata)),
        "minimum_cells_per_donor": int(metadata.groupby("sample_id").size().min()),
        "features": int(len(features)),
        "coarse_classes": sorted(metadata["coarse_subtype"].unique()),
        "fine_classes": sorted(metadata["cell_subtype"].unique()),
        "fold_sizes": {key: len(values) for key, values in folds.items()},
        "fold_pool_counts": {key: len(values) for key, values in fold_pools.items()},
        "eligibility_rule": "non-doublet donors with at least 800 cells; calibration pool excluded",
    }
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
