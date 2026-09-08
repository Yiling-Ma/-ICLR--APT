"""Prepare the public COMBAT CITE-seq cohort for donor-disjoint scaling.

The CELLxGENE object contains the RNA channel of the COMBAT CITE-seq assay.
Ten donors are reserved only for unsupervised feature calibration and are never
used for model fitting or evaluation. The evaluation cohort includes donors
with at least 1,600 valid cells so every fixed-total design cell is exact.
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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


RANDOM_STATE = 20270908
N_CALIBRATION_DONORS = 10
MIN_EVALUATION_CELLS = 1600
N_FEATURES = 293
N_OUTER_FOLDS = 5

LINEAGE_MAP = {
    "CD4": "T",
    "CD8": "T",
    "GDT": "T",
    "DP": "T",
    "DN": "T",
    "MAIT": "T",
    "iNKT": "T",
    "B": "B",
    "PB": "B",
    "NK": "NK",
    "cMono": "Myeloid",
    "ncMono": "Myeloid",
    "DC": "Myeloid",
    "HSC": "Other",
    "PLT": "Other",
    "RET": "Other",
    "Mast": "Other",
}


def stable_mode(values: pd.Series) -> str:
    counts = values.astype(str).value_counts()
    return str(sorted(counts[counts == counts.max()].index)[0])


def collapsed_source(value: str) -> str:
    if value == "COVID_LDN":
        return "COVID_MILD"
    return value


def valid_cell_mask(obs: pd.DataFrame) -> np.ndarray:
    minor = obs["minor_subset"].astype(str)
    major = obs["major_subset"].astype(str)
    return (
        minor.ne("nan")
        & major.ne("nan")
        & ~obs["GEX_region"].astype(str).str.startswith("E:")
        & major.isin(LINEAGE_MAP)
    ).to_numpy()


def donor_table(obs: pd.DataFrame, valid: np.ndarray) -> pd.DataFrame:
    frame = obs.loc[valid, ["donor_id", "Source", "Institute"]].copy()
    table = (
        frame.groupby("donor_id", observed=True)
        .agg(
            source=("Source", stable_mode),
            institute=("Institute", stable_mode),
            n_valid_cells=("donor_id", "size"),
        )
        .reset_index()
    )
    table["stratum"] = table["source"].map(collapsed_source)
    return table.sort_values("donor_id").reset_index(drop=True)


def choose_calibration_donors(table: pd.DataFrame) -> list[str]:
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=N_CALIBRATION_DONORS, random_state=RANDOM_STATE
    )
    _, calibration = next(splitter.split(table, table["stratum"]))
    return sorted(table.iloc[calibration]["donor_id"].astype(str))


def select_features(
    dataset: ad.AnnData,
    obs: pd.DataFrame,
    valid: np.ndarray,
    calibration_donors: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    sampled: list[np.ndarray] = []
    donors = obs["donor_id"].astype(str).to_numpy()
    for donor in calibration_donors:
        indices = np.flatnonzero(valid & (donors == donor))
        size = min(1000, len(indices))
        sampled.append(np.sort(rng.choice(indices, size=size, replace=False)))
    calibration_indices = np.sort(np.concatenate(sampled))

    names = dataset.var["feature_name"].astype(str)
    is_protein_coding = dataset.var["feature_type"].astype(str).eq("protein_coding")
    excludes_technical = ~names.str.startswith(("MT-", "RPL", "RPS"))
    candidate_indices = np.flatnonzero((is_protein_coding & excludes_technical).to_numpy())
    matrix = dataset.X[calibration_indices, :][:, candidate_indices]
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        second = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    else:
        values = np.asarray(matrix, dtype=np.float64)
        mean = values.mean(axis=0)
        second = np.square(values).mean(axis=0)
    variance = second - np.square(mean)
    order = np.argsort(-variance, kind="stable")[:N_FEATURES]
    selected = candidate_indices[order]
    feature_table = dataset.var.iloc[selected].copy()
    feature_table.insert(0, "feature_index", selected)
    feature_table["calibration_variance"] = variance[order]
    feature_table["calibration_cells"] = len(calibration_indices)
    return selected.astype(np.int64), feature_table.reset_index(names="feature_id")


def create_folds(table: pd.DataFrame) -> dict[str, list[str]]:
    splitter = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    folds: dict[str, list[str]] = {}
    for fold, (_, test) in enumerate(splitter.split(table, table["stratum"])):
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
    chunk_size = 20_000
    for start in range(0, len(evaluation_indices), chunk_size):
        stop = min(start + chunk_size, len(evaluation_indices))
        rows = evaluation_indices[start:stop]
        values = dataset.X[rows, :][:, feature_indices]
        if sparse.issparse(values):
            values = values.toarray()
        matrix[start:stop] = np.asarray(values, dtype=np.float32)
        matrix.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ad.read_h5ad(args.input, backed="r")
    obs = dataset.obs.copy()
    valid = valid_cell_mask(obs)
    donors = donor_table(obs, valid)
    calibration_donors = choose_calibration_donors(donors)
    evaluation_donors = donors[
        ~donors["donor_id"].astype(str).isin(calibration_donors)
        & donors["n_valid_cells"].ge(MIN_EVALUATION_CELLS)
    ].copy()
    if len(evaluation_donors) < 40:
        raise RuntimeError("COMBAT evaluation cohort has fewer than 40 eligible donors.")

    feature_indices, features = select_features(
        dataset, obs, valid, calibration_donors
    )
    donor_values = obs["donor_id"].astype(str)
    evaluation_mask = valid & donor_values.isin(evaluation_donors["donor_id"].astype(str)).to_numpy()
    evaluation_indices = np.flatnonzero(evaluation_mask)
    metadata = obs.iloc[evaluation_indices].copy()
    metadata = pd.DataFrame(
        {
            "cell_id": metadata.index.astype(str),
            "sample_id": metadata["donor_id"].astype(str).to_numpy(),
            "disease": metadata["donor_id"].astype(str).map(
                evaluation_donors.set_index("donor_id")["stratum"]
            ).to_numpy(),
            "source": metadata["Source"].astype(str).to_numpy(),
            "institute": metadata["Institute"].astype(str).to_numpy(),
            "cell_subtype": metadata["minor_subset"].astype(str).to_numpy(),
            "major_subset": metadata["major_subset"].astype(str).to_numpy(),
        }
    )
    metadata["coarse_subtype"] = metadata["major_subset"].map(LINEAGE_MAP)
    if metadata.isna().any().any():
        raise RuntimeError("Prepared COMBAT metadata contain missing values.")
    pairs = metadata[["cell_subtype", "coarse_subtype"]].drop_duplicates()
    if pairs["cell_subtype"].duplicated().any():
        raise RuntimeError("A COMBAT minor subtype maps to multiple lineages.")

    extract_matrix(
        dataset,
        evaluation_indices,
        feature_indices,
        args.output_dir / "expression.npy",
    )
    metadata.to_parquet(args.output_dir / "metadata.parquet", index=False)
    evaluation_donors.to_csv(args.output_dir / "donors.csv", index=False)
    features.to_csv(args.output_dir / "features.csv", index=False)
    folds = create_folds(evaluation_donors)
    (args.output_dir / "folds.json").write_text(
        json.dumps(folds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fine_coverage = metadata.groupby("cell_subtype")["sample_id"].nunique()
    audit = {
        "source": "COMBAT CITE-seq RNA, CELLxGENE curated object",
        "source_url": "https://datasets.cellxgene.cziscience.com/359ed2e3-7f6d-4ec4-9e4a-63113cfbcd84.h5ad",
        "source_sha256": sha256(args.input),
        "raw_cells": int(dataset.n_obs),
        "raw_donors": int(obs["donor_id"].nunique()),
        "calibration_donors": calibration_donors,
        "evaluation_donors": int(len(evaluation_donors)),
        "evaluation_cells": int(len(metadata)),
        "minimum_cells_per_evaluation_donor": int(evaluation_donors["n_valid_cells"].min()),
        "features": int(len(features)),
        "coarse_classes": sorted(metadata["coarse_subtype"].unique()),
        "fine_classes": sorted(metadata["cell_subtype"].unique()),
        "ubiquitous_fine_classes_80pct": sorted(
            fine_coverage[fine_coverage >= 0.8 * len(evaluation_donors)].index
        ),
        "fold_sizes": {key: len(value) for key, value in folds.items()},
        "protocol_note": "Calibration donors are excluded from every model fit and test fold.",
    }
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
