"""Extract the 192-antibody COMBAT CITE-seq channel for scaling sensitivity."""

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

from prepare_combat_citeseq import LINEAGE_MAP


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_donors = pd.read_csv(args.reference_prepared_dir / "donors.csv")
    reference_audit = json.loads(
        (args.reference_prepared_dir / "dataset_audit.json").read_text(encoding="utf-8")
    )
    folds = json.loads(
        (args.reference_prepared_dir / "folds.json").read_text(encoding="utf-8")
    )
    evaluation_donors = set(reference_donors["donor_id"].astype(str))

    dataset = ad.read_h5ad(args.input, backed="r")
    obs = dataset.obs
    donor = obs["COMBAT_ID"].astype(str)
    minor = obs["Annotation_minor_subset"].astype(str)
    major = obs["Annotation_major_subset"].astype(str)
    valid = (
        donor.isin(evaluation_donors)
        & minor.ne("nan")
        & major.ne("nan")
        & ~obs["GEX_region"].astype(str).str.startswith("E:")
        & major.isin(LINEAGE_MAP)
    ).to_numpy()
    evaluation_indices = np.flatnonzero(valid)
    feature_indices = np.flatnonzero(
        dataset.var_names.astype(str).str.startswith("AB_")
    )
    if len(feature_indices) != 192:
        raise RuntimeError(f"Expected 192 ADT features, found {len(feature_indices)}")

    selected_obs = obs.iloc[evaluation_indices]
    metadata = pd.DataFrame(
        {
            "cell_id": selected_obs.index.astype(str),
            "sample_id": selected_obs["COMBAT_ID"].astype(str).to_numpy(),
            "disease": selected_obs["COMBAT_ID"].astype(str).map(
                reference_donors.set_index("donor_id")["stratum"]
            ).to_numpy(),
            "source": selected_obs["Source"].astype(str).to_numpy(),
            "institute": selected_obs["Institute"].astype(str).to_numpy(),
            "cell_subtype": selected_obs["Annotation_minor_subset"].astype(str).to_numpy(),
            "major_subset": selected_obs["Annotation_major_subset"].astype(str).to_numpy(),
        }
    )
    metadata["coarse_subtype"] = metadata["major_subset"].map(LINEAGE_MAP)
    if metadata.isna().any().any():
        raise RuntimeError("Prepared ADT metadata contain missing values.")
    if len(metadata) != reference_audit["evaluation_cells"]:
        raise RuntimeError(
            "RNA and ADT channels do not retain the same evaluation cells: "
            f"{len(metadata)} versus {reference_audit['evaluation_cells']}"
        )

    destination = args.output_dir / "expression.npy"
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

    feature_names = dataset.var_names[feature_indices].astype(str)
    pd.DataFrame(
        {
            "feature_index": feature_indices,
            "feature_id": feature_names,
            "feature_name": feature_names,
        }
    ).to_csv(args.output_dir / "features.csv", index=False)
    metadata.to_parquet(args.output_dir / "metadata.parquet", index=False)
    reference_donors.to_csv(args.output_dir / "donors.csv", index=False)
    (args.output_dir / "folds.json").write_text(
        json.dumps(folds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        **reference_audit,
        "source": "COMBAT CITE-seq ADT channel, original Zenodo object",
        "source_url": "https://zenodo.org/records/5139561",
        "source_sha256": sha256(args.input),
        "features": len(feature_indices),
        "feature_names": feature_names.tolist(),
        "matched_rna_evaluation_cells": True,
        "matched_rna_donors_and_folds": True,
    }
    (args.output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
