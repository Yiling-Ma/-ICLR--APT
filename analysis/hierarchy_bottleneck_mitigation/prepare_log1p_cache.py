#!/usr/bin/env python3
"""Materialize an audited local-SSD cache of exact unified_v2 log1p(raw APT)."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import patient_cell_scaling as benchmark
import validate_paired_information as raw_input
from hbm_core import atomic_json, sha256_strings


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta, _, _ = benchmark.load_data()
    x = raw_input.raw_apt(meta)
    if x.shape != (361792, 293) or not np.isfinite(x).all() or (x < 0).any():
        raise AssertionError("Unexpected exact log1p APT matrix")
    temporary = args.output.with_suffix(".tmp.npy")
    np.save(temporary, x.astype(np.float32))
    os.replace(temporary, args.output)
    raw_matrix = raw_input.ROOT / "data/ALL_PBMC_APT_Cell_matrix.npz"
    barcodes = raw_input.ROOT / "data/cells.tsv"
    atomic_json(args.output.with_suffix(".json"), {
        "shape": list(x.shape), "dtype": "float32",
        "transform": "numpy.log1p on barcode-aligned raw nonnegative counts",
        "cell_id_sha256": sha256_strings(meta.cell_id.astype(str)),
        "raw_matrix_path": str(raw_matrix), "raw_matrix_sha256": file_sha256(raw_matrix),
        "barcodes_path": str(barcodes), "barcodes_sha256": file_sha256(barcodes),
    })
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
