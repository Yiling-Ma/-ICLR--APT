"""Audit partitions and prepare full-budget, train-only RNA features."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.preprocessing import StandardScaler

from common import ROOT, WIDTHS, atomic_json, freeze_protocol, partitions, protocol_sha256, core


def write_indices(output: Path) -> dict:
    meta, _, _ = core.load_data()
    assert len(meta) == 361792 and meta.cell_id.is_unique
    encoders = core.fit_label_encoders(meta)
    folds = core.load_folds()
    output.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output / "cells.csv", index=False)
    counts = {}
    for fold in range(5):
        split = partitions(meta, fold)
        counts[str(fold)] = {
            key: {"cells": int(len(ix)), "patients": int(meta.sample_id.iloc[ix].nunique())}
            for key, ix in split.items()
        }
        for stage, left, right in (("inner", "inner", "validation"),
                                   ("outer", "refit", "test")):
            np.savetxt(output / f"f{fold}_{stage}_train.txt", split[left], fmt="%d")
            np.savetxt(output / f"f{fold}_{stage}_test.txt", split[right], fmt="%d")
    audit = {
        "status": "PASS",
        "protocol_sha256": protocol_sha256(),
        "cells": len(meta),
        "patients": int(meta.sample_id.nunique()),
        "fine_classes": len(encoders["fine"].classes_),
        "coarse_classes": len(encoders["coarse"].classes_),
        "folds": folds,
        "counts": counts,
        "cell_order_sha256": hashlib.sha256("\n".join(meta.cell_id.astype(str)).encode()).hexdigest(),
    }
    atomic_json(output / "partition_audit.json", audit)
    return audit


def prepare_width(root: Path, indices: Path, width: int, rscript: str) -> None:
    destination = root / "prepared" / f"hvg{width}"
    destination.mkdir(parents=True, exist_ok=True)
    for source in [indices / "cells.csv", *indices.glob("f*_train.txt"), *indices.glob("f*_test.txt")]:
        target = destination / source.name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Existing prepared index differs: {target}")
        else:
            shutil.copy2(source, target)
    environment = dict(os.environ, APT_RNA_HVG=str(width))
    subprocess.run([rscript, str(ROOT / "analysis" / "prepare_fold_rna.R"),
                    str(ROOT), str(destination)], env=environment, check=True)
    manifest = []
    for fold in range(5):
        for stage in ("inner", "outer"):
            prefix = destination / f"f{fold}_{stage}"
            genes = Path(str(prefix) + "_genes.txt").read_text().splitlines()
            if len(genes) != width or len(set(genes)) != width:
                raise RuntimeError(f"Invalid gene list for fold {fold} {stage}")
            train = mmread(str(prefix) + "_train.mtx").tocsr().astype(np.float32)
            test = mmread(str(prefix) + "_test.mtx").tocsr().astype(np.float32)
            train_ix = np.loadtxt(str(prefix) + "_train.txt", dtype=int)
            test_ix = np.loadtxt(str(prefix) + "_test.txt", dtype=int)
            assert train.shape == (len(train_ix), width) and test.shape == (len(test_ix), width)
            scaler = StandardScaler(with_mean=False).fit(train)
            train = scaler.transform(train).astype(np.float32)
            test = scaler.transform(test).astype(np.float32)
            sparse.save_npz(str(prefix) + "_train_scaled.npz", train, compressed=True)
            sparse.save_npz(str(prefix) + "_test_scaled.npz", test, compressed=True)
            np.savez_compressed(str(prefix) + "_rna_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
            manifest.append({
                "fold": fold, "stage": stage, "width": width,
                "train_cells": len(train_ix), "evaluation_cells": len(test_ix),
                "gene_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
                "genes": genes,
            })
    atomic_json(destination / "manifest.json", {"status": "PASS", "entries": manifest})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "rna"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=WIDTHS)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    freeze_protocol(args.output)
    indices = args.output / "prepared" / "indices"
    write_indices(indices)
    if args.command == "rna":
        if args.width is None:
            parser.error("--width is required for RNA preparation")
        prepare_width(args.output, indices, args.width, args.rscript)


if __name__ == "__main__":
    main()
