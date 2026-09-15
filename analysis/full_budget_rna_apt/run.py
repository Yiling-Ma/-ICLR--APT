"""Select and refit matched full-budget RNA/RNA+APT MLPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from common import (
    CONTROLS, GRID, PRIMARY, ROOT, SEEDS, SHUFFLE_SEEDS, WIDTHS, MatchedMLP,
    atomic_json, dense_batch, freeze_protocol, grouped_permutation,
    partitions, patient_confusions, protocol_sha256, score, core, paired,
)


def apt_values(meta, cache: Path | None) -> np.ndarray:
    if cache is not None:
        values = np.load(cache, mmap_mode="r")
        if values.shape != (len(meta), 293):
            raise RuntimeError(f"APT cache shape mismatch: {values.shape}")
        manifest_path = cache.with_suffix(".json")
        if not manifest_path.exists():
            raise RuntimeError(f"Missing APT cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        cell_hash = hashlib.sha256("\n".join(meta.cell_id.astype(str)).encode()).hexdigest()
        if manifest.get("cell_id_sha256") != cell_hash:
            raise RuntimeError("APT cache cell order does not match benchmark metadata")
        if manifest.get("transform") != "numpy.log1p on barcode-aligned raw nonnegative counts":
            raise RuntimeError("APT cache transform does not match the frozen protocol")
        return values
    return paired.raw_apt(meta)


def control_seed(seed: int, width: int, fold: int, stage: str, side: int) -> int:
    base = SHUFFLE_SEEDS[seed]
    return base + width * 100 + fold * 10 + (stage == "outer") * 2 + side


def load_inputs(args, meta, lineage, split, stage: str, seed: int):
    prepared = args.output / "prepared" / f"hvg{args.width}"
    prefix = prepared / f"f{args.fold}_{stage}"
    left_name, right_name = (("inner", "validation") if stage == "inner" else ("refit", "test"))
    left_ix, right_ix = split[left_name], split[right_name]
    stored_left = np.loadtxt(str(prefix) + "_train.txt", dtype=int)
    stored_right = np.loadtxt(str(prefix) + "_test.txt", dtype=int)
    np.testing.assert_array_equal(stored_left, left_ix)
    np.testing.assert_array_equal(stored_right, right_ix)
    rna_left = sparse.load_npz(str(prefix) + "_train_scaled.npz").tocsr()
    rna_right = sparse.load_npz(str(prefix) + "_test_scaled.npz").tocsr()
    assert rna_left.shape == (len(left_ix), args.width)
    assert rna_right.shape == (len(right_ix), args.width)

    extra_left = extra_right = None
    extra_audit = {"kind": "none"}
    if args.condition == "rna_noise":
        left_shape = (len(left_ix), 293); right_shape = (len(right_ix), 293)
        extra_left = np.random.default_rng(control_seed(seed, args.width, args.fold, stage, 0)).standard_normal(left_shape).astype(np.float32)
        extra_right = np.random.default_rng(control_seed(seed, args.width, args.fold, stage, 1)).standard_normal(right_shape).astype(np.float32)
        extra_audit = {"kind": "independent_standard_normal", "label_access": False,
                       "apt_value_access": False}
        apt_scaler = None
    elif args.condition != "rna":
        apt = apt_values(meta, args.apt_cache)
        scaler = StandardScaler().fit(np.asarray(apt[left_ix]))
        left = scaler.transform(np.asarray(apt[left_ix])).astype(np.float32)
        right = scaler.transform(np.asarray(apt[right_ix])).astype(np.float32)
        if args.condition == "rna_apt":
            extra_left, extra_right = left, right
            extra_audit = {"kind": "paired_apt"}
        else:
            conditional = args.condition == "lineage_shuffle"
            left_order, left_audit = grouped_permutation(
                meta.sample_id.iloc[left_ix].to_numpy(str), lineage[left_ix],
                control_seed(seed, args.width, args.fold, stage, 0), conditional)
            right_order, right_audit = grouped_permutation(
                meta.sample_id.iloc[right_ix].to_numpy(str), lineage[right_ix],
                control_seed(seed, args.width, args.fold, stage, 1), conditional)
            extra_left, extra_right = left[left_order], right[right_order]
            extra_audit = {"kind": args.condition, "left": left_audit, "right": right_audit}
        apt_scaler = {"mean": scaler.mean_.astype(np.float32), "scale": scaler.scale_.astype(np.float32)}
    else:
        apt_scaler = None
    return (rna_left, extra_left, left_ix), (rna_right, extra_right, right_ix), apt_scaler, extra_audit


def predict(model, rna, extra, device):
    model.eval(); fine = []; coarse = []
    with torch.no_grad():
        for start in range(0, len(rna), 4096):
            indices = np.arange(start, min(start + 4096, len(rna)))
            values = torch.as_tensor(dense_batch(rna, extra, indices), device=device)
            f, c = model(values)
            fine.append(f.softmax(1).cpu().numpy())
            coarse.append(c.softmax(1).cpu().numpy())
    return np.concatenate(fine), np.concatenate(coarse)


def train(rna, extra, fine_y, coarse_y, patient_ids, config, seed, epochs,
          patience, device, validation=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = MatchedMLP(rna.shape[1] + (0 if extra is None else extra.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    rng = np.random.default_rng(seed)
    best_score, best_epoch, history = -np.inf, 1, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(rna))
        for start in range(0, len(order), 1024):
            indices = order[start:start + 1024]
            values = torch.as_tensor(dense_batch(rna, extra, indices), device=device)
            fy = torch.as_tensor(fine_y[indices], dtype=torch.long, device=device)
            cy = torch.as_tensor(coarse_y[indices], dtype=torch.long, device=device)
            fine_logits, coarse_logits = model(values)
            loss = (torch.nn.functional.cross_entropy(fine_logits, fy)
                    + 0.5 * torch.nn.functional.cross_entropy(coarse_logits, cy))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if validation is not None:
            vrna, vextra, vy, vpatients = validation
            fine_prob, _ = predict(model, vrna, vextra, device)
            _, cms = patient_confusions(vy, fine_prob.argmax(1), vpatients, 27)
            value = score(cms)
            history.append(float(value))
            if value > best_score:
                best_score, best_epoch = float(value), epoch
            print("epoch", epoch, "validation", value, flush=True)
            if epoch - best_epoch >= patience:
                break
        else:
            print("epoch", epoch, "refit", flush=True)
    return model, {"score": float(best_score), "epochs": best_epoch, "history": history}


def run(args):
    freeze_protocol(args.output)
    if not str(args.device).startswith("cuda"):
        raise ValueError("The frozen protocol requires GPU training; CPU is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to fall back to CPU")
    if args.condition in CONTROLS and args.width != 10000:
        raise ValueError("Predeclared controls are restricted to the 10k-HVG experiment")
    meta, _, _ = core.load_data()
    assert len(meta) == 361792 and meta.cell_id.is_unique
    enc = core.fit_label_encoders(meta)
    fine_y = enc["fine"].transform(meta.cell_subtype.astype(str))
    coarse_y = enc["coarse"].transform(meta.coarse_subtype.astype(str))
    ids = meta.sample_id.to_numpy(str)
    split = partitions(meta, args.fold)
    destination = args.output / "runs" / f"hvg{args.width}" / args.condition / f"f{args.fold}"
    destination.mkdir(parents=True, exist_ok=True)
    selection_path = destination / "selection.json"
    device = torch.device(args.device)

    if args.command == "select":
        if selection_path.exists():
            print("EXISTS", selection_path); return
        train_data, validation_data, apt_scaler, control_audit = load_inputs(
            args, meta, coarse_y, split, "inner", 17)
        rna, extra, train_ix = train_data
        vrna, vextra, validation_ix = validation_data
        trials = []
        started = time.time()
        for config in GRID:
            model, result = train(
                rna, extra, fine_y[train_ix], coarse_y[train_ix], ids[train_ix],
                config, 17, 50, 8, device,
                (vrna, vextra, fine_y[validation_ix], ids[validation_ix]))
            trials.append({"config": config, **result}); del model
        selected = max(trials, key=lambda item: item["score"])
        atomic_json(selection_path, {
            "protocol_sha256": protocol_sha256(), "fold": args.fold,
            "width": args.width, "condition": args.condition,
            "selection_seed": 17, "trials": trials, "selected": selected,
            "train_patients": sorted(set(ids[train_ix])),
            "validation_patients": sorted(set(ids[validation_ix])),
            "train_cells": len(train_ix), "validation_cells": len(validation_ix),
            "control_audit": control_audit, "seconds": time.time() - started,
        })
        print("COMPLETE", selection_path, flush=True)
        return

    if args.seed not in SEEDS:
        raise ValueError(f"Final seed must be one of {SEEDS}")
    if not selection_path.exists():
        raise RuntimeError(f"Missing frozen validation selection: {selection_path}")
    selection = json.loads(selection_path.read_text())
    if selection["protocol_sha256"] != protocol_sha256():
        raise RuntimeError("Selection was made under a different protocol")
    result_path = destination / f"s{args.seed}.npz"
    if result_path.exists():
        print("EXISTS", result_path); return
    started = time.time()
    train_data, test_data, apt_scaler, control_audit = load_inputs(
        args, meta, coarse_y, split, "outer", args.seed)
    rna, extra, train_ix = train_data
    trna, textra, test_ix = test_data
    chosen = selection["selected"]
    model, _ = train(rna, extra, fine_y[train_ix], coarse_y[train_ix], ids[train_ix],
                     chosen["config"], args.seed, chosen["epochs"], 8, device)
    fine_prob, coarse_prob = predict(model, trna, textra, device)
    assert fine_prob.shape == (len(test_ix), 27) and coarse_prob.shape == (len(test_ix), 5)
    assert np.allclose(fine_prob.sum(1), 1, atol=1e-5)
    assert np.allclose(coarse_prob.sum(1), 1, atol=1e-5)
    parameter_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    genes = (args.output / "prepared" / f"hvg{args.width}" /
             f"f{args.fold}_outer_genes.txt").read_text().splitlines()
    checkpoint = {
        "state": model.state_dict(), "input_dim": int(rna.shape[1] + (0 if extra is None else 293)),
        "parameter_count": parameter_count, "genes": genes, "selection": chosen,
        "fold": args.fold, "seed": args.seed, "condition": args.condition,
        "protocol_sha256": protocol_sha256(),
    }
    torch.save(checkpoint, result_path.with_suffix(".pt"))
    temporary = result_path.with_name(result_path.stem + ".tmp.npz")
    np.savez_compressed(temporary, fine_prob=fine_prob, coarse_prob=coarse_prob,
                        fine_truth=fine_y[test_ix], coarse_truth=coarse_y[test_ix],
                        cell_ids=meta.cell_id.iloc[test_ix].to_numpy(str), patient_ids=ids[test_ix],
                        fine_classes=enc["fine"].classes_.astype(str),
                        coarse_classes=enc["coarse"].classes_.astype(str))
    os.replace(temporary, result_path)
    if apt_scaler is not None:
        np.savez_compressed(destination / f"s{args.seed}_apt_scaler.npz", **apt_scaler)
    atomic_json(result_path.with_suffix(".json"), {
        "protocol_sha256": protocol_sha256(), "source_git_hash": args.source_git_hash,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fold": args.fold, "seed": args.seed, "condition": args.condition,
        "width": args.width, "selected": chosen, "parameter_count": parameter_count,
        "architecture": [int(checkpoint["input_dim"]), 256, 128, 27, 5],
        "optimizer": "AdamW", "sample_weighting": "none",
        "train_patients": sorted(set(ids[train_ix])), "test_patients": sorted(set(ids[test_ix])),
        "train_cells": len(train_ix), "test_cells": len(test_ix),
        "selected_gene_file": str(args.output / "prepared" / f"hvg{args.width}" / f"f{args.fold}_outer_genes.txt"),
        "selected_gene_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "control_audit": control_audit, "runtime_seconds": time.time() - started,
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
    })
    print("COMPLETE", result_path, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "refit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=WIDTHS, required=True)
    parser.add_argument("--condition", choices=PRIMARY + CONTROLS, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--apt-cache", type=Path)
    parser.add_argument("--source-git-hash", default="unknown")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == "__main__":
    main()
