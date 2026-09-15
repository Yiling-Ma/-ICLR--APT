#!/usr/bin/env python3
"""Leakage-safe, resumable fold runner for HBM-MLP.

Run from the apt_agent repository root.  This script imports the audited
unified_v2 data, frozen-fold, metric, and oracle functions instead of
reimplementing the benchmark protocol.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import patient_cell_scaling as benchmark
import run_full_budget_mlp as unified
import validate_paired_information as raw_input
from hbm_core import (
    FINAL_SEEDS,
    HBMStudent,
    N_FEATURES,
    N_LINEAGES,
    N_SUBTYPES,
    OracleTeacher,
    aggregate_parent_log_probs,
    assert_patient_partition,
    atomic_json,
    diagnostic_counts,
    hierarchy_consistency_loss,
    mask_logits_to_true_lineage,
    oracle_predictions,
    parent_mass_loss,
    set_seed,
    sha256_strings,
    within_lineage_kd_loss,
)


EPOCHS = 50
DECAYS = (1e-5, 1e-3)
BATCH_SIZE = 1024
LAMBDA_COARSE = 0.5
CONDITIONS = ("plain", "parent", "parent_cons", "kd", "parent_kd", "full")
DEPENDENCY = {
    "parent": ("plain",),
    "parent_cons": ("plain", "parent"),
    "kd": ("plain", "teacher"),
    "parent_kd": ("plain", "parent", "kd", "teacher"),
    "full": ("plain", "parent", "parent_cons", "kd", "teacher"),
}


def source_hash() -> str:
    digest = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ANALYSIS.parent, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", str(HERE.relative_to(ANALYSIS.parent))],
        cwd=ANALYSIS.parent, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=False,
    ).stdout
    return f"{digest or 'unavailable'}{'-dirty' if status else ''}"


def load_protocol(log1p_cache: Path | None = None) -> dict[str, Any]:
    meta, _, features = benchmark.load_data()
    # Exact unified_v2 input path: barcode-aligned raw sparse counts, log1p,
    # with no use of the historical pre-standardized apt_expression table.
    if log1p_cache is not None and log1p_cache.exists():
        manifest = json.loads(log1p_cache.with_suffix(".json").read_text())
        if manifest["cell_id_sha256"] != sha256_strings(meta.cell_id.astype(str)):
            raise AssertionError("log1p cache cell order does not match benchmark metadata")
        if manifest.get("shape") != [361792, N_FEATURES] or manifest.get("dtype") != "float32":
            raise AssertionError("log1p cache manifest has an unexpected shape or dtype")
        if manifest.get("transform") != "numpy.log1p on barcode-aligned raw nonnegative counts":
            raise AssertionError("log1p cache manifest has an unexpected transform")
        x = np.load(log1p_cache, mmap_mode="r")
    else:
        x = raw_input.raw_apt(meta)
    encoders = benchmark.fit_label_encoders(meta)
    folds = benchmark.load_folds()
    ids = meta.sample_id.astype(str).to_numpy()
    pairs = meta[["cell_subtype", "coarse_subtype"]].drop_duplicates()
    if not pairs.cell_subtype.is_unique:
        raise AssertionError("Each subtype must have exactly one lineage")
    mapping = pairs.set_index("cell_subtype").coarse_subtype
    parent = encoders["coarse"].transform(mapping.loc[encoders["fine"].classes_])
    if len(meta) != 361792 or x.shape != (361792, N_FEATURES):
        raise AssertionError(f"Unexpected benchmark shape: {len(meta)}, {x.shape}")
    if not meta.cell_id.is_unique or not np.isfinite(x).all():
        raise AssertionError("Invalid cell IDs or non-finite APT matrix")
    if len(encoders["coarse"].classes_) != N_LINEAGES or len(encoders["fine"].classes_) != N_SUBTYPES:
        raise AssertionError("Ontology is not the frozen 5-lineage/27-subtype taxonomy")
    if len(set(ids)) != 40 or sorted(set(ids)) != sorted(sum(folds.values(), [])):
        raise AssertionError("Data patients disagree with frozen folds")
    return dict(meta=meta, x=x, features=features, encoders=encoders, folds=folds,
                ids=ids, parent=parent.astype(np.int64))


def indices_for_fold(protocol: dict[str, Any], fold: int) -> dict[str, np.ndarray]:
    folds, ids = protocol["folds"], protocol["ids"]
    test_patients = folds[fold]
    val_patients = folds[(fold + 1) % 5]
    train_patients = sum([folds[index] for index in range(5) if index not in (fold, (fold + 1) % 5)], [])
    dev_patients = train_patients + val_patients
    assert_patient_partition(train_patients, val_patients, test_patients)
    result = {
        "train": np.flatnonzero(np.isin(ids, train_patients)),
        "val": np.flatnonzero(np.isin(ids, val_patients)),
        "test": np.flatnonzero(np.isin(ids, test_patients)),
        "dev": np.flatnonzero(np.isin(ids, dev_patients)),
    }
    for name, index in result.items():
        if len(index) == 0:
            raise AssertionError(f"Empty {name} partition")
    if set(result["train"]) & set(result["val"]) or set(result["dev"]) & set(result["test"]):
        raise AssertionError("Cell leakage across partitions")
    return result


def labels(protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    meta, encoders = protocol["meta"], protocol["encoders"]
    fine = encoders["fine"].transform(meta.cell_subtype.astype(str)).astype(np.int64)
    coarse = encoders["coarse"].transform(meta.coarse_subtype.astype(str)).astype(np.int64)
    np.testing.assert_array_equal(protocol["parent"][fine], coarse)
    return fine, coarse


def predict_student(model: HBMStudent, x: np.ndarray, device: torch.device):
    model.eval()
    fine, coarse = [], []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            output = model(torch.as_tensor(x[start:start + 128], dtype=torch.float32, device=device))
            fine.append(F.softmax(output["fine_logits"], dim=1).cpu().numpy())
            coarse.append(F.softmax(output["coarse_logits"], dim=1).cpu().numpy())
    return np.concatenate(fine), np.concatenate(coarse)


def predict_teacher_logits(
    model: OracleTeacher, x: np.ndarray, true_coarse: np.ndarray, device: torch.device
) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            batch_x = torch.as_tensor(x[start:start + 128], dtype=torch.float32, device=device)
            batch_g = torch.as_tensor(true_coarse[start:start + 128], dtype=torch.long, device=device)
            output.append(model(batch_x, batch_g)["fine_logits"].cpu().numpy())
    return np.concatenate(output)


def validation_score(
    truth: np.ndarray, pred: np.ndarray, patient_ids: np.ndarray, n_classes: int
) -> float:
    _, confusion = unified.matrices(truth, pred, patient_ids, n_classes)
    return float(unified.score(confusion))


def fit_teacher(
    x: np.ndarray, y_fine: np.ndarray, y_coarse: np.ndarray, parent: np.ndarray,
    seed: int, lr: float, decay: float, epochs: int, device: torch.device, validation=None,
) -> tuple[OracleTeacher, float, int, list[float]]:
    set_seed(seed)
    model = OracleTeacher(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    parent_t = torch.as_tensor(parent, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    best, best_epoch, history, best_state = -np.inf, 1, [], None
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(y_fine))
        for start in range(0, len(y_fine), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            bx = torch.as_tensor(x[batch], dtype=torch.float32, device=device)
            by = torch.as_tensor(y_fine[batch], dtype=torch.long, device=device)
            bg = torch.as_tensor(y_coarse[batch], dtype=torch.long, device=device)
            logits = model(bx, bg)["fine_logits"]
            loss = F.cross_entropy(mask_logits_to_true_lineage(logits, bg, parent_t), by)
            loss.backward()
            optimizer.step()
        if validation is not None:
            vx, vy, vg, patient_ids = validation
            logits = predict_teacher_logits(model, vx, vg, device)
            pred = np.where(parent[None, :] == vg[:, None], logits, -np.inf).argmax(1)
            score = validation_score(vy, pred, patient_ids, N_SUBTYPES)
            history.append(score)
            if score > best:
                best, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch - best_epoch >= 8:
                break
        if epoch % 10 == 0:
            print("EPOCH", epoch, "teacher", "val" if validation else "refit", flush=True)
    if validation is not None:
        if best_state is None:
            raise AssertionError("Teacher validation never produced a best checkpoint")
        model.load_state_dict(best_state)
    return model, float(best), int(best_epoch), history


def fit_student(
    x: np.ndarray, y_fine: np.ndarray, y_coarse: np.ndarray, parent: np.ndarray,
    seed: int, lr: float, decay: float, epochs: int, device: torch.device, weights: dict[str, float],
    temperature: float = 1.0, teacher_logits: np.ndarray | None = None, validation=None,
) -> tuple[HBMStudent, float, int, list[dict[str, float]]]:
    set_seed(seed)
    model = HBMStudent(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    parent_t = torch.as_tensor(parent, dtype=torch.long, device=device)
    if weights.get("kd", 0) > 0 and teacher_logits is None:
        raise ValueError("KD requested without fold-specific teacher logits")
    rng = np.random.default_rng(seed)
    best, best_epoch, history, best_state = -np.inf, 1, [], None
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("fine", "coarse", "parent", "cons", "kd", "total")}
        seen = 0
        order = rng.permutation(len(y_fine))
        for start in range(0, len(y_fine), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            bx = torch.as_tensor(x[batch], dtype=torch.float32, device=device)
            by = torch.as_tensor(y_fine[batch], dtype=torch.long, device=device)
            bg = torch.as_tensor(y_coarse[batch], dtype=torch.long, device=device)
            output = model(bx)
            losses = {
                "fine": F.cross_entropy(output["fine_logits"], by),
                "coarse": F.cross_entropy(output["coarse_logits"], bg),
                "parent": parent_mass_loss(output["fine_logits"], bg, parent_t),
                "cons": hierarchy_consistency_loss(output["coarse_logits"], output["fine_logits"], parent_t),
                "kd": output["fine_logits"].new_zeros(()),
            }
            if teacher_logits is not None and weights.get("kd", 0) > 0:
                batch_teacher = torch.as_tensor(teacher_logits[batch], dtype=torch.float32, device=device)
                losses["kd"] = within_lineage_kd_loss(
                    output["fine_logits"], batch_teacher, bg, parent_t, temperature
                )
            loss = (
                losses["fine"] + weights.get("coarse", LAMBDA_COARSE) * losses["coarse"]
                + weights.get("parent", 0.0) * losses["parent"]
                + weights.get("cons", 0.0) * losses["cons"]
                + weights.get("kd", 0.0) * losses["kd"]
            )
            loss.backward()
            optimizer.step()
            size = len(batch)
            seen += size
            for name in losses:
                totals[name] += float(losses[name].detach()) * size
            totals["total"] += float(loss.detach()) * size
        record = {name: value / seen for name, value in totals.items()}
        if validation is not None:
            vx, vy, _, patient_ids = validation
            fine_prob, _ = predict_student(model, vx, device)
            score = validation_score(vy, fine_prob.argmax(1), patient_ids, N_SUBTYPES)
            record["val_fine_sb_macro_f1"] = score
            if score > best:
                best, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)
        if epoch % 10 == 0:
            print("EPOCH", epoch, "student", "score", record.get("val_fine_sb_macro_f1"), flush=True)
        if validation is not None and epoch - best_epoch >= 8:
            break
    if validation is not None:
        if best_state is None:
            raise AssertionError("Student validation never produced a best checkpoint")
        model.load_state_dict(best_state)
    return model, float(best), int(best_epoch), history


def checkpoint(path: Path, model, scaler: StandardScaler, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save({"model": model.state_dict(), "mean": scaler.mean_, "scale": scaler.scale_, **payload}, temporary)
    os.replace(temporary, path)


def load_selection(output: Path, condition: str, seed: int, fold: int) -> dict[str, Any]:
    # Unified_v2 freezes selection once with seed 17 and reuses it for all
    # three randomized outer refits.
    path = output / condition / "seed17" / f"fold{fold}" / "selection.json"
    if not path.exists():
        raise FileNotFoundError(f"Run selection dependency first: {path}")
    return json.loads(path.read_text())


def selected_student_trials(condition: str, dependencies: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base = dependencies["plain"]["selected"]
    lr, decay = base["lr"], base["decay"]
    if condition == "parent":
        return [{"lr": lr, "decay": decay, "weights": {"coarse": LAMBDA_COARSE, "parent": value}}
                for value in (0.25, 0.5, 1.0)]
    if condition == "parent_cons":
        parent = dependencies["parent"]["selected"]["weights"]["parent"]
        return [{"lr": lr, "decay": decay, "weights": {"coarse": LAMBDA_COARSE, "parent": parent, "cons": value}}
                for value in (0.1, 0.25)]
    if condition == "kd":
        return [{"lr": lr, "decay": decay, "weights": {"coarse": LAMBDA_COARSE, "kd": 0.5}, "temperature": value}
                for value in (1.0, 2.0, 4.0)]
    if condition == "parent_kd":
        parent = dependencies["parent"]["selected"]["weights"]["parent"]
        kd = dependencies["kd"]["selected"]
        return [{"lr": lr, "decay": decay, "weights": {"coarse": LAMBDA_COARSE, "parent": parent,
                 "kd": kd["weights"]["kd"]}, "temperature": kd["temperature"]}]
    if condition == "full":
        parent = dependencies["parent"]["selected"]["weights"]["parent"]
        cons = dependencies["parent_cons"]["selected"]["weights"]["cons"]
        kd = dependencies["kd"]["selected"]
        return [{"lr": lr, "decay": decay, "weights": {"coarse": LAMBDA_COARSE, "parent": parent,
                 "cons": cons, "kd": kd["weights"]["kd"]}, "temperature": kd["temperature"]}]
    raise ValueError(condition)


def prepare_fold(protocol: dict[str, Any], fold: int, fit_partition: str):
    index = indices_for_fold(protocol, fold)
    fine, coarse = labels(protocol)
    fit_index = index[fit_partition]
    scaler = StandardScaler().fit(protocol["x"][fit_index])
    transformed = {name: scaler.transform(protocol["x"][ix]).astype(np.float32)
                   for name, ix in index.items() if name in (fit_partition, "val", "test")}
    return index, fine, coarse, scaler, transformed


def run_plain_selection(args, protocol, destination: Path, device):
    index, fine, coarse, scaler, x = prepare_fold(protocol, args.fold, "train")
    trials, models = [], []
    for lr in (3e-4, 1e-3):
        for decay in DECAYS:
            model, score, epoch, history = fit_student(
                x["train"], fine[index["train"]], coarse[index["train"]], protocol["parent"],
                17, lr, decay, EPOCHS, device, {"coarse": LAMBDA_COARSE},
                validation=(x["val"], fine[index["val"]], coarse[index["val"]],
                            protocol["ids"][index["val"]]),
            )
            trials.append({"lr": lr, "decay": decay, "weights": {"coarse": LAMBDA_COARSE},
                           "score": score, "epoch": epoch, "history": history})
            models.append(model)
    best_index = max(range(len(trials)), key=lambda value: trials[value]["score"])
    selected = copy.deepcopy(trials[best_index])
    checkpoint(destination / "inner_student.pt", models[best_index], scaler,
               {"seed": 17, "fold": args.fold, "condition": "plain", "selected": selected})
    payload = {"trials": trials, "selected": selected, "seed": 17, "fold": args.fold,
               "source_hash": source_hash(), "selection_uses": "inner validation patients only"}
    atomic_json(destination / "selection.json", payload)


def run_teacher_selection(args, protocol, destination: Path, device):
    index, fine, coarse, scaler, x = prepare_fold(protocol, args.fold, "train")
    trials, models = [], []
    for lr in (3e-4, 1e-3):
        for decay in DECAYS:
            model, score, epoch, history = fit_teacher(
                x["train"], fine[index["train"]], coarse[index["train"]], protocol["parent"],
                17, lr, decay, EPOCHS, device,
                (x["val"], fine[index["val"]], coarse[index["val"]], protocol["ids"][index["val"]]),
            )
            trials.append({"lr": lr, "decay": decay, "score": score, "epoch": epoch, "history": history})
            models.append(model)
    best_index = max(range(len(trials)), key=lambda value: trials[value]["score"])
    selected = copy.deepcopy(trials[best_index])
    checkpoint(destination / "inner_teacher.pt", models[best_index], scaler,
               {"seed": 17, "fold": args.fold, "selected": selected,
                "train_patients": sorted(set(protocol["ids"][index["train"]])),
                "val_patients": sorted(set(protocol["ids"][index["val"]]))})
    atomic_json(destination / "selection.json", {"trials": trials, "selected": selected,
                "seed": 17, "fold": args.fold, "source_hash": source_hash(),
                "teacher_scope": "inner-train only; true lineage used only inside teacher"})


def load_teacher_for_partition(output: Path, seed: int, fold: int, kind: str, device):
    path = output / "teacher" / f"seed{seed}" / f"fold{fold}" / f"{kind}_teacher.pt"
    saved = torch.load(path, map_location=device, weights_only=False)
    model = OracleTeacher().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, saved


def run_student_selection(args, protocol, destination: Path, device):
    dependencies = {name: load_selection(args.output, name, args.seed, args.fold)
                    for name in DEPENDENCY[args.condition]}
    index, fine, coarse, scaler, x = prepare_fold(protocol, args.fold, "train")
    teacher_logits = None
    if "teacher" in DEPENDENCY[args.condition]:
        teacher, saved = load_teacher_for_partition(args.output, args.seed, args.fold, "inner", device)
        np.testing.assert_allclose(saved["mean"], scaler.mean_)
        np.testing.assert_allclose(saved["scale"], scaler.scale_)
        teacher_logits = predict_teacher_logits(
            teacher, x["train"], coarse[index["train"]], device
        )
        del teacher
    specifications = selected_student_trials(args.condition, dependencies)
    trials, models = [], []
    for specification in specifications:
        model, score, epoch, history = fit_student(
            x["train"], fine[index["train"]], coarse[index["train"]], protocol["parent"],
            17, specification["lr"], specification["decay"], EPOCHS, device, specification["weights"],
            specification.get("temperature", 1.0), teacher_logits,
            (x["val"], fine[index["val"]], coarse[index["val"]], protocol["ids"][index["val"]]),
        )
        trials.append({**specification, "score": score, "epoch": epoch, "history": history})
        models.append(model)
    # KD temperature stage, then lambda stage without a Cartesian search.
    if args.condition == "kd":
        first_best = max(trials, key=lambda value: value["score"])
        for kd_weight in (0.25, 1.0):
            specification = {"lr": first_best["lr"], "decay": first_best["decay"],
                             "temperature": first_best["temperature"],
                             "weights": {"coarse": LAMBDA_COARSE, "kd": kd_weight}}
            model, score, epoch, history = fit_student(
                x["train"], fine[index["train"]], coarse[index["train"]], protocol["parent"],
                17, specification["lr"], specification["decay"], EPOCHS, device, specification["weights"],
                specification["temperature"], teacher_logits,
                (x["val"], fine[index["val"]], coarse[index["val"]], protocol["ids"][index["val"]]),
            )
            trials.append({**specification, "score": score, "epoch": epoch, "history": history})
            models.append(model)
    best_index = max(range(len(trials)), key=lambda value: trials[value]["score"])
    selected = copy.deepcopy(trials[best_index])
    checkpoint(destination / "inner_student.pt", models[best_index], scaler,
               {"seed": 17, "fold": args.fold, "condition": args.condition, "selected": selected})
    atomic_json(destination / "selection.json", {"trials": trials, "selected": selected,
                "seed": 17, "fold": args.fold, "source_hash": source_hash(),
                "selection_uses": "inner validation fine subject-balanced Macro-F1 only"})


def refit_teacher(args, protocol, destination: Path, device):
    selection = load_selection(args.output, "teacher", args.seed, args.fold)["selected"]
    index, fine, coarse, scaler, x = prepare_fold(protocol, args.fold, "dev")
    model, _, _, history = fit_teacher(
        x["dev"], fine[index["dev"]], coarse[index["dev"]], protocol["parent"], args.seed,
        selection["lr"], selection["decay"], selection["epoch"], device,
    )
    checkpoint(destination / "refit_teacher.pt", model, scaler,
               {"seed": args.seed, "fold": args.fold, "selected": selection,
                "dev_patients": sorted(set(protocol["ids"][index["dev"]])),
                "test_patients": sorted(set(protocol["ids"][index["test"]])), "history": history})


def save_predictions(args, protocol, destination, fine_prob, coarse_prob, model_payload):
    index = indices_for_fold(protocol, args.fold)
    fine, coarse = labels(protocol)
    test = index["test"]
    fine_pred, coarse_pred = fine_prob.argmax(1), coarse_prob.argmax(1)
    oracle_pred = oracle_predictions(fine_prob, fine[test], protocol["parent"])
    patients, fine_cm = unified.matrices(fine[test], fine_pred, protocol["ids"][test], N_SUBTYPES)
    _, coarse_cm = unified.matrices(coarse[test], coarse_pred, protocol["ids"][test], N_LINEAGES)
    _, oracle_cm = unified.matrices(fine[test], oracle_pred, protocol["ids"][test], N_SUBTYPES)
    counts = []
    for patient in patients:
        keep = protocol["ids"][test] == patient
        counts.append(diagnostic_counts(fine[test][keep], fine_pred[keep], coarse[test][keep],
                                        coarse_pred[keep], protocol["parent"]))
    temporary = destination / "predictions.tmp.npz"
    np.savez_compressed(
        temporary, cell_ids=protocol["meta"].cell_id.iloc[test].astype(str).to_numpy(),
        sample_ids=np.asarray(protocol["ids"][test], dtype=str), truth_fine=fine[test], truth_coarse=coarse[test],
        fine_prob=fine_prob.astype(np.float32), coarse_prob=coarse_prob.astype(np.float32),
        pred_fine=fine_pred, pred_coarse=coarse_pred, oracle_pred_fine=oracle_pred,
        patients=np.asarray(patients, dtype=str), fine_cm=fine_cm, coarse_cm=coarse_cm, oracle_cm=oracle_cm,
        diagnostic_counts=np.asarray([[row[key] for key in sorted(row)] for row in counts]),
        diagnostic_keys=np.asarray(sorted(counts[0]), dtype=str), parent=protocol["parent"],
        fine_classes=protocol["encoders"]["fine"].classes_.astype(str),
        coarse_classes=protocol["encoders"]["coarse"].classes_.astype(str),
    )
    os.replace(temporary, destination / "predictions.npz")
    metrics = {
        "fine_sb_macro_f1": unified.score(fine_cm),
        "coarse_sb_macro_f1": unified.score(coarse_cm),
        "oracle_fine_sb_macro_f1": unified.score(oracle_cm),
        "hierarchy_gap": unified.score(oracle_cm) - unified.score(fine_cm),
    }
    atomic_json(destination / "metrics.json", {**metrics, **model_payload,
                "seed": args.seed, "fold": args.fold, "source_hash": source_hash(),
                "feature_count": len(protocol["features"]),
                "cell_id_sha256": sha256_strings(protocol["meta"].cell_id.astype(str)),
                "test_labels_used_for": "evaluation only", "n_test_cells": int(len(test))})


def refit_plain(args, protocol, destination: Path, device):
    selected = load_selection(args.output, "plain", args.seed, args.fold)["selected"]
    index, fine, coarse, scaler, x = prepare_fold(protocol, args.fold, "dev")
    model, _, _, history = fit_student(
        x["dev"], fine[index["dev"]], coarse[index["dev"]], protocol["parent"], args.seed,
        selected["lr"], selected["decay"], selected["epoch"], device,
        {"coarse": LAMBDA_COARSE},
    )
    fine_prob, coarse_prob = predict_student(model, x["test"], device)
    checkpoint(destination / "refit_student.pt", model, scaler,
               {"seed": args.seed, "fold": args.fold, "condition": "plain",
                "selected": selected, "history": history})
    save_predictions(args, protocol, destination, fine_prob, coarse_prob,
                     {"condition": "plain", "selection": selected})


def refit_student(args, protocol, destination: Path, device):
    selected = load_selection(args.output, args.condition, args.seed, args.fold)["selected"]
    index, fine, coarse, scaler, x = prepare_fold(protocol, args.fold, "dev")
    teacher_logits = None
    if "teacher" in DEPENDENCY[args.condition]:
        teacher, saved = load_teacher_for_partition(args.output, args.seed, args.fold, "refit", device)
        if set(saved["dev_patients"]) & set(saved["test_patients"]):
            raise AssertionError("Teacher checkpoint has outer-test leakage")
        np.testing.assert_allclose(saved["mean"], scaler.mean_)
        np.testing.assert_allclose(saved["scale"], scaler.scale_)
        teacher_logits = predict_teacher_logits(teacher, x["dev"], coarse[index["dev"]], device)
        del teacher
    model, _, _, history = fit_student(
        x["dev"], fine[index["dev"]], coarse[index["dev"]], protocol["parent"], args.seed,
        selected["lr"], selected["decay"], selected["epoch"], device, selected["weights"],
        selected.get("temperature", 1.0), teacher_logits,
    )
    fine_prob, coarse_prob = predict_student(model, x["test"], device)
    checkpoint(destination / "refit_student.pt", model, scaler,
               {"seed": args.seed, "fold": args.fold, "condition": args.condition,
                "selected": selected, "history": history})
    save_predictions(args, protocol, destination, fine_prob, coarse_prob,
                     {"condition": args.condition, "selection": selected})


def audit(args, protocol):
    payload = {
        "status": "PASS", "cells": len(protocol["meta"]), "features": len(protocol["features"]),
        "patients": len(set(protocol["ids"])), "outer_folds": protocol["folds"],
        "fine_classes": protocol["encoders"]["fine"].classes_.tolist(),
        "coarse_classes": protocol["encoders"]["coarse"].classes_.tolist(),
        "subtype_to_lineage": protocol["parent"].tolist(),
        "cell_id_sha256": sha256_strings(protocol["meta"].cell_id.astype(str)),
        "source_hash": source_hash(),
        "preprocessing": "barcode-aligned raw sparse counts; log1p; StandardScaler fit only on inner-train or outer-development, exactly unified_v2",
        "model_selection": "seed 17 only; inner fold (outer+1)%5; fine subject-balanced Macro-F1; patience 8; earliest strict maximum",
        "final_seeds": list(FINAL_SEEDS), "test_role": "evaluation only after frozen validation selection",
    }
    atomic_json(args.output / "protocol_audit.json", payload)
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "select", "refit"))
    parser.add_argument("--condition", choices=CONDITIONS + ("teacher",))
    parser.add_argument("--seed", type=int, choices=FINAL_SEEDS)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log1p-cache", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command != "audit" and (args.condition is None or args.seed is None or args.fold is None):
        parser.error("select/refit require --condition, --seed, and --fold")
    if args.command == "select" and args.seed != 17:
        parser.error("unified_v2 selection is frozen to seed 17; seeds 29/43 are refit-only")
    protocol = load_protocol(args.log1p_cache)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "audit":
        audit(args, protocol)
        return
    destination = args.output / args.condition / f"seed{args.seed}" / f"fold{args.fold}"
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ("selection.json" if args.command == "select" else
                            ("refit_teacher.pt" if args.condition == "teacher" else "predictions.npz"))
    if marker.exists() and not args.force:
        print("SKIP existing", marker)
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.time()
    if args.command == "select":
        if args.condition == "plain":
            run_plain_selection(args, protocol, destination, device)
        elif args.condition == "teacher":
            run_teacher_selection(args, protocol, destination, device)
        else:
            run_student_selection(args, protocol, destination, device)
    elif args.condition == "plain":
        refit_plain(args, protocol, destination, device)
    elif args.condition == "teacher":
        refit_teacher(args, protocol, destination, device)
    else:
        refit_student(args, protocol, destination, device)
    print("COMPLETE", args.command, args.condition, args.seed, args.fold,
          "seconds", round(time.time() - started, 1), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
