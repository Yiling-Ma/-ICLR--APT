#!/usr/bin/env python3
"""CUDA-only, resumable runner for RNA-privileged APT distillation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    CONDITIONS, RNATeacher, apt_partition, assert_frozen, atomic_json,
    cosine_alignment, dense_rna_batch, fold_indices, grouped_target_permutation,
    load_benchmark, rna_partition, routing_kd_loss, validation_score,
)
from hbm_core import (
    FINAL_SEEDS, HBMStudent, diagnostic_counts, oracle_predictions, set_seed,
    sha256_strings, within_lineage_kd_loss,
)
import run_full_budget_mlp as unified

EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 1024
TEACHER_GRID = ((3e-4, 1e-5), (3e-4, 1e-3), (1e-3, 1e-5), (1e-3, 1e-3))


def source_hash() -> str:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                            capture_output=True, check=False).stdout.strip() or "unavailable"
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(Path(__file__).parent)],
                           cwd=root, text=True, capture_output=True, check=False).stdout
    return commit + ("-dirty" if dirty else "")


def require_cuda(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RPH neural fits require an available CUDA GPU; CPU fallback is forbidden")
    torch.empty(1, device=device)
    return device


def predict_teacher(model, matrix, device):
    model.eval(); fine, coarse, representation = [], [], []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], BATCH_SIZE):
            rows = np.arange(start, min(start + BATCH_SIZE, matrix.shape[0]))
            x = torch.as_tensor(dense_rna_batch(matrix, rows), device=device)
            output = model(x)
            fine.append(output["fine_logits"].cpu().numpy().astype(np.float32))
            coarse.append(output["coarse_logits"].cpu().numpy().astype(np.float32))
            representation.append(output["representation"].cpu().numpy().astype(np.float32))
    return np.concatenate(fine), np.concatenate(coarse), np.concatenate(representation)


def predict_student(model, matrix, device):
    model.eval(); fine, coarse = [], []
    with torch.no_grad():
        for start in range(0, len(matrix), BATCH_SIZE):
            x = torch.as_tensor(matrix[start:start + BATCH_SIZE], device=device)
            output = model(x)
            fine.append(F.softmax(output["fine_logits"], 1).cpu().numpy().astype(np.float32))
            coarse.append(F.softmax(output["coarse_logits"], 1).cpu().numpy().astype(np.float32))
    return np.concatenate(fine), np.concatenate(coarse)


def fit_teacher(matrix, fine, coarse, patients, seed, lr, decay, epochs, device,
                validation=None):
    set_seed(seed); model = RNATeacher(matrix.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    rng = np.random.default_rng(seed); history = []; best = -np.inf; best_epoch = 1; state = None
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(fine)); total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            x = torch.as_tensor(dense_rna_batch(matrix, batch), device=device)
            yf = torch.as_tensor(fine[batch], dtype=torch.long, device=device)
            yc = torch.as_tensor(coarse[batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True); output = model(x)
            loss = F.cross_entropy(output["fine_logits"], yf) + .5 * F.cross_entropy(output["coarse_logits"], yc)
            loss.backward(); optimizer.step(); total += float(loss.detach()) * len(batch)
        record = {"epoch": epoch, "train_loss": total / len(fine)}
        if validation is not None:
            vm, vf, _, vp = validation
            logits, _, _ = predict_teacher(model, vm, device)
            score = validation_score(vf, logits.argmax(1), vp)
            record["val_fine_sb_macro_f1"] = score
            if score > best:
                best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        history.append(record)
        print("TEACHER", epoch, record, flush=True)
        if validation is not None and epoch - best_epoch >= PATIENCE:
            break
    if validation is not None:
        if state is None: raise AssertionError("No valid teacher checkpoint")
        model.load_state_dict(state)
    return model, float(best), int(best_epoch), history


def load_targets(path: Path, expected_indices: np.ndarray):
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["indices"], expected_indices)
        return {key: saved[key] for key in ("fine_logits", "coarse_logits", "representation")}


def student_loss(output, yf, yc, parent, targets, batch, spec):
    losses = {
        "fine": F.cross_entropy(output["fine_logits"], yf),
        "coarse": F.cross_entropy(output["coarse_logits"], yc),
        "route": output["fine_logits"].new_zeros(()),
        "within": output["fine_logits"].new_zeros(()),
        "align": output["fine_logits"].new_zeros(()),
    }
    if targets is not None:
        tf = torch.as_tensor(targets["fine_logits"][batch], device=output["fine_logits"].device)
        tc = torch.as_tensor(targets["coarse_logits"][batch], device=output["fine_logits"].device)
        tr = torch.as_tensor(targets["representation"][batch], device=output["fine_logits"].device)
        if spec.get("lambda_route", 0):
            losses["route"] = routing_kd_loss(output["coarse_logits"], tc, spec["temperature_route"])
        if spec.get("lambda_within", 0):
            losses["within"] = within_lineage_kd_loss(
                output["fine_logits"], tf, yc, parent, spec["temperature_within"])
        if spec.get("lambda_align", 0):
            losses["align"] = cosine_alignment(output["representation"], tr)
    total = (losses["fine"] + spec.get("lambda_coarse", .5) * losses["coarse"]
             + spec.get("lambda_route", 0) * losses["route"]
             + spec.get("lambda_within", 0) * losses["within"]
             + spec.get("lambda_align", 0) * losses["align"])
    return total, losses


def fit_student(matrix, fine, coarse, patients, parent_np, targets, spec, seed, epochs,
                device, validation=None):
    set_seed(seed); model = HBMStudent(matrix.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["decay"])
    parent = torch.as_tensor(parent_np, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed); history = []; best = -np.inf; best_epoch = 1; state = None
    for epoch in range(1, epochs + 1):
        model.train(); order = rng.permutation(len(fine)); totals = {}; seen = 0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            x = torch.as_tensor(matrix[batch], device=device)
            yf = torch.as_tensor(fine[batch], dtype=torch.long, device=device)
            yc = torch.as_tensor(coarse[batch], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True); output = model(x)
            loss, pieces = student_loss(output, yf, yc, parent, targets, batch, spec)
            loss.backward(); optimizer.step(); seen += len(batch)
            for key, value in {**pieces, "total": loss}.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(batch)
        record = {"epoch": epoch, **{key: value / seen for key, value in totals.items()}}
        if validation is not None:
            vm, vf, _, vp = validation
            probability, _ = predict_student(model, vm, device)
            score = validation_score(vf, probability.argmax(1), vp)
            record["val_fine_sb_macro_f1"] = score
            if score > best:
                best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
        history.append(record); print("STUDENT", epoch, record, flush=True)
        if validation is not None and epoch - best_epoch >= PATIENCE: break
    if validation is not None:
        if state is None: raise AssertionError("No valid student checkpoint")
        model.load_state_dict(state)
    return model, float(best), int(best_epoch), history


def plain_selection(output: Path, fold: int):
    path = output / "plain" / "seed17" / f"fold{fold}" / "selection.json"
    if not path.exists(): raise FileNotFoundError(path)
    return json.loads(path.read_text())["selected"]


def selection(output: Path, condition: str, fold: int):
    path = output / condition / "seed17" / f"fold{fold}" / "selection.json"
    if not path.exists(): raise FileNotFoundError(path)
    return json.loads(path.read_text())["selected"]


def base_spec(output, fold):
    base = plain_selection(output, fold)
    return {"lr": base["lr"], "decay": base["decay"], "lambda_coarse": .5}


def component_spec(output, condition, fold):
    spec = base_spec(output, fold)
    if condition in ("route", "route_align", "route_within", "full", "full_shuffled"):
        chosen = selection(output, "route", fold)
        spec.update(lambda_route=chosen["lambda_route"], temperature_route=chosen["temperature_route"])
    if condition in ("within", "route_within", "full", "full_shuffled"):
        chosen = selection(output, "within", fold)
        spec.update(lambda_within=chosen["lambda_within"], temperature_within=chosen["temperature_within"])
    if condition in ("align", "route_align", "full", "full_shuffled"):
        spec["lambda_align"] = selection(output, "align", fold)["lambda_align"]
    if condition == "strong_lineage": spec["lambda_coarse"] = 1.0
    return spec


def trial_specs(output, condition, fold):
    base = base_spec(output, fold)
    if condition == "route":
        return [dict(base, lambda_route=.25, temperature_route=t) for t in (1., 2., 4.)]
    if condition == "within":
        return [dict(base, lambda_within=.25, temperature_within=t) for t in (1., 2., 4.)]
    if condition == "align":
        return [dict(base, lambda_align=value) for value in (.01, .05, .1, .25)]
    return [component_spec(output, condition, fold)]


def save_checkpoint(path, model, payload):
    temporary = path.with_suffix(".tmp.pt")
    torch.save({"model": model.state_dict(), **payload}, temporary); os.replace(temporary, path)


def save_target_file(path, indices, outputs):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, indices=indices, fine_logits=outputs[0],
                        coarse_logits=outputs[1], representation=outputs[2])
    os.replace(temporary, path)


def select_teacher(args, data, index, destination, device):
    matrices, genes = rna_partition(args.rna_root, index, args.fold, "inner")
    trials = []; models = []
    for lr, decay in TEACHER_GRID:
        model, score, epoch, history = fit_teacher(
            matrices["train"], data["fine"][index["train"]], data["coarse"][index["train"]],
            data["ids"][index["train"]], 17, lr, decay, EPOCHS, device,
            (matrices["val"], data["fine"][index["val"]], data["coarse"][index["val"]], data["ids"][index["val"]]))
        trials.append({"lr": lr, "decay": decay, "score": score, "epoch": epoch, "history": history})
        models.append(model)
    winner = max(range(len(trials)), key=lambda value: trials[value]["score"]); selected = trials[winner]
    model = models[winner]
    save_checkpoint(destination / "inner_teacher.pt", model, {"selected": selected, "genes": genes})
    save_target_file(destination / "inner_train_targets.npz", index["train"], predict_teacher(model, matrices["train"], device))
    save_target_file(destination / "inner_val_targets.npz", index["val"], predict_teacher(model, matrices["val"], device))
    atomic_json(destination / "selection.json", {"selected": selected, "trials": trials, "fold": args.fold,
                "seed": 17, "source_hash": source_hash(), "selection_uses": "inner validation only"})


def refit_teacher(args, data, index, destination, device):
    chosen = selection(args.output, "teacher", args.fold)
    matrices, genes = rna_partition(args.rna_root, index, args.fold, "outer")
    model, _, _, history = fit_teacher(matrices["dev"], data["fine"][index["dev"]],
        data["coarse"][index["dev"]], data["ids"][index["dev"]], args.seed,
        chosen["lr"], chosen["decay"], chosen["epoch"], device)
    save_checkpoint(destination / "refit_teacher.pt", model, {"selected": chosen, "genes": genes,
        "history": history, "dev_patients": sorted(set(data["ids"][index["dev"]])),
        "test_patients": sorted(set(data["ids"][index["test"]]))})
    save_target_file(destination / "dev_targets.npz", index["dev"], predict_teacher(model, matrices["dev"], device))
    save_target_file(destination / "test_diagnostics.npz", index["test"], predict_teacher(model, matrices["test"], device))


def add_lambda_stage(args, data, index, x, targets, device, trials, models):
    first = max(trials, key=lambda value: value["score"])
    name = args.condition; temperature = first[f"temperature_{name}"]
    for value in (.1, .5, 1.0):
        spec = base_spec(args.output, args.fold); spec.update({f"lambda_{name}": value, f"temperature_{name}": temperature})
        model, score, epoch, history = fit_student(x["train"], data["fine"][index["train"]],
            data["coarse"][index["train"]], data["ids"][index["train"]], data["parent"], targets,
            spec, 17, EPOCHS, device, (x["val"], data["fine"][index["val"]], data["coarse"][index["val"]], data["ids"][index["val"]]))
        trials.append({**spec, "score": score, "epoch": epoch, "history": history}); models.append(model)


def select_student(args, data, index, destination, device):
    if args.condition == "full_shuffled":
        chosen = copy.deepcopy(selection(args.output, "full", args.fold))
        atomic_json(destination / "selection.json", {
            "selected": chosen, "trials": [], "fold": args.fold, "seed": 17,
            "source_hash": source_hash(),
            "selection_uses": "exact Full RPH configuration and epoch; no shuffled-control tuning",
        })
        return
    _, x = apt_partition(data, index, "train")
    targets = None
    if args.condition != "strong_lineage":
        targets = load_targets(args.output / "teacher/seed17" / f"fold{args.fold}" / "inner_train_targets.npz", index["train"])
    trials = []; models = []
    for spec in trial_specs(args.output, args.condition, args.fold):
        model, score, epoch, history = fit_student(x["train"], data["fine"][index["train"]],
            data["coarse"][index["train"]], data["ids"][index["train"]], data["parent"], targets,
            spec, 17, EPOCHS, device, (x["val"], data["fine"][index["val"]], data["coarse"][index["val"]], data["ids"][index["val"]]))
        trials.append({**spec, "score": score, "epoch": epoch, "history": history}); models.append(model)
    if args.condition in ("route", "within"):
        add_lambda_stage(args, data, index, x, targets, device, trials, models)
    winner = max(range(len(trials)), key=lambda value: trials[value]["score"]); selected = trials[winner]
    save_checkpoint(destination / "inner_student.pt", models[winner], {"selected": selected})
    atomic_json(destination / "selection.json", {"selected": selected, "trials": trials, "fold": args.fold,
                "seed": 17, "source_hash": source_hash(), "selection_uses": "inner validation Fine SB-Macro-F1 only"})


def save_predictions(args, data, index, destination, model, probability, selected, history):
    test = index["test"]; fine_prob, coarse_prob = probability
    pf, pc = fine_prob.argmax(1), coarse_prob.argmax(1)
    oracle = oracle_predictions(fine_prob, data["fine"][test], data["parent"])
    patients, fine_cm = unified.matrices(data["fine"][test], pf, data["ids"][test], 27)
    _, coarse_cm = unified.matrices(data["coarse"][test], pc, data["ids"][test], 5)
    _, oracle_cm = unified.matrices(data["fine"][test], oracle, data["ids"][test], 27)
    rows = [diagnostic_counts(data["fine"][test][data["ids"][test] == patient],
            pf[data["ids"][test] == patient], data["coarse"][test][data["ids"][test] == patient],
            pc[data["ids"][test] == patient], data["parent"]) for patient in patients]
    keys = sorted(rows[0]); temporary = destination / "predictions.tmp.npz"
    np.savez_compressed(temporary, indices=test,
        cell_ids=data["meta"].cell_id.iloc[test].astype(str).to_numpy(dtype=str),
        sample_ids=np.asarray(data["ids"][test], dtype=str), truth_fine=data["fine"][test],
        truth_coarse=data["coarse"][test], fine_prob=fine_prob, coarse_prob=coarse_prob,
        pred_fine=pf, pred_coarse=pc, oracle_pred_fine=oracle, patients=np.asarray(patients, dtype=str),
        fine_cm=fine_cm, coarse_cm=coarse_cm, oracle_cm=oracle_cm,
        diagnostic_counts=np.asarray([[row[key] for key in keys] for row in rows]),
        diagnostic_keys=np.asarray(keys, dtype=str), parent=data["parent"])
    os.replace(temporary, destination / "predictions.npz")
    save_checkpoint(destination / "refit_student.pt", model, {"selected": selected, "history": history})
    atomic_json(destination / "metrics.json", {"condition": args.condition, "fold": args.fold,
        "seed": args.seed, "fine_sb_macro_f1": unified.score(fine_cm),
        "coarse_sb_macro_f1": unified.score(coarse_cm), "oracle_fine_sb_macro_f1": unified.score(oracle_cm),
        "hierarchy_gap": unified.score(oracle_cm) - unified.score(fine_cm), "selection": selected,
        "n_test_cells": len(test), "source_hash": source_hash(),
        "cell_id_sha256": sha256_strings(data["meta"].cell_id.astype(str)),
        "student_test_inputs": "293 APT features only; no RNA or true lineage"})


def refit_student(args, data, index, destination, device):
    if args.condition == "full_shuffled":
        chosen = copy.deepcopy(selection(args.output, "full", args.fold))
    else:
        chosen = selection(args.output, args.condition, args.fold)
    _, x = apt_partition(data, index, "dev")
    targets = None
    if args.condition != "strong_lineage":
        target_path = args.output / "teacher" / f"seed{args.seed}" / f"fold{args.fold}" / "dev_targets.npz"
        targets = load_targets(target_path, index["dev"])
        if args.condition == "full_shuffled":
            order = grouped_target_permutation(data["ids"][index["dev"]], data["coarse"][index["dev"]],
                                               100000 + args.seed * 10 + args.fold)
            targets = {key: value[order] for key, value in targets.items()}
    model, _, _, history = fit_student(x["dev"], data["fine"][index["dev"]],
        data["coarse"][index["dev"]], data["ids"][index["dev"]], data["parent"], targets,
        chosen, args.seed, chosen["epoch"], device)
    probability = predict_student(model, x["test"], device)
    save_predictions(args, data, index, destination, model, probability, chosen, history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "refit")); parser.add_argument("--condition", required=True,
        choices=("teacher",) + CONDITIONS); parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--seed", required=True, type=int, choices=FINAL_SEEDS)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--apt-cache", required=True, type=Path)
    parser.add_argument("--rna-root", required=True, type=Path); parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    if args.command == "select" and args.seed != 17: parser.error("selection uses seed 17 only")
    args.output.mkdir(parents=True, exist_ok=True); assert_frozen(args.output)
    destination = args.output / args.condition / f"seed{args.seed}" / f"fold{args.fold}"
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ("selection.json" if args.command == "select" else
                            ("refit_teacher.pt" if args.condition == "teacher" else "predictions.npz"))
    if marker.exists() and not args.force: print("SKIP", marker); return
    device = require_cuda(args.device); data = load_benchmark(args.apt_cache); index = fold_indices(data, args.fold)
    started = time.time()
    if args.command == "select" and args.condition == "teacher": select_teacher(args, data, index, destination, device)
    elif args.command == "refit" and args.condition == "teacher": refit_teacher(args, data, index, destination, device)
    elif args.command == "select": select_student(args, data, index, destination, device)
    else: refit_student(args, data, index, destination, device)
    print("COMPLETE", args.command, args.condition, args.seed, args.fold, round(time.time() - started, 1), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4); main()
