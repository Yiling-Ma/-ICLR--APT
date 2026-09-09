"""Sequential sampling controls; descriptive contrasts, not causal mediation.

A samples uniformly from selected subjects' pooled cells. B keeps the P=8
class set with a one-cell support floor. C fixes P=8 integer quotas, sampling
uniformly within class. D keeps C's quotas but balances donors within class.
All conditions share random cell priorities and nested subject sets.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

import class_matched_scaling as legacy
from audit_granularity_chance import f1

core = legacy.core
CONDITIONS = ("A", "B", "C", "D")
VERSION = "sequential-controls-v1"


def sample(labels, subjects, selected, quotas, priority, donor_priority, condition):
    eligible = np.flatnonzero(np.isin(subjects, selected))
    eligible = eligible[np.argsort(priority[eligible], kind="stable")]
    total = int(quotas.sum())
    if condition == "A":
        chosen = eligible[:total]
    elif condition == "B":
        eligible = eligible[quotas[labels[eligible]] > 0]
        reserved = np.array([eligible[labels[eligible] == k][0]
                             for k in np.flatnonzero(quotas)], dtype=int)
        rest = eligible[~np.isin(eligible, reserved)]
        chosen = np.concatenate([reserved, rest[:total - len(reserved)]])
    elif condition in ("C", "D"):
        chunks = []
        for k in np.flatnonzero(quotas):
            rows = eligible[labels[eligible] == k]
            if len(rows) < quotas[k]:
                raise ValueError("Insufficient class capacity")
            if condition == "D":
                donors = sorted(set(subjects[rows]), key=lambda d: donor_priority[d])
                queues = [rows[subjects[rows] == d] for d in donors]
                # Random donor order avoids always favoring the earliest nested donor.
                rows = np.array([r for group in itertools.zip_longest(*queues)
                                 for r in group if r is not None], dtype=int)
            chunks.append(rows[:quotas[k]])
        chosen = np.concatenate(chunks)
    else:
        raise ValueError(condition)
    if len(chosen) != total or len(set(chosen)) != total:
        raise ValueError("Invalid total or repeated cells")
    counts = np.bincount(labels[chosen], minlength=len(quotas))
    if condition != "A" and not np.array_equal(counts > 0, quotas > 0):
        raise ValueError("Support mismatch")
    if condition in ("C", "D") and not np.array_equal(counts, quotas):
        raise ValueError("Quota mismatch")
    return chosen


def configuration(args):
    return dict(version=VERSION, dataset=args.dataset, seeds=args.seeds,
                xgboost_device=os.environ.get("APT_XGB_DEVICE", "cpu"),
                total=args.total, budgets=[8, 32], conditions=list(CONDITIONS),
                models=args.models.split(","), folds=5, tasks=["coarse", "fine"],
                quota_reference="P8 development capacity, largest remainder with support floor",
                metric="mean across seeds of subject-balanced pooled Macro-F1",
                mlp="128,64 ReLU Adam batch256 lr0.001 alpha0.0001 100 epochs; no early stopping",
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependency_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(legacy.__file__), Path(core.__file__),
                              Path(__file__).with_name("onek1k_scaling.py"),
                              Path(__file__).with_name("combat_citeseq_scaling.py")]})


def jobs(args):
    for fold, seed, task, model, p, condition in itertools.product(
            range(5), range(args.seeds), ("coarse", "fine"),
            args.models.split(","), (8, 32), CONDITIONS):
        yield dict(fold=fold, seed=seed, task=task, model=model, p=p, condition=condition)


def name(j):
    return "f{fold}_s{seed}_{task}_{model}_P{p}_{condition}".format(**j)


def predict(model_name, task, train, y, test):
    if model_name == "xgboost":
        observed, local = np.unique(y, return_inverse=True)
        if len(observed) == 1:
            return np.full(len(test), observed[0], dtype=int)
        model = core.build_model(model_name)
        model.fit(train, local)
        device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if os.environ.get("APT_XGB_DEVICE", "cpu").startswith("cuda") and not device.startswith("cuda"):
            raise RuntimeError("Requested GPU but XGBoost fell back to CPU")
        print("XGBOOST_DEVICE", device, "HOST", socket.gethostname(), flush=True)
        return np.concatenate([observed[model.predict(test[i:i + 8192]).astype(int)]
                               for i in range(0, len(test), 8192)])
    if model_name != "mlp":
        return core.fit_one_model(model_name, task, train, y, test)[0]
    if len(np.unique(y)) == 1:
        return np.full(len(test), y[0], dtype=int)
    model = MLPClassifier(hidden_layer_sizes=(128, 64), batch_size=256,
                          learning_rate_init=.001, alpha=.0001, max_iter=100,
                          early_stopping=False, tol=0, n_iter_no_change=101,
                          random_state=42)
    model.fit(train, y)
    return np.concatenate([model.predict(test[i:i + 8192])
                           for i in range(0, len(test), 8192)]).astype(int)


def run(args, output, config):
    legacy.configure_dataset(args.dataset)
    metadata, matrix, _ = core.load_data()
    encoders, folds = core.fit_label_encoders(metadata), core.load_folds()
    patients = core.patient_table(metadata)
    subjects = metadata.sample_id.astype(str).to_numpy()
    cell_ids = metadata.cell_id.astype(str).to_numpy()
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("Cell identifiers must be unique")
    label_arrays = {t: e.transform(metadata[core.TASKS[t]].astype(str)) for t, e in encoders.items()}
    for index, j in enumerate(jobs(args)):
        if index % args.shards != args.shard:
            continue
        stem = output / "runs" / name(j)
        if stem.with_suffix(".json").exists() and stem.with_suffix(".npz").exists():
            if json.loads(stem.with_suffix(".json").read_text())["config"] != config:
                raise ValueError("Existing run has a different configuration")
            continue
        start = time.time()
        labels = label_arrays[j["task"]]
        k = len(encoders[j["task"]].classes_)
        test_patients = folds[j["fold"]]
        development = patients[~patients.sample_id.isin(test_patients)]
        order = core.nested_patient_order(development, j["fold"], j["seed"])
        if len(order) < 32:
            raise ValueError("Fewer than 32 development subjects")
        reference = np.flatnonzero(np.isin(subjects, order[:8]))
        quotas = legacy.allocate_quotas(np.bincount(labels[reference], minlength=k), args.total)
        rng = np.random.default_rng(np.random.SeedSequence([20270909, j["fold"], j["seed"]]))
        priority = rng.random(len(labels))
        donor_priority = dict(zip(sorted(set(subjects)), rng.random(len(set(subjects)))))
        train_idx = sample(labels, subjects, order[:j["p"]], quotas, priority, donor_priority, j["condition"])
        test_idx = np.flatnonzero(np.isin(subjects, test_patients))
        if set(subjects[train_idx]) & set(test_patients):
            raise ValueError("Train/test subject overlap")
        raw = np.asarray(matrix[train_idx], dtype=np.float32)
        mean, std = core.fit_standardizer(raw)
        pred = predict(j["model"], j["task"], core.apply_standardizer(raw, mean, std), labels[train_idx],
                       core.apply_standardizer(np.asarray(matrix[test_idx], dtype=np.float32), mean, std))
        cm = np.stack([confusion_matrix(labels[test_idx][subjects[test_idx] == d],
                       pred[subjects[test_idx] == d], labels=np.arange(k)) for d in test_patients])
        counts = np.bincount(labels[train_idx], minlength=k)
        coverage = [len(set(subjects[train_idx][labels[train_idx] == c])) for c in range(k)]
        core.atomic_npz(stem.with_suffix(".npz"), patient_ids=np.asarray(test_patients, dtype=str),
                        confusions=cm, train_cell_ids=cell_ids[train_idx], train_subject_ids=subjects[train_idx],
                        class_names=encoders[j["task"]].classes_.astype(str))
        legacy.atomic_json(stem.with_suffix(".json"), dict(j, config=config, seconds=time.time()-start,
            hostname=socket.gethostname(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            actual_subjects=len(set(subjects[train_idx])), class_counts=counts.tolist(),
            class_donor_coverage=coverage, reference_quotas=quotas.tolist(),
            train_manifest_sha256=hashlib.sha256("\n".join(cell_ids[train_idx]).encode()).hexdigest()))
        print("COMPLETE", name(j), round(time.time()-start, 1), flush=True)


def joint_summary(deltas):
    return dict(estimate=float(np.mean(deltas)), lower=float(np.quantile(deltas,.025)),
                upper=float(np.quantile(deltas,.975)), bootstrap_positive_fraction=float(np.mean(deltas > 0)))


def aggregate(args, output):
    expected_config = configuration(args)
    missing = [name(j) for j in jobs(args) if not (output/"runs"/(name(j)+".json")).exists()
               or not (output/"runs"/(name(j)+".npz")).exists()]
    if missing:
        legacy.atomic_json(output/"completion.json", dict(complete=False, missing=len(missing), examples=missing[:20]))
        raise RuntimeError(f"Incomplete: {len(missing)} runs missing; refusing partial inference")
    report = []
    for task, model in itertools.product(("coarse", "fine"), args.models.split(",")):
        tensors = {}
        ids_ref = None
        for c, p in itertools.product(CONDITIONS, (8,32)):
            seeds = []
            for seed in range(args.seeds):
                pieces, ids = [], []
                for fold in range(5):
                    j = dict(fold=fold, seed=seed, task=task, model=model, p=p, condition=c)
                    meta = json.loads((output/"runs"/(name(j)+".json")).read_text())
                    if meta["config"] != expected_config:
                        raise ValueError("Mixed protocol results")
                    with np.load(output/"runs"/(name(j)+".npz"), allow_pickle=False) as a:
                        if (a["confusions"] < 0).any() or (a["confusions"].sum(axis=(1,2)) <= 0).any():
                            raise ValueError("Invalid confusion matrix")
                        pieces.append(a["confusions"])
                        ids.extend(a["patient_ids"].tolist())
                if len(ids) != len(set(ids)) or (ids_ref is not None and ids != ids_ref):
                    raise ValueError("Unpaired or duplicate OOF subjects")
                ids_ref = ids
                cm = np.concatenate(pieces).astype(float)
                seeds.append(cm/cm.sum(axis=(1,2), keepdims=True))
            tensors[c,p] = np.stack(seeds)
        n = len(ids_ref)
        rng = np.random.default_rng(20270909)
        draws = {c: [] for c in CONDITIONS}
        for _ in range(args.bootstrap):
            sw = rng.multinomial(args.seeds, np.full(args.seeds,1/args.seeds))/args.seeds
            pw = rng.multinomial(n, np.full(n,1/n))/n
            for c in CONDITIONS:
                lo, hi = [f1(np.einsum("spij,p->sij", tensors[c,p], pw)) for p in (8,32)]
                draws[c].append(float(sw @ (hi-lo)))
        points = {c: float(np.mean(f1(tensors[c,32].mean(axis=1))-f1(tensors[c,8].mean(axis=1)))) for c in CONDITIONS}
        row = dict(task=task,model=model,subject_count=n,bootstrap=args.bootstrap,conditions={},contrasts={})
        for c in CONDITIONS:
            row["conditions"][c] = joint_summary(np.asarray(draws[c])) | {"estimate":points[c]}
        for a,b in zip(CONDITIONS[:-1], CONDITIONS[1:]):
            row["contrasts"][a+"-"+b] = joint_summary(np.asarray(draws[a])-draws[b]) | {"estimate":points[a]-points[b]}
        report.append(row)
    legacy.atomic_json(output/"summary.json",dict(results=report,interpretation="Sequential descriptive contrasts; not causal mediation. Bootstrap conditional on fixed folds and fitted runs."))
    legacy.atomic_json(output/"completion.json", dict(complete=True, runs=len(list(jobs(args)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run","aggregate"])
    parser.add_argument("--dataset", choices=["apt","combat_rna","onek1k"], required=True)
    parser.add_argument("--models", default="logistic_regression,xgboost,mlp")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--total", type=int, default=6400)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds < 1 or args.total < 1 or args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid budgets/shard")
    if set(args.models.split(",")) - {"logistic_regression","xgboost","mlp"}:
        raise ValueError("Unknown model")
    output = args.output or core.PROJECT_ROOT/"outputs/sequential_scaling"/args.dataset
    config = configuration(args)
    path = output/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Protocol differs: use a new output directory")
    legacy.atomic_json(path,config)
    if args.command == "run":
        run(args,output,config)
    else:
        aggregate(args,output)


if __name__ == "__main__":
    main()
