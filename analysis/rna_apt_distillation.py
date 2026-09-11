"""Patient-cross-fitted RNA-to-APT distillation with frozen matched controls.

RNA preprocessing is fitted separately for every teacher training split.
The projector's teacher may see projector-training cells, but neither upstream
model nor either preprocessor sees the patients receiving cached targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.io import mmread
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

import patient_cell_scaling as core
import validate_paired_information as paired

VERSION = "rna-apt-crossfit-v1"
METHODS = ("supervised", "kd", "confidence", "projection", "shrink", "lineage")
TEMPERATURE = 2.0
EPOCHS = 60
SEEDS = (1701, 1702, 1703)


def save_json(path, value):
    core.atomic_json(Path(path), value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def patient_weights(ids):
    _, inverse, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return (weights / weights.mean()).astype(np.float32)


def taxonomy(meta):
    enc = core.fit_label_encoders(meta)
    y = enc["fine"].transform(meta["cell_subtype"].astype(str))
    coarse = enc["coarse"].transform(meta["coarse_subtype"].astype(str))
    parents = np.empty(27, dtype=int)
    for k in range(27):
        values = np.unique(coarse[y == k])
        if len(values) != 1:
            raise ValueError(f"Non-unique parent for subtype {k}")
        parents[k] = values[0]
    return y, coarse, parents


def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(Path(args.paired) / "cells.csv")
    folds = core.load_folds()
    meta.to_csv(out / "cells.csv", index=False)
    plans = []
    for fold in args.folds:
        stages = ["inner"] if args.pilot else ["outer"]
        for stage in stages:
            prefix = Path(args.paired) / f"f{fold}_{stage}"
            tr = np.atleast_1d(np.loadtxt(str(prefix) + "_train.txt", dtype=int))
            te = np.atleast_1d(np.loadtxt(str(prefix) + "_test.txt", dtype=int))
            train_ids = set(meta.sample_id.iloc[tr].astype(str))
            test_ids = set(meta.sample_id.iloc[te].astype(str))
            assert train_ids.isdisjoint(test_ids)
            assert train_ids.isdisjoint(set(map(str, folds[fold])))
            jobs = []
            for block, block_patients in folds.items():
                target_mask = meta.sample_id.iloc[tr].astype(str).isin(
                    list(map(str, block_patients))).to_numpy()
                target = tr[target_mask]
                if len(target) == 0:
                    continue
                source = tr[~target_mask]
                sid = set(meta.sample_id.iloc[source].astype(str))
                tid = set(meta.sample_id.iloc[target].astype(str))
                assert sid.isdisjoint(tid | test_ids)
                job = f"f{fold}_{stage}_g{block}"
                np.savetxt(out / f"{job}_train.txt", source, fmt="%d")
                np.savetxt(out / f"{job}_test.txt", target, fmt="%d")
                jobs.append(dict(name=job, source_patients=sorted(sid),
                                 target_patients=sorted(tid)))
            plans.append(dict(fold=fold, stage=stage, train=tr.tolist(),
                              test=te.tolist(), jobs=jobs))
    config = dict(version=VERSION, pilot=args.pilot, folds=args.folds,
                  epochs=EPOCHS, teacher_epochs=EPOCHS, seeds=list(SEEDS),
                  methods=list(METHODS), temperature=TEMPERATURE,
                  loss="fine CE + 0.5 aggregated-lineage CE + 0.5 T^2 KL",
                  weighting="equal patients; no class reweighting",
                  student="MLP 128-128, GELU, dropout 0.1",
                  teacher="same MLP applied to 2000 train-only RNA HVGs",
                  projector="same MLP, APT input, soft teacher CE",
                  optimizer="AdamW lr=0.001 weight_decay=0.001 batch=512",
                  schedule="fixed 60 epochs; no checkpoint selection",
                  tuning="none; six fixed methods, no outer-test selection",
                  interval="paired seed and test-patient bootstrap; fixed splits",
                  seeds_scope="student seeds; teacher/projector seed fixed per job",
                  lineage_weight="OOF target conditional NLL, donor-equal, shrinkage 0.05",
                  plans=plans)
    config["protocol_hash"] = digest(config)
    existing = out / "protocol.json"
    if existing.exists() and read_json(existing) != config:
        raise RuntimeError("Output has a different protocol; use a new directory")
    save_json(existing, config)
    print("PREPARED_MANIFEST", out, len(plans), flush=True)


def coarse_probs(q, parents):
    return np.stack([q[:, parents == m].sum(1) for m in range(5)], axis=1)


def mix_targets(q, projected, parents, alpha):
    q = np.maximum(q, 1e-12)
    q = q / q.sum(1, keepdims=True)
    projected = np.maximum(projected, 1e-12)
    projected = projected / projected.sum(1, keepdims=True)
    result = np.empty_like(q)
    for m in range(5):
        ix = parents == m
        mass = q[:, ix].sum(1, keepdims=True)
        conditional = projected[:, ix] / np.maximum(
            projected[:, ix].sum(1, keepdims=True), 1e-12)
        result[:, ix] = alpha[m] * q[:, ix] + (1-alpha[m]) * mass * conditional
    return result / result.sum(1, keepdims=True)


def select_alpha(q, projected, y, ids, parents):
    # Selection is entirely inside student training data, using targets from
    # models whose whole upstream fitting chain excludes each target patient.
    grid = np.linspace(0, 1, 5)
    losses = []
    w = patient_weights(ids)
    for a in grid:
        mix = mix_targets(q, projected, parents, np.repeat(a, 5))
        losses.append(np.average(-np.log(mix[np.arange(len(y)), y] + 1e-12), weights=w))
    global_a = float(grid[np.argmin(losses)])
    result = np.repeat(global_a, 5)
    for m in range(5):
        take = parents[y] == m
        if len(np.unique(ids[take])) < 4:
            continue
        weights = patient_weights(ids[take])
        risks = []
        for a in grid:
            mix = mix_targets(q[take], projected[take], parents, np.repeat(a, 5))
            mass = mix[:, parents == m].sum(1)
            prob = mix[np.arange(take.sum()), y[take]] / np.maximum(mass, 1e-12)
            risks.append(np.average(-np.log(prob + 1e-12), weights=weights)
                         + 0.05 * (a-global_a)**2)
        result[m] = grid[np.argmin(risks)]
    return result


def network(dim):
    return nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(.1),
                         nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1),
                         nn.Linear(128, 27))


def train(x, y, ids, parents, seed, device, targets=None, confidence=None,
          projector=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = network(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    w = torch.as_tensor(patient_weights(ids), device=device)
    parent = torch.as_tensor(parents, dtype=torch.long, device=device)
    grouping = F.one_hot(parent, 5).float()
    soft = None if targets is None else torch.as_tensor(targets, dtype=torch.float32, device=device)
    confidence = np.ones(len(y), np.float32) if confidence is None else confidence
    c = torch.as_tensor(confidence, dtype=torch.float32, device=device)
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(y), 512):
            ix = order[start:start+512]
            logits = model(x[ix])
            if projector:
                loss = -(soft[ix] * F.log_softmax(logits, dim=1)).sum(1)
            else:
                loss = F.cross_entropy(logits, y[ix], reduction="none")
                coarse = F.softmax(logits, dim=1) @ grouping
                loss = loss + .5 * F.nll_loss(coarse.clamp_min(1e-12).log(),
                                            parent[y[ix]], reduction="none")
                if soft is not None:
                    kd = F.kl_div(F.log_softmax(logits/TEMPERATURE, dim=1),
                                  soft[ix], reduction="none").sum(1)
                    loss = loss + .5 * TEMPERATURE**2 * c[ix] * kd
            objective = (loss*w[ix]).mean()
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    return model


def predict(model, x, device, temperature=1):
    model.eval()
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(x), 2048):
            inputs = torch.as_tensor(x[start:start+2048], dtype=torch.float32, device=device)
            blocks.append(F.softmax(model(inputs)/temperature, 1).cpu().numpy())
    return np.concatenate(blocks)


def standardize(source, target):
    scaler = StandardScaler().fit(source)
    return scaler.transform(source).astype(np.float32), scaler.transform(target).astype(np.float32)


def get_targets(out, plan, meta, apt, y, parents, device, protocol_hash):
    cache = out / f"f{plan['fold']}_{plan['stage']}_targets.npz"
    if cache.exists():
        data = np.load(cache)
        assert str(data["protocol_hash"]) == protocol_hash
        assert np.array_equal(data["train"], plan["train"])
        return data["teacher"], data["projected"], data["alpha"]
    tr = np.asarray(plan["train"])
    q = np.full((len(tr), 27), np.nan, np.float32)
    projected = q.copy()
    positions = {int(cell): i for i, cell in enumerate(tr)}
    seen = set()
    for j, job in enumerate(plan["jobs"]):
        name = job["name"]
        destination = out / f"{name}_targets.npz"
        source = np.atleast_1d(np.loadtxt(out/f"{name}_train.txt", dtype=int))
        target = np.atleast_1d(np.loadtxt(out/f"{name}_test.txt", dtype=int))
        assert not seen.intersection(target.tolist())
        seen.update(target.tolist())
        assert set(meta.sample_id.iloc[source]).isdisjoint(set(meta.sample_id.iloc[target]))
        if destination.exists():
            data = np.load(destination)
            assert str(data["protocol_hash"]) == protocol_hash
            assert np.array_equal(data["target"], target)
            teacher_target, projector_target = data["teacher"], data["projected"]
        else:
            begin = time.time()
            if not (out/f"{name}_done.txt").exists():
                raise RuntimeError(f"RNA preparation incomplete: {name}")
            rna = [mmread(out/f"{name}_{s}.mtx").toarray().astype(np.float32)
                   for s in ("train", "test")]
            rx, rv = standardize(*rna)
            seed = 10000 + plan["fold"]*10 + j
            ids = meta.sample_id.iloc[source].to_numpy()
            teacher = train(rx, y[source], ids, parents, seed, device)
            teacher_source = predict(teacher, rx, device, TEMPERATURE)
            teacher_target = predict(teacher, rv, device, TEMPERATURE)
            del teacher, rx, rv, rna
            ax, av = standardize(apt[source], apt[target])
            projection = train(ax, y[source], ids, parents, seed+100, device,
                               targets=teacher_source, projector=True)
            projector_target = predict(projection, av, device)
            del projection, ax, av
            core.atomic_npz(destination, teacher=teacher_target, projected=projector_target,
                            source=source, target=target, protocol_hash=protocol_hash)
            print("TARGETS", name, round(time.time()-begin, 1), flush=True)
        ix = [positions[int(i)] for i in target]
        q[ix], projected[ix] = teacher_target, projector_target
    assert seen == set(tr.tolist())
    assert np.isfinite(q).all() and np.isfinite(projected).all()
    alpha = select_alpha(q, projected, y[tr], meta.sample_id.iloc[tr].to_numpy(), parents)
    core.atomic_npz(cache, teacher=q, projected=projected, alpha=alpha,
                    train=tr, protocol_hash=protocol_hash)
    return q, projected, alpha


def evaluate(q, y, coarse, ids, parents):
    patients = np.unique(ids)
    fine_pred = q.argmax(1)
    coarse_pred = coarse_probs(q, parents).argmax(1)
    cm = {}
    for task, truth, prediction, k in (("fine", y, fine_pred, 27),
                                       ("coarse", coarse, coarse_pred, 5)):
        cm[task] = np.stack([np.bincount(k*truth[ids == p]+prediction[ids == p],
                             minlength=k*k).reshape(k, k) for p in patients])
    return patients, cm


def run(args):
    out = Path(args.output)
    config = read_json(out / "protocol.json")
    config_without_hash = {k: v for k, v in config.items() if k != "protocol_hash"}
    assert digest(config_without_hash) == config["protocol_hash"]
    assert config["version"] == VERSION and config["epochs"] == EPOCHS
    meta = pd.read_csv(out / "cells.csv")
    y, coarse, parents = taxonomy(meta)
    apt = paired.raw_apt(meta)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    print("DEVICE", device, "VISIBLE", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    for i, plan in enumerate(config["plans"]):
        if i % args.shards != args.shard:
            continue
        tr, te = np.asarray(plan["train"]), np.asarray(plan["test"])
        teacher, projected, alpha = get_targets(out, plan, meta, apt, y, parents,
                                                device, config["protocol_hash"])
        x, xt = standardize(apt[tr], apt[te])
        ids = meta.sample_id.iloc[tr].to_numpy()
        test_ids = meta.sample_id.iloc[te].to_numpy()
        confidence = 1 + (teacher*np.log(teacher+1e-12)).sum(1)/np.log(27)
        confidence /= np.average(confidence, weights=patient_weights(ids))
        target_sets = dict(supervised=None, kd=teacher, confidence=teacher,
            projection=mix_targets(teacher, projected, parents, np.zeros(5)),
            shrink=mix_targets(teacher, projected, parents, np.repeat(.5, 5)),
            lineage=mix_targets(teacher, projected, parents, alpha))
        for seed in SEEDS:
            for method in METHODS:
                dest = out/f"f{plan['fold']}_{plan['stage']}_s{seed}_{method}.npz"
                if dest.exists():
                    assert str(np.load(dest)["protocol_hash"]) == config["protocol_hash"]
                    continue
                begin = time.time()
                model = train(x, y[tr], ids, parents, seed, device,
                              targets=target_sets[method],
                              confidence=confidence if method == "confidence" else None)
                q = predict(model, xt, device)
                patients, cm = evaluate(q, y[te], coarse[te], test_ids, parents)
                core.atomic_npz(dest, fine=cm["fine"], coarse=cm["coarse"],
                    patients=patients.astype(str), probabilities=q, cells=te,
                    alpha=alpha, protocol_hash=config["protocol_hash"])
                score = core.f1_from_confusion(core.patient_balanced_matrix(cm["fine"]))
                print("COMPLETE", dest.name, "fine", round(score, 5),
                      "seconds", round(time.time()-begin, 1), flush=True)
                del model, q


def aggregate(args):
    out = Path(args.output)
    config = read_json(out/"protocol.json")
    arrays = {}
    canonical = None
    for method in METHODS:
        arrays[method] = {}
        for task in ("fine", "coarse"):
            seeds = []
            for seed in SEEDS:
                ids, cms = [], []
                for plan in config["plans"]:
                    name = f"f{plan['fold']}_{plan['stage']}_s{seed}_{method}.npz"
                    data = np.load(out/name)
                    assert str(data["protocol_hash"]) == config["protocol_hash"]
                    ids.extend(data["patients"].tolist())
                    cms.append(data[task])
                assert len(ids) == len(set(ids))
                order = np.argsort(ids)
                ids = np.asarray(ids)[order]
                if canonical is None:
                    canonical = ids
                assert np.array_equal(canonical, ids)
                seeds.append(np.concatenate(cms)[order])
            arrays[method][task] = np.stack(seeds).astype(float)
    assert len(canonical) == (8 if config["pilot"] else 8*len(config["folds"]))
    rng = np.random.default_rng(99441)
    seed_draws = rng.integers(0, len(SEEDS), (2000, len(SEEDS)))
    patient_draws = rng.integers(0, len(canonical), (2000, len(canonical)))
    def score(cm):
        return np.mean([core.f1_from_confusion(core.patient_balanced_matrix(c)) for c in cm])
    records, draws = [], {}
    for task in ("fine", "coarse"):
        for method in METHODS:
            cm = arrays[method][task]
            values = np.array([score(cm[s][:, p]) for s, p in zip(seed_draws, patient_draws)])
            draws[method, task] = values
            records.append(dict(task=task, comparison=method, estimate=score(cm),
                low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975))))
        for method in METHODS[1:]:
            for ref in ("supervised", "kd"):
                if method == ref:
                    continue
                values = draws[method, task]-draws[ref, task]
                records.append(dict(task=task, comparison=method+"-minus-"+ref,
                    estimate=score(arrays[method][task])-score(arrays[ref][task]),
                    low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                    bootstrap_positive_fraction=float(np.mean(values > 0))))
    pd.DataFrame(records).to_csv(out/"summary.csv", index=False)
    save_json(out/"completion.json", dict(status="PASS", protocol_hash=config["protocol_hash"],
        student_fits=len(config["plans"])*len(SEEDS)*len(METHODS),
        test_patients=len(canonical), scope="development pilot" if config["pilot"] else "retrospective fixed-fold validation",
        limitations=["fixed teacher/projector seed, not all upstream uncertainty",
                     "no C2KD/BioKD/TabM or independent-cohort comparison yet",
                     "lineage coefficient uses target-risk proxy, not observed student treatment effect",
                     "pointwise intervals, no multiplicity-adjusted superiority claim"]))
    print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "aggregate"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired", default="/ssd3/mayiling/apt_agent_runtime/paired_information_validation")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    options = parser.parse_args()
    if len(set(options.folds)) != len(options.folds) or not set(options.folds) <= set(range(5)):
        parser.error("folds must be distinct values from 0 to 4")
    if options.pilot and options.folds != [0]:
        parser.error("pilot uses fold 0 inner split only")
    globals()[options.command](options)
