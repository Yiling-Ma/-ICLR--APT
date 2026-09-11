"""Development-only APT patient-context controls; no outer-test evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

import rna_apt_distillation as common

VERSION = "patient-context-pilot-v1"
METHODS = ("raw", "centered", "global_context", "soft_context",
           "hierarchy_uniform", "hierarchy_reliable")
LOSSES = ("ce", "balanced")
SEEDS = (1701, 1702, 1703)
BUDGET = 128
EPOCHS = 60


def model(dim, classes):
    return nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(.1),
                         nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1),
                         nn.Linear(128, classes))


def fit(x, y, ids, seed, classes, balanced):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model(x.shape[1], classes).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    weights = common.patient_weights(ids)
    counts = np.bincount(y, weights=weights, minlength=classes) + 1.
    prior = torch.tensor(np.log(counts/counts.sum()), dtype=torch.float32, device=device)
    tx = torch.as_tensor(x, device=device)
    ty = torch.as_tensor(y, dtype=torch.long, device=device)
    tw = torch.as_tensor(weights, device=device)
    for _ in range(EPOCHS):
        net.train()
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(y), 512):
            ix = order[start:start+512]
            logits = net(tx[ix])
            loss = F.cross_entropy(logits + prior if balanced else logits,
                                   ty[ix], reduction="none")
            objective = (loss*tw[ix]).mean()
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite loss")
            opt.zero_grad(set_to_none=True)
            objective.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
    return net


def predict(net, x):
    return common.predict(net, x, next(net.parameters()).device)


def references(x, q, ids):
    means = []
    for p in np.unique(ids):
        take = ids == p
        mass = q[take].sum(0)
        means.append((q[take].T @ x[take])/np.maximum(mass[:, None], 1e-8))
    return np.mean(means, axis=0).astype(np.float32)


def statistics(x, q, indices, reference):
    """Use a fixed context set and remove each query's own contribution."""
    n, d = x.shape
    context = x[indices].astype(np.float64)
    probs = q[indices].astype(np.float64)
    self_mask = np.zeros(n, bool)
    self_mask[indices] = True
    count = len(indices)-self_mask.astype(int)
    good = count > 0
    den = np.maximum(count, 1)[:, None]
    mean = (context.sum(0)-self_mask[:, None]*x)/den
    variance = (np.square(context).sum(0)-self_mask[:, None]*np.square(x))/den-mean**2
    std = np.sqrt(np.maximum(variance, 0))
    mass = np.maximum(probs.sum(0)[None, :] - self_mask[:, None]*q, 0)
    squares = np.maximum(np.square(probs).sum(0)[None, :] - self_mask[:, None]*np.square(q), 0)
    weighted = probs.T @ context
    centers = np.empty((n, 5, d), np.float32)
    for m in range(5):
        centers[:, m] = (weighted[m]-self_mask[:, None]*q[:, m, None]*x)/np.maximum(mass[:, m, None], 1e-8)
        centers[mass[:, m] <= 1e-8, m] = reference[m]
    residual = centers-reference[None, :, :]
    present = mass > 1e-8
    rho = mass/(mass+10.) * squares/np.maximum(mass, 1e-8)
    uniform = (residual*present[:, :, None]).sum(1)/5.
    reliable = (residual*rho[:, :, None]).sum(1)/5.
    for value in (mean, std, uniform, reliable):
        value[~good] = 0
    centers[~good] = reference
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32),
                centers=centers, uniform=uniform.astype(np.float32),
                reliable=reliable.astype(np.float32))


def context_for_patients(x, q, ids, reference, cell_ids):
    parts = {k: np.empty((len(x), 5, x.shape[1]) if k == "centers" else x.shape,
                         np.float32) for k in ("mean", "std", "centers", "uniform", "reliable")}
    manifests = {}
    for p in np.unique(ids):
        ix = np.flatnonzero(ids == p)
        seed = int(hashlib.sha256(str(p).encode()).hexdigest()[:8], 16)
        chosen = np.random.default_rng(seed).choice(len(ix), min(BUDGET, len(ix)), replace=False)
        manifests[str(p)] = cell_ids[ix[chosen]].tolist()
        values = statistics(x[ix], q[ix], chosen, reference)
        for k, v in values.items():
            parts[k][ix] = v
    return parts, manifests


def design(x, stats, method):
    z = np.zeros_like(x)
    if method == "raw":
        blocks = [x]+[z]*7
    elif method == "centered":
        blocks = [x-stats["mean"]]+[z]*7
    elif method == "global_context":
        blocks = [x, x-stats["mean"], stats["mean"], stats["std"]]+[z]*4
    elif method == "soft_context":
        blocks = [x, stats["mean"], stats["std"]]+[stats["centers"][:, m] for m in range(5)]
    else:
        b = stats["uniform" if method == "hierarchy_uniform" else "reliable"]
        blocks = [x, x-b, b, stats["std"]]+[z]*4
    return np.concatenate(blocks, axis=1).astype(np.float32)


def prepare(out, paired):
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(paired/"cells.csv")
    tr = np.loadtxt(paired/"f0_inner_train.txt", dtype=int)
    va = np.loadtxt(paired/"f0_inner_test.txt", dtype=int)
    ids = meta.sample_id.to_numpy()
    folds = common.core.load_folds()
    assert set(ids[tr]).isdisjoint(ids[va])
    assert set(map(str, np.r_[ids[tr], ids[va]])).isdisjoint(map(str, folds[0]))
    config = dict(version=VERSION, methods=METHODS, losses=LOSSES, seeds=SEEDS,
        epochs=EPOCHS, budget=BUDGET, scope="retrospective inner-validation development only",
        train=tr.tolist(), validation=va.tolist(), train_patients=sorted(set(map(str, ids[tr]))),
        validation_patients=sorted(set(map(str, ids[va]))),
        loss="fine CE or donor-prior Balanced Softmax; no coarse auxiliary loss",
        upstream="cross-fitted 5-way balanced MLP; source-only scaling and donor-equal soft references",
        reliability="mass/(mass+10) * sum(q^2)/mass; sum residuals divided by fixed 5",
        context="same 128-cell subset per patient, query contribution removed",
        network="8*293 input slots (unused padded zero); 128-128 GELU dropout .1",
        optimization="AdamW .001 wd .001, batch512, 60fixed epochs, donor-equal weights")
    config = json.loads(json.dumps(config))
    config["protocol_hash"] = common.digest(config)
    if (out/"protocol.json").exists():
        assert common.read_json(out/"protocol.json") == config
    common.save_json(out/"protocol.json", config)
    if (out/"prepared.json").exists():
        assert common.read_json(out/"prepared.json")["protocol_hash"] == config["protocol_hash"]
        return
    x = common.paired.raw_apt(meta)
    y, coarse, parents = common.taxonomy(meta)
    assembled = {}
    manifests, upstream = {}, []
    for block, patients in folds.items():
        take = np.isin(ids[tr].astype(str), list(map(str, patients)))
        target, source = tr[take], tr[~take]
        if not len(target):
            continue
        assert set(ids[source]).isdisjoint(set(ids[target]) | set(ids[va]))
        sx, tx = common.standardize(x[source], x[target])
        net = fit(sx, coarse[source], ids[source], 81100+int(block), 5, True)
        reference = references(x[source], predict(net, sx), ids[source])
        stats, selected = context_for_patients(x[target], predict(net, tx), ids[target], reference, target)
        for key, value in stats.items():
            if key not in assembled:
                assembled[key] = np.empty((len(tr),)+value.shape[1:], np.float32)
            assembled[key][take] = value
        manifests.update(selected)
        upstream.append(dict(source=source.tolist(), target=target.tolist()))
        del net
        print("CROSSFIT", block, "done", flush=True)
    sx, vx = common.standardize(x[tr], x[va])
    net = fit(sx, coarse[tr], ids[tr], 81110, 5, True)
    ref = references(x[tr], predict(net, sx), ids[tr])
    vstats, selected = context_for_patients(x[va], predict(net, vx), ids[va], ref, va)
    manifests.update(selected)
    common.core.atomic_npz(out/"train.npz", x=x[tr], y=y[tr], coarse=coarse[tr],
                           patients=ids[tr].astype(str), cells=tr, parents=parents, **assembled)
    common.core.atomic_npz(out/"validation.npz", x=x[va], y=y[va], coarse=coarse[va],
                           patients=ids[va].astype(str), cells=va, parents=parents, **vstats)
    common.save_json(out/"context_manifest.json", manifests)
    common.save_json(out/"upstream_splits.json", upstream)
    common.save_json(out/"prepared.json", dict(protocol_hash=config["protocol_hash"], status="PASS"))
    print("PREPARED", flush=True)


def run(out, shard):
    cfg = common.read_json(out/"protocol.json")
    assert cfg["version"] == VERSION
    train = dict(np.load(out/"train.npz"))
    val = dict(np.load(out/"validation.npz"))
    counter = 0
    for method in METHODS:
        x, xv = design(train["x"], train, method), design(val["x"], val, method)
        x, xv = common.standardize(x, xv)
        for loss in LOSSES:
            for seed in SEEDS:
                counter += 1
                if (counter-1) % 2 != shard:
                    continue
                path = out/f"{method}_{loss}_{seed}.npz"
                if path.exists():
                    assert str(np.load(path)["protocol_hash"]) == cfg["protocol_hash"]
                    continue
                start = time.time()
                net = fit(x, train["y"], train["patients"], seed, 27, loss == "balanced")
                q = predict(net, xv)
                patients, cms = common.evaluate(q, val["y"], val["coarse"], val["patients"], val["parents"])
                assert np.isfinite(q).all()
                np.testing.assert_allclose(q.sum(1), 1., atol=1e-5)
                common.core.atomic_npz(path, fine=cms["fine"], coarse=cms["coarse"],
                    patients=patients, probabilities=q, cells=val["cells"], protocol_hash=cfg["protocol_hash"])
                score = common.core.f1_from_confusion(common.core.patient_balanced_matrix(cms["fine"]))
                print("COMPLETE", path.name, score, "seconds", round(time.time()-start, 1), flush=True)
                del net


def aggregate(out):
    cfg = common.read_json(out/"protocol.json")
    rng = np.random.default_rng(28831)
    sd = rng.integers(0, 3, (2000, 3))
    pdx = rng.integers(0, 8, (2000, 8))
    scores, draws, rows = {}, {}, []
    canonical = None
    for method in METHODS:
        for loss in LOSSES:
            for task in ("fine", "coarse"):
                mats = []
                for seed in SEEDS:
                    d = np.load(out/f"{method}_{loss}_{seed}.npz")
                    assert str(d["protocol_hash"]) == cfg["protocol_hash"]
                    if canonical is None:
                        canonical = d["patients"]
                    np.testing.assert_array_equal(d["patients"], canonical)
                    assert len(canonical) == 8
                    mats.append(d[task])
                mats = np.stack(mats)
                def score(m):
                    return np.mean([common.core.f1_from_confusion(common.core.patient_balanced_matrix(c)) for c in m])
                key = (method, loss, task)
                scores[key] = score(mats)
                draws[key] = np.array([score(mats[s][:, p]) for s, p in zip(sd, pdx)])
                lo, hi = np.quantile(draws[key], [.025, .975])
                rows.append(dict(method=method, loss=loss, task=task, comparison="score", estimate=scores[key], low=lo, high=hi))
    for key in scores:
        method, loss, task = key
        for ref in ("raw", "global_context", "soft_context", "hierarchy_uniform"):
            if ref == method:
                continue
            delta = draws[key]-draws[(ref, loss, task)]
            lo, hi = np.quantile(delta, [.025, .975])
            rows.append(dict(method=method, loss=loss, task=task, comparison="minus_"+ref,
                estimate=scores[key]-scores[(ref, loss, task)], low=lo, high=hi))
    pd.DataFrame(rows).to_csv(out/"summary.csv", index=False)
    common.save_json(out/"completion.json", dict(status="PASS", fits=36, validation_patients=8,
        protocol_hash=cfg["protocol_hash"], scope="development; no superiority claim",
        limitations=["fixed context and upstream seeds", "pointwise intervals, multiple candidates",
                     "nominal input dimensions equal, active information differs", "no outer or external confirmation"]))
    print(pd.DataFrame(rows).query("comparison == 'score'").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired", type=Path, default=Path("/ssd3/mayiling/apt_agent_runtime/paired_information_validation"))
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    print("DEVICE", "cuda" if torch.cuda.is_available() else "cpu", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    if args.command == "prepare":
        prepare(args.output, args.paired)
    elif args.command == "run":
        run(args.output, args.shard)
    else:
        aggregate(args.output)
