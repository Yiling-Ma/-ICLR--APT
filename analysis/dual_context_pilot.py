"""Composition-stable / shift-equivariant correction: inner development only."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import rna_apt_distillation as common

VERSION = "dual-context-v1"
METHODS = ("raw", "context", "correction", "composition", "equivariance", "dual")
SEEDS = (1701, 1702, 1703)
STEPS = 1000
SUPPORT = 128
QUERIES = 32
PATIENTS_PER_STEP = 4
CONDITIONS = ("natural", "composition", "shift", "both")


def composition_view(pool, labels, original, rng):
    classes = np.unique(labels[original])
    allowed = pool[np.isin(labels[pool], classes)]
    anchors = np.array([rng.choice(allowed[labels[allowed] == k]) for k in classes])
    rest = allowed[~np.isin(allowed, anchors)]
    proportions = np.maximum(rng.dirichlet(np.repeat(.25, len(classes))), 1e-8)
    weights = np.zeros(len(rest))
    for k, share in zip(classes, proportions):
        ix = labels[rest] == k
        weights[ix] = share/max(ix.sum(), 1)
    need = len(original)-len(anchors)
    more = rng.choice(rest, need, replace=False, p=weights/weights.sum()) if need else np.array([], int)
    result = np.concatenate([anchors, more]).astype(int)
    assert len(np.unique(result)) == len(original)
    assert np.array_equal(np.unique(labels[result]), classes)
    return result


class ContextModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shape_encoder = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.offset = nn.Sequential(nn.Linear(dim+64, 64), nn.GELU(), nn.Linear(64, dim))
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.classifier = nn.Sequential(nn.Linear(2*dim+64, 128), nn.GELU(), nn.Dropout(.1),
                                        nn.Linear(128, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 27))

    def summarize(self, support):
        mean = support.mean(1)
        shape = self.shape_encoder(support-mean[:, None]).mean(1)
        offset = self.offset(torch.cat([mean, shape], -1))
        return mean, shape, offset

    def classify(self, queries, summary, method):
        mean, shape, offset = summary
        if method == "raw":
            corrected, other, shape = queries, torch.zeros_like(mean), torch.zeros_like(shape)
        elif method == "context":
            corrected, other = queries, mean
        else:
            corrected, other = queries-offset[:, None], torch.zeros_like(mean)
        inputs = torch.cat([corrected, other[:, None].expand(-1, queries.shape[1], -1),
                            shape[:, None].expand(-1, queries.shape[1], -1)], -1)
        return self.classifier(inputs)


def constraint_terms(a, b, shifted, shift):
    return F.mse_loss(a, b), F.mse_loss(shifted-a, shift)


def prepare(out, source):
    out.mkdir(parents=True, exist_ok=True)
    train, val = [dict(np.load(source/f"{s}.npz")) for s in ("train", "validation")]
    assert len(np.unique(train["patients"])) == 24 and len(np.unique(val["patients"])) == 8
    assert set(train["patients"]).isdisjoint(val["patients"])
    cfg = dict(version=VERSION, methods=METHODS, seeds=SEEDS, steps=STEPS,
        support=SUPPORT, queries=QUERIES, patients_per_step=PATIENTS_PER_STEP,
        scope="retrospective eight-patient inner-validation development; outer fold 0 excluded",
        source_protocol=common.read_json(source/"protocol.json")["protocol_hash"],
        train_patients=np.unique(train["patients"]).tolist(), validation_patients=np.unique(val["patients"]).tolist(),
        optimizer="AdamW lr .001 wd .001 gradient_clip5; fixed1000steps",
        augmentation="all six methods: mean CE over natural/composition/shifted query episodes",
        constraints="composition MSE and offset-increment MSE, each weight1; shared .001 offset anchor",
        shift="Gaussian standardized APT offsets, train SD.25, diagnostic SD.5",
        context="training labels only select support-matched composition views; no labels enter model",
        evaluation="independent 256-cell reference pool; natural support128; queries exclude whole pool",
        diagnostics="validation labels construct composition stress views only; same fixed query cells",
        tuning="none; six fixed variants; no automatic outer runs")
    cfg = json.loads(json.dumps(cfg))
    cfg["protocol_hash"] = common.digest(cfg)
    if (out/"protocol.json").exists():
        assert common.read_json(out/"protocol.json") == cfg
    common.save_json(out/"protocol.json", cfg)
    if (out/"prepared.json").exists():
        assert common.read_json(out/"prepared.json")["protocol_hash"] == cfg["protocol_hash"]
        return
    tx, vx = common.standardize(train["x"], val["x"])
    patients = np.unique(train["patients"])
    groups = [np.flatnonzero(train["patients"] == p) for p in patients]
    for seed in SEEDS:
        rng = np.random.default_rng(seed+70000)
        q, a, b = [], [], []
        for _ in range(STEPS):
            qq, aa, bb = [], [], []
            for patient in rng.choice(len(groups), PATIENTS_PER_STEP, replace=False):
                available = groups[patient]
                query = rng.choice(available, QUERIES, replace=False)
                pool = available[~np.isin(available, query)]
                first = rng.choice(pool, SUPPORT, replace=False)
                second = composition_view(pool, train["y"], first, rng)
                assert not np.intersect1d(query, np.r_[first, second]).size
                qq.append(query); aa.append(first); bb.append(second)
            q.append(qq); a.append(aa); b.append(bb)
        shifts = rng.normal(0, .25, (STEPS, PATIENTS_PER_STEP, tx.shape[1])).astype(np.float32)
        common.core.atomic_npz(out/f"schedule_{seed}.npz", query=np.asarray(q), first=np.asarray(a),
                               second=np.asarray(b), shift=shifts, protocol_hash=cfg["protocol_hash"])
    rng = np.random.default_rng(431881)
    supports, alternatives, pools, queries = [], [], [], []
    validation_patients = np.unique(val["patients"])
    for patient in validation_patients:
        ix = np.flatnonzero(val["patients"] == patient)
        assert len(ix) > 256
        pool = rng.choice(ix, 256, replace=False)
        first = pool[:SUPPORT]
        second = composition_view(pool, val["y"], first, rng)
        supports.append(first); alternatives.append(second); pools.append(pool)
        queries.extend(ix[~np.isin(ix, pool)])
    common.core.atomic_npz(out/"data.npz", train_x=tx, val_x=vx, train_y=train["y"], val_y=val["y"],
        train_patients=train["patients"], val_patients=val["patients"], val_coarse=val["coarse"],
        train_cells=train["cells"], val_cells=val["cells"], parents=train["parents"],
        support=np.asarray(supports), alternative=np.asarray(alternatives), pool=np.asarray(pools),
        queries=np.asarray(queries), patients=validation_patients,
        shift=rng.normal(0, .5, (8, tx.shape[1])).astype(np.float32), protocol_hash=cfg["protocol_hash"])
    common.save_json(out/"prepared.json", dict(status="PASS", protocol_hash=cfg["protocol_hash"]))
    print("PREPARED", flush=True)


def train_model(data, schedule, seed, method, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = ContextModel(data["train_x"].shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    x = torch.as_tensor(data["train_x"], device=device)
    y = torch.as_tensor(data["train_y"], dtype=torch.long, device=device)
    schedule = {k: torch.as_tensor(schedule[k], device=device) for k in ("query", "first", "second", "shift")}
    total = np.zeros(4)
    for step in range(STEPS):
        query = x[schedule["query"][step]]
        first, second = x[schedule["first"][step]], x[schedule["second"][step]]
        shift = schedule["shift"][step]
        a, b, shifted = net.summarize(first), net.summarize(second), net.summarize(first+shift[:, None])
        truth = y[schedule["query"][step]].reshape(-1)
        ce = sum(F.cross_entropy(net.classify(q, s, method).reshape(-1, 27), truth)
                 for q, s in ((query, a), (query, b), (query+shift[:, None], shifted)))/3
        comp, equiv = constraint_terms(a[2], b[2], shifted[2], shift)
        penalty = .001*a[2].square().mean() if method not in ("raw", "context") else ce*0
        loss = ce+penalty
        if method in ("composition", "dual"):
            loss = loss+comp
        if method in ("equivariance", "dual"):
            loss = loss+equiv
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
        total += [float(ce.detach()), float(comp.detach()), float(equiv.detach()), float(loss.detach())]
    return net, total/STEPS


def evaluate(net, data, method, device):
    net.eval()
    results = {}
    diagnostics = []
    selected = data["queries"]
    with torch.inference_mode():
        for condition in CONDITIONS:
            prob = np.empty((len(selected), 27), np.float32)
            for j, patient in enumerate(data["patients"]):
                take = data["val_patients"][selected] == patient
                idx = selected[take]
                support = data["alternative"][j] if condition in ("composition", "both") else data["support"][j]
                shifted = condition in ("shift", "both")
                delta = data["shift"][j] if shifted else np.zeros(data["val_x"].shape[1], np.float32)
                summary = net.summarize(torch.as_tensor(data["val_x"][support]+delta, device=device)[None])
                blocks = []
                for start in range(0, len(idx), 2048):
                    q = torch.as_tensor(data["val_x"][idx[start:start+2048]]+delta, device=device)[None]
                    blocks.append(F.softmax(net.classify(q, summary, method), -1)[0].cpu().numpy())
                prob[take] = np.concatenate(blocks)
                if condition == "natural" and method not in ("raw", "context"):
                    alt = net.summarize(torch.as_tensor(data["val_x"][data["alternative"][j]], device=device)[None])
                    delta_t = torch.as_tensor(data["shift"][j], device=device)[None]
                    moved = net.summarize(torch.as_tensor(data["val_x"][support], device=device)[None]+delta_t[:, None])
                    comp, equiv = constraint_terms(summary[2], alt[2], moved[2], delta_t)
                    diagnostics.append([float(comp), float(equiv)])
            np.testing.assert_allclose(prob.sum(1), 1., atol=1e-5)
            patients, cms = common.evaluate(prob, data["val_y"][selected], data["val_coarse"][selected],
                data["val_patients"][selected], data["parents"])
            results[condition+"_probabilities"] = prob
            for task in ("fine", "coarse"):
                results[condition+"_"+task] = cms[task]
    results.update(patients=patients, cells=data["val_cells"][selected],
                   diagnostics=np.asarray(diagnostics, dtype=float).reshape(-1, 2))
    return results


def run(out, shard):
    cfg = common.read_json(out/"protocol.json")
    assert cfg["version"] == VERSION and cfg["steps"] == STEPS
    data = dict(np.load(out/"data.npz"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mi, method in enumerate(METHODS):
        for si, seed in enumerate(SEEDS):
            if (mi*len(SEEDS)+si) % 2 != shard:
                continue
            dest = out/f"{method}_{seed}.npz"
            if dest.exists():
                assert str(np.load(dest)["protocol_hash"]) == cfg["protocol_hash"]
                continue
            schedule = dict(np.load(out/f"schedule_{seed}.npz"))
            assert str(schedule["protocol_hash"]) == cfg["protocol_hash"]
            start = time.time()
            net, losses = train_model(data, schedule, seed, method, device)
            result = evaluate(net, data, method, device)
            common.core.atomic_npz(dest, **result, training_losses=losses, protocol_hash=cfg["protocol_hash"])
            score = common.core.f1_from_confusion(common.core.patient_balanced_matrix(result["natural_fine"]))
            print("COMPLETE", method, seed, score, "seconds", round(time.time()-start, 1), flush=True)
            del net


def aggregate(out):
    cfg = common.read_json(out/"protocol.json")
    rng = np.random.default_rng(86283)
    sd = rng.integers(0, 3, (2000, 3)); pdx = rng.integers(0, 8, (2000, 8))
    records, seed_scores, diagnostics, scores, draws = [], [], [], {}, {}
    canonical = None
    for method in METHODS:
        fits = [dict(np.load(out/f"{method}_{seed}.npz")) for seed in SEEDS]
        for seed, fit in zip(SEEDS, fits):
            assert str(fit["protocol_hash"]) == cfg["protocol_hash"]
            if canonical is None:
                canonical = fit["patients"]
            np.testing.assert_array_equal(canonical, fit["patients"])
            for patient_index, (comp, equiv) in enumerate(fit["diagnostics"]):
                diagnostics.append(dict(method=method, seed=seed, patient_index=patient_index, composition_mse=comp, shift_increment_mse=equiv))
        for condition in CONDITIONS:
            for task in ("fine", "coarse"):
                cms = np.stack([f[condition+"_"+task] for f in fits])
                def score(m):
                    return np.mean([common.core.f1_from_confusion(common.core.patient_balanced_matrix(c)) for c in m])
                key = method, condition, task
                scores[key] = score(cms)
                draws[key] = np.array([score(cms[s][:, p]) for s, p in zip(sd, pdx)])
                lo, hi = np.quantile(draws[key], [.025, .975])
                records.append(dict(method=method, condition=condition, task=task, comparison="score", estimate=scores[key], low=lo, high=hi))
                for seed, cm in zip(SEEDS, cms):
                    seed_scores.append(dict(method=method, seed=seed, condition=condition, task=task, score=score(cm[None])))
    for key in scores:
        method, condition, task = key
        for ref in ("raw", "context", "correction", "composition", "equivariance"):
            if method == ref:
                continue
            baseline = ref, condition, task
            lo, hi = np.quantile(draws[key]-draws[baseline], [.025, .975])
            records.append(dict(method=method, condition=condition, task=task, comparison="minus_"+ref,
                                estimate=scores[key]-scores[baseline], low=lo, high=hi))
    pd.DataFrame(records).to_csv(out/"summary.csv", index=False)
    pd.DataFrame(seed_scores).to_csv(out/"seed_scores.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out/"diagnostics.csv", index=False)
    common.save_json(out/"completion.json", dict(status="PASS", fits=18, validation_patients=8,
        protocol_hash=cfg["protocol_hash"], limitations=["development only, pointwise intervals",
        "fixed support and synthetic diagnostic shift seeds", "composition stress uses validation labels to construct views",
        "offset biological/technical origin not identifiable", "nominal architecture matched, active branches differ"]))
    print(pd.DataFrame(records).query("comparison == 'score' and task == 'fine'").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "aggregate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/ssd3/mayiling/apt_agent_runtime/patient_context_pilot_v1"))
    parser.add_argument("--shard", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    print("DEVICE", "cuda" if torch.cuda.is_available() else "cpu", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    if args.command == "prepare": prepare(args.output, args.source)
    elif args.command == "run": run(args.output, args.shard)
    else: aggregate(args.output)
