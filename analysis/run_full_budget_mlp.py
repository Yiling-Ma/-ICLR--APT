"""Full-cell APT baseline; validation-only epoch selection and explicit OOF IDs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import patient_cell_scaling as core

SEEDS = [20260912, 20260913, 20260914]
DECAYS = [1e-5, 1e-3]


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2))
    os.replace(temp, path)


def matrices(y, pred, ids, k):
    patients = np.unique(ids)
    return patients, np.stack([confusion_matrix(y[ids == p], pred[ids == p],
                            labels=np.arange(k)) for p in patients])


def score(cm):
    return core.f1_from_confusion(core.patient_balanced_matrix(cm))


def oracle(prob, truth, parent):
    allowed = parent[None, :] == parent[truth, None]
    return np.where(allowed, prob, -np.inf).argmax(1)


def fit(x, y, k, seed, decay, epochs, device, validation=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(),
        torch.nn.Dropout(.1), torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Dropout(.1), torch.nn.Linear(128, k)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=decay)
    tx = torch.as_tensor(x, device=device)
    ty = torch.as_tensor(y, dtype=torch.long, device=device)
    best, best_epoch, history = -np.inf, 1, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(y), device=device)
        for ix in order.split(1024):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(tx[ix]), ty[ix])
            loss.backward()
            optimizer.step()
        if validation is not None:
            vx, vy, ids = validation
            _, cm = matrices(vy, predict(model, vx, device).argmax(1), ids, k)
            s = score(cm)
            history.append(float(s))
            if s > best:
                best, best_epoch = s, epoch
        if epoch % 10 == 0:
            print('EPOCH', epoch, 'validation' if validation else 'refit', flush=True)
    return model, best, best_epoch, history


def predict(model, x, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(torch.as_tensor(
            x[i:i+4096], device=device)), dim=1).cpu().numpy()
            for i in range(0, len(x), 4096)])


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(version='full-budget-mlp-v1', seeds=SEEDS, epochs=50,
        hidden=[256,128], dropout=.1, batch=1024, lr=.001, decay=DECAYS,
        training='all available development cells; unweighted CE; no disease supervision',
        input='provided apt_expression, train-only StandardScaler; fixed 293 features',
        selection='inner fold (outer+1)%5; SB Macro-F1 chooses decay and epoch; earliest tie',
        refit='all 32 outer-development patients for selected epochs, initialized afresh',
        scope='full data budget, not optimizer/objective matched to other architectures',
        uncertainty='paired seed and held-out patient resampling, conditional on frozen folds')
    path = out/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        atomic_json(path, protocol)
    meta, x, features = core.load_data()
    assert len(meta) == 361792 and meta.cell_id.is_unique and np.isfinite(x).all()
    enc = core.fit_label_encoders(meta)
    folds = core.load_folds()
    ids = meta.sample_id.to_numpy(dtype=str)
    device = torch.device(args.device)
    pairs = meta[['cell_subtype','coarse_subtype']].drop_duplicates()
    assert pairs.cell_subtype.is_unique
    mapping = pairs.set_index('cell_subtype').coarse_subtype
    parent = enc['coarse'].transform(mapping.loc[enc['fine'].classes_])
    for seed in SEEDS:
        for f in range(5):
            te = np.flatnonzero(np.isin(ids, folds[f]))
            va = np.flatnonzero(np.isin(ids, folds[(f+1)%5]))
            tr = np.flatnonzero(~np.isin(ids, folds[f]+folds[(f+1)%5]))
            dev = np.flatnonzero(~np.isin(ids, folds[f]))
            for task in core.TASKS:
                dest = out/f's{seed}_f{f}_{task}.npz'
                if dest.exists():
                    continue
                started = time.time()
                y = enc[task].transform(meta[core.TASKS[task]].astype(str))
                k = len(enc[task].classes_)
                scaler = StandardScaler().fit(x[tr])
                train, valid = [scaler.transform(x[ix]).astype(np.float32) for ix in [tr,va]]
                trials = []
                for decay in DECAYS:
                    model, best, ep, history = fit(train, y[tr], k, seed, decay, 50,
                        device, (valid, y[va], ids[va]))
                    trials.append(dict(decay=decay, score=float(best), epoch=ep, history=history))
                    del model
                selected = max(trials, key=lambda r:r['score'])
                scaler = StandardScaler().fit(x[dev])
                model, _, _, _ = fit(scaler.transform(x[dev]).astype(np.float32), y[dev],
                    k, seed, selected['decay'], selected['epoch'], device)
                prob = predict(model, scaler.transform(x[te]).astype(np.float32), device)
                assert prob.shape == (len(te), k) and np.allclose(prob.sum(1), 1, atol=1e-5)
                patients, cm = matrices(y[te], prob.argmax(1), ids[te], k)
                extra = {}
                if task == 'fine':
                    op = oracle(prob, y[te], parent)
                    _, ocm = matrices(y[te], op, ids[te], k)
                    extra = dict(oracle_cm=ocm, parent=parent)
                torch.save(dict(model=model.state_dict(), features=features,
                    classes=enc[task].classes_.tolist(), mean=scaler.mean_, scale=scaler.scale_,
                    seed=seed, fold=f, selected=selected), dest.with_suffix('.pt'))
                np.savez_compressed(dest.with_suffix('.tmp.npz'), cm=cm, patients=patients,
                    prob=prob, truth=y[te], cell_ids=meta.cell_id.iloc[te].to_numpy(dtype=str),
                    sample_ids=ids[te], classes=enc[task].classes_.astype(str), **extra)
                os.replace(dest.with_suffix('.tmp.npz'), dest)
                atomic_json(dest.with_suffix('.json'), dict(seed=seed, fold=f, task=task,
                    trials=trials, selected=selected, train_patients=sorted(set(ids[dev])),
                    test_patients=patients.tolist(), train_cells=len(dev), test_cells=len(te),
                    seconds=time.time()-started,
                    cell_hash=hashlib.sha256('\n'.join(meta.cell_id.astype(str)).encode()).hexdigest()))
                print('COMPLETE', dest.name, 'seconds', round(time.time()-started,1), flush=True)
                del model


def aggregate(args):
    rows=[]
    rng=np.random.default_rng(20260912)
    meta, _, _ = core.load_data()
    enc = core.fit_label_encoders(meta)
    folds = core.load_folds()
    index = pd.Index(meta.cell_id.astype(str))
    assert index.is_unique
    for task in core.TASKS:
        all_cm, all_oracle=[],[]
        order=None
        for seed in SEEDS:
            blocks=[np.load(args.output/f's{seed}_f{f}_{task}.npz') for f in range(5)]
            # Legacy NPZ string arrays used object dtype. Recover IDs from the
            # JSON sidecar, then validate them against keyed cell metadata and CMs.
            patient_blocks=[]
            for f,b in enumerate(blocks):
                audit=json.loads((args.output/f's{seed}_f{f}_{task}.json').read_text())
                patients=np.asarray(audit['test_patients'],dtype=str)
                assert np.array_equal(patients,sorted(folds[f]))
                position=index.get_indexer(b['cell_ids'].astype(str))
                assert (position>=0).all()
                sample_ids=meta.sample_id.iloc[position].to_numpy(dtype=str)
                truth=enc[task].transform(meta[core.TASKS[task]].iloc[position].astype(str))
                np.testing.assert_array_equal(truth,b['truth'])
                np.testing.assert_array_equal(enc[task].classes_.astype(str),b['classes'])
                assert np.allclose(b['prob'].sum(1),1,atol=1e-5)
                recovered,cm=matrices(truth,b['prob'].argmax(1),sample_ids,len(enc[task].classes_))
                np.testing.assert_array_equal(recovered,patients)
                np.testing.assert_array_equal(cm,b['cm'])
                if task=='fine':
                    mapping=meta[['cell_subtype','coarse_subtype']].drop_duplicates().set_index('cell_subtype')
                    parent=enc['coarse'].transform(mapping.loc[enc['fine'].classes_,'coarse_subtype'])
                    np.testing.assert_array_equal(parent,b['parent'])
                    _,ocm=matrices(truth,oracle(b['prob'],truth,parent),sample_ids,len(enc[task].classes_))
                    np.testing.assert_array_equal(ocm,b['oracle_cm'])
                patient_blocks.append(patients)
            ids=np.concatenate(patient_blocks)
            cell_ids=np.concatenate([b['cell_ids'] for b in blocks])
            assert len(ids)==40 and len(set(ids))==40 and len(set(cell_ids))==361792
            if order is None: order=ids
            assert np.array_equal(ids,order)
            all_cm.append(np.concatenate([b['cm'] for b in blocks]))
            if task=='fine': all_oracle.append(np.concatenate([b['oracle_cm'] for b in blocks]))
            for block in blocks: block.close()
        a=np.stack(all_cm)
        for name,b in [('unconstrained',a)]+([('true_lineage_oracle',np.stack(all_oracle))] if all_oracle else []):
            values=[score(cm) for cm in b]
            draws=[]
            diff=[]
            for _ in range(2000):
                ss=rng.integers(0,3,3); pp=rng.integers(0,40,40)
                draws.append(np.mean([score(b[s,pp]) for s in ss]))
                diff.append(np.mean([score(b[s,pp])-score(a[s,pp]) for s in ss]))
            rows.append(dict(task=task,method=name,estimate=np.mean(values),
                low=np.quantile(draws,.025),high=np.quantile(draws,.975),seed_values=values,
                paired_delta=np.mean(values)-np.mean([score(c) for c in a]),
                delta_low=np.quantile(diff,.025),delta_high=np.quantile(diff,.975)))
    pd.DataFrame(rows).to_csv(args.output/'summary.csv',index=False)
    atomic_json(args.output/'completion.json',dict(status='PASS',outer_fits=30,
        seed_count=3,patients=40,cells=361792,keyed_cell_label_alignment=True,
        confusion_reconstruction=True,oracle_confusion_reconstruction=True,
        patient_ids_source='JSON sidecars checked against keyed cell metadata; pickle never enabled',
        oracle_scope='privileged-label diagnostic, not deployable'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['run','aggregate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    torch.set_num_threads(4)
    globals()[a.command](a)
