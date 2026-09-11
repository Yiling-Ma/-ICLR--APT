"""Nested train-only RNA feature selection with matched augmentation controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import patient_cell_scaling as core
from paired_rna_oracle import patient_class_weights

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get(
    'APT_PAIRED_OUT',
    str(ROOT / 'outputs/paired_information_validation'),
)).resolve()
SEED = 20260911
CAP = 1000
MODES = ['rna', 'apt', 'rna_apt', 'rna_noise', 'rna_shuffled_apt']
MODELS = {'sgd': [1e-4, 1e-3], 'mlp': [1e-3, 1e-1]}


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, _, _ = core.load_data()
    meta.to_csv(OUT/'cells.csv', index=False)
    folds = core.load_folds()
    for f in range(5):
        for stage in ['inner', 'outer']:
            test_patients = folds[(f+1)%5] if stage == 'inner' else folds[f]
            forbidden = set(test_patients) | set(folds[f])
            train = []
            for patient in sorted(set(meta.sample_id)-forbidden):
                candidates = np.flatnonzero(meta.sample_id.to_numpy() == patient)
                seed = int(hashlib.sha256(f'{SEED}:{f}:{patient}'.encode()).hexdigest()[:8],16)
                train.extend(np.random.default_rng(seed).permutation(candidates)[:CAP])
            test = np.flatnonzero(meta.sample_id.isin(test_patients))
            prefix = OUT / f'f{f}_{stage}'
            np.savetxt(str(prefix)+'_train.txt', sorted(train), fmt='%d')
            np.savetxt(str(prefix)+'_test.txt', test, fmt='%d')
    save_json(OUT/'protocol.json', dict(seed=SEED, cap=CAP, modes=MODES, models=MODELS,
        inner='one fixed patient-disjoint validation fold; no outer-test exposure',
        HVG='2000 mean-binned log-count dispersion features, fitted per inner/outer training subset',
        noise='293 standard normal features; fixed per fold/stage',
        shuffle='APT vectors permuted within patient separately in training and evaluation',
        interval='paired patient bootstrap conditional on fitted models; not training-seed uncertainty',
        scope='post-hoc validation; RNA-derived operational labels; not external clinical validation'))
    subprocess.run(['/home/mayiling/opt/apt-rna-r/bin/Rscript',
                    str(ROOT/'analysis/prepare_fold_rna.R'), str(ROOT), str(OUT)], check=True)


def raw_apt(meta):
    ids = pd.read_csv(ROOT/'data/cells.tsv', header=None, sep='\t')[0].astype(str)
    assert ids.is_unique
    ix = pd.Index(ids).get_indexer(meta.cell_id.astype(str))
    assert np.all(ix >= 0)
    x = sparse.load_npz(ROOT/'data/ALL_PBMC_APT_Cell_matrix.npz')
    if x.shape[0] != len(ids) and x.shape[1] == len(ids):
        x = x.T
    assert x.shape == (len(ids), 293)
    x = x.tocsr()[ix].toarray().astype(np.float32)
    assert np.all(x >= 0) and np.all(np.isfinite(x))
    return np.log1p(x)


def inputs(meta, apt, fold, stage):
    prefix = str(OUT/f'f{fold}_{stage}')
    tr = np.loadtxt(prefix+'_train.txt', dtype=int)
    te = np.loadtxt(prefix+'_test.txt', dtype=int)
    assert not set(meta.sample_id.iloc[tr]) & set(meta.sample_id.iloc[te])
    rtr, rte = [mmread(prefix+'_'+s+'.mtx').tocsr().astype(np.float32)
                for s in ['train', 'test']]
    scaler = StandardScaler(with_mean=False).fit(rtr)
    rtr, rte = scaler.transform(rtr), scaler.transform(rte)
    scaler = StandardScaler().fit(apt[tr])
    atr, ate = scaler.transform(apt[tr]), scaler.transform(apt[te])
    rng = np.random.default_rng(SEED+fold*10+(stage=='outer'))
    noise = [rng.standard_normal(a.shape).astype(np.float32) for a in [atr, ate]]
    shuffled = []
    for a, ix in [(atr,tr), (ate,te)]:
        b = a.copy()
        for patient in np.unique(meta.sample_id.iloc[ix]):
            pos = np.flatnonzero(meta.sample_id.iloc[ix].to_numpy()==patient)
            b[pos] = a[rng.permutation(pos)]
        shuffled.append(b)
    def join(a, b):
        return sparse.hstack([a, sparse.csr_matrix(b)],format='csr')
    return tr, te, dict(rna=(rtr,rte), apt=(atr,ate),
        rna_apt=(join(rtr,atr),join(rte,ate)),
        rna_noise=(join(rtr,noise[0]),join(rte,noise[1])),
        rna_shuffled_apt=(join(rtr,shuffled[0]),join(rte,shuffled[1])))


def fit(name, alpha, x, y, weights):
    if name == 'sgd':
        model = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=40, tol=None,
                              average=True, random_state=SEED)
    else:
        model = MLPClassifier(hidden_layer_sizes=(64,), alpha=alpha, max_iter=40,
            batch_size=512, early_stopping=False, n_iter_no_change=41, random_state=SEED)
    model.fit(x,y,sample_weight=weights)
    return model


def cms(y, pred, ids, classes):
    patients = np.unique(ids)
    return patients, np.stack([confusion_matrix(y[ids==p], pred[ids==p],
                                  labels=np.arange(classes)) for p in patients])


def run():
    meta = pd.read_csv(OUT/'cells.csv')
    apt = raw_apt(meta)
    enc = core.fit_label_encoders(meta)
    for fold in range(5):
        inner = inputs(meta, apt, fold, 'inner')
        outer = inputs(meta, apt, fold, 'outer')
        for task in ['coarse','fine']:
            y = enc[task].transform(meta[core.TASKS[task]].astype(str))
            k = len(enc[task].classes_)
            for name, grid in MODELS.items():
                for mode in MODES:
                    dest=OUT/f'f{fold}_{task}_{name}_{mode}.npz'
                    if dest.exists():
                        continue
                    scores=[]
                    for alpha in grid:
                        tr,te,xx=inner
                        model=fit(name,alpha,xx[mode][0],y[tr],
                                  patient_class_weights(meta.sample_id.iloc[tr].to_numpy(),y[tr]))
                        _, cm=cms(y[te],model.predict(xx[mode][1]),meta.sample_id.iloc[te].to_numpy(),k)
                        scores.append(core.f1_from_confusion(core.patient_balanced_matrix(cm)))
                    alpha=grid[int(np.argmax(scores))]
                    tr,te,xx=outer
                    model=fit(name,alpha,xx[mode][0],y[tr],
                              patient_class_weights(meta.sample_id.iloc[tr].to_numpy(),y[tr]))
                    patients,cm=cms(y[te],model.predict(xx[mode][1]),meta.sample_id.iloc[te].to_numpy(),k)
                    tmp=dest.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp,cm=cm,patients=patients.astype(str),alpha=alpha,
                                        validation_scores=scores, iterations=model.n_iter_)
                    os.replace(tmp,dest)
                    print('COMPLETE',dest.name,flush=True)


def aggregate():
    rows=[]
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,40,(2000,40))
    for task in ['coarse','fine']:
        for model in MODELS:
            matrices={}
            patient_order=None
            for mode in MODES:
                blocks=[np.load(OUT/f'f{f}_{task}_{model}_{mode}.npz') for f in range(5)]
                ids=np.concatenate([b['patients'] for b in blocks])
                assert len(ids)==40 and len(set(ids))==40
                if patient_order is None: patient_order=ids
                assert np.array_equal(patient_order,ids)
                matrices[mode]=np.concatenate([b['cm'] for b in blocks])
            def score(cm): return core.f1_from_confusion(core.patient_balanced_matrix(cm))
            for mode,cm in matrices.items():
                rows.append(dict(task=task,model=model,comparison=mode,estimate=score(cm)))
            for ref in ['rna','rna_noise','rna_shuffled_apt']:
                high,low=matrices['rna_apt'],matrices[ref]
                delta=np.array([score(high[d])-score(low[d]) for d in draws])
                rows.append(dict(task=task,model=model,comparison='rna_apt-minus-'+ref,
                    estimate=score(high)-score(low),low=np.quantile(delta,.025),high=np.quantile(delta,.975)))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    save_json(OUT/'completion.json',dict(status='PASS', outer_fits=100, inner_fits=200,
        unique_test_patients=40, paired_patient_alignment=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','aggregate'])
    a=parser.parse_args()
    with threadpool_limits(limits=4):
        globals()[a.command]()
