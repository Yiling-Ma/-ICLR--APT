"""Validate completed v2 artifacts against source labels, folds and raw-input scalers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from run import MODELS, SEEDS, core, raw


def main(root):
    meta, _, _ = core.load_data()
    x = raw.raw_apt(meta)
    enc = core.fit_label_encoders(meta)
    y = enc['fine'].transform(meta.cell_subtype)
    c = enc['coarse'].transform(meta.coarse_subtype)
    ids = meta.sample_id.to_numpy(str)
    cells = meta.cell_id.to_numpy(str)
    folds = core.load_folds()
    assert len(cells) == len(set(cells)) == 361792
    assert len(set(ids)) == 40
    protocol_hash = hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest()
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [raw.ROOT/'data/ALL_PBMC_APT_Cell_matrix.npz', raw.ROOT/'data/cells.tsv']}
    rows, sources, checked = [], [], []
    for f in range(5):
        ix = {'test': np.flatnonzero(np.isin(ids, folds[f])),
              'validation': np.flatnonzero(np.isin(ids, folds[(f+1)%5]))}
        ix['inner'] = np.flatnonzero(~np.isin(ids, folds[f]+folds[(f+1)%5]))
        ix['refit'] = np.sort(np.r_[ix['inner'], ix['validation']])
        expected_scalers = {s: StandardScaler().fit(x[ix[t]]) for s, t in [('inner','inner'), ('outer','refit')]}
        for model in MODELS:
            directory = root/f'{model}_f{f}'
            audit = json.loads((directory/'audit.json').read_text())
            assert audit['protocol_sha256'] == protocol_hash and audit['code_sha256'] == code_hash
            for p, digest in hashes.items(): assert audit[p] == digest
            for stage, indices in ix.items():
                assert audit['patients'][stage] == sorted(set(ids[indices]))
                assert audit['cells'][stage] == len(indices)
            with np.load(directory/'scalers.npz') as scaler:
                for s, expected in expected_scalers.items():
                    np.testing.assert_allclose(scaler[s+'_mean'], expected.mean_, rtol=1e-7, atol=1e-8)
                    np.testing.assert_allclose(scaler[s+'_scale'], expected.scale_, rtol=1e-7, atol=1e-8)
            for task in (['fine','coarse'] if model in ['lr','xgb'] else ['joint']):
                selection = json.loads((directory/f'{task}_selection.json').read_text())
                trials = selection['trials']; best = selection['selected']
                assert len(trials) == (12 if model == 'xgb' else 4)
                assert best == max(trials, key=lambda t: t['score'])
                if model not in ['lr','xgb']:
                    for trial in trials:
                        assert trial['score'] == max(trial['history'])
                        assert trial['epochs'] == int(np.argmax(trial['history']))+1
                        assert len(trial['history']) <= 50
                for stage in ['inner','validation']:
                    rows.append(dict(model=model,fold=f,task=task,stage=stage,seed=17,
                        patient_count=len(set(ids[ix[stage]])),cell_count=len(ix[stage]),
                        configuration_count=len(trials),selected=json.dumps(best),
                        selection_seconds=selection['seconds'],refit_seconds='',hardware=audit['hardware'],
                        artifact_source=str(directory/f'{task}_selection.json')))
                for seed in ([17] if model == 'lr' else SEEDS):
                    path = directory/f'{task}_s{seed}.npz'
                    side = json.loads(path.with_suffix('.json').read_text())
                    assert side['seed'] == seed and side['task'] == task and side['selected'] == best
                    assert side['training_cells'] == len(ix['refit']) and side['test_cells'] == len(ix['test'])
                    k = 5 if task == 'coarse' else 27
                    with np.load(path, allow_pickle=False) as a:
                        np.testing.assert_array_equal(a['cell_ids'], cells[ix['test']])
                        np.testing.assert_array_equal(a['patient_ids'], ids[ix['test']])
                        np.testing.assert_array_equal(a['truth'], (c if task=='coarse' else y)[ix['test']])
                        np.testing.assert_array_equal(a['classes'], enc['coarse' if task=='coarse' else 'fine'].classes_)
                        p = a['prob']; assert p.shape == (len(ix['test']), k)
                        assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1),1,atol=1e-5)
                        if task == 'joint':
                            np.testing.assert_array_equal(a['coarse_truth'], c[ix['test']])
                            np.testing.assert_array_equal(a['parent'][a['truth']], a['coarse_truth'])
                            cp = a['coarse_prob']
                            assert cp.shape == (len(p),5) and np.isfinite(cp).all() and np.allclose(cp.sum(1),1,atol=1e-5)
                    for stage in ['refit','test']:
                        rows.append(dict(model=model,fold=f,task=task,stage=stage,seed=seed,
                            patient_count=len(set(ids[ix[stage]])),cell_count=len(ix[stage]),
                            configuration_count=len(trials),selected=json.dumps(best),
                            selection_seconds='',refit_seconds=side['seconds'],hardware=audit['hardware'],
                            artifact_source=str(path)))
                    sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                    checked.append(str(path))
    assert len(checked) == 130
    with (root/'protocol_inventory.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
    (root/'artifact_audit.json').write_text(json.dumps(dict(status='PASS',prediction_count=len(checked),
        cells=361792,patients=40,checks=['exact source-label and fold alignment for all 130 predictions',
        'train-only raw-count log1p scalers independently reconstructed',
        'audit split patient sets and cell counts', 'recorded selection maximum and earliest epoch ties',
        'fixed class order and probability shape/normalization', 'code and protocol hashes'],
        input_sha256=hashes,code_sha256=code_hash,protocol_sha256=protocol_hash,sources=sources),indent=2))
    print('PASS: 130 predictions; source OOF alignment, split counts, scalers, selection and hashes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
