"""Independently verify saved permutation, scaler, label and fit records."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from run_signal import base, permutation


def main(a):
    meta,_,_=base.core.load_data(); x=base.raw.raw_apt(meta); folds=base.core.load_folds()
    ids=meta.sample_id.to_numpy(str); cells=meta.cell_id.to_numpy(str); enc=base.core.fit_label_encoders(meta)
    y=enc['fine'].transform(meta.cell_subtype); c=enc['coarse'].transform(meta.coarse_subtype)
    parent=enc['coarse'].transform(meta[['cell_subtype','coarse_subtype']].drop_duplicates().set_index('cell_subtype').loc[enc['fine'].classes_,'coarse_subtype'])
    records=[]
    settings=[('shuffle',r) for r in [101,211,307]]+([('conditional',0)] if a.model=='mlp' else [])
    for kind,r in settings:
        for f in range(5):
            root=a.results/f'{a.model}_{kind}_r{r}_f{f}'
            audit=json.loads((root/'audit.json').read_text())
            assert audit['code_sha256']==hashlib.sha256(Path(__file__).with_name('run_signal.py').read_bytes()).hexdigest()
            assert audit['base_code_sha256']==hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
            assert audit['protocol_sha256']==hashlib.sha256(Path(__file__).with_name('PROTOCOL.md').read_bytes()).hexdigest()
            te=np.flatnonzero(np.isin(ids,folds[f])); va=np.flatnonzero(np.isin(ids,folds[(f+1)%5]))
            tr=np.flatnonzero(~np.isin(ids,folds[f]+folds[(f+1)%5])); dev=np.sort(np.r_[tr,va])
            parts=dict(inner=tr,validation=va,refit=dev,test=te)
            with np.load(root/'partition_indices.npz',allow_pickle=False) as b:
                indices={k:b[k].copy() for k in b.files}
            for j,(s,ix) in enumerate(parts.items()):
                np.testing.assert_array_equal(indices[s],ix)
                expected=permutation(ix,ids,c,r,f,j) if r else ix
                np.testing.assert_array_equal(indices[s+'_source'],expected)
                assert audit['cells'][s]==len(ix) and audit['patients'][s]==sorted(set(ids[ix]))
            with np.load(root/'scalers.npz',allow_pickle=False) as scaler:
                for s,label in [('inner','inner'),('refit','outer')]:
                    ref=StandardScaler().fit(x[indices[s+'_source']])
                    np.testing.assert_allclose(scaler[label+'_mean'],ref.mean_,rtol=1e-12,atol=1e-12)
                    np.testing.assert_allclose(scaler[label+'_scale'],ref.scale_,rtol=1e-12,atol=1e-12)
            selection=json.loads((root/'selection.json').read_text()); trials=selection['trials']
            assert len(trials)==4 and selection['selected']==max(trials,key=lambda t:t['score'])
            for trial in trials:
                h=np.array(trial['history']); assert np.isfinite(h).all() and len(h)<=50
                assert trial['epochs']==int(h.argmax())+1 and np.isclose(trial['score'],h.max())
            prior=np.bincount(y[dev],minlength=27).astype(float)
            for m in range(5): prior[parent==m]/=max(prior[parent==m].sum(),1)
            for seed in [17,29,43]:
                p=root/f'joint_s{seed}.npz'; record=json.loads(p.with_suffix('.json').read_text())
                assert record['seed']==seed and record['selected']==selection['selected']
                assert record['training_cells']==len(dev) and record['test_cells']==len(te)
                assert p.with_suffix('.pt').exists()
                with np.load(p,allow_pickle=False) as b:
                    for k,expected in [('truth',y[te]),('coarse_truth',c[te]),('parent',parent),('classes',enc['fine'].classes_),
                        ('cell_ids',cells[te]),('patient_ids',ids[te]),('source_cell_ids',cells[indices['test_source']])]:
                        np.testing.assert_array_equal(b[k],expected)
                    np.testing.assert_allclose(b['prior'],prior)
                    for k,n in [('prob',27),('coarse_prob',5)]:
                        assert b[k].shape==(len(te),n) and np.isfinite(b[k]).all()
                        assert (b[k]>=0).all() and np.allclose(b[k].sum(1),1,atol=1e-5)
                records.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    out=a.results/f'{a.model}_source_audit.json'
    out.write_text(json.dumps(dict(status='PASS',predictions=len(records),records=records),indent=2))
    print('PASS',a.model,len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,required=True); p.add_argument('--model',choices=['mlp','flat'],default='mlp'); main(p.parse_args())
