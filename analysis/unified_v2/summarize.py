"""Fail closed on incomplete runs; score all conditions and paired contrasts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from run import MODELS, SEEDS, matrices, score


def decode(fine, coarse, parent, truth):
    mass=np.stack([fine[:,parent==i].sum(1) for i in range(5)],1)
    consistent=fine/np.maximum(mass[:,parent],1e-30)*coarse[:,parent]
    return dict(D0=fine.argmax(1),D1=np.where(parent[None,:]==coarse.argmax(1)[:,None],fine,-np.inf).argmax(1),
        D2=consistent.argmax(1),D4=np.where(parent[None,:]==parent[truth,None],fine,-np.inf).argmax(1))


def main(root):
    cms={}; canonical={}; sources=[]
    for model in MODELS:
        for seed in ([17] if model=='lr' else SEEDS):
            blocks={}
            for fold in range(5):
                prefix=root/f'{model}_f{fold}'
                path=prefix/f'{"fine" if model in ["lr","xgb"] else "joint"}_s{seed}.npz'
                with np.load(path,allow_pickle=False) as b:
                    fine=b['prob']; truth=b['truth']; ids=b['patient_ids']; cells=b['cell_ids']
                    if fold not in canonical: canonical[fold]=(cells.copy(),truth.copy(),ids.copy(),b['classes'].copy())
                    for observed,expected in zip([cells,truth,ids,b['classes']],canonical[fold]): np.testing.assert_array_equal(observed,expected)
                    assert np.isfinite(fine).all() and np.allclose(fine.sum(1),1,atol=1e-5)
                    if model in ['lr','xgb']:
                        with np.load(prefix/f'coarse_s{seed}.npz',allow_pickle=False) as cb:
                            np.testing.assert_array_equal(cb['cell_ids'],cells)
                            coarse=cb['prob']; ct=cb['truth']
                        # Recover the fixed mapping from the complete paired OOF labels.
                        parent=np.array([np.unique(ct[truth==i])[0] if np.any(truth==i) else -1 for i in range(27)])
                        if (parent<0).any():
                            with np.load(root/f'mlp_f{fold}/joint_s17.npz',allow_pickle=False) as mb: parent=mb['parent']
                    else: coarse=b['coarse_prob']; ct=b['coarse_truth']; parent=b['parent']
                    assert np.array_equal(parent[truth],ct)
                    preds=decode(fine,coarse,parent,truth)
                    assert np.array_equal(preds['D4'][preds['D0']==truth],truth[preds['D0']==truth])
                    for name,pred in preds.items(): blocks.setdefault(name,[]).append(matrices(truth,pred,ids,27)[1])
                    blocks.setdefault('coarse',[]).append(matrices(ct,coarse.argmax(1),ids,5)[1])
                    sources.append(str(path))
            for name,values in blocks.items(): cms.setdefault((model,name),[]).append(np.concatenate(values))
    assert sum(len(v[0]) for v in canonical.values())==361792
    assert len(set(np.concatenate([v[0] for v in canonical.values()])))==361792
    rng=np.random.default_rng(20260915); pp=rng.integers(0,40,(5000,40)); ss=rng.integers(0,3,(5000,3))
    estimates={}; draws={}; rows=[]
    for key,blocks in cms.items():
        a=np.stack(blocks); values=np.array([score(cm) for cm in a]); estimates[key]=values.mean()
        draws[key]=np.array([np.mean([score(a[s if len(a)>1 else 0,p]) for s in seeds]) for p,seeds in zip(pp,ss)])
        rows.append(dict(model=key[0],decoding=key[1],mean=values.mean(),low=np.quantile(draws[key],.025),high=np.quantile(draws[key],.975),seed_values=values.tolist()))
    pairs=[('flat','mlp'),('cascade','flat'),('flat_contrast','flat'),('cascade_contrast','cascade')]
    contrasts=[]
    for a,b in pairs:
        delta=draws[(a,'D0')]-draws[(b,'D0')]
        contrasts.append(dict(comparison=a+' minus '+b,mean=estimates[(a,'D0')]-estimates[(b,'D0')],low=np.quantile(delta,.025),high=np.quantile(delta,.975)))
    for model in MODELS:
        for d in ['D1','D2','D4']:
            delta=draws[(model,d)]-draws[(model,'D0')]
            contrasts.append(dict(comparison=model+' '+d+' minus D0',mean=estimates[(model,d)]-estimates[(model,'D0')],low=np.quantile(delta,.025),high=np.quantile(delta,.975)))
    interaction=draws[('cascade_contrast','D0')]-draws[('cascade','D0')]-draws[('flat_contrast','D0')]+draws[('flat','D0')]
    contrasts.append(dict(comparison='contrastive interaction (D-C)-(B-A)',mean=estimates[('cascade_contrast','D0')]-estimates[('cascade','D0')]-estimates[('flat_contrast','D0')]+estimates[('flat','D0')],low=np.quantile(interaction,.025),high=np.quantile(interaction,.975)))
    for name,data in [('scores.csv',rows),('paired_contrasts.csv',contrasts)]:
        with (root/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]),lineterminator='\n'); writer.writeheader();writer.writerows(data)
    (root/'summary_validation.json').write_text(json.dumps(dict(status='PASS',source_predictions=sources,bootstrap=5000,scope='pointwise exploratory fixed-selection seed/patient intervals'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
