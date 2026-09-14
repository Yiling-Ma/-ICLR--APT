"""Fail-closed paired diagnostics for completed hierarchy-signal experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_existing import load, write_csv


def f1(cm,children):
    den=cm.sum(-1)+cm.sum(-2)
    value=np.divide(2*np.diagonal(cm,axis1=-2,axis2=-1),den,out=np.zeros_like(den),where=den>0)
    return value[...,children].mean(-1)


def cm_by_patient(y,pred,ids,scope,parent,patients,prior=None):
    cms=[]
    for patient in patients:
        ix=ids==patient
        if scope>=0: ix &= parent[y]==scope
        n=ix.sum(); assert n>0
        if prior is None:
            cm=np.bincount(y[ix]*27+pred[ix],minlength=729).reshape(27,27).astype(float)
        else:
            cm=np.zeros((27,27))
            for k in range(27):
                rows=ix&(y==k)
                cm[k]=prior[rows].sum(0)*(parent==parent[k])
        cms.append(cm/n)
    return np.stack(cms)


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    assert json.loads((a.results/f'{a.model}_source_audit.json').read_text())['status']=='PASS'
    blocks={}; priors=[]; sources=[]; canonical=None
    for seed in [17,29,43]:
        real,src=load(a.source,a.model,seed); sources+=src
        if canonical is None: canonical=real
        for k in ['truth','coarse_truth','cell_ids','patient_ids','classes','parent']:
            np.testing.assert_array_equal(real[k],canonical[k])
        blocks.setdefault('real',[]).append([real])
        shuffled=[]
        for rep in [101,211,307]:
            arrays=[]
            for fold in range(5):
                root=a.results/f'{a.model}_shuffle_r{rep}_f{fold}'
                path=root/f'joint_s{seed}.npz'
                with np.load(path,allow_pickle=False) as b: arrays.append({k:b[k].copy() for k in b.files})
                assert path.with_suffix('.json').exists()
                audit=json.loads((root/'audit.json').read_text()); assert audit['shuffle_seed']==rep
                with np.load(root/'partition_indices.npz',allow_pickle=False) as ix:
                    for stage in ['inner','validation','refit','test']:
                        np.testing.assert_array_equal(np.sort(ix[stage]),np.sort(ix[stage+'_source']))
                sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            keys=['prob','coarse_prob','truth','coarse_truth','cell_ids','patient_ids']
            b={k:np.concatenate([v[k] for v in arrays]) for k in keys}
            for k in ['truth','coarse_truth','cell_ids','patient_ids']: np.testing.assert_array_equal(b[k],real[k])
            for v in arrays:
                for k in ['parent','classes']: np.testing.assert_array_equal(v[k],real[k])
            for k in ['prob','coarse_prob']: assert np.isfinite(b[k]).all() and np.allclose(b[k].sum(1),1,atol=1e-5)
            b.update(parent=real['parent'],classes=real['classes']); shuffled.append(b)
            if seed==17 and rep==101:
                priors=np.concatenate([np.broadcast_to(v['prior'],(len(v['truth']),27)) for v in arrays])
        blocks.setdefault('shuffle',[]).append(shuffled)
        if a.model=='mlp':
            arrays=[]
            for fold in range(5):
                path=a.results/f'mlp_conditional_r0_f{fold}'/f'joint_s{seed}.npz'
                with np.load(path,allow_pickle=False) as b: arrays.append({k:b[k].copy() for k in b.files})
                assert path.with_suffix('.json').exists()
                sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            b={k:np.concatenate([v[k] for v in arrays]) for k in ['prob','coarse_prob','truth','coarse_truth','cell_ids','patient_ids']}
            for k in ['truth','coarse_truth','cell_ids','patient_ids']: np.testing.assert_array_equal(b[k],real[k])
            b.update(parent=real['parent'],classes=real['classes']); blocks.setdefault('conditional',[]).append([b])
    parent=canonical['parent']; y=canonical['truth']; ids=canonical['patient_ids']
    rows=[]; contrasts=[]; rng=np.random.default_rng(20260916)
    for scope in [-1,0,1,2,3,4]:
        children=np.arange(27) if scope==-1 else np.flatnonzero(parent==scope)
        patients=np.unique(ids if scope==-1 else ids[parent[y]==scope]); n=len(patients)
        matrices={}
        for condition,seeds in blocks.items():
            for decoding in (['ordinary','oracle','D2'] if condition=='real' else ['ordinary','oracle']):
                out=[]
                for replicas in seeds:
                    rr=[]
                    for b in replicas:
                        p=b['prob']; q=b['coarse_prob']
                        if decoding=='oracle': pred=np.where(parent[None,:]==parent[y,None],p,-np.inf).argmax(1)
                        elif decoding=='D2':
                            mass=np.stack([p[:,parent==m].sum(1) for m in range(5)],1)
                            pred=(p/np.maximum(mass[:,parent],1e-30)*q[:,parent]).argmax(1)
                        else: pred=p.argmax(1)
                        rr.append(cm_by_patient(y,pred,ids,scope,parent,patients))
                    out.append(rr)
                matrices[condition+'_'+decoding]=np.array(out)
        matrices['prior']=cm_by_patient(y,None,ids,scope,parent,patients,priors)[None,None]
        draws=rng.integers(0,n,(5000,n)); counts=np.stack([np.bincount(p,minlength=n) for p in draws])
        seedslots=rng.integers(0,3,(5000,3)); shuffleslots=rng.integers(0,3,(5000,3,3))
        means={}; intervals={}
        for name,cm in matrices.items():
            means[name]=float(f1(cm.sum(2),children).mean()); values=[]
            for start in range(0,5000,100):
                pooled=np.einsum('bp,srpkj->bsrkj',counts[start:start+100],cm,optimize=True)
                v=f1(pooled,children)
                for j,vs in enumerate(v):
                    k=start+j; take=seedslots[k] if len(cm)>1 else np.zeros(3,dtype=int)
                    values.append(float(np.mean([vs[s,shuffleslots[k,t]].mean() if cm.shape[1]>1 else vs[s,0] for t,s in enumerate(take)])))
            intervals[name]=np.array(values)
            rows.append(dict(model=a.model,scope=scope,eligible_patients=n,condition=name,mean=means[name],low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975))))
        pairs=[('cell_signal','real_oracle','shuffle_oracle'),('oracle_minus_prior','real_oracle','prior')]
        if a.model=='mlp': pairs += [('deploy_gain','conditional_ordinary','real_ordinary'),('conditional_oracle_gap','conditional_oracle','conditional_ordinary'),('conditional_oracle_gain','conditional_oracle','real_oracle')]
        for name,left,right in pairs:
            delta=intervals[left]-intervals[right]
            contrasts.append(dict(model=a.model,scope=scope,comparison=name,mean=means[left]-means[right],low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),eligible_patients=n))
    write_csv(a.output/'scores.csv',rows); write_csv(a.output/'contrasts.csv',contrasts)
    gate=next(r for r in contrasts if r['scope']==-1 and r['comparison']=='cell_signal')
    (a.output/'validation.json').write_text(json.dumps(dict(status='PASS',model=a.model,training_seeds=[17,29,43],shuffle_seeds=[101,211,307],bootstrap=5000,
        flat_replication_gate=bool(gate['low']>0),primary_cell_signal=gate,sources=sources),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--model',choices=['mlp','flat'],default='mlp'); main(p.parse_args())
