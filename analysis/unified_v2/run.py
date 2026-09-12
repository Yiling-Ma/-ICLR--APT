"""Frozen v2 reference selection/refit, with explicit keyed prediction artifacts."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

HOME = Path('/home/mayiling/projs/apt_agent')
sys.path[:0] = [str(HOME/'analysis'), str(HOME/'cell_JEPA')]
import patient_cell_scaling as core
import validate_paired_information as raw
from run_full_budget_mlp import matrices, score, atomic_json
from apt_jepa.models.soft_lineage_cascade import SoftLineageCascade
from apt_jepa.losses.hierarchical_cross_entropy import build_two_level_reachability, hierarchical_nll

SEEDS = [17,29,43]
MODELS = ['lr','xgb','mlp','flat','hce','cascade','flat_contrast','cascade_contrast']


def contrast(z, y, d):
    z = torch.nn.functional.normalize(z, dim=1)
    sim = z @ z.T / .1
    mask = ~torch.eye(len(y),dtype=torch.bool,device=z.device)
    pos = (y[:,None] == y[None,:]) & (d[:,None] != d[None,:]) & mask
    valid = pos.any(1)
    if not valid.any(): return z.sum()*0
    logp = sim - sim.masked_fill(~mask,-torch.inf).logsumexp(1,keepdim=True)
    return -(logp.masked_fill(~pos,0).sum(1)/pos.sum(1).clamp_min(1))[valid].mean()


class Network(torch.nn.Module):
    def __init__(self, name, parent):
        super().__init__(); self.name=name
        self.register_buffer('reach',build_two_level_reachability(parent,5,27))
        if name=='mlp':
            self.encoder=torch.nn.Sequential(torch.nn.Linear(293,256),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(256,128),torch.nn.ReLU(),torch.nn.Dropout(.1))
            self.fine=torch.nn.Linear(128,27); self.coarse=torch.nn.Linear(128,5)
        else:
            self.net=SoftLineageCascade(parent,5,hidden_dim=128,n_layers=2,n_heads=4,ff_dim=256)

    def forward(self,x):
        if self.name=='mlp':
            z=self.encoder(x); return self.fine(z),self.coarse(z),z
        if self.name.startswith('cascade'):
            o=self.net(x); return o['fine_logits'],o['coarse_logits'],o['projection']
        z=self.net.encode(x)
        return self.net.global_fine_head(z),self.net.coarse_head(z),self.net.projection_head(z)


def predict(model,x,device):
    model.eval(); fp=[]; cp=[]
    with torch.no_grad():
        for start in range(0,len(x),128):
            f,c,_=model(torch.as_tensor(x[start:start+128],device=device))
            fp.append(f.softmax(1).cpu().numpy()); cp.append(c.softmax(1).cpu().numpy())
    return np.concatenate(fp),np.concatenate(cp)


def train(name,x,y,c,d,parent,cfg,seed,epochs,device,val=None):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=Network(name,parent).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['decay'])
    rng=np.random.default_rng(seed); best=-1.; best_ep=1; history=[]
    batch=1024 if name=='mlp' else 128
    for epoch in range(1,epochs+1):
        model.train(); order=rng.permutation(len(y))
        for start in range(0,len(y),batch):
            ix=order[start:start+batch]
            bx=torch.as_tensor(x[ix],device=device)
            by=torch.as_tensor(y[ix],device=device); bc=torch.as_tensor(c[ix],device=device)
            f,co,z=model(bx)
            if name=='hce':
                nodes=torch.cat([co,f],1)
                loss=hierarchical_nll(nodes,by+5,model.reach)+.5*hierarchical_nll(nodes,bc,model.reach)
            else: loss=torch.nn.functional.cross_entropy(f,by)+.5*torch.nn.functional.cross_entropy(co,bc)
            if name.endswith('contrast'): loss=loss+.1*contrast(z,by,torch.as_tensor(d[ix],device=device))
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if val is not None:
            vx,vy,vid=val; p,_=predict(model,vx,device)
            s=score(matrices(vy,p.argmax(1),vid,27)[1]); history.append(float(s))
            if s>best: best=float(s); best_ep=epoch
            print(name,seed,epoch,'validation',s,flush=True)
            if epoch-best_ep>=8: break
        else: print(name,seed,epoch,'refit',flush=True)
    return model,dict(score=best,epochs=best_ep,history=history)


def aligned_prob(model,x,k):
    p=np.zeros((len(x),k)); p[:,model.classes_.astype(int)]=model.predict_proba(x)
    return p


def run(a):
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=Path(__file__).with_name('PROTOCOL.md').read_text()
    marker=a.output/'PROTOCOL.md'
    if marker.exists(): assert marker.read_text()==protocol
    else: marker.write_text(protocol)
    meta,_,_=core.load_data(); x=raw.raw_apt(meta)
    assert len(meta)==361792 and meta.cell_id.is_unique and x.shape==(361792,293)
    enc=core.fit_label_encoders(meta); ids=meta.sample_id.to_numpy(str)
    y=enc['fine'].transform(meta.cell_subtype); c=enc['coarse'].transform(meta.coarse_subtype)
    disease=meta.disease.astype('category').cat.codes.to_numpy(dtype=np.int64)
    mapping=meta[['cell_subtype','coarse_subtype']].drop_duplicates().set_index('cell_subtype')
    assert mapping.index.is_unique
    parent=enc['coarse'].transform(mapping.loc[enc['fine'].classes_,'coarse_subtype'])
    folds=core.load_folds(); f=a.fold
    te=np.flatnonzero(np.isin(ids,folds[f])); va=np.flatnonzero(np.isin(ids,folds[(f+1)%5]))
    tr=np.flatnonzero(~np.isin(ids,folds[f]+folds[(f+1)%5])); dev=np.sort(np.r_[tr,va])
    assert len(set(ids[tr]))==24 and len(set(ids[va]))==8 and len(set(ids[te]))==8
    assert not set(ids[dev])&set(ids[te])
    root=a.output/f'{a.model}_f{f}'; root.mkdir(exist_ok=True)
    audit=dict(model=a.model,fold=f,source='raw counts log1p',feature_order='source matrix columns 0..292',
        patients={s:sorted(set(ids[ix])) for s,ix in [('inner',tr),('validation',va),('refit',dev),('test',te)]},
        cells={s:len(ix) for s,ix in [('inner',tr),('validation',va),('refit',dev),('test',te)]},
        protocol_sha256=hashlib.sha256(protocol.encode()).hexdigest(),device=a.device,
        hardware=torch.cuda.get_device_name(0) if a.device=='cuda' else 'CPU',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for path in [raw.ROOT/'data/ALL_PBMC_APT_Cell_matrix.npz',raw.ROOT/'data/cells.tsv']:
        audit[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(root/'audit.json',audit)
    inner=StandardScaler().fit(x[tr]); outer=StandardScaler().fit(x[dev])
    np.savez_compressed(root/'scalers.npz',inner_mean=inner.mean_,inner_scale=inner.scale_,outer_mean=outer.mean_,outer_scale=outer.scale_)
    xt,xv,xd,xe=[s.transform(x[ix]).astype(np.float32) for s,ix in [(inner,tr),(inner,va),(outer,dev),(outer,te)]]
    tasks=['fine','coarse'] if a.model in ['lr','xgb'] else ['joint']
    for task in tasks:
        labels=c if task=='coarse' else y; k=5 if task=='coarse' else 27
        selected_path=root/f'{task}_selection.json'
        started=time.time()
        if selected_path.exists(): selection=json.loads(selected_path.read_text())
        else:
            trials=[]
            if a.model in ['lr','xgb']:
                grid=[dict(C=v) for v in [.01,.1,1,10]] if a.model=='lr' else [dict(max_depth=d,learning_rate=l,n_estimators=n) for d,l,n in itertools.product([3,6],[.03,.1],[100,300,500])]
                for cfg in grid:
                    m=classical(a.model,cfg,17); m.fit(xt,labels[tr]); p=aligned_prob(m,xv,k)
                    trials.append(dict(config=cfg,score=float(score(matrices(labels[va],p.argmax(1),ids[va],k)[1]))))
            else:
                for lr,decay in itertools.product([.0003,.001],[1e-5,1e-3]):
                    cfg=dict(lr=lr,decay=decay)
                    m,result=train(a.model,xt,y[tr],c[tr],disease[tr],parent,cfg,17,50,a.device,(xv,y[va],ids[va]))
                    trials.append(dict(config=cfg,**result)); del m
            selection=dict(trials=trials,selected=max(trials,key=lambda t:t['score']),seconds=time.time()-started)
            atomic_json(selected_path,selection)
        best=selection['selected']
        for seed in ([17] if a.model=='lr' else SEEDS):
            path=root/f'{task}_s{seed}.npz'
            if path.exists(): continue
            started=time.time()
            if a.model in ['lr','xgb']:
                import joblib
                m=classical(a.model,best['config'],seed); m.fit(xd,labels[dev]); p=aligned_prob(m,xe,k)
                extra={}; joblib.dump(dict(model=m,scaler=outer),path.with_suffix('.joblib'))
            else:
                m,_=train(a.model,xd,y[dev],c[dev],disease[dev],parent,best['config'],seed,best['epochs'],a.device)
                p,cp=predict(m,xe,a.device); extra=dict(coarse_prob=cp,coarse_truth=c[te],parent=parent)
                torch.save(dict(state=m.state_dict(),mean=outer.mean_,scale=outer.scale_,config=best,parent=parent),path.with_suffix('.pt'))
            assert p.shape==(len(te),k) and np.allclose(p.sum(1),1,atol=1e-5)
            np.savez_compressed(path.with_suffix('.tmp.npz'),prob=p,truth=labels[te],cell_ids=meta.cell_id.iloc[te].to_numpy(str),patient_ids=ids[te],classes=enc['coarse' if task=='coarse' else 'fine'].classes_.astype(str),**extra)
            os.replace(path.with_suffix('.tmp.npz'),path)
            atomic_json(path.with_suffix('.json'),dict(seed=seed,task=task,seconds=time.time()-started,selected=best,training_cells=len(dev),test_cells=len(te)))
            print('COMPLETE',path,flush=True)


def classical(name,cfg,seed):
    if name=='lr': return LogisticRegression(**cfg,max_iter=1000,solver='lbfgs')
    return XGBClassifier(**cfg,tree_method='hist',n_jobs=4,random_state=seed,subsample=.8,colsample_bytree=.8)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--fold',type=int,choices=range(5),required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda'); a=p.parse_args(); torch.set_num_threads(4); run(a)
