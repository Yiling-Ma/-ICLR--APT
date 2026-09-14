"""Matched-budget conditional-shuffle and conditional-subtype follow-up."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'unified_v2'))
import run as base


def conditional_logp(fine, parent):
    out=torch.empty_like(fine)
    for m in range(5):
        mask=parent==m
        out[:,mask]=fine[:,mask].log_softmax(1)
    return out


def permutation(ix, ids, coarse, seed, fold, stage):
    rng=np.random.default_rng(np.random.SeedSequence([seed,fold,stage]))
    source=ix.copy()
    for patient in sorted(set(ids[ix])):
        for lineage in range(5):
            pos=np.flatnonzero((ids[ix]==patient)&(coarse[ix]==lineage))
            source[pos]=rng.permutation(ix[pos])
    assert np.array_equal(np.sort(source),np.sort(ix))
    assert np.array_equal(ids[source],ids[ix])
    assert np.array_equal(coarse[source],coarse[ix])
    return source


def predict(model,x,parent,device,conditional):
    model.eval(); fp=[]; cp=[]
    pt=torch.as_tensor(parent,device=device)
    with torch.no_grad():
        for start in range(0,len(x),128):
            f,c,_=model(torch.as_tensor(x[start:start+128],device=device))
            q=c.softmax(1)
            p=conditional_logp(f,pt).exp()*q[:,pt] if conditional else f.softmax(1)
            fp.append(p.cpu().numpy()); cp.append(q.cpu().numpy())
    return np.concatenate(fp),np.concatenate(cp)


def train(x,y,c,parent,cfg,seed,epochs,device,conditional,backbone,val=None):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=base.Network(backbone,parent).to(device)
    pt=torch.as_tensor(parent,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['decay'])
    rng=np.random.default_rng(seed); best=-1.; best_ep=1; history=[]
    batch=1024 if backbone=='mlp' else 128
    for epoch in range(1,epochs+1):
        model.train(); order=rng.permutation(len(y))
        for start in range(0,len(y),batch):
            ix=order[start:start+batch]; by=torch.as_tensor(y[ix],device=device)
            f,co,_=model(torch.as_tensor(x[ix],device=device))
            fine_loss=(-conditional_logp(f,pt).gather(1,by[:,None]).mean()
                       if conditional else torch.nn.functional.cross_entropy(f,by))
            loss=fine_loss+.5*torch.nn.functional.cross_entropy(co,torch.as_tensor(c[ix],device=device))
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if val is not None:
            vx,vy,vid=val; p,_=predict(model,vx,parent,device,conditional)
            s=float(base.score(base.matrices(vy,p.argmax(1),vid,27)[1])); history.append(s)
            if s>best: best=s; best_ep=epoch
            print('SELECT',seed,epoch,s,flush=True)
            if epoch-best_ep>=8: break
        else: print('REFIT',seed,epoch,flush=True)
    return model,dict(score=best,epochs=best_ep,history=history)


def main(a):
    torch.set_num_threads(4)
    conditional=a.kind=='conditional'
    assert not conditional or (a.backbone=='mlp' and a.shuffle==0)
    assert conditional or a.shuffle in [101,211,307]
    root=a.output/f'{a.backbone}_{a.kind}_r{a.shuffle}_f{a.fold}'
    root.mkdir(parents=True,exist_ok=True)
    codehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol=Path(__file__).with_name('PROTOCOL.md').read_text()
    marker=root/'PROTOCOL.md'
    if marker.exists(): assert marker.read_text()==protocol
    else: marker.write_text(protocol)
    if (root/'audit.json').exists():
        assert json.loads((root/'audit.json').read_text())['code_sha256']==codehash
    meta,_,_=base.core.load_data(); x=base.raw.raw_apt(meta)
    enc=base.core.fit_label_encoders(meta); ids=meta.sample_id.to_numpy(str)
    y=enc['fine'].transform(meta.cell_subtype); c=enc['coarse'].transform(meta.coarse_subtype)
    parent=enc['coarse'].transform(meta[['cell_subtype','coarse_subtype']].drop_duplicates().set_index('cell_subtype').loc[enc['fine'].classes_,'coarse_subtype'])
    folds=base.core.load_folds(); f=a.fold
    te=np.flatnonzero(np.isin(ids,folds[f])); va=np.flatnonzero(np.isin(ids,folds[(f+1)%5]))
    tr=np.flatnonzero(~np.isin(ids,folds[f]+folds[(f+1)%5])); dev=np.sort(np.r_[tr,va])
    parts=dict(inner=tr,validation=va,refit=dev,test=te)
    assert meta.cell_id.is_unique and len(meta)==361792 and x.shape==(361792,293)
    assert [len(set(ids[ix])) for ix in parts.values()]==[24,8,32,8]
    assert not set(ids[dev])&set(ids[te])
    sources={s:permutation(ix,ids,c,a.shuffle,f,j) if a.shuffle else ix.copy()
             for j,(s,ix) in enumerate(parts.items())}
    np.savez_compressed(root/'partition_indices.npz',**parts,**{s+'_source':ix for s,ix in sources.items()})
    inner=StandardScaler().fit(x[sources['inner']]); outer=StandardScaler().fit(x[sources['refit']])
    np.savez_compressed(root/'scalers.npz',inner_mean=inner.mean_,inner_scale=inner.scale_,outer_mean=outer.mean_,outer_scale=outer.scale_)
    xs={s:(inner if s in ['inner','validation'] else outer).transform(x[ix]).astype(np.float32)
        for s,ix in sources.items()}
    audit=dict(kind=a.kind,backbone=a.backbone,fold=f,shuffle_seed=a.shuffle,
        training_seeds=base.SEEDS,selection_seed=17,code_sha256=codehash,
        base_code_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(protocol.encode()).hexdigest(),
        patients={s:sorted(set(ids[ix])) for s,ix in parts.items()},cells={s:len(ix) for s,ix in parts.items()},
        parameter_count=sum(p.numel() for p in base.Network(a.backbone,parent).parameters()),
        hardware=torch.cuda.get_device_name(0) if a.device=='cuda' else 'CPU')
    for path in [base.raw.ROOT/'data/ALL_PBMC_APT_Cell_matrix.npz',base.raw.ROOT/'data/cells.tsv']:
        audit[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    base.atomic_json(root/'audit.json',audit)
    sp=root/'selection.json'; started=time.time()
    if sp.exists(): selection=json.loads(sp.read_text())
    else:
        trials=[]
        for lr,decay in itertools.product([.0003,.001],[1e-5,1e-3]):
            cfg=dict(lr=lr,decay=decay)
            m,result=train(xs['inner'],y[tr],c[tr],parent,cfg,17,50,a.device,conditional,a.backbone,(xs['validation'],y[va],ids[va]))
            trials.append(dict(config=cfg,**result)); del m
        selection=dict(trials=trials,selected=max(trials,key=lambda t:t['score']),seconds=time.time()-started)
        base.atomic_json(sp,selection)
    best=selection['selected']; prior=np.bincount(y[dev],minlength=27).astype(float)
    for m in range(5): prior[parent==m]/=max(prior[parent==m].sum(),1)
    for seed in base.SEEDS:
        path=root/f'joint_s{seed}.npz'
        if path.exists() and path.with_suffix('.json').exists(): continue
        started=time.time()
        model,_=train(xs['refit'],y[dev],c[dev],parent,best['config'],seed,best['epochs'],a.device,conditional,a.backbone)
        p,q=predict(model,xs['test'],parent,a.device,conditional)
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        torch.save(dict(state=model.state_dict(),mean=outer.mean_,scale=outer.scale_,config=best,parent=parent,conditional=conditional),path.with_suffix('.pt'))
        np.savez_compressed(path.with_suffix('.tmp.npz'),prob=p,coarse_prob=q,truth=y[te],coarse_truth=c[te],parent=parent,
            cell_ids=meta.cell_id.iloc[te].to_numpy(str),patient_ids=ids[te],classes=enc['fine'].classes_.astype(str),prior=prior,
            source_cell_ids=meta.cell_id.iloc[sources['test']].to_numpy(str))
        os.replace(path.with_suffix('.tmp.npz'),path)
        base.atomic_json(path.with_suffix('.json'),dict(seed=seed,seconds=time.time()-started,selected=best,training_cells=len(dev),test_cells=len(te)))
        print('COMPLETE',path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=['shuffle','conditional'],required=True)
    p.add_argument('--backbone',choices=['mlp','flat'],default='mlp'); p.add_argument('--shuffle',type=int,default=0)
    p.add_argument('--fold',type=int,choices=range(5),required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda'); main(p.parse_args())
