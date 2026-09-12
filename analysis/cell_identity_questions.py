"""Matched-budget hierarchy and conditional-pairing diagnostics; no outer tuning."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
import validate_paired_information as base
from run_modality_repeats import SEEDS, SHUFFLES
from run_full_budget_mlp import oracle


def permutation(ids, lineage, seed, conditional):
    rng=np.random.default_rng(seed)
    groups=pd.DataFrame(dict(patient=ids,lineage=lineage))
    keys=['patient','lineage'] if conditional else ['patient']
    order=np.arange(len(ids)); singletons=0
    for positions in groups.groupby(keys,sort=True).indices.values():
        order[positions]=rng.permutation(positions)
        singletons+=int(len(positions)==1)
    assert np.array_equal(np.sort(order),np.arange(len(ids)))
    assert np.array_equal(ids,ids[order])
    if conditional: assert np.array_equal(lineage,lineage[order])
    return order,dict(singletons=singletons,fixed_fraction=float(np.mean(order==np.arange(len(ids)))))


def prior_predictions(train_truth, truth, parent):
    counts=np.bincount(train_truth,minlength=len(parent))
    winners={p:np.flatnonzero(parent==p)[np.argmax(counts[parent==p])] for p in np.unique(parent)}
    return np.asarray([winners[p] for p in parent[truth]])


def margin(prob):
    top=np.sort(prob,axis=1)[:,-2:]
    return top[:,1]-top[:,0]


def probabilities(model,x,k):
    p=np.zeros((len(x) if not sparse.issparse(x) else x.shape[0],k),dtype=np.float64)
    p[:,model.classes_.astype(int)]=model.predict_proba(x)
    assert np.allclose(p.sum(1),1,atol=1e-5)
    return p


def scoped_cm(truth,pred,patients,mask,k):
    _,cm=base.cms(truth,pred,patients,k)
    # Keep zero-support patients and all columns, including outside-lineage errors.
    result=np.zeros_like(cm)
    for j,p in enumerate(np.unique(patients)):
        ii=mask & (patients==p)
        np.add.at(result[j],(truth[ii],pred[ii]),1)
    return result


def expected_prior_cm(train_truth,truth,parent,patients,mask):
    counts=np.bincount(train_truth,minlength=len(parent)).astype(float)
    q=np.zeros((len(parent),len(parent)))
    for p in np.unique(parent):
        child=np.flatnonzero(parent==p); mass=counts[child]
        mass=mass/mass.sum() if mass.sum()>0 else np.ones(len(child))/len(child)
        q[np.ix_(child,child)]=mass[None,:]
    result=[]
    for p in np.unique(patients):
        n=np.bincount(truth[mask & (patients==p)],minlength=len(parent))
        result.append(n[:,None]*q)
    return np.stack(result)


def scoped_score(cm,labels):
    sizes=cm.sum((1,2)); eligible=sizes>0
    if not eligible.any(): return np.nan
    pooled=(cm[eligible]/sizes[eligible,None,None]).sum(0)
    tp=np.diag(pooled); den=pooled.sum(0)+pooled.sum(1)
    f=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)
    return float(f[labels].mean())


def bootstrap_scores(cm,counts,labels):
    sizes=cm.sum((1,2)); eligible=sizes>0
    normalized=cm/np.maximum(sizes,1)[:,None,None]
    pooled=(counts@normalized.reshape(len(cm),-1)).reshape(len(counts),cm.shape[1],cm.shape[2])
    tp=np.diagonal(pooled,axis1=1,axis2=2); den=pooled.sum(1)+pooled.sum(2)
    f=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)[:,labels].mean(1)
    n=counts@eligible.astype(float)
    f[n==0]=np.nan
    acc=counts@(np.trace(cm,axis1=1,axis2=2)/np.maximum(sizes,1))
    acc=np.divide(acc,n,out=np.full_like(acc,np.nan),where=n>0)
    return {'sb_f1':f,'subject_accuracy':acc}


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    base.OUT=args.source/'prepared'
    while not (base.OUT/'cells.csv').exists(): time.sleep(60)
    meta=pd.read_csv(base.OUT/'cells.csv'); apt=base.raw_apt(meta)
    enc=base.core.fit_label_encoders(meta); k=27
    y=enc['fine'].transform(meta.cell_subtype.astype(str))
    lin=enc['coarse'].transform(meta.coarse_subtype.astype(str))
    mapping=meta[['cell_subtype','coarse_subtype']].drop_duplicates().set_index('cell_subtype')
    parent=enc['coarse'].transform(mapping.loc[enc['fine'].classes_,'coarse_subtype'])
    for seed,shuffle in zip(SEEDS,SHUFFLES):
        for fold in range(5):
            for name,grid in base.MODELS.items():
                dest=args.output/f's{seed}_f{fold}_{name}.npz'
                if dest.exists(): continue
                paths={m:args.source/f's{seed}_f{fold}_fine_{name}_{m}.npz' for m in base.MODES}
                while not all(p.exists() for p in paths.values()):
                    base.save_json(args.output/'waiting.json',dict(seed=seed,fold=fold,model=name,
                        missing=[str(p) for p in paths.values() if not p.exists()]))
                    time.sleep(60)
                base.SEED=shuffle
                inner=base.inputs(meta,apt,fold,'inner'); outer=base.inputs(meta,apt,fold,'outer')
                base.SEED=seed
                tr,te,xx=outer; it,iv,ix=inner
                patients=meta.sample_id.iloc[te].to_numpy(dtype=str)
                predictions={}; objects={}
                for mode,path in paths.items():
                    obj=np.load(path)
                    np.testing.assert_array_equal(obj['cell_ids'],meta.cell_id.iloc[te].to_numpy(dtype=str))
                    np.testing.assert_array_equal(obj['truth'],y[te])
                    np.testing.assert_array_equal(obj['classes'],enc['fine'].classes_.astype(str))
                    predictions[mode]=obj['pred']; objects[mode]=float(obj['alpha'])
                    obj.close()
                for mode in ['rna','apt']:
                    model=base.fit(name,objects[mode],xx[mode][0],y[tr],
                        base.patient_class_weights(meta.sample_id.iloc[tr].to_numpy(),y[tr]))
                    prob=probabilities(model,xx[mode][1],k)
                    np.testing.assert_array_equal(model.predict(xx[mode][1]),predictions[mode])
                    # SGD sigmoid probabilities can saturate and introduce argmax ties.
                    # Its native decision scores preserve the fitted classifier's order.
                    scores=prob
                    if name=='sgd':
                        scores=np.full_like(prob,-np.finfo(prob.dtype).max)
                        scores[:,model.classes_.astype(int)]=model.decision_function(xx[mode][1])
                    np.testing.assert_array_equal(scores.argmax(1),predictions[mode])
                    predictions[mode+'_oracle']=oracle(scores,y[te],parent)
                    if mode=='rna':
                        rna_margin=margin(prob)
                        im=base.fit(name,objects[mode],ix[mode][0],y[it],
                            base.patient_class_weights(meta.sample_id.iloc[it].to_numpy(),y[it]))
                        threshold=float(np.quantile(margin(probabilities(im,ix[mode][1],k)),.2))
                predictions['prior_oracle']=prior_predictions(y[tr],y[te],parent)
                audit=[]
                # The source patient shuffle is the first draw; add two independent draws.
                predictions['patient_shuffle_0']=predictions.pop('rna_shuffled_apt')
                for conditional in [False,True]:
                    kind='lineage_shuffle' if conditional else 'patient_shuffle'
                    for repeat in range(3):
                        mode=f'{kind}_{repeat}'
                        if mode in predictions: continue
                        cache=args.output/f's{seed}_f{fold}_{name}_{mode}.npz'
                        if cache.exists():
                            with np.load(cache) as cached: predictions[mode]=cached['pred']
                            continue
                        data=[]; pa=[]
                        for stage,(train,test,features) in enumerate([inner,outer]):
                            changed=[]
                            for side,ids in enumerate([train,test]):
                                a=features['apt'][side]
                                pos,info=permutation(meta.sample_id.iloc[ids].to_numpy(),lin[ids],
                                    shuffle+repeat*10000+fold*100+stage*10+side,conditional)
                                changed.append(sparse.hstack([features['rna'][side],sparse.csr_matrix(a[pos])],format='csr'))
                                pa.append(dict(stage=stage,side=side,**info))
                            data.append(changed)
                        scores=[]
                        for alpha in grid:
                            model=base.fit(name,alpha,data[0][0],y[it],
                                base.patient_class_weights(meta.sample_id.iloc[it].to_numpy(),y[it]))
                            _,cm=base.cms(y[iv],model.predict(data[0][1]),meta.sample_id.iloc[iv].to_numpy(),k)
                            scores.append(base.core.f1_from_confusion(base.core.patient_balanced_matrix(cm)))
                        alpha=grid[int(np.argmax(scores))]
                        model=base.fit(name,alpha,data[1][0],y[tr],
                            base.patient_class_weights(meta.sample_id.iloc[tr].to_numpy(),y[tr]))
                        pred=model.predict(data[1][1]); predictions[mode]=pred
                        np.savez_compressed(cache.with_suffix('.tmp.npz'),pred=pred,alpha=alpha,scores=scores)
                        os.replace(cache.with_suffix('.tmp.npz'),cache)
                        base.save_json(cache.with_suffix('.json'),dict(permutation_audit=pa))
                        print('CONTROL',cache.name,flush=True)
                scopes={'all':np.ones(len(te),bool),'rna_low_margin':rna_margin<=threshold,
                    'rna_error':predictions['rna']!=y[te], 'rna_correct':predictions['rna']==y[te]}
                scopes.update({str(label):lin[te]==j for j,label in enumerate(enc['coarse'].classes_)})
                arrays=dict(patients=np.unique(patients).astype(str),classes=enc['fine'].classes_.astype(str),
                    parent=parent,lineages=enc['coarse'].classes_.astype(str),threshold=threshold,
                    cell_ids=meta.cell_id.iloc[te].to_numpy(dtype=str),truth=y[te])
                for mode,pred in predictions.items():
                    arrays['pred__'+mode]=pred
                    for scope,mask in scopes.items():
                        arrays[scope+'__'+mode]=scoped_cm(y[te],pred,patients,mask,k)
                for scope,mask in scopes.items():
                    arrays[scope+'__prior_expected_cm']=expected_prior_cm(y[tr],y[te],parent,patients,mask)
                np.savez_compressed(dest.with_suffix('.tmp.npz'),**arrays)
                os.replace(dest.with_suffix('.tmp.npz'),dest)
                print('COMPLETE',dest.name,flush=True)


def aggregate(args):
    rng=np.random.default_rng(20260912); rows=[]; contrasts=[]
    counts=rng.multinomial(40,np.ones(40)/40,size=2000)
    seed_draws=rng.integers(0,3,(2000,3))
    shuffle_draws=rng.integers(0,3,(2000,3,3))
    for name in base.MODELS:
        blocks=[[np.load(args.output/f's{s}_f{f}_{name}.npz') for f in range(5)] for s in SEEDS]
        keys=[key for key in blocks[0][0].files if '__' in key and not key.startswith('pred__')]
        matrices={key:np.stack([np.concatenate([b[key] for b in seed]) for seed in blocks]) for key in keys}
        for seed in blocks:
            ids=np.concatenate([b['patients'] for b in seed]); cells=np.concatenate([b['cell_ids'] for b in seed])
            assert len(set(ids))==len(ids)==40 and len(set(cells))==len(cells)==361792
            np.testing.assert_array_equal(ids,np.concatenate([b['patients'] for b in blocks[0]]))
        lineages=blocks[0][0]['lineages']; parent=blocks[0][0]['parent']
        scopes=sorted(set(key.split('__')[0] for key in keys))
        def labels(scope):
            return np.flatnonzero(parent==list(lineages).index(scope)) if scope in lineages else np.arange(27)
        for scope in scopes:
            targets=labels(scope)
            boot={}
            for key,a in matrices.items():
                if not key.startswith(scope+'__'): continue
                boot[key]=[bootstrap_scores(cm,counts,targets) for cm in a]
                val=[scoped_score(cm,targets) for cm in a]
                rows.append(dict(model=name,scope=scope,method=key.split('__')[1],mean=np.mean(val),
                    seed_values=val,patient_counts=[int((cm.sum((1,2))>0).sum()) for cm in a],
                    cell_counts=[int(round(cm.sum())) for cm in a]))
            for high,low in [('rna_apt','rna'),('rna_apt','rna_noise'),('rna_apt','patient_shuffle'),
                ('rna_apt','lineage_shuffle'),('apt_oracle','prior_oracle'),('rna_oracle','prior_oracle'),
                ('apt_oracle','prior_expected_cm'),('rna_oracle','prior_expected_cm')]:
                h=matrices[scope+'__'+high]
                l=(np.stack([matrices[scope+'__'+low+'_'+str(r)] for r in range(3)],axis=1)
                   if low.endswith('shuffle') else matrices[scope+'__'+low][:,None])
                def accuracy(cm):
                    n=cm.sum((1,2)); valid=n>0
                    return float(np.mean(np.trace(cm[valid],axis1=1,axis2=2)/n[valid])) if valid.any() else np.nan
                for metric in ['sb_f1','subject_accuracy']:
                    scorer=(lambda cm:scoped_score(cm,targets)) if metric=='sb_f1' else accuracy
                    values=[scorer(h[s])-np.mean([scorer(c) for c in l[s]]) for s in range(3)]
                    hb=np.stack([b[metric] for b in boot[scope+'__'+high]])
                    low_keys=([scope+'__'+low+'_'+str(r) for r in range(3)]
                              if low.endswith('shuffle') else [scope+'__'+low])
                    lb=np.stack([np.stack([b[metric] for b in boot[key]]) for key in low_keys],axis=1)
                    d=np.arange(2000); samples=np.zeros(2000)
                    for slot in range(3):
                        ss=seed_draws[:,slot]
                        null=np.mean([lb[ss,shuffle_draws[:,slot,r] if len(low_keys)==3 else 0,d]
                                      for r in range(len(low_keys))],axis=0)
                        samples+=(hb[ss,d]-null)/3
                    finite=samples[np.isfinite(samples)]
                    contrasts.append(dict(model=name,scope=scope,metric=metric,comparison=high+'-minus-'+low,
                        mean=np.mean(values),seed_values=values,valid_bootstrap=len(finite),
                        low=np.quantile(finite,.025) if len(finite) else np.nan,
                        high=np.quantile(finite,.975) if len(finite) else np.nan))
        for seed in blocks:
            for b in seed: b.close()
    pd.DataFrame(rows).to_csv(args.output/'scope_scores.csv',index=False)
    pd.DataFrame(contrasts).to_csv(args.output/'paired_contrasts.csv',index=False)
    base.save_json(args.output/'completion.json',dict(status='PASS',blocks=30,training_seeds=SEEDS,
        shuffle_draws_per_seed=3,scope='exploratory; true lineage controls are privileged'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','aggregate'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=4): globals()[args.command](args)
