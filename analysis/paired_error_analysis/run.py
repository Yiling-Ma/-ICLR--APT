"""Post-hoc paired error accounting and patient-stratified feature associations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SEEDS = [20260912, 20260913, 20260914]
B = 2000


def joint(a, seed=91822):
    """Mean of seed-specific eligible-patient means; shared patient bootstrap."""
    a = np.asarray(a, float)
    shape = a.shape[2:]
    a = a.reshape(a.shape[0], a.shape[1], -1)
    valid = np.isfinite(a)
    sums = np.nan_to_num(a)
    point = np.nanmean(np.divide(sums.sum(1), valid.sum(1),
                       out=np.full_like(sums.sum(1), np.nan), where=valid.sum(1)>0), axis=0)
    rng = np.random.default_rng(seed)
    pw = rng.multinomial(a.shape[1], np.ones(a.shape[1])/a.shape[1], B)
    sw = rng.multinomial(a.shape[0], np.ones(a.shape[0])/a.shape[0], B)
    num = np.einsum('bp,spf->bsf', pw, sums)
    den = np.einsum('bp,spf->bsf', pw, valid.astype(float))
    means = np.divide(num, den, out=np.full_like(num, np.nan), where=den>0)
    weights = sw[:,:,None] * np.isfinite(means)
    boots = np.divide((np.nan_to_num(means)*weights).sum(1), weights.sum(1),
                      out=np.full((B,a.shape[2]),np.nan), where=weights.sum(1)>0)
    lo, hi = np.nanquantile(boots,[.025,.975],axis=0)
    return [z.reshape(shape) for z in (point,lo,hi)]


def counts(p,y,pred,P,K,mask=None):
    if mask is None: mask = np.ones(len(y),bool)
    return np.bincount((p[mask]*K+y[mask])*K+pred[mask],
                       minlength=P*K*K).reshape(P,K,K)


def error_counts(p,y,r,a,P,K):
    rc = counts(p,y,r,P,K); ac = counts(p,y,a,P,K)
    rescued = counts(p,y,r,P,K,(r!=y)&(a==y))
    harmed = counts(p,y,a,P,K,(r==y)&(a!=y))
    changed = (r!=y)&(a!=y)&(r!=a)
    outgoing = counts(p,y,r,P,K,changed)
    incoming = counts(p,y,a,P,K,changed)
    off = ~np.eye(K,dtype=bool)
    assert np.array_equal((rc-ac)[:,off],(rescued-harmed+outgoing-incoming)[:,off])
    return dict(rna=rc, paired=ac, rescued=rescued, harmed=harmed,
                rerouted_out=outgoing, rerouted_in=incoming)


def load_predictions(base,meta,model,seed):
    ids = pd.Index(meta.cell_id)
    pred = {}; seen = np.zeros(len(meta),int); classes = None
    for fold in range(5):
        for mode in ['rna','rna_apt']:
            path = base/f's{seed}_f{fold}_fine_{model}_{mode}.npz'
            with np.load(path,allow_pickle=False) as z:
                cl=z['classes'].astype(str)
                if classes is None: classes=cl
                assert np.array_equal(classes,cl)
                ix=ids.get_indexer(z['cell_ids'].astype(str))
                assert np.all(ix>=0) and len(np.unique(ix))==len(ix)
                assert np.array_equal(classes[z['truth']],meta.cell_subtype.to_numpy()[ix])
                assert np.all((z['pred']>=0)&(z['pred']<len(classes)))
                if mode not in pred: pred[mode]=np.full(len(meta),-1,int)
                pred[mode][ix]=z['pred']
                if mode=='rna': seen[ix]+=1
    assert np.all(seen==1) and all(np.all(v>=0) for v in pred.values())
    return classes,pred['rna'],pred['rna_apt']


def feature_matrix(root,meta,private):
    frame=pd.read_csv(private/'markers_qc.csv')
    assert frame.cell_id.is_unique
    frame=frame.set_index('cell_id').loc[meta.cell_id]
    names=[]; arrays=[]
    for col in frame.columns:
        x=frame[col].to_numpy(float)
        name=col if col.startswith('RNA:') else 'QC:'+col
        if col in ['RNA_total','RNA_detected']:
            x=np.log1p(x); name='QC:log1p_'+col
        names.append(name); arrays.append(x)
    ids=pd.read_csv(root/'data/cells.tsv',sep='\t',header=None)[0].astype(str)
    assert ids.is_unique
    ix=pd.Index(ids).get_indexer(meta.cell_id); assert np.all(ix>=0)
    apt=sparse.load_npz(root/'data/ALL_PBMC_APT_Cell_matrix.npz')
    if apt.shape[0]!=len(ids): apt=apt.T
    apt=apt.tocsr()[ix].toarray().astype(np.float32)
    features=(root/'data/apt_features.txt').read_text().splitlines()
    assert apt.shape==(len(meta),293) and len(features)==293 and len(set(features))==293
    assert np.all(apt>=0) and np.isfinite(apt).all()
    total=apt.sum(1); detected=(apt>0).sum(1)
    names+=['QC:log1p_APT_total','QC:log1p_APT_detected']
    arrays += [np.log1p(total),np.log1p(detected)]
    x=np.column_stack(arrays).astype(np.float32)
    names += ['APT_raw:'+f for f in features]+['APT_normalized:'+f for f in features]
    x=np.column_stack([x,np.log1p(apt),np.log1p(apt/np.maximum(total[:,None],1)*1e4)])
    assert np.isfinite(x).all()
    ann=pd.read_csv(root/'data/cell_annotation.csv').rename(columns={'Sample':'cell_id'})
    assert ann.cell_id.is_unique
    ann=ann.set_index('cell_id').loc[meta.cell_id]
    audit={c:float(np.corrcoef(frame[c],ann[d])[0,1]) for c,d in
           [('RNA_total','nCount_RNA'),('RNA_detected','nFeature_RNA')]}
    np.save(private/'cell_features.npy',x)
    (private/'feature_names.json').write_text(json.dumps(names))
    meta.to_csv(private/'cell_index.csv',index=False)
    return x,names,audit


def stratified_features(x,p,y,r,a,P,K,kind):
    if kind=='rescue':
        g=(p*K+y)*K+r; G=P*K*K
        pos=(r!=y)&(a==y); neg=(r!=y)&(a!=y)
    else:
        g=p*K+y; G=P*K
        pos=(r==y)&(a!=y); neg=(r==y)&(a==y)
    mus=[]; ns=[]
    for mask in [pos,neg]:
        n=np.bincount(g[mask],minlength=G)
        indicator=sparse.csr_matrix((np.ones(mask.sum()),(g[mask],np.flatnonzero(mask))),
                                    shape=(G,len(p)))
        mus.append((indicator@x)/np.maximum(n[:,None],1)); ns.append(n)
    valid=(ns[0]>=5)&(ns[1]>=5)
    dif=mus[0]-mus[1]; dif[~valid]=np.nan
    shaped=dif.reshape(P,-1,x.shape[1])
    nvalid=np.isfinite(shaped[:,:,0]).sum(1)
    average=np.divide(np.nansum(shaped,axis=1),nvalid[:,None],
                      out=np.full((P,x.shape[1]),np.nan),where=nvalid[:,None]>0)
    audit=dict(positive_cells=int(pos.sum()),negative_cells=int(neg.sum()),
               eligible_strata=int(valid.sum()),candidate_strata=int(((ns[0]+ns[1])>0).sum()),
               eligible_patients=int((nvalid>0).sum()),
               retained_positive_cells=int(ns[0][valid].sum()),
               retained_negative_cells=int(ns[1][valid].sum()))
    return average,shaped,audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); public=args.out/'public'; private=args.out/'private'
    public.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(args.runtime/'remaining_v1/modality_hvg2000/prepared/cells.csv')
    assert meta.cell_id.is_unique
    p,patients=pd.factorize(meta.sample_id,sort=True); P=len(patients)
    x,names,qcaudit=feature_matrix(args.root,meta,private)
    all_edges=[]; summaries=[]; feature_rows=[]; exclusions=[]; audits=[]
    for width in [2000,5000]:
        base=args.runtime/f'remaining_v1/modality_hvg{width}'
        for model in ['sgd','mlp']:
            blocks=[]; feature_blocks={'rescue':[],'harm':[]}; edge_features=[]
            for seed in SEEDS:
                classes,r,a=load_predictions(base,meta,model,seed); K=len(classes)
                y=pd.Index(classes).get_indexer(meta.cell_subtype); assert np.all(y>=0)
                blocks.append(error_counts(p,y,r,a,P,K))
                np.savez_compressed(private/f'pred_{width}_{model}_{seed}.npz',
                    truth=y,rna=r,paired=a,classes=classes,
                    rescued=(r!=y)&(a==y),harmed=(r==y)&(a!=y))
                for k,label in enumerate(classes):
                    sel=y==k
                    summaries.append(dict(width=width,model=model,seed=seed,subtype=label,
                        cells=int(sel.sum()),patients=int(len(np.unique(p[sel]))),
                        rna_correct=int(((r==y)&sel).sum()),paired_correct=int(((a==y)&sel).sum()),
                        rescued=int(((r!=y)&(a==y)&sel).sum()),harmed=int(((r==y)&(a!=y)&sel).sum())))
                if model=='mlp':
                    for kind in ['rescue','harm']:
                        means,details,audit=stratified_features(x,p,y,r,a,P,K,kind)
                        feature_blocks[kind].append(means)
                        if kind=='rescue': edge_features.append(details.reshape(P,K,K,-1))
                        exclusions.append(dict(width=width,seed=seed,contrast=kind,**audit))
            arr={key:np.stack([v[key] for v in blocks]) for key in blocks[0]}
            denom=arr['rna'].sum(-1,keepdims=True)
            rates={key:np.divide(v,denom,out=np.full(v.shape,np.nan),where=denom>0)
                   for key,v in arr.items()}
            delta,lo,hi=joint(rates['rna']-rates['paired'])
            for i in range(K):
                for j in range(K):
                    row=dict(width=width,model=model,true_subtype=classes[i],predicted_subtype=classes[j],
                             delta_rna_minus_paired=delta[i,j],ci_low=lo[i,j],ci_high=hi[i,j],
                             true_class_patients=int((denom[0,:,i,0]>0).sum()))
                    for key in arr:
                        row[key+'_cells_mean']=float(arr[key][:,:,i,j].sum(1).mean())
                        row[key+'_patient_rate']=float(np.nanmean(np.nanmean(rates[key][:,:,i,j],axis=1)))
                    for s,seed in enumerate(SEEDS):
                        row[f'delta_seed_{seed}']=float(np.nanmean((rates['rna']-rates['paired'])[s,:,i,j]))
                    row['rescue_patients_union']=int((arr['rescued'][:,:,i,j].sum(0)>0).sum())
                    row['harm_patients_union']=int((arr['harmed'][:,:,i,j].sum(0)>0).sum())
                    all_edges.append(row)
            if model=='mlp':
                for kind,values in feature_blocks.items():
                    v=np.stack(values); point,l,h=joint(v)
                    for f,name in enumerate(names):
                        feature_rows.append(dict(width=width,contrast=kind,scope='all_eligible_strata',
                            true_subtype='',rna_wrong_subtype='',feature=name,difference=point[f],
                            ci_low=l[f],ci_high=h[f],eligible_patients_min=int(np.isfinite(v[:,:,f]).sum(1).min()),
                            eligible_patients_max=int(np.isfinite(v[:,:,f]).sum(1).max())))
                ef=np.stack(edge_features)
                for i in range(K):
                    for j in range(K):
                        v=ef[:,:,i,j,:]
                        n=np.isfinite(v[:,:,0]).sum(1)
                        if n.max()<5: continue
                        point,l,h=joint(v)
                        for f,name in enumerate(names):
                            feature_rows.append(dict(width=width,contrast='rescue',scope='supported_edge_posthoc',
                                true_subtype=classes[i],rna_wrong_subtype=classes[j],feature=name,
                                difference=point[f],ci_low=l[f],ci_high=h[f],
                                eligible_patients_min=int(n.min()),eligible_patients_max=int(n.max())))
            audits.append(dict(width=width,model=model,seeds=SEEDS,cells=len(meta),patients=P,
                               classes=K,alignment='passed',edge_conservation='passed'))
            print('Completed',width,model,flush=True)
    pd.DataFrame(all_edges).to_csv(public/'all_confusion_edges.csv',index=False)
    pd.DataFrame(summaries).to_csv(public/'subtype_seed_counts.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(public/'feature_associations.csv',index=False)
    pd.DataFrame(exclusions).to_csv(public/'feature_matching_coverage.csv',index=False)
    audit=dict(prediction_audits=audits,raw_vs_annotation_qc_correlations=qcaudit,
        features=len(names),feature_names=names,bootstrap_replicates=B,
        metadata_sha256=hashlib.sha256(pd.util.hash_pandas_object(meta,index=False).values.tobytes()).hexdigest(),
        private_location=str(private),missing_markers=(private/'missing_markers.txt').read_text().splitlines())
    (public/'audit.json').write_text(json.dumps(audit,indent=2))
    (public/'COMPLETE').write_text('Analysis completed; no model fitting or paper modifications.\n')


if __name__=='__main__': main()
