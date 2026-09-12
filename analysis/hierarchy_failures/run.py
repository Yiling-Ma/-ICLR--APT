"""Aggregate verified saved full-budget MLP scores; emit no patient/cell IDs."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HOME=Path('/home/mayiling/projs/apt_agent')
SOURCE=Path('/ssd3/mayiling/apt_agent_runtime/remaining_v1/full_mlp')
sys.path.insert(0,str(HOME/'analysis'))
import patient_cell_scaling as core

SEEDS=[20260912,20260913,20260914]


def per_class(cm):
    den=cm.sum(0)+cm.sum(1)
    return np.divide(2*cm.diagonal(),den,out=np.zeros(len(cm)),where=den>0)


def main():
    meta,_,_=core.load_data(); enc=core.fit_label_encoders(meta); folds=core.load_folds()
    index=meta.set_index('cell_id'); classes=enc['fine'].classes_.astype(str)
    lineages=enc['coarse'].classes_.astype(str)
    mapping=meta[['cell_subtype','coarse_subtype']].drop_duplicates().set_index('cell_subtype')
    parent=enc['coarse'].transform(mapping.loc[classes,'coarse_subtype'])
    all_y=enc['fine'].transform(meta.cell_subtype)
    order=np.lexsort((np.arange(27),parent))
    summary=[]; class_rows=[]; matrices=[]; sources={}; invariants={}
    support=np.bincount(all_y,minlength=27)
    coverage=[int(meta.loc[all_y==k,'sample_id'].nunique()) for k in range(27)]
    for seed in SEEDS:
        blocks=[]; prior_cms=[]
        for f in range(5):
            path=SOURCE/f's{seed}_f{f}_fine.npz'
            sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path,allow_pickle=False) as a:
                assert np.array_equal(a['classes'],classes) and np.array_equal(a['parent'],parent)
                assert np.isfinite(a['prob']).all() and np.allclose(a['prob'].sum(1),1,atol=1e-5)
                assert len(set(a['cell_ids']))==len(a['cell_ids'])
                target=index.loc[a['cell_ids']]
                assert set(a['cell_ids'])==set(meta.loc[meta.sample_id.isin(folds[f]),'cell_id'])
                y=enc['fine'].transform(target.cell_subtype); ids=target.sample_id.to_numpy(str)
                assert np.array_equal(y,a['truth'])
                ordinary=a['prob'].argmax(1)
                oracle=np.where(parent[None,:]==parent[y,None],a['prob'],-np.inf).argmax(1)
                correct=ordinary==y; cross=parent[ordinary]!=parent[y]; within=~correct & ~cross
                assert np.all(correct.astype(int)+cross+within==1)
                assert np.array_equal(oracle[correct],ordinary[correct])
                assert np.array_equal(oracle[within],ordinary[within])
                assert np.all(parent[oracle]==parent[y])
                patients=np.unique(ids)
                for pred,key in [(ordinary,'cm'),(oracle,'oracle_cm')]:
                    cms=np.stack([np.bincount(27*y[ids==p]+pred[ids==p],minlength=729).reshape(27,27) for p in patients])
                    assert np.array_equal(cms,a[key])
                train=all_y[~meta.sample_id.isin(folds[f]).to_numpy()]
                train_counts=np.bincount(train,minlength=27).astype(float)
                for p in patients:
                    counts=np.bincount(y[ids==p],minlength=27).astype(float)
                    expected=np.zeros((27,27))
                    for lineage in range(5):
                        ix=np.flatnonzero(parent==lineage)
                        q=train_counts[ix]/train_counts[ix].sum() if train_counts[ix].sum() else np.ones(len(ix))/len(ix)
                        expected[np.ix_(ix,ix)]=counts[ix,None]*q[None,:]
                    prior_cms.append(expected/expected.sum())
                blocks.append((y,ordinary,oracle,ids))
        y,ordinary,oracle,ids=[np.concatenate([b[i] for b in blocks]) for i in range(4)]
        assert len(y)==361792 and len(set(ids))==40
        correct=ordinary==y; cross=parent[ordinary]!=parent[y]; within=~correct & ~cross
        repair=cross & (oracle==y)
        for scope in ['All',*lineages]:
            scope_mask=np.ones(len(y),bool) if scope=='All' else parent[y]==list(lineages).index(scope)
            rows=[]; conditional=[]
            for p in sorted(set(ids[scope_mask])):
                ix=(ids==p)&scope_mask; denom=ix.sum()
                rates=[float(v[ix].sum()/denom) for v in [correct,cross,within,repair,oracle==y]]
                assert np.isclose(sum(rates[:3]),1)
                assert np.isclose(rates[0]+rates[3],rates[4])
                rows.append(rates)
                if cross[ix].sum(): conditional.append(float(repair[ix].sum()/cross[ix].sum()))
            vals=np.mean(rows,axis=0)
            summary.append(dict(seed=seed,scope=scope,eligible_patients=len(rows),cells=int(scope_mask.sum()),
                correct=vals[0],cross_error=vals[1],within_error=vals[2],repair=vals[3],oracle_correct=vals[4],
                cross_repair_fraction=float(np.mean(conditional)) if conditional else None,
                cross_ratio_eligible_patients=len(conditional)))
        for mode,pred in [('ordinary',ordinary),('oracle',oracle)]:
            cm=np.zeros((27,27))
            for p in sorted(set(ids)):
                ix=ids==p
                cm+=np.bincount(27*y[ix]+pred[ix],minlength=729).reshape(27,27)/ix.sum()
            assert np.isclose(cm.sum(),40)
            if mode=='oracle': assert np.all(cm[parent[:,None]!=parent[None,:]]==0)
            f1=per_class(cm); display=cm[np.ix_(order,order)]
            den=display.sum(1,keepdims=True)
            row=np.divide(display,den,out=np.zeros_like(display),where=den>0)
            matrices.append(dict(seed=seed,mode=mode,sb_cm=display.tolist(),row_normalized=row.tolist(),
                zero_rows=np.flatnonzero(den[:,0]==0).tolist(),macro_f1=float(f1.mean())))
            for k in order:
                candidates=[j for j in order if j!=k]
                dest=max(candidates,key=lambda j:cm[k,j])
                class_rows.append(dict(seed=seed,mode=mode,subtype=classes[k],lineage=lineages[parent[k]],
                    cell_count=int(support[k]),patient_coverage=coverage[k],sb_f1=float(f1[k]),
                    largest_wrong_destination=classes[dest],wrong_destination_row_fraction=float(cm[k,dest]/cm[k].sum()),
                    wrong_destination_all_cell_mass=float(cm[k,dest]/40)))
        invariants[str(seed)]={'ordinary_correct_preserved':True,'within_error_prediction_unchanged':True,
            'oracle_has_no_cross_parent_predictions':True,'three_way_partition_sum_one':True,
            'ordinary_correct_plus_repair_equals_oracle_correct':True,'saved_confusion_exact':True,
            'prior_expected_sb_f1':float(per_class(np.sum(prior_cms,axis=0)).mean())}
    print(json.dumps(dict(model='full-budget Plain MLP',seeds=SEEDS,cells=361792,patients=40,
        display_classes=classes[order].tolist(),display_lineages=lineages[parent[order]].tolist(),
        summary=summary,per_class=class_rows,matrices=matrices,checks=invariants,sources=sources,
        new_training=0,new_inference=0,scope='descriptive post-hoc saved-score analysis'),indent=2))


if __name__=='__main__': main()
