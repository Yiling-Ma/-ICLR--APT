"""Complete v2 failure diagnostics, without model inference or training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)


def per_class(cm):
    den=cm.sum(0)+cm.sum(1)
    return np.divide(2*cm.diagonal(),den,out=np.zeros(len(cm)),where=den>0)


def load(root,model,seed):
    arrays=[]; sources=[]
    for fold in range(5):
        path=root/f'{model}_f{fold}'/f'joint_s{seed}.npz'
        with np.load(path,allow_pickle=False) as b: arrays.append({k:b[k].copy() for k in b.files})
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for a in arrays[1:]:
        for k in ['classes','parent']: np.testing.assert_array_equal(a[k],arrays[0][k])
    keys=['prob','coarse_prob','truth','coarse_truth','cell_ids','patient_ids']
    result={k:np.concatenate([a[k] for a in arrays]) for k in keys}
    result.update({k:arrays[0][k] for k in ['parent','classes']})
    assert len(result['truth'])==361792 and len(np.unique(result['cell_ids']))==361792
    assert len(np.unique(result['patient_ids']))==40
    np.testing.assert_array_equal(result['parent'][result['truth']],result['coarse_truth'])
    for k in ['prob','coarse_prob']:
        assert np.isfinite(result[k]).all() and np.allclose(result[k].sum(1),1,atol=1e-5)
    return result,sources


def main(a):
    a.output.mkdir(parents=True,exist_ok=True)
    assert json.loads((a.source/'hce_decoding_correction.json').read_text())['status']=='PASS'
    models=['mlp','flat','hce','cascade']; records=[]; subtype=[]; sources=[]; canonical=None
    summaries=[]; pairs=[]; all_cm={}; score_rows=[]
    for model in models:
        cm_seeds=[]
        for seed in [17,29,43]:
            b,src=load(a.source,model,seed); sources+=src
            if canonical is None: canonical=b
            for k in ['truth','coarse_truth','cell_ids','patient_ids','parent','classes']:
                np.testing.assert_array_equal(b[k],canonical[k])
            y=b['truth']; parent=b['parent']; ids=b['patient_ids']; p=b['prob']; q=b['coarse_prob']
            ordinary=p.argmax(1); oracle=np.where(parent[None,:]==parent[y,None],p,-np.inf).argmax(1)
            correct=ordinary==y; cross=parent[ordinary]!=parent[y]; within=~correct&~cross
            assert np.all(correct.astype(int)+cross+within==1)
            np.testing.assert_array_equal(oracle[~cross],ordinary[~cross])
            assert np.all(parent[oracle]==parent[y])
            coarse_ok=q.argmax(1)==parent[y]; fixed=cross&(oracle==y)
            patient_cms=[]
            for patient in sorted(set(ids)):
                ix=ids==patient
                patient_cms.append(np.stack([np.bincount(y[ix]*27+pred[ix],minlength=729).reshape(27,27)/ix.sum() for pred in [ordinary,oracle]]))
            cms=np.stack(patient_cms); cm_seeds.append(cms)
            for scope in [-1,0,1,2,3,4]:
                scope_mask=np.ones(len(y),bool) if scope==-1 else parent[y]==scope
                rr=[]
                for patient in sorted(set(ids[scope_mask])):
                    ix=(ids==patient)&scope_mask; n=ix.sum()
                    row=dict(model=model,seed=seed,scope=scope,patient=patient,cells=int(n),
                        correct=float(correct[ix].mean()),cross=float(cross[ix].mean()),within=float(within[ix].mean()),
                        oracle_correct=float((oracle[ix]==y[ix]).mean()),fixed_mass=float(fixed[ix].mean()))
                    for name,num,den in [('cross_among_errors',cross,~correct),('repair_among_cross',fixed,cross),
                                         ('fine_cross_given_coarse_correct',cross,coarse_ok),('fine_cross_given_coarse_wrong',cross,~coarse_ok)]:
                        count=int((ix&den).sum()); row[name]=float(num[ix&den].mean()) if count else None; row[name+'_n']=count
                    assert np.isclose(row['correct']+row['cross']+row['within'],1)
                    assert np.isclose(row['correct']+row['fixed_mass'],row['oracle_correct'])
                    records.append(row); rr.append(row)
                for metric in ['correct','cross','within','oracle_correct','fixed_mass','cross_among_errors','repair_among_cross','fine_cross_given_coarse_correct','fine_cross_given_coarse_wrong']:
                    values=[r[metric] for r in rr if r[metric] is not None]
                    summaries.append(dict(model=model,seed=seed,scope=scope,metric=metric,mean=float(np.mean(values)) if values else None,eligible_patients=len(values)))
            aggregate=cms.sum(0)
            for d,cm in zip(['ordinary','oracle'],aggregate):
                f1=per_class(cm); score_rows.append(dict(model=model,seed=seed,decoding=d,sb_f1=float(f1.mean())))
                for k,label in enumerate(b['classes']):
                    row=cm[k].copy(); row[k]=0; target=int(row.argmax())
                    subtype.append(dict(model=model,seed=seed,decoding=d,subtype=str(label),parent=int(parent[k]),cells=int((y==k).sum()),
                        patient_coverage=len(np.unique(ids[y==k])),sb_per_class_f1=float(f1[k]),largest_error_target=str(b['classes'][target]) if row.sum() else '',largest_error_mass=float(row[target])))
        matrices=np.stack(cm_seeds); all_cm[model]=matrices
        average=matrices.sum(1).mean(0); den=average.sum(2,keepdims=True)
        normalized=np.divide(average,den,out=np.zeros_like(average),where=den>0)
        np.savez_compressed(a.output/f'{model}_confusions.npz',patient_matrices=matrices,seed_mean_pooled=average,row_normalized=normalized,
            zero_rows=den[:,:,0]==0,seeds=[17,29,43],patients=sorted(set(ids)),classes=b['classes'],parent=parent)
        rank=average[1].copy(); np.fill_diagonal(rank,0)
        order=np.argsort(-rank.ravel(),kind='stable')
        for position,idx in enumerate(order):
            i,j=divmod(int(idx),27)
            if i==j or rank[i,j]==0: continue
            seedmass=matrices[:,:,1,i,j].sum(1)
            pairs.append(dict(model=model,rank=position+1,true_subtype=str(b['classes'][i]),predicted_subtype=str(b['classes'][j]),mass=float(rank[i,j]),
                per_seed_mass=json.dumps(seedmass.tolist()),true_cells=int((y==i).sum()),true_patients=len(np.unique(ids[y==i]))))
    for name,rows in [('patient_errors',records),('error_summary',summaries),('subtype_results',subtype),('oracle_pairs',pairs),('scores',score_rows)]:
        write_csv(a.output/f'{name}.csv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=np.argsort(canonical['parent'],kind='stable'); names=canonical['classes'][order]
    boundaries=np.flatnonzero(np.diff(canonical['parent'][order]))+.5
    for model,cms in all_cm.items():
        pooled=cms.sum(1).mean(0); denom=pooled.sum(2,keepdims=True)
        norm=np.divide(pooled,denom,out=np.zeros_like(pooled),where=denom>0)
        fig,axes=plt.subplots(1,2,figsize=(16,8),layout='constrained')
        for ax,data,title in zip(axes,norm,['Ordinary','True-lineage oracle (privileged)']):
            im=ax.imshow(data[np.ix_(order,order)],vmin=0,vmax=1,cmap='Blues')
            ax.set_xticks(range(27),names,rotation=90,fontsize=7); ax.set_yticks(range(27),names,fontsize=7)
            for v in boundaries: ax.axhline(v,color='gray',lw=.5); ax.axvline(v,color='gray',lw=.5)
            ax.set(title=title,xlabel='Predicted subtype',ylabel='True subtype')
        fig.colorbar(im,ax=axes,shrink=.6,label='Row-normalized seed-mean SB confusion')
        fig.suptitle(f'Unified v2 {model}: all 27 subtypes, seeds 17/29/43')
        fig.savefig(a.output/f'{model}_confusions.pdf'); plt.close(fig)
    # Primary compact panel; right-hand F1 means come from the same v2 fits.
    old=list(csv.DictReader((a.source/'scores.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    bottom=np.zeros(4)
    for metric,color in [('correct','#357A85'),('cross','#D18A57'),('within','#AAB5BD')]:
        values=np.array([np.mean([r['mean'] for r in summaries if r['model']==m and r['scope']==-1 and r['metric']==metric]) for m in models])
        axes[0].bar(models,values,bottom=bottom,label=metric,color=color); bottom+=values
    axes[0].set(ylabel='Patient-balanced cell fraction',ylim=(0,1)); axes[0].legend(fontsize=8)
    for d,label in [('D0','Ordinary'),('D1','Predicted-parent mask'),('D4','True-parent oracle')]:
        values=[float(next(r['mean'] for r in old if r['model']==m and r['decoding']==d)) for m in models]
        axes[1].plot(models,values,'o-',label=label)
    axes[1].set(ylabel='Fine SB-Macro-F1'); axes[1].legend(fontsize=8)
    fig.savefig(a.output/'failure_overview.pdf'); plt.close(fig)
    (a.output/'validation.json').write_text(json.dumps(dict(status='PASS',cells=361792,patients=40,models=models,seeds=[17,29,43],new_training=0,sources=sources,
        aggregation='Patient fractions first; conditional denominators omit zero-support patients. Heatmaps are seed-mean summaries, not seed-pooled F1.'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True); main(p.parse_args())
