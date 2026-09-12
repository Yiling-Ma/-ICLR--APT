"""All-lineage structure for completed full-cell MLP, including training priors."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import patient_cell_scaling as core
from run_full_budget_mlp import SEEDS,oracle
from cell_identity_questions import scoped_cm,scoped_score,expected_prior_cm,prior_predictions,bootstrap_scores

def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    meta,_,_=core.load_data(); enc=core.fit_label_encoders(meta); folds=core.load_folds()
    y=enc['fine'].transform(meta.cell_subtype.astype(str))
    mapping=meta[['cell_subtype','coarse_subtype']].drop_duplicates().set_index('cell_subtype')
    parent=enc['coarse'].transform(mapping.loc[enc['fine'].classes_,'coarse_subtype'])
    index=pd.Index(meta.cell_id.astype(str)); lineages=enc['coarse'].classes_.astype(str)
    stores={}; reference_ids=None
    for s in SEEDS:
        blocks={}; pblocks=[]
        for f in range(5):
            with np.load(args.source/f's{s}_f{f}_fine.npz') as b:
                pos=index.get_indexer(b['cell_ids']); assert (pos>=0).all()
                truth=y[pos]; np.testing.assert_array_equal(truth,b['truth'])
                ids=meta.sample_id.iloc[pos].to_numpy(dtype=str)
                assert sorted(set(ids))==sorted(folds[f]); pblocks.extend(np.unique(ids))
                train=y[~meta.sample_id.isin(folds[f]).to_numpy()]
                predictions={'APT':b['prob'].argmax(1),'APT_oracle':oracle(b['prob'],truth,parent),
                    'prior_MAP':prior_predictions(train,truth,parent)}
                for scope in ['all',*lineages]:
                    mask=np.ones(len(pos),bool) if scope=='all' else parent[truth]==list(lineages).index(scope)
                    for mode,pred in predictions.items():
                        blocks.setdefault((scope,mode),[]).append(scoped_cm(truth,pred,ids,mask,27))
                    blocks.setdefault((scope,'prior_expected'),[]).append(expected_prior_cm(train,truth,parent,ids,mask))
        if reference_ids is None: reference_ids=pblocks
        assert pblocks==reference_ids and len(set(pblocks))==40
        for key,value in blocks.items(): stores.setdefault(key,[]).append(np.concatenate(value))
    rng=np.random.default_rng(20260912); counts=rng.multinomial(40,np.ones(40)/40,2000)
    ss=rng.integers(0,3,(2000,3)); d=np.arange(2000)
    rows=[]; contrasts=[]; boots={}
    for (scope,mode),values in stores.items():
        labels=np.arange(27) if scope=='all' else np.flatnonzero(parent==list(lineages).index(scope))
        point=[scoped_score(cm,labels) for cm in values]
        samples=np.stack([bootstrap_scores(cm,counts,labels)['sb_f1'] for cm in values])
        boot=np.mean([samples[ss[:,j],d] for j in range(3)],axis=0)
        boots[(scope,mode)]=boot
        finite=boot[np.isfinite(boot)]
        rows.append(dict(lineage=scope,method=mode,estimate=np.mean(point),
            low=np.quantile(finite,.025),high=np.quantile(finite,.975),
            classes=len(labels),patients=int((values[0].sum((1,2))>0).sum()),cells=int(round(values[0].sum())),seed_values=point))
    frame=pd.DataFrame(rows); frame.to_csv(args.output/'lineage_scores.csv',index=False)
    for scope in ['all',*lineages]:
        for ref in ['APT','prior_MAP','prior_expected']:
            diff=boots[(scope,'APT_oracle')]-boots[(scope,ref)]
            table=frame[frame.lineage==scope].set_index('method')
            contrasts.append(dict(lineage=scope,comparison='APT_oracle-minus-'+ref,
                estimate=table.loc['APT_oracle','estimate']-table.loc[ref,'estimate'],
                low=np.nanquantile(diff,.025),high=np.nanquantile(diff,.975)))
    pd.DataFrame(contrasts).to_csv(args.output/'paired_contrasts.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
    for j,(mode,label,color) in enumerate([('APT','APT, ordinary','#2563a6'),
        ('APT_oracle','APT + true lineage','#16846b'),('prior_expected','Training-prior expected CM','#b07736')]):
        table=frame[frame.method==mode].set_index('lineage').loc[lineages]
        yy=np.arange(5)+(j-1)*.2
        ax.hlines(yy,table.low,table.high,color=color,lw=1.3)
        ax.scatter(table.estimate,yy,s=22,color=color,label=label,zorder=3)
    ax.set_yticks(np.arange(5),lineages);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('Within-lineage subject-balanced Macro-F1')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
    ax.legend(loc='lower right',fontsize=8,frameon=False)
    fig.savefig(args.output/'lineage_structure.pdf');fig.savefig(args.output/'lineage_structure.png',dpi=220)
    (args.output/'completion.json').write_text(json.dumps(dict(status='PASS',seeds=SEEDS,
        cells=361792,patients=40,all_lineages_reported=True,
        estimand='subject-normalized confusion, fixed child classes; outside predictions remain false negatives',
        prior='training-subset conditional proportions; expected CM is not expected finite-sample F1'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
