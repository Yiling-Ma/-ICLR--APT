"""Independent fit/shuffle replicates on frozen cell subsets, with RNA-width sensitivity."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import validate_paired_information as base

SEEDS=[20260912,20260913,20260914]
SHUFFLES=[73001,73002,73003]


def prepare(args):
    cache=args.output/'prepared'
    cache.mkdir(parents=True,exist_ok=True)
    for source in [args.source/'cells.csv',*args.source.glob('f*_train.txt'),*args.source.glob('f*_test.txt')]:
        dest=cache/source.name
        if dest.exists():
            assert dest.read_bytes()==source.read_bytes()
        else: shutil.copy2(source,dest)
    if args.hvg==2000:
        for source in [*args.source.glob('f*.mtx'),*args.source.glob('f*_genes.txt'),*args.source.glob('f*_done.txt')]:
            dest=cache/source.name
            if not dest.exists(): dest.symlink_to(source.resolve())
    else:
        subprocess.run(['/home/mayiling/opt/apt-rna-r/bin/Rscript',
            str(base.ROOT/'analysis/prepare_fold_rna.R'),str(base.ROOT),str(cache)],
            env=dict(os.environ,APT_RNA_HVG=str(args.hvg)),check=True)
    for f in range(5):
        for stage in ['inner','outer']:
            genes=(cache/f'f{f}_{stage}_genes.txt').read_text().splitlines()
            assert len(genes)==args.hvg and len(set(genes))==args.hvg
    protocol=dict(version='paired-repeat-v1',training_seeds=SEEDS,shuffle_seeds=SHUFFLES,
        hvg=args.hvg,source=str(args.source.resolve()),cell_subsets='frozen original subset',
        models=base.MODELS,modes=base.MODES,epochs=40,
        inference='same old sklearn SGD/MLP recipes; no GPU estimator substitution',
        uncertainty='paired seed and patient bootstrap, conditional on frozen subsets/folds',
        selection='same inner fold alpha selection for every fit, no outer selection')
    path=args.output/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==protocol
    else: base.save_json(path,protocol)


def run(args):
    base.OUT=args.output/'prepared'
    meta=pd.read_csv(base.OUT/'cells.csv')
    apt=base.raw_apt(meta)
    enc=base.core.fit_label_encoders(meta)
    for seed,shuffle_seed in zip(SEEDS,SHUFFLES):
        for fold in range(5):
            base.SEED=shuffle_seed
            inner=base.inputs(meta,apt,fold,'inner')
            outer=base.inputs(meta,apt,fold,'outer')
            base.SEED=seed
            for task in ['coarse','fine']:
                y=enc[task].transform(meta[base.core.TASKS[task]].astype(str))
                k=len(enc[task].classes_)
                for name,grid in base.MODELS.items():
                    for mode in base.MODES:
                        dest=args.output/f's{seed}_f{fold}_{task}_{name}_{mode}.npz'
                        if dest.exists(): continue
                        started=time.time(); scores=[]
                        for alpha in grid:
                            tr,te,xx=inner
                            model=base.fit(name,alpha,xx[mode][0],y[tr],
                                base.patient_class_weights(meta.sample_id.iloc[tr].to_numpy(),y[tr]))
                            _,cm=base.cms(y[te],model.predict(xx[mode][1]),meta.sample_id.iloc[te].to_numpy(),k)
                            scores.append(base.core.f1_from_confusion(base.core.patient_balanced_matrix(cm)))
                        alpha=grid[int(np.argmax(scores))]
                        tr,te,xx=outer
                        model=base.fit(name,alpha,xx[mode][0],y[tr],
                            base.patient_class_weights(meta.sample_id.iloc[tr].to_numpy(),y[tr]))
                        pred=model.predict(xx[mode][1])
                        patients,cm=base.cms(y[te],pred,meta.sample_id.iloc[te].to_numpy(),k)
                        np.savez_compressed(dest.with_suffix('.tmp.npz'),cm=cm,patients=patients.astype(str),
                            alpha=alpha,validation_scores=scores,iterations=model.n_iter_,
                            cell_ids=meta.cell_id.iloc[te].to_numpy(dtype=str),truth=y[te],pred=pred,
                            classes=enc[task].classes_.astype(str),seed=seed,shuffle_seed=shuffle_seed)
                        os.replace(dest.with_suffix('.tmp.npz'),dest)
                        print('COMPLETE',dest.name,'seconds',round(time.time()-started,1),flush=True)


def aggregate(args):
    rng=np.random.default_rng(20260912); rows=[]
    def score(cm): return base.core.f1_from_confusion(base.core.patient_balanced_matrix(cm))
    for task in ['coarse','fine']:
        for model in base.MODELS:
            arrays={}; order=None
            for mode in base.MODES:
                seeds=[]
                for seed in SEEDS:
                    blocks=[np.load(args.output/f's{seed}_f{f}_{task}_{model}_{mode}.npz') for f in range(5)]
                    ids=np.concatenate([b['patients'] for b in blocks])
                    cells=np.concatenate([b['cell_ids'] for b in blocks])
                    assert len(ids)==40 and len(set(ids))==40 and len(set(cells))==361792
                    if order is None: order=ids
                    assert np.array_equal(order,ids)
                    seeds.append(np.concatenate([b['cm'] for b in blocks]))
                arrays[mode]=np.stack(seeds)
                rows.append(dict(task=task,model=model,comparison=mode,
                    estimate=np.mean([score(c) for c in arrays[mode]]),hvg=args.hvg))
            for ref in ['rna','rna_noise','rna_shuffled_apt']:
                high,low=arrays['rna_apt'],arrays[ref]
                values=[score(h)-score(l) for h,l in zip(high,low)]
                draws=[]
                for _ in range(2000):
                    ss=rng.integers(0,3,3); pp=rng.integers(0,40,40)
                    draws.append(np.mean([score(high[s,pp])-score(low[s,pp]) for s in ss]))
                rows.append(dict(task=task,model=model,comparison='rna_apt-minus-'+ref,
                    estimate=np.mean(values),low=np.quantile(draws,.025),high=np.quantile(draws,.975),
                    seed_values=values,hvg=args.hvg))
    pd.DataFrame(rows).to_csv(args.output/'summary.csv',index=False)
    base.save_json(args.output/'completion.json',dict(status='PASS',outer_fits=300,
        inner_fits=600,seeds=SEEDS,hvg=args.hvg))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','run','aggregate'])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hvg',type=int,choices=[2000,5000],required=True)
    a=p.parse_args()
    with threadpool_limits(limits=4): globals()[a.command](a)
