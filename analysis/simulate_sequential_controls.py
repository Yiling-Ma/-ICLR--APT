"""Synthetic mechanism check, not evidence about clinical biology.

Factorial donor offset, donor-varying label mixture, and separation.
The zero-offset/fixed-mixture setting has exchangeable donors. Offsets do not
guarantee a breadth benefit: its sign is an empirical outcome, never asserted.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sequential_scaling_controls import sample
from class_matched_scaling import allocate_quotas, atomic_json
from audit_granularity_chance import f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds",type=int,default=20)
    ap.add_argument("--output",type=Path,required=True)
    args = ap.parse_args()
    rows = []
    for offset, mixture, separation in itertools.product((0.,1.5),(0.,1.5),(0.5,1.5)):
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed+1000)
            centers = rng.normal(size=(6,12))*separation
            x,y,donors = [],[],[]
            for donor in range(48):
                logits = rng.normal(size=6)*mixture
                probs = np.exp(logits-logits.max()); probs /= probs.sum()
                labels = rng.choice(6,size=400,p=probs)
                features = centers[labels]+rng.normal(size=(400,12))+offset*rng.normal(size=12)
                x.append(features); y.append(labels); donors.extend([donor]*400)
            x,y,d = np.concatenate(x),np.concatenate(y),np.asarray(donors)
            priority = rng.random(len(y)); dp = dict(zip(range(48),rng.random(48)))
            quotas = allocate_quotas(np.bincount(y[d<8],minlength=6),1600)
            scores = {}
            for condition,p in itertools.product("ABCD",(8,32)):
                ix = sample(y,d,list(range(p)),quotas,priority,dp,condition)
                scaler = StandardScaler().fit(x[ix])
                model = LogisticRegression(max_iter=500).fit(scaler.transform(x[ix]),y[ix])
                pred = model.predict(scaler.transform(x[d>=32]))
                cm = np.stack([confusion_matrix(y[d>=32][d[d>=32]==donor],
                    pred[d[d>=32]==donor],labels=np.arange(6)) for donor in range(32,48)])
                scores[condition,p] = float(f1((cm/cm.sum(axis=(1,2),keepdims=True)).mean(axis=0)))
            rows.append(dict(offset=offset,mixture=mixture,separation=separation,seed=seed,
                             effects={c:scores[c,32]-scores[c,8] for c in "ABCD"}))
            print(offset,mixture,separation,seed,flush=True)
    summary = []
    for setting in itertools.product((0.,1.5),(0.,1.5),(0.5,1.5)):
        group = [r for r in rows if (r["offset"],r["mixture"],r["separation"])==setting]
        summary.append(dict(offset=setting[0],mixture=setting[1],separation=setting[2],
            conditions={c:dict(mean=float(np.mean([r["effects"][c] for r in group])),
                 empirical_seed_interval=np.quantile([r["effects"][c] for r in group],[.025,.975]).tolist())
                 for c in "ABCD"}))
    atomic_json(args.output,dict(seeds=args.seeds,rows=rows,summary=summary,
        scope="Synthetic additive-offset model; seed intervals are not confidence intervals of the mean; no clinical or causal mediation claim."))


if __name__ == "__main__":
    main()
