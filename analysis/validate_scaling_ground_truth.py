"""Known-null and independent Monte Carlo target validation.

Target: expected P32-P8 score under an exact-quota, uniform-within-class
allocation policy. This is not the unrestricted benefit of more subjects.
The no-shift null is exactly zero by conditional feature exchangeability.
Non-null targets are independently estimated, never assumed from offset size.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sequential_scaling_controls import sample
from class_matched_scaling import allocate_quotas, atomic_json
from audit_granularity_chance import f1

SCENARIOS = [
    dict(name="known_null", donor=0., block=0., batch=0., separation=.8),
    dict(name="mild_donor", donor=.5, block=0., batch=0., separation=.8),
    dict(name="strong_donor", donor=1.5, block=0., batch=0., separation=.8),
    dict(name="correlated_cells", donor=.5, block=.7, batch=0., separation=.8),
    dict(name="weak_subtypes", donor=.5, block=0., batch=0., separation=.3),
    dict(name="batch_shift", donor=.5, block=0., batch=1., separation=.8),
]
N_TRAIN, N_TEST, CELLS, TOTAL, K = 32, 40, 120, 480, 6


def generate(seed, setting):
    rng = np.random.default_rng(seed)
    # Three lineages, two sibling subtypes each. Centers fixed across replicates.
    centers = np.zeros((K, 8))
    for k in range(K):
        centers[k,k//2] = 2.
        centers[k,3+k//2] = setting["separation"] * (-1 if k%2==0 else 1)
    x, y, d = [], [], []
    batch_effects = rng.normal(size=((N_TRAIN+N_TEST)//4,8))*setting["batch"]
    for donor in range(N_TRAIN+N_TEST):
        logits = rng.normal(size=K)*1.5
        # The sixth subtype is available in only 10% of donors.
        if rng.random() >= .1:
            logits[-1] = -np.inf
        weights = np.exp(logits-np.max(logits)); weights /= weights.sum()
        labels = rng.choice(K,CELLS,p=weights)
        offset = rng.normal(size=8)*setting["donor"]
        blocks = np.repeat(rng.normal(size=(CELLS//10,8))*setting["block"],10,axis=0)
        features = centers[labels]+offset+blocks+batch_effects[donor//4]+rng.normal(size=(CELLS,8))
        x.append(features); y.append(labels); d.extend([donor]*CELLS)
    return np.concatenate(x),np.concatenate(y),np.asarray(d)


def score(x,y,d,ix):
    if np.any(d[ix] >= N_TRAIN) or len(ix)!=TOTAL or len(set(ix))!=TOTAL:
        raise ValueError("Training manifest invalid")
    test = d >= N_TRAIN
    scaler = StandardScaler().fit(x[ix])
    model = LogisticRegression(max_iter=500,random_state=42).fit(scaler.transform(x[ix]),y[ix])
    pred = model.predict(scaler.transform(x[test])).astype(int)
    cells = np.bincount((d[test]-N_TRAIN)*K*K+y[test]*K+pred,
                        minlength=N_TEST*K*K).reshape(N_TEST,K,K)
    return float(f1((cells/cells.sum(axis=(1,2),keepdims=True)).mean(axis=0)))


def oracle_indices(y,d,p,quotas,rng):
    # Separate implementation from the evaluated priority sampler.
    parts = []
    for k in np.flatnonzero(quotas):
        available = np.flatnonzero((d<p)&(y==k))
        parts.append(rng.choice(available,size=int(quotas[k]),replace=False))
    result = np.concatenate(parts)
    if not np.array_equal(np.bincount(y[result],minlength=K),quotas):
        raise ValueError("Oracle quota mismatch")
    return result


def replicate(seed,setting,oracle=False):
    x,y,d = generate(seed,setting)
    quotas = allocate_quotas(np.bincount(y[d<8],minlength=K),TOTAL)
    rng = np.random.default_rng(seed+10000000)
    priority=rng.random(len(y)); dp=dict(zip(range(N_TRAIN+N_TEST),rng.random(N_TRAIN+N_TEST)))
    effects = {}
    for condition in ("oracle",) if oracle else ("A","B","C"):
        values = []
        for p in (8,32):
            ix = oracle_indices(y,d,p,quotas,rng) if oracle else sample(y,d,list(range(p)),quotas,priority,dp,condition)
            values.append(score(x,y,d,ix))
        effects[condition] = values[1]-values[0]
    return dict(seed=seed,effects=effects,reference_support=int(np.count_nonzero(quotas)))


def summarize(reference,evaluation,exact_null):
    r=np.array([v["effects"]["oracle"] for v in reference])
    target=0. if exact_null else float(r.mean())
    target_se=0. if exact_null else float(r.std(ddof=1)/np.sqrt(len(r)))
    out=dict(target=target,target_mcse=target_se,target_kind="exact" if exact_null else "independent Monte Carlo",
             reference_mean=float(r.mean()),reference_mcse=float(r.std(ddof=1)/np.sqrt(len(r))),estimators={})
    for c in "ABC":
        v=np.array([a["effects"][c] for a in evaluation])
        bias=float(v.mean()-target)
        se=float(np.sqrt(v.var(ddof=1)/len(v)+target_se**2))
        out["estimators"][c]=dict(mean=float(v.mean()),bias=bias,bias_mcse=se,
            bias_interval=[bias-1.96*se,bias+1.96*se],rmse=float(np.sqrt(np.mean((v-target)**2))),
            empirical_replicate_interval=np.quantile(v,[.025,.975]).tolist())
    out["paired_contrasts"]={}
    for a,b in (("A","C"),("B","C")):
        v=np.array([r["effects"][a]-r["effects"][b] for r in evaluation])
        se=float(v.std(ddof=1)/np.sqrt(len(v)))
        out["paired_contrasts"][a+"-"+b]=dict(mean=float(v.mean()),mcse=se,interval=[float(v.mean()-1.96*se),float(v.mean()+1.96*se)])
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=int,default=512)
    p.add_argument("--replicates",type=int,default=100)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if min(args.reference,args.replicates)<2:
        raise ValueError("At least two independent replicates required")
    config=dict(reference=args.reference,replicates=args.replicates,total=TOTAL,cells=CELLS,
                train_subjects=N_TRAIN,test_subjects=N_TEST,classes=K,scenarios=SCENARIOS,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir(parents=True,exist_ok=True)
    all_results=[]
    for setting in SCENARIOS:
        path=args.output/(setting["name"]+".json")
        if path.exists():
            result=json.loads(path.read_text())
            if result["config"]!=config:
                raise ValueError("Configuration changed; choose a new output directory")
        else:
            start=time.time()
            reference=[]; evaluation=[]
            for i in range(args.reference):
                reference.append(replicate(200000+i,setting,oracle=True))
                if i%100==0: print(setting["name"],"reference",i,flush=True)
            for i in range(args.replicates):
                evaluation.append(replicate(400000+i,setting))
            assert not ({r["seed"] for r in reference}&{r["seed"] for r in evaluation})
            result=dict(config=config,scenario=setting,reference=reference,evaluation=evaluation,
                        summary=summarize(reference,evaluation,setting["name"]=="known_null"),seconds=time.time()-start)
            atomic_json(path,result)
        all_results.append(dict(scenario=setting,summary=result["summary"]))
        print(setting["name"],json.dumps(result["summary"]),flush=True)
    atomic_json(args.output/"summary.json",dict(config=config,results=all_results,
        scope="Bias relative to a specific overlap-quota policy; not total subject value or real-cohort causal mediation. Bias intervals are Monte Carlo normal intervals, not evaluated patient-bootstrap coverage."))


if __name__=="__main__":
    main()
