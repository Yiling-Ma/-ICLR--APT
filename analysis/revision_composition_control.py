"""E3-direct-v1: frozen-fold median APT ridge-to-simplex composition control."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def simplex(x):
    u = np.sort(x, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1) - 1
    j = np.arange(1, x.shape[1] + 1)
    rho = (u - css / j > 0).sum(axis=1) - 1
    theta = css[np.arange(len(x)), rho] / (rho + 1)
    return np.maximum(x - theta[:, None], 0)


def predict(x, y, test, alpha):
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    return simplex(model.predict(scaler.transform(test)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-summary', type=Path)
    args = parser.parse_args()
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(version='E3-direct-v1', alpha=[.1, 1, 10, 100, 1000],
                    seeds=[20260912], bootstrap=2000, metric='equal-patient composition MAE',
                    normalization='provided APT expression; no new log transform',
                    selection='remaining four frozen patient folds; ridge followed by Euclidean simplex projection',
                    supervision='development reference composition; disease only for bootstrap strata')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    source = root/'outputs/oof_composition_bridge/patient_composition_features.parquet'
    frame = pd.read_parquet(source)
    test = frame[frame.role == 'outer_test'].sort_values('patient_id').reset_index(drop=True)
    assert len(test) == 40 and test.patient_id.is_unique
    meta = pd.read_csv(root/'data/metadata.csv')
    ann = pd.read_csv(root/'data/cell_annotation.csv').rename(columns={'Sample': 'cell_id'})
    ann = ann[ann.Celltypes_new.notna() & ~ann.Celltypes_new.str.strip().str.lower().eq('unknown')]
    cells = meta.merge(ann[['cell_id', 'Celltypes_new']], on='cell_id', validate='one_to_one')
    path = root/'data/apt_expression.parquet'
    expr = pd.read_parquet(path) if path.exists() else pd.read_csv(root/'data/apt_expression.csv')
    features = sorted(c for c in expr if c.upper().startswith('APT-'))
    assert len(features) == 293 and expr.cell_id.is_unique and len(cells) == 361792
    cells = cells.merge(expr[['cell_id', *features]], on='cell_id', validate='one_to_one', how='left')
    assert not cells[features].isna().any().any()
    medians = cells.groupby('sample_id')[features].median()
    x = medians.loc[test.patient_id].to_numpy()
    folds = test.outer_fold.to_numpy()
    assert sorted(pd.Series(folds).value_counts()) == [8]*5 and np.isfinite(x).all()
    rng = np.random.default_rng(20260912)
    groups = test.disease.to_numpy()
    draws = np.concatenate([rng.choice(np.flatnonzero(groups == g),
                (2000, sum(groups == g)), replace=True) for g in np.unique(groups)], axis=1)
    summary, classes, selections = [], [], []
    for level in ['lineage', 'subtype']:
        cols = sorted(c for c in test if c.startswith('true_'+level+'::'))
        names = [c.split('::')[1] for c in cols]
        y = test[cols].to_numpy()
        soft = test[['pred_soft_'+level+'::'+c for c in names]].to_numpy()
        direct, baseline = np.zeros_like(y), np.zeros_like(y)
        for fold in sorted(set(folds)):
            tr, te = folds != fold, folds == fold
            scores = []
            for alpha in protocol['alpha']:
                inner_pred = np.zeros_like(y)
                for inner in sorted(set(folds[tr])):
                    itr, ite = tr & (folds != inner), tr & (folds == inner)
                    inner_pred[ite] = predict(x[itr], y[itr], x[ite], alpha)
                scores.append(abs(inner_pred[tr]-y[tr]).mean())
            alpha = protocol['alpha'][int(np.argmin(scores))]
            direct[te] = predict(x[tr], y[tr], x[te], alpha)
            baseline[te] = y[tr].mean(0)
            selections.append(dict(level=level, fold=int(fold), alpha=alpha, inner_mae=min(scores)))
        for a in [y, soft, direct, baseline]:
            assert np.isfinite(a).all() and a.min() >= -1e-8 and np.allclose(a.sum(1), 1, atol=1e-6)
        np.savez_compressed(out/(level+'_private_predictions.npz'), truth=y, soft=soft,
            direct=direct, baseline=baseline, patient_ids=test.patient_id.to_numpy(dtype=str),
            folds=folds, classes=np.asarray(names))
        errors = {n: abs(p-y).mean(1) for n,p in [('constant',baseline),('soft',soft),('direct',direct)]}
        errors['soft_minus_direct_mae'] = errors['soft']-errors['direct']
        errors['constant_minus_direct_mae'] = errors['constant']-errors['direct']
        for name, e in errors.items():
            for metric, scale in [('mae',1), ('tv',len(names)/2)]:
                b = e[draws].mean(1)*scale
                summary.append(dict(level=level, method=name, metric=metric,
                    estimate=e.mean()*scale, low=np.quantile(b,.025), high=np.quantile(b,.975)))
        for k, label in enumerate(names):
            for name,p in [('constant',baseline),('soft',soft),('direct',direct)]:
                classes.append(dict(level=level,label=label,method=name,mae=abs(p[:,k]-y[:,k]).mean(),
                    bias=(p[:,k]-y[:,k]).mean(),prevalence=y[:,k].mean(),
                    centered_mae=abs((p[:,k]-p[:,k].mean())-(y[:,k]-y[:,k].mean())).mean()))
    old = pd.read_csv(args.reference_summary or root/'outputs/composition_recovery_validation/summary.csv')
    for level in ['lineage','subtype']:
        expected = old[(old.level==level)&(old.metric=='macro_mae')].estimate.iloc[0]
        actual = next(r['estimate'] for r in summary if r['level']==level and r['method']=='soft' and r['metric']=='mae')
        assert np.isclose(actual,expected,atol=1e-10)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.DataFrame(classes).to_csv(out/'per_class.csv',index=False)
    pd.DataFrame(selections).to_csv(out/'selection.csv',index=False)
    (out/'qa.json').write_text(json.dumps(dict(status='PASS',patients=40,cells=len(cells),
        alignment='one-to-one cell-key merges and patient-key median alignment',
        existing_soft_mae_reproduced=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='fixed outer folds; direct inner preprocessing and fitting exclude validation patients; no new cell model'),indent=2))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
