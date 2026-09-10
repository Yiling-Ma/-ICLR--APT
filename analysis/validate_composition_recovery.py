"""Patient-balanced OOF composition recovery against training-only baselines."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/composition_recovery_validation'


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def within_group_rho(x, y, groups):
    x, y = rankdata(x).astype(float), rankdata(y).astype(float)
    for g in np.unique(groups):
        mask = groups == g
        x[mask] -= x[mask].mean()
        y[mask] -= y[mask].mean()
    return corr(x, y)


def main():
    source = ROOT / 'outputs/oof_composition_bridge/patient_composition_features.parquet'
    frame = pd.read_parquet(source)
    test = frame[frame.role == 'outer_test'].sort_values('patient_id').copy()
    assert len(test) == 40 and test.patient_id.nunique() == 40
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260911)
    # Stratified patient bootstrap, paired across predictions and baselines.
    disease = test.disease.to_numpy()
    draws = np.concatenate([rng.choice(np.flatnonzero(disease == g),
                            (2000, sum(disease == g)), replace=True)
                            for g in np.unique(disease)], axis=1)
    summary, per_class = [], []
    for level in ['lineage', 'subtype']:
        truth_cols = sorted(c for c in frame if c.startswith('true_' + level + '::'))
        names = [c.split('::')[1] for c in truth_cols]
        pred_cols = ['pred_soft_' + level + '::' + c for c in names]
        y, p = test[truth_cols].to_numpy(), test[pred_cols].to_numpy()
        baseline = np.empty_like(y)
        for i, (_, row) in enumerate(test.iterrows()):
            train = frame[(frame.outer_fold == row.outer_fold) & (frame.role == 'inner_oof')]
            assert len(train) == 32 and row.patient_id not in set(train.patient_id)
            # Equal weight per development patient, never test labels.
            baseline[i] = train[truth_cols].mean().to_numpy()
        assert np.allclose(y.sum(1), 1) and np.allclose(p.sum(1), 1)
        error, base_error = abs(p-y), abs(baseline-y)
        for name, values in [('macro_mae', error.mean(1)),
                             ('baseline_macro_mae', base_error.mean(1)),
                             ('mae_improvement', (base_error-error).mean(1))]:
            boot = values[draws].mean(1)
            summary.append(dict(level=level, metric=name, estimate=values.mean(),
                                low=np.quantile(boot, .025), high=np.quantile(boot, .975)))
        for k, name in enumerate(names):
            improvement = (base_error-error)[:, k]
            boot = improvement[draws].mean(1)
            per_class.append(dict(level=level, label=name, mae=error[:, k].mean(),
                baseline_mae=base_error[:, k].mean(), bias=(p-y)[:, k].mean(),
                mae_improvement=improvement.mean(), low=np.quantile(boot, .025),
                high=np.quantile(boot, .975), prevalence=y[:, k].mean(),
                spearman=corr(rankdata(p[:, k]), rankdata(y[:, k])),
                disease_adjusted_rank_correlation=within_group_rho(p[:, k], y[:, k], disease)))
        np.savez_compressed(OUT / (level + '_patient_arrays.npz'), truth=y, prediction=p,
                            baseline=baseline, patients=test.patient_id.to_numpy(dtype=str))
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(per_class).to_csv(OUT / 'per_class.csv', index=False)
    (OUT / 'qa.json').write_text(json.dumps(dict(status='PASS', patients=40,
        bootstrap=2000, baseline='outer-development patient mean reference composition',
        scope=('conditional on completed OOF fits; per-class intervals exploratory, not simultaneous; '
               'disease-adjusted ranks are descriptive, not causal')), indent=2))


if __name__ == '__main__':
    main()
