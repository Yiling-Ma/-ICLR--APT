"""Re-score frozen OOF predictions; do not retrain or change checkpoints."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import generate_hierarchy_main_table as old


def summarize(models, n_boot=2000, seed=42):
    rows = []
    for task, recipes in models.items():
        ref = recipes['DropCascade']
        patients = sorted(ref.sample_id.unique())
        labels = sorted(ref.y_true.unique())
        target = ref.groupby(['sample_id', 'y_true']).size().sort_index()
        counts = ref.groupby('sample_id').size().reindex(patients).to_numpy()[:, None]
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(len(patients), np.ones(len(patients))/len(patients), size=n_boot)
        for name, frame in recipes.items():
            if not frame.groupby(['sample_id', 'y_true']).size().sort_index().equals(target):
                raise ValueError(f'Mismatched held-out targets: {task}/{name}')
            tp, fp, fn = old.patient_confusions(frame, patients, labels)
            for metric in ('cell_weighted', 'subject_balanced'):
                arrays = (tp, fp, fn) if metric == 'cell_weighted' else tuple(a/counts for a in (tp, fp, fn))
                point = float(old.macro_f1(*(a.sum(0) for a in arrays)))
                samples = old.macro_f1(*(draws @ a for a in arrays))
                lo, hi = np.quantile(samples, [.025, .975])
                rows.append(dict(task=task, model=name, metric=metric, estimate=point,
                                 lower=float(lo), upper=float(hi), patients=len(patients),
                                 cells=len(frame), bootstrap=n_boot, seed=seed))
    return pd.DataFrame(rows)


def write_table(frame, path, metric, label, caption, models):
    data = frame[frame.metric == metric].set_index(['task', 'model'])
    heading = 'SB-Macro-F1' if metric == 'subject_balanced' else 'CW-Macro-F1'
    names = dict(old.TEX_NAMES, **{'Flat FT-style Transformer': 'Flat Transformer',
                                  'FT-style Transformer + HCE': 'HCE Transformer'})
    lines = [r'\begin{table}[t]', r'\centering\small', r'\setlength{\tabcolsep}{6pt}',
             r'\begin{tabular}{llcc}', r'\toprule',
             'Model & Input & Coarse ' + heading + r' $\uparrow$ & Fine ' + heading + r' $\uparrow$ \\', r'\midrule']
    for model in models:
        values = []
        for task in ('coarse', 'fine'):
            row = data.loc[(task, model)]
            values.append(f'{row.estimate:.3f} [{row.lower:.3f}, {row.upper:.3f}]')
        lines.append(names.get(model, model) + ' & APT & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '\\caption{' + caption + '}',
              '\\label{' + label + '}', r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')


def main():
    args = old.parse_args()
    models, _ = old.load_predictions(args)
    result = summarize(models, args.n_boot, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir/'benchmark_reweighted_metrics.csv', index=False)
    # Reproduce the historical estimates before changing the presentation metric.
    legacy, _ = old.evaluate(models, args.n_boot, args.seed)
    check = result[result.metric == 'cell_weighted'].merge(legacy, on=['task', 'model'])
    if not np.allclose(check.estimate, check.macro_f1, atol=1e-12):
        raise AssertionError('Historical metric reconstruction failed')
    sources = [args.dropcascade]
    sources += [args.classical_root/task/model/'pooled_oof_predictions.csv'
                for task in ('coarse', 'fine') for model in old.CLASSICAL_NAMES]
    sources += [args.matched_root/model/'pooled_oof_predictions.csv' for model in ('flat_ce', 'hce')]
    reference_folds = None
    for source in sources:
        frame = pd.read_csv(source, usecols=['sample_id', 'fold'])
        assignments = frame.drop_duplicates().sort_values('sample_id').reset_index(drop=True)
        if len(frame) != 361792 or len(assignments) != 40 or not assignments.sample_id.is_unique:
            raise ValueError(f'Invalid full-cohort OOF assignment: {source}')
        if sorted(assignments.groupby('fold').size()) != [8]*5:
            raise ValueError(f'Unexpected fold sizes: {source}')
        if reference_folds is None:
            reference_folds = assignments
        elif not reference_folds.equals(assignments):
            raise ValueError(f'Inconsistent patient-fold mapping: {source}')
    audit = dict(operation='Re-evaluation of frozen predictions, no refitting',
                 historical_metric_check=True, patient_fold_alignment=True,
                 neural_barcode_alignment='Not independently checked; neural pooled exports lack cell IDs',
                 full_mlp_available=False,
                 sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (args.output_dir/'benchmark_reweighted_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    write_table(result, args.table_path, 'subject_balanced', 'tab:main',
        r'Primary APT-only cell-typing comparison: five patient-disjoint folds, 40 patients, 361{,}792 cells. Entries are subject-balanced pooled Macro-F1 with pointwise 95\% patient-bootstrap intervals (2{,}000 resamples). Frozen predictions are retrospectively re-scored with equal total patient weight; checkpoints are unchanged. Comparisons are descriptive, conditional on fitted models. Transformer auxiliary supervision differs (Table~\ref{tab:transformer_components}); full-budget MLP coverage remains unavailable.',
        ['Logistic Regression', 'XGBoost', 'Flat FT-style Transformer', 'FT-style Transformer + HCE', 'DropCascade'])
    write_table(result, args.table_path.parent/'hierarchy_cell_weighted.tex', 'cell_weighted', 'tab:hierarchy_cell_weighted',
        r'Historical cell-weighted pooled Macro-F1, with pointwise 95\% patient-bootstrap intervals, for the frozen APT-only hierarchy predictions. The same predictions are reweighted in Table~\ref{tab:main}; the estimands differ. Classical auxiliary references are retained here rather than omitted from the benchmark.', old.MODEL_ORDER)
    write_table(result, args.table_path.parent/'hierarchy_subject_balanced_all.tex', 'subject_balanced', 'tab:hierarchy_sb_all',
        r'Complete subject-balanced pooled Macro-F1 reference inventory, including auxiliary classical baselines. Definitions, resampling, and scope match Table~\ref{tab:main}.', old.MODEL_ORDER)


if __name__ == '__main__':
    main()
