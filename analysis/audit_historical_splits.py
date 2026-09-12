"""Read historical artifacts without fitting; emit aggregate provenance as JSON.

Run on the artifact host: python3 audit_historical_splits.py /path/to/apt_agent
No checkpoints are deserialized and no cell IDs are exported.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def digest(values):
    return hashlib.sha256('\n'.join(sorted(values)).encode()).hexdigest()


def audit(root):
    home = Path(root)
    base = home / 'cell_JEPA'
    sources = {}

    def file(name):
        path = base / name
        if path.exists():
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(block)
            sources[name] = {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}
        return path

    def rows(name):
        with file(name).open() as handle:
            return list(csv.DictReader(handle))

    def js(name):
        return json.loads(file(name).read_text())

    master_rows = rows('outputs/classical_baselines_kfold5/fine/xgboost/pooled_oof_predictions.csv')
    master = {r['cell_id']: r['sample_id'] for r in master_rows}
    assert len(master) == len(master_rows) == 361792
    patient_counts = Counter(master.values())
    assert len(patient_counts) == 40
    folds = js('outputs/classical_baselines_kfold5/fold_assignment.json')
    assert set().union(*(set(v) for v in folds.values())) == set(patient_counts)
    assert sum(map(len, folds.values())) == 40

    def summarize(parts):
        patients = {k: set(master[c] for c in v) for k, v in parts.items()}
        return {
            'parts': {k: {'cells': len(v), 'patients': len(patients[k]),
                          'cell_set_sha256': digest(v)} for k, v in parts.items()},
            'patient_intersections': {a + '/' + b: len(patients[a] & patients[b])
                                     for a, b in [('train','val'),('train','test'),('val','test')]},
            'cell_intersections': {a + '/' + b: len(parts[a] & parts[b])
                                  for a, b in [('train','val'),('train','test'),('val','test')]}}

    cell_rows = rows('outputs/classical_baselines_cell/cell_split_seed42.csv')
    cell_parts = {s: {r['cell_id'] for r in cell_rows if r['split'] == s}
                  for s in ['train','val','test']}
    assert set().union(*cell_parts.values()) == set(master)
    assert sum(map(len, cell_parts.values())) == len(master)
    soft_cell_rows = rows('outputs/soft_lineage_cascade_cell_matched/split.csv')
    assert {(r['cell_id'], r['split']) for r in soft_cell_rows} == {
        (r['cell_id'], r['split']) for r in cell_rows}
    split_summary = {'cell_both_models': summarize(cell_parts), 'xgb_patient': [], 'soft_patient': []}
    for fold in range(5):
        test_patients = set(folds[str(fold)])
        xparts = {'train': {c for c,p in master.items() if p not in test_patients},
                  'val': set(), 'test': {c for c,p in master.items() if p in test_patients}}
        split_summary['xgb_patient'].append({'fold': fold, **summarize(xparts)})
        manifest = rows(f'outputs/dropcascade_kfold5/fold{fold}/split.csv')
        sets = {s: {r['sample_id'] for r in manifest if r['split'] == s}
                for s in ['train','val','test']}
        assert sets['test'] == test_patients
        sparts = {s: {c for c,p in master.items() if p in pats} for s,pats in sets.items()}
        stat = summarize(sparts)
        assert not any(stat['patient_intersections'].values())
        assert sum(v['cells'] for v in stat['parts'].values()) == len(master)
        split_summary['soft_patient'].append({'fold': fold, **stat})

    def prediction_score(data, truth, prediction, classes):
        tp, actual, predicted = Counter(), Counter(), Counter()
        for row in data:
            y, p = row[truth], row[prediction]
            actual[y] += 1; predicted[p] += 1
            if y == p: tp[y] += 1
        labels = set(actual) | set(predicted)
        assert len(labels) == classes and len(actual) == classes
        value = sum(2*tp[k]/(actual[k]+predicted[k]) if actual[k]+predicted[k] else 0
                    for k in labels)/classes
        return value

    scores = []
    soft_pd = rows('outputs/dropcascade_kfold5/pooled_oof_predictions.csv')
    soft_cell = rows('outputs/soft_lineage_cascade_cell_matched/test_predictions.csv')
    neural_headers = {'patient': list(soft_pd[0]), 'cell': list(soft_cell[0])}
    assert Counter(r['sample_id'] for r in soft_pd) == patient_counts
    assert Counter(r['sample_id'] for r in soft_cell) == Counter(master[c] for c in cell_parts['test'])
    fine_cell = None
    for task, k in [('coarse',5),('fine',27)]:
        pd_path = f'outputs/classical_baselines_kfold5/{task}/xgboost/pooled_oof_predictions.csv'
        cell_path = f'outputs/classical_baselines_cell/{task}/xgboost/test_predictions.csv'
        pd_rows, cr = rows(pd_path), rows(cell_path)
        assert len(pd_rows) == len(master) and {r['cell_id'] for r in pd_rows} == set(master)
        assert len(cr) == len(cell_parts['test']) and {r['cell_id'] for r in cr} == cell_parts['test']
        assert all(master[r['cell_id']] == r['sample_id'] for r in pd_rows + cr)
        for row in pd_rows:
            if 'fold' in row: assert row['sample_id'] in folds[str(int(row['fold']))]
        if task == 'fine': fine_cell = cr
        pvalue = prediction_score(pd_rows,'y_true','y_pred',k)
        cvalue = prediction_score(cr,'y_true','y_pred',k)
        metric = js(f'outputs/classical_baselines_cell/{task}/xgboost/test_metrics.json')
        assert abs(cvalue-metric['macro_f1']) < 1e-12
        scores.append(dict(model='XGBoost',task=task,classes=k,patient=pvalue,cell=cvalue,
                           difference=cvalue-pvalue,patient_source=pd_path,cell_source=cell_path))
        pvalue = prediction_score(soft_pd,task+'_true_id',task+'_pred_id',k)
        cvalue = prediction_score(soft_cell,task+'_true_id',task+'_pred_id',k)
        metric = js('outputs/soft_lineage_cascade_cell_matched/test_metrics.json')
        assert abs(cvalue-metric['coarse_macro_f1' if task=='coarse' else 'macro_f1_all_classes']) < 1e-12
        scores.append(dict(model='Soft-cascade',task=task,classes=k,patient=pvalue,cell=cvalue,
                           difference=cvalue-pvalue,
                           patient_source='outputs/dropcascade_kfold5/pooled_oof_predictions.csv',
                           cell_source='outputs/soft_lineage_cascade_cell_matched/test_predictions.csv'))
    # Consistency check only: without neural barcodes this cannot prove cell identity.
    ordered_match = [(r['sample_id'],r['y_true']) for r in fine_cell] == [
        (r['sample_id'],r['subtype_true']) for r in soft_cell]
    configs, logs = {}, {}
    for name in ['soft_lineage_cascade_cell_matched'] + [f'dropcascade_kfold5/fold{i}' for i in range(5)]:
        config = js(f'outputs/{name}/resolved_config.json')
        configs[name] = config
        history = rows(f'outputs/{name}/training_log.csv')
        best = max(history,key=lambda r:float(r['val_macro_f1_all_classes']))
        logs[name] = {'epochs_logged': len(history), 'last_epoch': int(history[-1]['epoch']),
                      'validation_best_epoch': int(best['epoch']),
                      'validation_best_fine_cw': float(best['val_macro_f1_all_classes']),
                      'resolved_epoch_budget': config['train']['epochs']}
    for path in ['apt_jepa/scripts/train_classical_baselines.py',
                 'apt_jepa/scripts/train_classical_baselines_kfold.py',
                 'apt_jepa/scripts/train_soft_lineage_cascade.py',
                 'apt_jepa/data/dataset.py','apt_jepa/data/preprocessing.py','apt_jepa/data/split.py',
                 'apt_jepa/configs/classical_baselines_cell.yaml']:
        file(path)
    for name in ['classical_baselines_cell.log','soft_lineage_cascade_cell_matched.log',
                 'dropcascade_cell_best_eval.log']:
        file('../'+name)
    return {'route':'A','new_training_runs':0,'source_root':str(base),'sources':sources,
            'splits':split_summary,'contrasts':scores,'soft_configs':configs,'soft_training_logs':logs,
            'neural_prediction_headers':neural_headers,
            'neural_cell_order_patient_truth_matches_xgb':ordered_match,
            'test_cell_identity':{'xgb':'explicit IDs checked against manifests',
             'soft':'unavailable in predictions; split-implied sets and patient counts verified, not barcode linkage'},
            'intervals':'Protocol-difference uncertainty not estimated',
            'source_version_limit':'Current source and surviving records, not immutable original environment or CLI snapshot',
            'metric':'CW pooled Macro-F1, all 5/27 classes supported; zero denominator contributes zero'}


if __name__ == '__main__':
    print(json.dumps(audit(sys.argv[1]), indent=2))
