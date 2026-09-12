"""Canonical Table 1 refresh: historical frozen scores plus completed full MLP.

Validate prediction-derived audit points against retained CI summaries, export
stage records, and refresh only numeric table rows. Never fit or select models.
"""
import csv
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    'logistic_regression': ('Logistic Regression', 'Logistic Regression'),
    'xgboost': ('XGBoost', 'XGBoost'),
    'plain_mlp': ('Plain MLP (3 seeds)', None),
    'flat_ce': ('Flat Transformer', 'Flat FT-style Transformer'),
    'hce': ('HCE Transformer', 'FT-style Transformer + HCE'),
    'soft_cascade': (r'\ourmethod{}', 'DropCascade'),
}


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def build():
    a = json.loads((ROOT/'analysis/generated/main_protocol_audit.json').read_text())
    old = csv_rows(ROOT/'analysis/generated/benchmark_reweighted_metrics.csv')
    mlp = csv_rows(ROOT/'outputs/remaining_full_mlp_v1/summary.csv')
    assert a['new_training'] == 0 and len(a['records']) == 95
    text = (ROOT/'tables/main_result.tex').read_text()
    for model, (label, old_name) in NAMES.items():
        values=[]
        for task in ['coarse','fine']:
            points=[r['sb'] for r in a['scores'] if r['model']==model and r['task']==task]
            assert len(points)==(3 if model=='plain_mlp' else 1)
            point=sum(points)/len(points)
            if model=='plain_mlp':
                row=next(r for r in mlp if r['task']==task and r['method']=='unconstrained')
                lo,hi=float(row['low']),float(row['high'])
            else:
                row=next(r for r in old if r['task']==task and r['model']==old_name and r['metric']=='subject_balanced')
                lo,hi=float(row['lower']),float(row['upper'])
            assert math.isclose(point,float(row['estimate']),abs_tol=1e-12)
            values.append(f'{point:.3f} [{lo:.3f}, {hi:.3f}]')
        ci='SP' if model=='plain_mlp' else 'P'
        replacement=label+' & '+ci+' & '+' & '.join(values)+r' \\'
        pattern=r'^'+re.escape(label)+r' & .*?$'
        text,n=re.subn(pattern,lambda _:replacement,text,flags=re.MULTILINE)
        assert n==1
    (ROOT/'tables/main_result.tex').write_text(text)
    fields=['model','task','fold','stage','seed','patient_count','cell_count',
            'validation_patients','validation_cells','test_patients','test_cells',
            'preprocessing','selection','refit','sampling','supervision','ci_type',
            'bootstrap_seed','bootstrap_replicates','prediction_metric','artifact_source',
            'config_source','log_source','checkpoint_source','manifest_source','details']
    with (ROOT/'analysis/generated/main_training_protocols.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for rec in a['records']:
            row={k:rec.get(k,'unavailable') for k in fields}
            if rec['seed'] is None: row['seed']='unavailable / not applicable; see details'
            row.update(validation_patients=rec['validation']['patient_count'],
                validation_cells=rec['validation']['cell_count'],test_patients=rec['test']['patient_count'],
                test_cells=rec['test']['cell_count'],ci_type='SP' if rec['model']=='plain_mlp' else 'P',
                bootstrap_seed=20260912 if rec['model']=='plain_mlp' else 42,
                bootstrap_replicates=2000,prediction_metric='subject-normalized pooled fixed-class Macro-F1',
                details=json.dumps({k:v for k,v in rec.items() if k not in fields},sort_keys=True))
            writer.writerow(row)
    return a


if __name__=='__main__':
    build()
    print('PASS: six main-table recipes, 12 displayed points, 95 fold/stage records; no training.')
