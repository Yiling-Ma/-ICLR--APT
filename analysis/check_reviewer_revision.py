"""Read-only numerical and retention checks for the reviewer revision."""
import csv
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def read(path):
    with (root/path).open() as f:
        return list(csv.DictReader(f))

rows = read('outputs/revision_composition_control_v1/summary.csv')
assert json.loads((root/'outputs/revision_composition_control_v1/qa.json').read_text())['status'] == 'PASS'
for level,k in [('lineage',5),('subtype',27)]:
    for method in ['constant','soft','direct','soft_minus_direct_mae','constant_minus_direct_mae']:
        a = next(r for r in rows if r['level']==level and r['method']==method and r['metric']=='mae')
        b = next(r for r in rows if r['level']==level and r['method']==method and r['metric']=='tv')
        for field in ['estimate','low','high']:
            assert math.isclose(float(b[field]),float(a[field])*k/2,abs_tol=1e-12)
    diff = next(r for r in rows if r['level']==level and r['method']=='soft_minus_direct_mae' and r['metric']=='mae')
    assert float(diff['low']) < 0 < float(diff['high'])
classes = read('outputs/revision_composition_control_v1/per_class.csv')
assert len(classes)==(5+27)*3
subtypes = read('analysis/generated/subtype_per_class.csv')
assert len(subtypes)==27 and sum(int(r['cell_count']) for r in subtypes)==361792
paper = (root/'iclr2027_conference.tex').read_text()
assert r'\input{tables/paired_rna_oracle}' in paper
assert r'\input{tables/subtype_per_class}' in paper
assert r'\input{tables/disease_cell_level}' not in paper
assert 'Normal' in (root/'tables/LOD.tex').read_text()
assert 'Macro-average' not in (root/'tables/LOD.tex').read_text()
for name in ['claim_evidence_map.csv','pruning_manifest.csv']:
    assert len(read(name))>=10
print('PASS: TV identities, paired-difference scope, all-class retention, control retention, manifests.')

oracle = read('outputs/remaining_oracle_v1/summary.csv')
assert len(oracle) == 3
names = {'flat_ce':'Flat FT-style Transformer', 'hce':'FT-style Transformer + HCE',
         'soft_cascade':'DropCascade'}
reference = read('analysis/generated/benchmark_reweighted_metrics.csv')
for row in oracle:
    old = next(r for r in reference if r['task']=='fine' and r['metric']=='subject_balanced'
               and r['model']==names[row['model']])
    assert math.isclose(float(row['ordinary_sb']),float(old['estimate']),abs_tol=1e-12)
    assert math.isclose(float(row['delta']),float(row['oracle_sb'])-float(row['ordinary_sb']),abs_tol=1e-12)
assert json.loads((root/'outputs/remaining_oracle_v1/completion.json').read_text())['status']=='PASS'
print('PASS: oracle ordinary scores reconstruct the frozen main-table SB scores.')
