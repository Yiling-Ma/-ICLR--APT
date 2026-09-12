"""Read-only cross-width/diagnostic identity audit; publish aggregate status only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_modality_repeats import SEEDS

ROOT=Path('/ssd3/mayiling/apt_agent_runtime')
sources=[ROOT/f'remaining_v1/modality_hvg{w}' for w in (2000,5000)]
meta=pd.read_csv(sources[0]/'prepared/cells.csv')
assert len(meta)==361792 and meta.cell_id.is_unique and meta.sample_id.nunique()==40
for width,source in zip((2000,5000),sources):
    assert json.loads((source/'completion.json').read_text())['status']=='PASS'
    files=list(source.glob('s*_f*.npz')); assert len(files)==300
    for fold in range(5):
        for stage in ('inner','outer'):
            stem=source/'prepared'/f'f{fold}_{stage}'
            train=np.loadtxt(str(stem)+'_train.txt',dtype=int)
            test=np.loadtxt(str(stem)+'_test.txt',dtype=int)
            assert not set(meta.sample_id.iloc[train]) & set(meta.sample_id.iloc[test])
            genes=Path(str(stem)+'_genes.txt').read_text().splitlines()
            assert len(genes)==len(set(genes))==width
    for path in files:
        with np.load(path) as b:
            assert b['cm'].sum()==len(b['pred'])==len(b['cell_ids'])
            with np.load(sources[0]/path.name) as ref:
                for key in ('cell_ids','truth','classes','patients'):
                    np.testing.assert_array_equal(b[key],ref[key])
target=ROOT/'cell_identity_questions_v1/hvg2000'
assert json.loads((target/'completion.json').read_text())['status']=='PASS'
for seed in SEEDS:
    for model in ('sgd','mlp'):
        cells=[]; patients=[]
        for fold in range(5):
            with np.load(target/f's{seed}_f{fold}_{model}.npz') as b:
                cells.extend(b['cell_ids']); patients.extend(b['patients'])
                for mode in ('rna','apt','rna_apt','rna_noise'):
                    with np.load(sources[0]/f's{seed}_f{fold}_fine_{model}_{mode}.npz') as ref:
                        for key in ('cell_ids','truth','classes'):
                            np.testing.assert_array_equal(b[key],ref[key])
                        np.testing.assert_array_equal(b['pred__'+mode],ref['pred'])
                for mode in ('rna_oracle','apt_oracle','prior_oracle'):
                    np.testing.assert_array_equal(b['parent'][b['pred__'+mode]],b['parent'][b['truth']])
        assert len(cells)==len(set(cells))==361792
        assert len(patients)==len(set(patients))==40
report=dict(status='PASS',reference_outer_fits=600,targeted_blocks=30,
    checked=['cross-width barcodes/truth/classes/patient order','patient-disjoint preprocessing partitions',
             'unique per-stage HVG count','stored confusion mass','targeted source-prediction agreement',
             'oracle parent constraints','complete OOF cell and patient coverage'],
    limitation='HVG computation is additionally documented by the train-only preparation implementation; this audit does not independently reconstruct raw gene matrices.')
(target/'final_artifact_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
