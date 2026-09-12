"""Read-only metadata audit of the frozen sequential runs; no model fitting."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def audit(root):
    results = {}
    for dataset in ('apt', 'combat_rna', 'onek1k'):
        base = root/dataset
        protocol = json.loads((base/'protocol.json').read_text())
        assert protocol['seeds'] == 10 and protocol['total'] == 6400
        completion = json.loads((base/'completion.json').read_text())
        assert completion == dict(complete=True, runs=2400)
        manifests, quotas, realized = {}, {}, {8: [], 32: []}
        for fold, seed, task, model, p, c in itertools.product(
                range(5), range(10), ('coarse','fine'),
                ('logistic_regression','xgboost','mlp'), (8,32), 'ABCD'):
            stem = base/'runs'/f'f{fold}_s{seed}_{task}_{model}_P{p}_{c}'
            row = json.loads(stem.with_suffix('.json').read_text())
            assert stem.with_suffix('.npz').stat().st_size > 0
            assert row['config'] == protocol
            assert all(row[k] == v for k,v in dict(fold=fold,seed=seed,task=task,model=model,p=p,condition=c).items())
            counts, quota = row['class_counts'], row['reference_quotas']
            assert sum(counts) == sum(quota) == 6400
            assert 0 < row['actual_subjects'] <= p
            assert len(counts) == len(quota) == len(row['class_donor_coverage'])
            assert all(0 <= d <= row['actual_subjects'] for d in row['class_donor_coverage'])
            if c != 'A':
                assert [x>0 for x in counts] == [x>0 for x in quota]
            if c in 'CD':
                assert counts == quota
            key = (fold,seed,task,p,c)
            assert manifests.setdefault(key,row['train_manifest_sha256']) == row['train_manifest_sha256']
            assert quotas.setdefault((fold,seed,task),quota) == quota
            realized[p].append(row['actual_subjects'])
        summary = json.loads((base/'summary.json').read_text())
        assert len(summary['results']) == 6
        for row in summary['results']:
            assert row['bootstrap'] == 2000
            assert set(row['conditions']) == set('ABCD')
            for group in ('conditions','contrasts'):
                for v in row[group].values():
                    assert v['lower'] <= v['upper'] and 0 <= v['bootstrap_positive_fraction'] <= 1
            for contrast, v in row['contrasts'].items():
                a,b = contrast.split('-')
                assert abs(v['estimate']-(row['conditions'][a]['estimate']-row['conditions'][b]['estimate'])) < 1e-12
        results[dataset] = dict(status='PASS', fits=2400,
            realized_subject_range={p:[min(v),max(v)] for p,v in realized.items()},
            summary_sha256=hashlib.sha256((base/'summary.json').read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256((base/'protocol.json').read_bytes()).hexdigest())
    return dict(status='PASS', datasets=results, fits=7200,
        scope='Independent metadata, quota, paired training-manifest, and summary audit. NPZ contents and training execution are not revalidated here; the frozen runner and aggregator enforce disjointness and OOF pairing.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
