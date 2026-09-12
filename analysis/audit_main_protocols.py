"""Read owned run artifacts on vllab11; never fit or select a model.

Run from the runtime environment with numpy/pandas/torch/sklearn available.
Checkpoint loading is restricted to this project's known training outputs.
Only aggregate metadata, not cell IDs or tensors, are emitted.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HOME = Path('/home/mayiling/projs/apt_agent')
LEGACY = HOME / 'cell_JEPA/outputs'
RUN = Path('/ssd3/mayiling/apt_agent_runtime/remaining_v1')
sys.path.insert(0, str(HOME / 'analysis'))
import patient_cell_scaling as core


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    meta, x, features = core.load_data()
    folds = core.load_folds()
    enc = core.fit_label_encoders(meta)
    ids = meta.sample_id.to_numpy(str)
    cells = meta.cell_id.to_numpy(str)
    indexed = meta.set_index('cell_id')
    counts = meta.groupby('sample_id').size()
    records, sources, scores = [], {}, []

    def source(path):
        path = Path(path)
        if path.exists():
            sources[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
            return str(path)
        return 'unavailable: ' + str(path)

    def read(path):
        source(path)
        return json.loads(Path(path).read_text())

    def part(patients):
        return {'patient_count': len(patients),
                'cell_count': int(counts.loc[patients].sum()) if patients else 0}

    def sb(truth, pred, patients, k):
        cm = np.zeros((k, k), float)
        for p in np.unique(patients):
            ix = patients == p
            c = np.bincount(truth[ix] * k + pred[ix], minlength=k*k).reshape(k,k)
            cm += c / c.sum()
        den = cm.sum(0) + cm.sum(1)
        return float(np.divide(2 * cm.diagonal(), den, out=np.zeros(k), where=den > 0).mean())

    for model in ['logistic_regression', 'xgboost']:
        for task, col in [('coarse','coarse_subtype'), ('fine','cell_subtype')]:
            path = LEGACY / f'classical_baselines_kfold5/{task}/{model}/pooled_oof_predictions.csv'
            frame = pd.read_csv(source(path))
            assert frame.cell_id.is_unique and set(frame.cell_id) == set(cells)
            keyed = indexed.loc[frame.cell_id]
            assert np.array_equal(frame.sample_id.astype(str), keyed.sample_id.astype(str))
            assert np.array_equal(frame.y_true.astype(str), keyed[col].astype(str))
            scores.append(dict(model=model, task=task, seed=None,
                sb=sb(enc[task].transform(frame.y_true), enc[task].transform(frame.y_pred),
                      frame.sample_id.to_numpy(str), len(enc[task].classes_))))
            for f in range(5):
                assert set(frame.loc[frame.fold == f, 'sample_id']) == set(folds[f])
                dev = [p for j in range(5) if j != f for p in folds[j]]
                records.append(dict(model=model, task=task, fold=f, stage='final_fit',
                    seed=None, seed_status='unavailable original CLI; source default 42, LR lbfgs has no RNG seed',
                    **part(dev), validation=part([]), test=part(folds[f]),
                    refit='no separate selection stage in surviving source',
                    selection='fixed recipe in surviving source; original search history unavailable',
                    preprocessing='provided apt_expression, train-only mean/std; source evidence, original scaler unavailable',
                    sampling='all fit cells; LR class_weight=balanced' if model=='logistic_regression' else 'all fit cells; no sample weights; tree subsample=.8',
                    supervision=task+' labels only', artifact_source=str(path),
                    manifest_source=source(LEGACY/'classical_baselines_kfold5/fold_assignment.json'),
                    input_features_status='current keyed matrix and surviving source; original estimator feature snapshot unavailable'))

    for variant in ['flat_ce','hce','soft_cascade']:
        base = LEGACY / ('dropcascade_kfold5' if variant == 'soft_cascade' else 'matched_ft_transformer/'+variant)
        pooled = pd.read_csv(source(base/'pooled_oof_predictions.csv'))
        task_blocks = {'fine':[], 'coarse':[]}
        for f in range(5):
            directory = base/f'fold{f}'
            cfg = read(directory/'resolved_config.json')
            split = pd.read_csv(source(directory/'split.csv'))
            pats = {s:split.loc[split.split == s,'sample_id'].astype(str).tolist() for s in ['train','val','test']}
            assert set(pats['test']) == set(folds[f])
            assert not (set(pats['train']) & set(pats['test']) or set(pats['val']) & set(pats['test']))
            assert set(pats['train']) | set(pats['val']) | set(pats['test']) == set(ids)
            replay = RUN/f'oracle/{variant}_f{f}.npz'
            audit = read(replay.with_suffix('.json'))
            assert audit['ordinary_predictions_exact']
            checkpoint = directory/'final.pt'
            assert sha(checkpoint) == audit['checkpoint_sha256']
            source(checkpoint)
            final_state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            best_path = directory/'best.pt'
            best_state = torch.load(best_path, map_location='cpu', weights_only=False)
            source(best_path)
            assert final_state['model'].keys() == best_state['model'].keys()
            assert all(torch.equal(value, best_state['model'][key]) for key,value in final_state['model'].items())
            training_cfg = best_state['config']
            a = np.load(source(replay), allow_pickle=False)
            positions = indexed.index.get_indexer(a['cell_ids'])
            assert (positions >= 0).all() and len(set(a['cell_ids'])) == len(positions)
            target = indexed.iloc[positions]
            assert set(a['cell_ids']) == set(meta.loc[meta.sample_id.isin(folds[f]),'cell_id'])
            assert np.array_equal(a['sample_ids'].astype(str), target.sample_id.astype(str))
            # Verify the historical pooled rows, not merely the later replay summary.
            old = pooled[pooled.fold == f].reset_index(drop=True)
            assert np.array_equal(old.sample_id.astype(str), a['sample_ids'].astype(str))
            mapping = read(directory/'label_mapping.json')
            for task, truth_key, pred, labels, col in [
                ('fine','truth',a['prob'].argmax(1),mapping['subtypes'],'cell_subtype'),
                ('coarse','coarse_truth',a['coarse_pred'],mapping['lineages'],'coarse_subtype')]:
                labels = np.asarray(labels)
                assert np.array_equal(labels[a[truth_key]], target[col].to_numpy(str))
                assert np.array_equal(old[task+'_true_id'], a[truth_key])
                assert np.array_equal(old[task+'_pred_id'], pred)
                task_blocks[task].append((enc[task].transform(labels[a[truth_key]]),
                    enc[task].transform(labels[pred]), a['sample_ids'].astype(str)))
            logpath = directory/'training_log.csv'
            log = pd.read_csv(source(logpath))
            valcol = 'val_macro_f1_all_classes' if variant == 'soft_cascade' else 'val_fine_macro_f1'
            best = log.loc[log[valcol].idxmax()]
            assert int(best_state['epoch']) == int(best.epoch)
            assert abs(float(best_state['score']) - float(best[valcol])) < 1e-10
            records.append(dict(model=variant, task='coarse+fine', fold=f, stage='selected_checkpoint',
                seed=cfg['train']['seed'], **part(pats['train']), validation=part(pats['val']), test=part(pats['test']),
                refit='no development refit; selected checkpoint, calibration_epochs=0',
                selection='validation fine CW Macro-F1, first strict maximum, patience 10; '+('all 27 classes' if variant=='soft_cascade' else 'observed-label union'),
                final_equals_best_tensors=True,
                original_calibration_budget=training_cfg['schedule'].get('calibration_epochs'),
                final_replay_calibration_budget=cfg['schedule'].get('calibration_epochs'),
                observed_epochs=len(log), best_logged_epoch=int(best.epoch),
                declared_max_epochs=training_cfg['train']['epochs'],
                max_epoch_status='recovered from selected best.pt training config; final model tensors identical',
                evaluated_recipe_count='one retained configuration; complete historical search count unavailable',
                config=training_cfg, resolved_config=cfg, preprocessing='provided apt_expression, train-only mean/std, no quantile binning',
                sampling='all fit cells, shuffled mini-batches; no weighted sampler',
                supervision='fine/coarse and disease contrastive labels' if variant=='soft_cascade' else 'fine/coarse hierarchy labels, no disease objective',
                artifact_source=source(directory/'test_predictions.csv'), pooled_source=str(base/'pooled_oof_predictions.csv'),
                config_source=str(directory/'resolved_config.json'), log_source=str(logpath),
                manifest_source=str(directory/'split.csv'), checkpoint_source=str(checkpoint),
                alignment_source=str(replay), alignment='exact replay and keyed test set, no intersection filtering',
                input_features_status='293 features in saved config; exact historical feature-ID list not serialized in this audit'))
            del final_state, best_state
        for task, blocks in task_blocks.items():
            y,p,s = [np.concatenate([b[i] for b in blocks]) for i in range(3)]
            scores.append(dict(model=variant, task=task, seed=42, sb=sb(y,p,s,len(enc[task].classes_))))

    protocol = read(RUN/'full_mlp/protocol.json')
    for seed in protocol['seeds']:
        for task,col in [('coarse','coarse_subtype'),('fine','cell_subtype')]:
            blocks=[]
            for f in range(5):
                path = RUN/f'full_mlp/s{seed}_f{f}_{task}.npz'
                a=np.load(source(path),allow_pickle=False)
                info=read(path.with_suffix('.json'))
                dev=[p for j in range(5) if j!=f for p in folds[j]]
                tr=[p for j in range(5) if j not in [f,(f+1)%5] for p in folds[j]]
                va=folds[(f+1)%5]
                assert set(info['train_patients'])==set(dev) and info['train_cells']==part(dev)['cell_count']
                assert set(a['cell_ids'])==set(meta.loc[meta.sample_id.isin(folds[f]),'cell_id'])
                target=indexed.loc[a['cell_ids']]
                y=enc[task].transform(target[col]); s=target.sample_id.to_numpy(str)
                assert np.array_equal(y,a['truth']) and np.array_equal(a['classes'],enc[task].classes_)
                assert len(info['trials'])==2 and all(len(t['history'])==50 for t in info['trials'])
                selected=max(info['trials'],key=lambda t:t['score'])
                assert selected==info['selected']
                # Known owned checkpoint: inspect saved feature/scaler metadata without inference.
                state=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
                assert state['features']==features and state['seed']==seed and state['fold']==f
                dix=np.flatnonzero(np.isin(ids,dev))
                from sklearn.preprocessing import StandardScaler
                scaler=StandardScaler().fit(x[dix])
                assert np.allclose(scaler.mean_,state['mean'],rtol=1e-10,atol=1e-10)
                assert np.allclose(scaler.scale_,state['scale'],rtol=1e-10,atol=1e-10)
                source(path.with_suffix('.pt'))
                blocks.append((y,a['prob'].argmax(1),s))
                for stage,patients in [('inner_selection',tr),('outer_refit',dev)]:
                    records.append(dict(model='plain_mlp',task=task,fold=f,stage=stage,seed=seed,
                        **part(patients),validation=part(va) if stage=='inner_selection' else part([]),test=part(folds[f]),
                        refit='fresh initialization and scaler on all development cells',
                        selection='task-specific validation SB Macro-F1; two decays x 50 epochs; first strict maximum',
                        candidates=[t['decay'] for t in info['trials']],evaluated_configs=len(info['trials']),
                        selected_decay=selected['decay'],selected_epochs=selected['epoch'],
                        preprocessing='provided apt_expression; stage-specific training-only StandardScaler; outer saved scaler verified',
                        sampling='all stage fit cells, shuffled mini-batches; unweighted CE',supervision=task+' labels only',
                        artifact_source=str(path),config_source=str(RUN/'full_mlp/protocol.json'),log_source=str(path.with_suffix('.json')),
                        checkpoint_source=str(path.with_suffix('.pt')),input_features_status='saved checkpoint features exactly match keyed current 293-column input'))
            y,p,s=[np.concatenate([b[i] for b in blocks]) for i in range(3)]
            scores.append(dict(model='plain_mlp',task=task,seed=seed,sb=sb(y,p,s,len(enc[task].classes_))))
    for name in ['run_full_budget_mlp.py','patient_cell_scaling.py','export_lineage_oracle.py']:
        source(HOME/'analysis'/name)
    for name in ['train_classical_baselines.py','train_classical_baselines_kfold.py','train_matched_ft_transformer.py','train_soft_lineage_cascade.py']:
        source(HOME/'cell_JEPA/apt_jepa/scripts'/name)
    for pattern in ['classical_baselines_kfold5.log','matched_ft_*fold*.log','matched_best_eval*.log','dropcascade*log']:
        for path in HOME.glob(pattern):
            source(path)
    input_paths=[core.METADATA_PATH,core.ANNOTATION_PATH,core.MAPPING_PATH,core.FOLD_PATH]
    matrix=core.DATA_DIR/'apt_expression.parquet'
    if not matrix.exists(): matrix=core.DATA_DIR/'apt_expression.csv'
    input_paths.append(matrix)
    for path in input_paths: source(path)
    print(json.dumps(dict(records=records,scores=scores,sources=sources,mlp_protocol=protocol,
        input_sources=list(map(str,input_paths)),
        features=features,feature_status='current input checked; historical serialized feature IDs unavailable except MLP',
        test_alignment='all six models verified on identical keyed outer test sets; neural via exact replay',
        metric='subject-normalized pooled fixed-class Macro-F1; no refitting; no test selection',new_training=0),indent=2))


if __name__=='__main__':
    main()
