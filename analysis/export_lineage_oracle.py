"""Read frozen neural checkpoints; verify ordinary predictions before oracle scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cell_JEPA'))
from apt_jepa.data.dataset import build_dataloaders, load_merged_dataframe
from apt_jepa.scripts.train_soft_lineage_cascade import apply_explicit_lineage_mapping, build_subtype_to_lineage
from apt_jepa.scripts.train_matched_ft_transformer import MatchedFTTransformer, decode
from apt_jepa.models.soft_lineage_cascade import SoftLineageCascade
from apt_jepa.losses.hierarchical_cross_entropy import build_two_level_reachability
from run_full_budget_mlp import oracle, matrices, score, atomic_json


def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


def export(args):
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda')
    for variant in ['flat_ce','hce','soft_cascade']:
        for fold in range(5):
            dest=args.output/f'{variant}_f{fold}.npz'
            if dest.exists(): continue
            directory=(ROOT/'cell_JEPA/outputs/dropcascade_kfold5' if variant=='soft_cascade'
                else ROOT/'cell_JEPA/outputs/matched_ft_transformer'/variant)/f'fold{fold}'
            checkpoint=directory/'final.pt'
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            cfg=state['config']; d=cfg['data']
            meta,x,_=load_merged_dataframe(str(resolve(d['data_dir'])),
                str(resolve(d['metadata_path'])),str(resolve(d['annotation_path'])),
                subtype_col_in_annotation=d.get('subtype_col_in_annotation'),
                drop_missing_subtype=d.get('drop_missing_subtype',True),
                drop_unknown=d.get('drop_unknown',True),coarse_mapping_config=None)
            assert meta.cell_id.is_unique and len(meta)==361792
            meta,_=apply_explicit_lineage_mapping(meta,str(resolve(d['coarse_mapping_config'])))
            split=pd.read_csv(directory/'split.csv')
            assert split.sample_id.is_unique
            bundle=build_dataloaders(meta,x,split,batch_size=64,num_workers=0,
                use_quantile_binning=d.get('use_quantile_binning',False),split_mode='sample')
            assert np.array_equal(bundle.merged_df.cell_id.to_numpy(),meta.cell_id.to_numpy())
            labels=json.loads((directory/'label_mapping.json').read_text())
            assert labels['subtypes']==bundle.subtype_encoder.classes_.tolist()
            assert labels['lineages']==bundle.coarse_encoder.classes_.tolist()
            parent=np.asarray(build_subtype_to_lineage(bundle),dtype=int)
            k=len(labels['subtypes']); nc=len(labels['lineages'])
            assert (nc,k)==(5,27)
            m=cfg['model']
            reach=build_two_level_reachability(parent,nc,k).to(device)
            if variant=='soft_cascade':
                keys=['num_features','hidden_dim','n_layers','n_heads','ff_dim','dropout',
                    'projection_dim','routing_temperature','routing_epsilon','expert_scale','routing_strength']
                model=SoftLineageCascade(parent,nc,**{key:m[key] for key in keys if key in m})
                model.routing_strength=state.get('routing_strength',m.get('routing_strength',.5))
            else: model=MatchedFTTransformer(nc,k,m)
            model.load_state_dict(state['model'],strict=True)
            model=model.to(device).eval()
            store={key:[] for key in ['prob','truth','coarse_truth','coarse_pred','cell_ids','sample_ids']}
            with torch.no_grad():
                for batch in bundle.test_loader:
                    output=model(batch['x'].to(device))
                    if variant=='soft_cascade':
                        logits=output['fine_logits']
                        cp=output['coarse_logits'].argmax(1)
                    else:
                        logits=model.split_logits(output)[1]
                        cp,_=decode(model,output,reach,variant)
                    store['prob'].append(torch.softmax(logits,dim=1).cpu().numpy())
                    store['truth'].append(batch['subtype_label'].numpy())
                    store['coarse_truth'].append(batch['coarse_label'].numpy())
                    store['coarse_pred'].append(cp.cpu().numpy())
                    store['cell_ids'].append(np.asarray(batch['cell_id'],dtype=str))
                    store['sample_ids'].append(np.asarray(batch['sample_id'],dtype=str))
            a={key:np.concatenate(value) for key,value in store.items()}
            assert np.allclose(a['prob'].sum(1),1,atol=1e-5)
            assert np.array_equal(parent[a['truth']],a['coarse_truth'])
            old=pd.read_csv(directory/'test_predictions.csv')
            checks={'sample_id':a['sample_ids'],'fine_true_id':a['truth'],
                'coarse_true_id':a['coarse_truth'],'fine_pred_id':a['prob'].argmax(1),
                'coarse_pred_id':a['coarse_pred']}
            for key,values in checks.items():
                if not np.array_equal(old[key].to_numpy(),values):
                    raise ValueError(f'Frozen prediction reconstruction failed: {variant}/{fold}/{key}')
            op=oracle(a['prob'],a['truth'],parent)
            patients,cm=matrices(a['truth'],a['prob'].argmax(1),a['sample_ids'],k)
            _,ocm=matrices(a['truth'],op,a['sample_ids'],k)
            np.savez_compressed(dest.with_suffix('.tmp.npz'),**a,patients=patients,cm=cm,
                oracle_cm=ocm,parent=parent,classes=np.asarray(labels['subtypes'],dtype=str))
            os.replace(dest.with_suffix('.tmp.npz'),dest)
            atomic_json(dest.with_suffix('.json'),dict(status='PASS',model=variant,fold=fold,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                checkpoint=str(checkpoint),ordinary_predictions_exact=True,
                cell_ids_unique=len(set(a['cell_ids']))==len(a['cell_ids']),
                diagnostic='mask frozen scores to true lineage; no routing override or refit'))
            print('COMPLETE',dest.name,flush=True)
            del model,bundle,x,meta,state
            torch.cuda.empty_cache()
    rows=[]; order=None
    for variant in ['flat_ce','hce','soft_cascade']:
        blocks=[np.load(args.output/f'{variant}_f{f}.npz') for f in range(5)]
        ids=np.concatenate([b['patients'] for b in blocks])
        assert len(set(ids))==len(ids)==40
        cells=np.concatenate([b['cell_ids'] for b in blocks])
        assert len(set(cells))==len(cells)==361792
        if order is None: order=cells
        assert np.array_equal(cells,order)
        cm=np.concatenate([b['cm'] for b in blocks]); oc=np.concatenate([b['oracle_cm'] for b in blocks])
        rng=np.random.default_rng(20260912)
        draws=rng.integers(0,40,(2000,40))
        delta=np.array([score(oc[d])-score(cm[d]) for d in draws])
        rows.append(dict(model=variant,ordinary_sb=score(cm),oracle_sb=score(oc),
            delta=score(oc)-score(cm),low=np.quantile(delta,.025),high=np.quantile(delta,.975)))
    pd.DataFrame(rows).to_csv(args.output/'summary.csv',index=False)
    atomic_json(args.output/'completion.json',dict(status='PASS',checkpoints=15,
        cells_per_model=361792,patients=40,ordinary_prediction_reconstruction='exact',
        uncertainty='paired patient bootstrap conditional on frozen models'))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    torch.set_num_threads(4)
    export(p.parse_args())
