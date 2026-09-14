"""Correct HCE subtree decoding from frozen weights; retain original exports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from run import Network, SEEDS, core, raw


def main(root):
    torch.set_num_threads(4)
    meta, _, _ = core.load_data()
    x = raw.raw_apt(meta)
    cell_index = {str(c): i for i,c in enumerate(meta.cell_id)}
    reports=[]
    for fold in range(5):
        for seed in SEEDS:
            started=time.time()
            p=root/f'hce_f{fold}'/f'joint_s{seed}.npz'
            archive=p.parent/'original_head_exports'; archive.mkdir(exist_ok=True)
            if not (archive/p.name).exists(): shutil.copy2(p,archive/p.name)
            with np.load(archive/p.name,allow_pickle=False) as data:
                a={k:data[k].copy() for k in data.files}
            ckpt=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False)
            model=Network('hce',a['parent']).to('cuda').eval()
            model.load_state_dict(ckpt['state'],strict=True)
            ix=np.array([cell_index[str(c)] for c in a['cell_ids']])
            xe=((x[ix]-ckpt['mean'])/ckpt['scale']).astype(np.float32)
            fine,coarse=[],[]
            with torch.no_grad():
                for start in range(0,len(xe),128):
                    f,c,_=model(torch.as_tensor(xe[start:start+128],device='cuda'))
                    nodes=torch.softmax(torch.cat([c,f],1),1)
                    subtree=nodes @ model.reach.T
                    fine.append(torch.softmax(f,1).cpu().numpy())
                    coarse.append(subtree[:,:5].cpu().numpy())
            fp=np.concatenate(fine); cp=np.concatenate(coarse)
            np.testing.assert_allclose(fp,a['prob'],rtol=1e-3,atol=1e-4)
            assert np.array_equal(fp.argmax(1),a['prob'].argmax(1))
            assert np.allclose(cp.sum(1),1,atol=1e-5) and np.isfinite(cp).all()
            # Fine scores stay exactly as exported; only the invalid parent readout changes.
            a['coarse_prob']=cp
            np.savez_compressed(p,**a)
            reports.append(dict(fold=fold,seed=seed,checkpoint_sha256=hashlib.sha256(p.with_suffix('.pt').read_bytes()).hexdigest(),
                original_npz_sha256=hashlib.sha256((archive/p.name).read_bytes()).hexdigest(),
                corrected_npz_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                fine_argmax_exact=True,seconds=time.time()-started))
            print('REPLAY',fold,seed,flush=True)
    (root/'hce_decoding_correction.json').write_text(json.dumps(dict(status='PASS',new_training=0,
        checkpoint_replays=15,reason='HCE parent probability must include node and descendant probability mass',
        original_exports='hce_f*/original_head_exports/',affected=['coarse','D1','D2'],
        unchanged=['D0','D4','fine selection','all model weights'],records=reports),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
