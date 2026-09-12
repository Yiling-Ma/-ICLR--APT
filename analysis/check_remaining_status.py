"""Compact read-only status for the four independent experiment outputs."""
import json
from pathlib import Path

root=Path('/ssd3/mayiling/apt_agent_runtime/remaining_v1')
jobs=[('full_mlp',30,'s*_f*_*.npz','mlp.log'),
      ('oracle',15,'*_f*.npz','oracle.log'),
      ('modality_hvg2000',300,'s*_f*_*.npz','modality.log'),
      ('modality_hvg5000',300,'s*_f*_*.npz','modality.log')]
rows=[]
for name,expected,pattern,log in jobs:
    directory=root/name
    count=sum('.tmp.' not in p.name for p in directory.glob(pattern))
    complete=directory/'completion.json'
    payload=json.loads(complete.read_text()) if complete.exists() else None
    rows.append(dict(job=name,completed=count,expected=expected,
        validated_complete=bool(payload and payload.get('status')=='PASS' and count==expected),
        completion=payload,log=str(root/log)))
print(json.dumps(rows,indent=2))
