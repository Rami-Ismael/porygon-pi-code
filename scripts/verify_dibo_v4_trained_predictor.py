"""Check ownership sensitivity and permutation invariance on saved trained models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import dibo_v4 as D
from checkpoints import load_checkpoint


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path)
    root=ap.parse_args().directory.resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    runtime=D.Runtime(protocol['config'],root/'dibo.json')
    results=[]
    try:
        for generation in [0,protocol['config']['rounds']-1]:
            ck=load_checkpoint(runtime.checkdir/f'round{generation}.pt',map_location=D.L.DEV)
            ensemble=runtime.ensemble(protocol['config']['seed'])
            ensemble.load_state_dict(ck['ensemble']);ensemble.eval()
            x=torch.tensor([r['x'] for r in ck['rows'][:128]],device=D.L.DEV).reshape(-1,6,8)
            moved=x.clone();moved[:,:,2]=torch.roll(x[:,:,2],1,1)
            reordered=x[:,[5,4,3,2,1,0],:]
            moves=x.clone();moves[:,:,3:7]=x[:,:,3:7].flip(2)
            with torch.no_grad():
                base=ensemble(x.flatten(1))[0]
                deltas={name:(ensemble(value.flatten(1))[0]-base).abs() for name,value in [('item_ownership',moved),('slot_order',reordered),('move_order',moves)]}
            row=dict(round=generation,team_count=len(x),units='predicted win-rate probability',
                     differences={k:dict(mean=float(v.mean()),maximum=float(v.max())) for k,v in deltas.items()})
            assert row['differences']['slot_order']['maximum']<1e-5
            assert row['differences']['move_order']['maximum']<1e-5
            assert row['differences']['item_ownership']['maximum']>1e-5
            results.append(row)
    finally:
        runtime.close()
    (root/'trained-predictor-check.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
