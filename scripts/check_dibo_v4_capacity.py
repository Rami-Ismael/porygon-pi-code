"""One real-size posterior update and refinement batch; no battle calls."""
import hashlib
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import dibo_comparison_v2 as C
import dibo_v4 as D
import dibo_core_v4 as Core
from checkpoints import load_checkpoint

root=Path(__file__).resolve().parents[1]
cfg=C.config()
runtime=D.Runtime(cfg,root/'results/dibo_v4_capacity_check.json')
try:
    harness=root/'results/dibo_v4_harness_20261013/dibo.json'
    ckdir=D.L.CK/('dibo_v4_'+hashlib.sha256(str(harness).encode()).hexdigest()[:12])
    ck=load_checkpoint(ckdir/'round0.pt',map_location=D.L.DEV)
    prior=Core.posterior_copy(Core.PathSampler(runtime.base,D.D.ORDER,D.L.DEV))
    prior.model.load_state_dict(ck['prior'])
    ensemble=runtime.ensemble(cfg['seed']);ensemble.load_state_dict(ck['ensemble']);ensemble.eval()
    posterior=Core.posterior_copy(prior)
    x=torch.tensor([r['x'] for r in ck['rows']],device=D.L.DEV)
    def tilt(x):
        mu,sd=ensemble.standardized(x)
        return cfg['beta']*(mu+sd)
    before=time.time()
    Core.train_rtb(prior,posterior,tilt,1,256,cfg['rtb_lr'],replay=x,temperature=cfg['beta'],microbatch=128)
    print(f'256-trajectory update with microbatch 128 passed in {time.time()-before:.1f}s',flush=True)
    before=time.time()
    records=runtime.records(posterior.sample(128),prior,ensemble)
    runtime.cfg=dict(cfg,local_steps=1)
    refined=runtime.refine(records,prior,ensemble)
    assert all(b['log_target']>=a['log_target'] for a,b in zip(records,refined))
    print(f'128-candidate exact-acceptance refinement passed in {time.time()-before:.1f}s',flush=True)
finally:
    runtime.close()
