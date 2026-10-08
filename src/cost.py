import time, numpy as np, torch, random
from collections import defaultdict
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from model import MaskedFieldModel, resample_fields
from pilot import op_uniform, COLS
DEV="mps" if torch.backends.mps.is_available() else "cpu"
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V=Vocab(teams); X=np.stack([V.encode(t) for t in teams])
m=MaskedFieldModel(V).to(DEV).eval()
rng=np.random.default_rng(1)
src=X[rng.integers(0,len(X),256)]
cols=[list(rng.choice(COLS,3,replace=False)) for _ in range(256)]
t=time.time(); resample_fields(m,torch.tensor(src,device=DEV),cols,gen=None); tm=time.time()-t
slotcols=[list(range(s*NF,(s+1)*NF)) for s in rng.integers(0,NSLOT,256)]
t=time.time(); resample_fields(m,torch.tensor(src,device=DEV),slotcols,gen=None); ts=time.time()-t
t=time.time()
for _ in range(50): op_uniform(src,cols,V,rng)
tu=(time.time()-t)/50
print("256 proposals, k=3 scatter: model %.2fs  uniform %.4fs  -> model is %.0fx slower per proposal"%(tm,tu,tm/tu))
print("256 proposals, 8-field slot: model %.2fs"%ts)
print("plus 578.8s one-off training (results_s0.json train_time_s)")
print("model forward passes per k=3 proposal: 3 (batch size 1, one per field)  vs 0 for every baseline")
