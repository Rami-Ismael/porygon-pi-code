import json, random, time
import numpy as np, torch
from corpus import load_corpus
from encode import Vocab, Legality, FIELDS, NF, NSLOT
from model import MaskedFieldModel, random_mask, permute_slots
from pilot import set_seed, COLS
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
set_seed(0); rng = np.random.default_rng(0)
teams,_ = load_corpus(); teams=[t for t in teams if len(t)==6]
V,L = Vocab(teams), Legality(teams)
X = np.stack([V.encode(t) for t in teams])
idx = rng.permutation(len(X)); ntest=int(len(X)*0.15)
Xte,Xtr = X[idx[:ntest]], X[idx[ntest:]]
np.save("/tmp/vgc-pilot/Xtr.npy",Xtr); np.save("/tmp/vgc-pilot/Xte.npy",Xte)
model = MaskedFieldModel(V).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
xtr = torch.tensor(Xtr, device=DEV)
t0=time.time()
for ep in range(600):
    model.train()
    perm = torch.randperm(xtr.shape[0], device=DEV)
    for i in range(0, xtr.shape[0], 64):
        xb = permute_slots(xtr[perm[i:i+64]])
        loss = model.loss(xb, random_mask(xb))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    sched.step()
    if (ep+1)%100==0: print(ep+1, float(loss), time.time()-t0, flush=True)
torch.save(model.state_dict(), "/tmp/vgc-pilot/probe_model.pt")
print("done", time.time()-t0)
