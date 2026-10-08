"""Reproduce pilot seed 0 and CACHE every generated array for post-hoc metric audit."""
import json, random, time
from collections import Counter, defaultdict
import numpy as np, torch
from corpus import load_corpus
from encode import Vocab, Legality, team_fields, FIELDS, NF, NSLOT
from model import MaskedFieldModel, random_mask, permute_slots, resample_fields
from pilot import set_seed, evaluate, op_uniform, op_marginal, op_slotcopy, COLS, memorization_stats

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SEED, NPROP, K, EPOCHS = 0, 256, 3, 600
set_seed(SEED); rng = np.random.default_rng(SEED)
teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
V, L = Vocab(teams), Legality(teams)
X = np.stack([V.encode(t) for t in teams])
idx = rng.permutation(len(X)); ntest = int(len(X)*0.15)
Xte, Xtr = X[idx[:ntest]], X[idx[ntest:]]
colvals = defaultdict(list)
for c in range(COLS): colvals[V.key(c)] += list(Xtr[:, c])
colvals = {k: np.array(v) for k, v in colvals.items()}
model = MaskedFieldModel(V).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
xtr = torch.tensor(Xtr, device=DEV)
t0=time.time()
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(xtr.shape[0], device=DEV)
    for i in range(0, xtr.shape[0], 64):
        xb = permute_slots(xtr[perm[i:i+64]])
        loss = model.loss(xb, random_mask(xb))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    sched.step()
    if (ep+1)%100==0: print(ep+1, float(loss), time.time()-t0, flush=True)
model.eval()
torch.save(model.state_dict(), "/tmp/vgc-pilot/model_s0.pt")
store = dict(Xtr=Xtr, Xte=Xte)
for regime in ["scatter","slot"]:
    src_i = rng.integers(0, len(Xtr), NPROP); X_src = Xtr[src_i]
    if regime=="scatter":
        cols = [list(rng.choice(COLS, K, replace=False)) for _ in range(NPROP)]; slot_of=None
    else:
        slot_of = rng.integers(0, NSLOT, NPROP)
        cols = [list(range(s*NF,(s+1)*NF)) for s in slot_of]
    g = torch.Generator(device=DEV); g.manual_seed(SEED)
    Xm = resample_fields(model, torch.tensor(X_src, device=DEV), cols, gen=g).cpu().numpy()
    arms = {"model":Xm, "uniform":op_uniform(X_src,cols,V,rng), "marginal":op_marginal(X_src,cols,V,rng,colvals)}
    if regime=="slot": arms["slotcopy"]=op_slotcopy(X_src, slot_of, Xtr, rng)
    store[regime+"_src"]=X_src
    store[regime+"_cols"]=np.array([sorted(c) for c in cols])
    for n,A in arms.items():
        store[f"{regime}_{n}"]=A
        e=evaluate(V,L,A,X_src,Xtr); print(regime,n,round(e['legal'],4),round(e['legal_and_novel'],4),flush=True)
g = torch.Generator(device=DEV); g.manual_seed(SEED+7)
xz = torch.zeros(NPROP, COLS, dtype=torch.long, device=DEV)
Xu = resample_fields(model, xz, [list(range(COLS))]*NPROP, gen=g).cpu().numpy()
store["uncond"]=Xu
e=evaluate(V,L,Xu,np.full_like(Xu,-1),Xtr); print("uncond",e['legal'],memorization_stats(Xu,Xtr),flush=True)
np.savez("/tmp/vgc-pilot/arrays_s0.npz", **store)
json.dump({k:V.itos[k] for k in V.itos}, open("/tmp/vgc-pilot/vocab.json","w"))
print("DONE")
