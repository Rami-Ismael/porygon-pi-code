import json, numpy as np, torch, random
from collections import defaultdict
from corpus import load_corpus, norm
from encode import Vocab, Legality, NF, NSLOT
from model import MaskedFieldModel, random_mask, permute_slots, resample_fields
from pilot import evaluate, memorization_stats, COLS, set_seed
DEV="mps" if torch.backends.mps.is_available() else "cpu"
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams])
idx=rng.permutation(len(X)); ntest=int(len(X)*.15)
Xte,Xtr=X[idx[:ntest]],X[idx[ntest:]]
m=MaskedFieldModel(V).to(DEV)
opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,600)
xtr=torch.tensor(Xtr,device=DEV)
for ep in range(600):
    m.train(); perm=torch.randperm(xtr.shape[0],device=DEV)
    for i in range(0,xtr.shape[0],64):
        xb=permute_slots(xtr[perm[i:i+64]]); loss=m.loss(xb,random_mask(xb))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
    sch.step()
m.eval()
torch.save({"sd":m.state_dict()}, "/tmp/vgc-pilot/ck.pt")
g=torch.Generator(device=DEV); g.manual_seed(7)
xz=torch.zeros(256,COLS,dtype=torch.long,device=DEV)
Xu=resample_fields(m,xz,[list(range(COLS))]*256,gen=g).cpu().numpy()
def canon_rows(A):
    B=A.reshape(len(A),NSLOT,NF).copy()
    for i in range(len(B)):
        o=np.argsort([V.itos["species"][int(B[i,s,0])] for s in range(NSLOT)])
        B[i]=B[i][o]
    return B.reshape(len(A),NSLOT*NF)
raw=memorization_stats(Xu,Xtr); can=memorization_stats(canon_rows(Xu),Xtr)
e=evaluate(V,L,Xu,np.full_like(Xu,-1),Xtr)
print("UNCOND legal %.3f"%e["legal"])
print("  as scored by pilot.py (raw, no canon): mean NN %.1f  memorised %.3f  exact %.3f"%(raw["mean_nn_dist"],raw["memorized"],raw["exact_copy"]))
print("  re-canonicalised before Hamming      : mean NN %.1f  memorised %.3f  exact %.3f"%(can["mean_nn_dist"],can["memorized"],can["exact_copy"]))
json.dump(dict(raw=raw,canon=can,legal=e["legal"]),open("/tmp/vgc-pilot/c3.json","w"),indent=2)
