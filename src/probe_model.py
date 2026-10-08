import time, random, json
import numpy as np, torch
from collections import defaultdict
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from model import MaskedFieldModel, resample_fields
from pilot import evaluate, memorization_stats, set_seed, COLS
DEV="mps" if torch.backends.mps.is_available() else "cpu"
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
model=MaskedFieldModel(V).to(DEV); model.load_state_dict(torch.load("/tmp/vgc-pilot/probe_model.pt")); model.eval()

def recanon(R):
    out=R.copy()
    for b in range(len(R)):
        sl=R[b].reshape(NSLOT,NF); out[b]=sl[np.argsort(sl[:,0],kind="stable")].reshape(-1)
    return out

@torch.no_grad()
def resample_ordered(model,x,cols,order_fn,temp=1.0,gen=None):
    x=x.clone()
    for b in range(x.shape[0]):
        cs=order_fn(list(cols[b]))
        for c in cs: x[b,c]=0
        for c in cs:
            h=model(x[b:b+1]); lg=model.logits_at(h,c)[0]/temp; lg[0]=-1e9
            x[b,c]=torch.multinomial(torch.softmax(lg,-1),1,generator=gen).item()
    return x

def rand_order(cs): random.shuffle(cs); return cs
def species_first(cs): return sorted(cs, key=lambda c:(c%NF!=0, random.random()))

N=256
print("=== SLOT regime: does sampling ORDER explain model legal=0.023? ===")
X_src=Xtr[rng.integers(0,len(Xtr),N)]; so=rng.integers(0,NSLOT,N)
cols=[list(range(s*NF,(s+1)*NF)) for s in so]
xt=torch.tensor(X_src,device=DEV)
for label,ofn,temp in [("random order T=1.0 (pilot)",rand_order,1.0),
                       ("SPECIES-FIRST T=1.0",species_first,1.0),
                       ("SPECIES-FIRST T=0.5",species_first,0.5),
                       ("SPECIES-FIRST T=0.2",species_first,0.2)]:
    random.seed(0); g=torch.Generator(device=DEV); g.manual_seed(0)
    t0=time.time(); Xa=resample_ordered(model,xt,cols,ofn,temp,g).cpu().numpy(); dt=(time.time()-t0)/N
    e=evaluate(V,L,Xa,X_src,Xtr)
    print(f"  {label:28s} legal {e['legal']:.3f} changed {e['changed']:.3f} "
          f"legal&novel {e['legal_and_novel']:.3f} {dt*1e3:.0f} ms/proposal viol {e['violations']}")

print("\n=== SCATTER k=3: legality among proposals that actually CHANGED ===")
X_src=Xtr[rng.integers(0,len(Xtr),N)]
cols=[list(rng.choice(COLS,3,replace=False)) for _ in range(N)]
random.seed(0); g=torch.Generator(device=DEV); g.manual_seed(0)
t0=time.time(); Xa=resample_fields(model,torch.tensor(X_src,device=DEV),cols,gen=g).cpu().numpy()
dt=(time.time()-t0)/N
e=evaluate(V,L,Xa,X_src,Xtr)
ch=(Xa!=X_src).any(1)
leg=np.array([not L.violations([V.decode_field(c,int(Xa[i,c])) for c in range(COLS)]) for i in range(N)])
print(f"  model: legal {e['legal']:.3f}  changed {e['changed']:.3f}  "
      f"legal|changed {leg[ch].mean():.3f}  legal|unchanged {leg[~ch].mean():.3f}  {dt*1e3:.0f} ms/proposal")

print("\n=== UNCONDITIONAL: is legal=0.000 a decoder artifact? ===")
def uncond(sweeps,temp,order):
    xz=torch.zeros(64,COLS,dtype=torch.long,device=DEV)
    random.seed(0); g=torch.Generator(device=DEV); g.manual_seed(7)
    Xg=resample_ordered(model,xz,[list(range(COLS))]*64,order,temp,g)
    for _ in range(sweeps-1):
        Xg=resample_ordered(model,Xg,[list(range(COLS))]*64,order,temp,g)
    return Xg.cpu().numpy()
for sw,tp in [(1,1.0),(1,0.5),(3,1.0),(5,1.0),(5,0.5),(10,0.7)]:
    Xg=uncond(sw,tp,rand_order); e=evaluate(V,L,Xg,np.full_like(Xg,-1),Xtr)
    mr=memorization_stats(Xg,Xtr); mc=memorization_stats(recanon(Xg),Xtr)
    print(f"  sweeps={sw} T={tp}: legal {e['legal']:.3f} unique {e['unique']:.3f} "
          f"| NNdist raw {mr['mean_nn_dist']:.1f} CANONICALISED {mc['mean_nn_dist']:.1f} "
          f"| memorised raw {mr['memorized']:.3f} canon {mc['memorized']:.3f} exact {mc['exact_copy']:.3f}")
