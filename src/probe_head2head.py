"""Model vs the zero-parameter count table, IDENTICAL sources and masked columns,
scored under the pilot's oracle and under oracle+ItemClause, with wall-clock cost."""
import time, random
import numpy as np, torch
from collections import defaultdict
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from model import MaskedFieldModel, resample_fields
from pilot import evaluate, op_slotcopy, op_marginal, op_uniform, set_seed, COLS, decode_fields
DEV="mps"
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
model=MaskedFieldModel(V).to(DEV); model.load_state_dict(torch.load("/tmp/vgc-pilot/probe_model.pt")); model.eval()
def viol2(row):
    v=L.violations(decode_fields(V,row))
    its=[V.decode_field(s*NF+2,int(row[s*NF+2])) for s in range(NSLOT)]; its=[i for i in its if i]
    if len(set(its))!=len(its): v=v+["itemclause"]
    return v
pool=defaultdict(lambda: defaultdict(list)); sp_pool=[]
for b in range(len(Xtr)):
    for s in range(NSLOT):
        sl=Xtr[b,s*NF:(s+1)*NF]; sp=int(sl[0]); sp_pool.append(sp)
        for j in range(1,NF): pool[sp][j].append(int(sl[j]))
        pool[sp]['mv'] += [int(sl[j]) for j in range(3,7)]
sp_pool=np.array(sp_pool)
def op_cond(X_src,cols,rng):
    Xa=X_src.copy()
    for b in range(Xa.shape[0]):
        cs=sorted(cols[b])
        for c in cs:
            if c%NF==0: Xa[b,c]=sp_pool[rng.integers(0,len(sp_pool))]
        bys=defaultdict(list)
        for c in cs: bys[c//NF].append(c%NF)
        for s,js in bys.items():
            sp=int(Xa[b,s*NF]); mv=[j for j in js if 3<=j<=6]
            for j in [x for x in js if x not in mv and x!=0]:
                p=pool[sp][j] or [1]; Xa[b,s*NF+j]=p[rng.integers(0,len(p))]
            if mv:
                keep={int(Xa[b,s*NF+j]) for j in range(3,7) if j not in mv}
                p=[m for m in pool[sp]['mv'] if m not in keep] or [1]; chosen=set()
                for j in mv:
                    for _ in range(20):
                        v=p[rng.integers(0,len(p))]
                        if v not in chosen: break
                    chosen.add(v); Xa[b,s*NF+j]=v
    return Xa
cv=defaultdict(list)
for c in range(COLS): cv[V.key(c)] += list(Xtr[:,c])
colvals={k:np.array(v) for k,v in cv.items()}
N=256
for regime,k in [("scatter",3),("slot",8)]:
    X_src=Xtr[rng.integers(0,len(Xtr),N)]; so=rng.integers(0,NSLOT,N)
    cols=([list(rng.choice(COLS,k,replace=False)) for _ in range(N)] if regime=="scatter"
          else [list(range(s*NF,(s+1)*NF)) for s in so])
    random.seed(0); g=torch.Generator(device=DEV); g.manual_seed(0)
    t0=time.time(); Xm=resample_fields(model,torch.tensor(X_src,device=DEV),cols,gen=g).cpu().numpy(); tm=(time.time()-t0)/N
    t0=time.time(); Xc=op_cond(X_src,cols,rng); tc=(time.time()-t0)/N
    arms=[("model (2.06M param, 1296s train)",Xm,tm),("count-table (0 param, 10ms build)",Xc,tc)]
    if regime=="slot": arms.append(("slotcopy",op_slotcopy(X_src,so,Xtr,rng),1e-6))
    arms.append(("marginal",op_marginal(X_src,cols,V,rng,colvals),5e-6))
    print(f"\n=== {regime} (k={k}), N={N}, identical sources & masked columns ===")
    print(f"  {'arm':36s} {'legal':>6s} {'l&novel':>8s} {'+ItemClause':>12s} {'ms/prop':>9s} {'ms per legal&novel move':>24s}")
    for nm,Xa,t in arms:
        e=evaluate(V,L,Xa,X_src,Xtr)
        ic=np.mean([not viol2(Xa[i]) for i in range(N)])
        ch=(Xa!=X_src).any(1)
        icn=np.mean([(not viol2(Xa[i])) and ch[i] and np.min((Xa[i]!=Xtr).sum(1))>0 for i in range(N)])
        cost = t*1e3/max(icn,1e-9)
        print(f"  {nm:36s} {e['legal']:6.3f} {e['legal_and_novel']:8.3f} {ic:12.3f} {t*1e3:9.3f} {cost:24.1f}")
