"""A purely RULE-BASED mutation operator: no corpus statistics, no learning.
For a masked field, draw uniformly from the legal set for that slot's species
(the same Legality object every arm is scored against)."""
import numpy as np
from collections import defaultdict
from corpus import load_corpus, norm, NATURES
from encode import Vocab, Legality, NF, NSLOT
from pilot import evaluate, COLS
rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
colvals=defaultdict(list)
for c in range(COLS): colvals[V.key(c)]+=list(Xtr[:,c])
colvals={k:np.array(v) for k,v in colvals.items()}
# vocab-id sets that are legal for each species, per field type
sp_ids=sorted({int(v) for c in range(0,COLS,NF) for v in Xtr[:,c]})
LM={}; LA={}
for sid in sp_ids:
    sp=V.itos["species"][sid]
    LM[sid]=np.array([i for i in range(1,V.sizes["move"]) if V.itos["move"][i] in L.moves_for(sp)] or [1])
    LA[sid]=np.array([i for i in range(1,V.sizes["ability"]) if V.itos["ability"][i] in L.abils_for(sp)] or [1])
NAT=np.array([i for i in range(1,V.sizes["nature"]) if V.itos["nature"][i] in {norm(n) for n in NATURES}])
def rulemut(Xs,cols,r):
    Xo=Xs.copy()
    for b in range(Xo.shape[0]):
        for c in sorted(cols[b],key=lambda c:0 if c%NF==0 else 1):
            s=c//NF; j=c%NF
            if j==0:
                used={int(Xo[b,ss*NF]) for ss in range(NSLOT) if ss!=s}
                p=[i for i in sp_ids if i not in used]; Xo[b,c]=p[r.integers(0,len(p))]; continue
            sid=int(Xo[b,s*NF])
            if j==1: Xo[b,c]=LA[sid][r.integers(0,len(LA[sid]))]
            elif j==2: p=colvals["item"]; Xo[b,c]=p[r.integers(0,len(p))]
            elif j==7: Xo[b,c]=NAT[r.integers(0,len(NAT))]
            else:
                oth={int(Xo[b,s*NF+jj]) for jj in range(3,7) if jj!=j}
                pool=LM[sid]; 
                for _ in range(12):
                    val=int(pool[r.integers(0,len(pool))])
                    if val not in oth: break
                Xo[b,c]=val
    return Xo
for k in [3,8]:
    r=np.random.default_rng(5)
    src=Xtr[r.integers(0,len(Xtr),256)]
    cols=([list(r.choice(COLS,k,replace=False)) for _ in range(256)] if k==3
          else [list(range(s*NF,(s+1)*NF)) for s in r.integers(0,NSLOT,256)])
    e=evaluate(V,L,rulemut(src,cols,r),src,Xtr)
    print("rule-based mutation k=%d: legal %.3f  novel %.3f  changed %.3f  legal&novel %.3f  unique %.3f  viol %s"%(
        k,e["legal"],e["novel"],e["changed"],e["legal_and_novel"],e["unique"],e["violations"]))
