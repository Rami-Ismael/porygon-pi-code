import numpy as np
from collections import defaultdict
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import evaluate, op_uniform, COLS
rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
colvals=defaultdict(list)
for c in range(COLS): colvals[V.key(c)]+=list(Xtr[:,c])
colvals={k:np.array(v) for k,v in colvals.items()}
slotbank=defaultdict(list)
for r in Xtr:
    for s in range(NSLOT):
        b=s*NF; slotbank[int(r[b])].append(r[b:b+NF].copy())
def fcdd(Xs,cols,r2):
    Xo=Xs.copy()
    for b in range(Xo.shape[0]):
        for c in sorted(cols[b],key=lambda c:0 if c%NF==0 else 1):
            s=c//NF; j=c%NF
            if j==0:
                p=colvals["species"]; Xo[b,c]=p[r2.integers(0,len(p))]; continue
            bank=slotbank.get(int(Xo[b,s*NF]))
            for _ in range(8):
                if bank: val=int(bank[r2.integers(0,len(bank))][j])
                else:
                    p=colvals[V.key(c)]; val=int(p[r2.integers(0,len(p))])
                if not(3<=j<=6): break
                if val not in {int(Xo[b,s*NF+jj]) for jj in range(3,7) if jj!=j}: break
            Xo[b,c]=val
    return Xo
N=256
print(" k | field-copy+dedup legal / legal&novel | uniform legal | (results_s0 model @k=3: .285 / .172)")
for k in [1,2,3,4,6,8]:
    r=np.random.default_rng(42+k)
    src=Xtr[r.integers(0,len(Xtr),N)]
    cols=[list(r.choice(COLS,k,replace=False)) for _ in range(N)]
    a=evaluate(V,L,fcdd(src,cols,r),src,Xtr); u=evaluate(V,L,op_uniform(src,cols,V,r),src,Xtr)
    print(" %d |        %.3f / %.3f              |     %.3f"%(k,a["legal"],a["legal_and_novel"],u["legal"]))
# slot regime: whole-slot field-copy (== slotcopy but species-conditional is moot)
print("\nslot regime, all 8 fields of one slot:")
r=np.random.default_rng(7)
so=r.integers(0,NSLOT,N); src=Xtr[r.integers(0,len(Xtr),N)]
cols=[list(range(s*NF,(s+1)*NF)) for s in so]
a=evaluate(V,L,fcdd(src,cols,r),src,Xtr)
print("  field-copy+dedup legal %.3f  legal&novel %.3f  (slotcopy in json: .887/.883, model .023/.020)"%(a["legal"],a["legal_and_novel"]))
