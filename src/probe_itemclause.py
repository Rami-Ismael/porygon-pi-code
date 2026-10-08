"""Add the missing Item Clause to the oracle and re-score every model-free arm."""
import numpy as np
from collections import defaultdict
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import evaluate, op_uniform, op_marginal, op_slotcopy, set_seed, COLS, decode_fields
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
def viol2(row):
    v=L.violations(decode_fields(V,row))
    its=[V.decode_field(s*NF+2,int(row[s*NF+2])) for s in range(NSLOT)]
    its=[i for i in its if i]
    if len(set(its))!=len(its): v=v+["itemclause"]
    return v
print("corpus under oracle+ItemClause:", sum(1 for i in range(len(X)) if viol2(X[i])),"fail of",len(X))
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
N=512
for regime,k in [("scatter",3),("slot",8)]:
    X_src=Xtr[rng.integers(0,len(Xtr),N)]; so=rng.integers(0,NSLOT,N)
    cols=([list(rng.choice(COLS,k,replace=False)) for _ in range(N)] if regime=="scatter"
          else [list(range(s*NF,(s+1)*NF)) for s in so])
    arms={"cond-marginal(0 param)":op_cond(X_src,cols,rng),
          "marginal":op_marginal(X_src,cols,V,rng,colvals),
          "uniform":op_uniform(X_src,cols,V,rng)}
    if regime=="slot": arms["slotcopy"]=op_slotcopy(X_src,so,Xtr,rng)
    print(f"\n== {regime} ==   legal(pilot oracle) -> legal(+ItemClause)")
    for nm,Xa in arms.items():
        a=np.mean([not L.violations(decode_fields(V,Xa[i])) for i in range(N)])
        b=np.mean([not viol2(Xa[i]) for i in range(N)])
        print(f"  {nm:24s} {a:.3f}  ->  {b:.3f}")
