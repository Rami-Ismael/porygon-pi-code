import numpy as np
from collections import defaultdict
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import op_uniform, evaluate, COLS
rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams])
idx=rng.permutation(len(X)); nt=int(len(X)*.15); Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
pool=defaultdict(list); allslots=[]
for r in Xtr:
    for s in range(NSLOT):
        v=r[s*NF:(s+1)*NF]; pool[int(v[0])].append(v); allslots.append(v)
def condcopy(Xs,cols,rng,dedup=False):
    Xo=Xs.copy()
    for b in range(Xo.shape[0]):
        for c in sorted(cols[b]):
            s,f=c//NF,c%NF
            if f==0: Xo[b,s*NF:(s+1)*NF]=allslots[rng.integers(0,len(allslots))]
            else:
                cand=pool.get(int(Xo[b,s*NF]))
                if not cand: continue
                for _ in range(20 if dedup else 1):
                    val=cand[rng.integers(0,len(cand))][f]
                    if not dedup or not (3<=f<=6): break
                    cur=[Xo[b,s*NF+j] for j in range(3,7) if s*NF+j!=c]
                    if val not in cur: break
                Xo[b,c]=val
    return Xo
print(" k   model(reported)  uniform  condcopy  condcopy+dedup   [legal / legal&novel]")
rep={1:None,3:(0.285,0.172),8:None}
for k in [1,2,3,4,6,8]:
    src=Xtr[rng.integers(0,len(Xtr),256)]
    cols=[list(rng.choice(COLS,k,replace=False)) for _ in range(256)]
    eu=evaluate(V,L,op_uniform(src,cols,V,rng),src,Xtr)
    ec=evaluate(V,L,condcopy(src,cols,rng),src,Xtr)
    ed=evaluate(V,L,condcopy(src,cols,rng,dedup=True),src,Xtr)
    r=rep.get(k)
    print(f"{k:2d}   {('%.3f/%.3f'%r) if r else '     -      ':12s}  {eu['legal']:.3f}/{eu['legal_and_novel']:.3f}  "
          f"{ec['legal']:.3f}/{ec['legal_and_novel']:.3f}  {ed['legal']:.3f}/{ed['legal_and_novel']:.3f}")
