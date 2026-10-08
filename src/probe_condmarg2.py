"""cond-marginal + the 2-line fix: draw the 4 moves WITHOUT replacement from the
species' seen-move pool. Still zero parameters, still a count table."""
import numpy as np, time
from collections import defaultdict
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import evaluate, set_seed, COLS
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
pool=defaultdict(lambda: defaultdict(list)); spec_pool=[]
for b in range(len(Xtr)):
    for s in range(NSLOT):
        sl=Xtr[b,s*NF:(s+1)*NF]; sp=int(sl[0]); spec_pool.append(sp)
        for j in range(1,NF): pool[sp][j].append(int(sl[j]))
        pool[sp]['mv'] += [int(sl[j]) for j in range(3,7)]
spec_pool=np.array(spec_pool)

def op(X_src, cols, rng):
    Xa=X_src.copy()
    for b in range(Xa.shape[0]):
        cs=sorted(cols[b])
        for c in cs:
            if c%NF==0: Xa[b,c]=spec_pool[rng.integers(0,len(spec_pool))]
        by_slot=defaultdict(list)
        for c in cs: by_slot[c//NF].append(c%NF)
        for s,js in by_slot.items():
            sp=int(Xa[b,s*NF])
            mv=[j for j in js if 3<=j<=6]; other=[j for j in js if j not in mv and j!=0]
            for j in other:
                p=pool[sp][j] or [1]; Xa[b,s*NF+j]=p[rng.integers(0,len(p))]
            if mv:
                keep={int(Xa[b,s*NF+j]) for j in range(3,7) if j not in mv}
                p=[m for m in pool[sp]['mv'] if m not in keep] or [1]
                chosen=set()
                for j in mv:
                    for _ in range(20):
                        v=p[rng.integers(0,len(p))]
                        if v not in chosen: break
                    chosen.add(v); Xa[b,s*NF+j]=v
    return Xa

N=256
for regime,k in [("scatter",3),("slot",8)]:
    X_src=Xtr[rng.integers(0,len(Xtr),N)]
    cols=([list(rng.choice(COLS,k,replace=False)) for _ in range(N)] if regime=="scatter"
          else [list(range(s*NF,(s+1)*NF)) for s in rng.integers(0,NSLOT,N)])
    t0=time.time(); Xa=op(X_src,cols,rng); dt=(time.time()-t0)/N
    e=evaluate(V,L,Xa,X_src,Xtr)
    print(f"{regime:8s} cond-marginal+distinct-moves: legal {e['legal']:.3f} changed {e['changed']:.3f} "
          f"legal&novel {e['legal_and_novel']:.3f} unique {e['unique']:.3f} "
          f"{dt*1e6:.0f} us/proposal  viol {e['violations']}")
