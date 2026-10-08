"""The baseline the pilot never ran: p(field | species of that slot) as a plain
count table from the training corpus. Zero parameters, zero training, ~microseconds."""
import numpy as np
from collections import defaultdict, Counter
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import evaluate, op_uniform, op_marginal, set_seed, COLS
set_seed(0); rng = np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams])
idx=rng.permutation(len(X)); ntest=int(len(X)*0.15)
Xte,Xtr=X[idx[:ntest]],X[idx[ntest:]]
print("vocab sizes:", V.sizes)

# count table: for each species id, the pool of values seen in each within-slot field j
pool=defaultdict(lambda: defaultdict(list))
spec_pool=[]
for b in range(len(Xtr)):
    for s in range(NSLOT):
        sl=Xtr[b,s*NF:(s+1)*NF]; sp=int(sl[0]); spec_pool.append(sp)
        for j in range(1,NF): pool[sp][j].append(int(sl[j]))
        # the 4 move columns pool together (a move can appear in any move column)
        pool[sp]['mv'] += [int(sl[j]) for j in range(3,7)]
spec_pool=np.array(spec_pool)

def op_condmarg(X_src, cols, rng):
    Xa=X_src.copy()
    for b in range(Xa.shape[0]):
        cs=sorted(cols[b])
        # if species masked, redraw species first from the corpus species marginal
        for c in cs:
            if c%NF==0: Xa[b,c]=spec_pool[rng.integers(0,len(spec_pool))]
        for c in cs:
            j=c%NF
            if j==0: continue
            sp=int(Xa[b,(c//NF)*NF])
            p = pool[sp]['mv'] if 3<=j<=6 else pool[sp][j]
            if not p: p = pool[spec_pool[rng.integers(0,len(spec_pool))]][j] or [1]
            Xa[b,c]=p[rng.integers(0,len(p))]
    return Xa

from collections import defaultdict as dd
cv=dd(list)
for c in range(COLS): cv[V.key(c)] += list(Xtr[:,c])
colvals={k:np.array(v) for k,v in cv.items()}

N=256
for regime,k in [("scatter",3),("slot",8)]:
    X_src=Xtr[rng.integers(0,len(Xtr),N)]
    if regime=="scatter":
        cols=[list(rng.choice(COLS,k,replace=False)) for _ in range(N)]
    else:
        so=rng.integers(0,NSLOT,N); cols=[list(range(s*NF,(s+1)*NF)) for s in so]
    print(f"\n== {regime} (k={k}) ==")
    for name,Xa in [("cond-marginal(0 param)",op_condmarg(X_src,cols,rng)),
                    ("marginal",op_marginal(X_src,cols,V,rng,colvals)),
                    ("uniform",op_uniform(X_src,cols,V,rng))]:
        e=evaluate(V,L,Xa,X_src,Xtr)
        print(f"  {name:24s} legal {e['legal']:.3f} changed {e['changed']:.3f} "
              f"legal&novel {e['legal_and_novel']:.3f} unique {e['unique']:.3f} viol {e['violations']}")
