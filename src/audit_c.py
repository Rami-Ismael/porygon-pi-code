"""(C) duplicates/leakage, (D) the missing species-conditioned copy baseline for scatter."""
import numpy as np
from collections import defaultdict, Counter
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import op_uniform, op_marginal, evaluate, hamming, COLS

rng = np.random.default_rng(0)
teams,_ = load_corpus(); teams=[t for t in teams if len(t)==6]
V,L = Vocab(teams), Legality(teams)
X = np.stack([V.encode(t) for t in teams])
idx = rng.permutation(len(X)); ntest=int(len(X)*0.15)
Xte, Xtr = X[idx[:ntest]], X[idx[ntest:]]

print("[C] corpus duplication / split leakage")
print("  teams", len(X), "unique grids", len(np.unique(X,axis=0)))
dup = sum(1 for i in range(len(Xte)) if hamming(Xte[i],Xtr).min()==0)
print(f"  held-out teams that are an EXACT copy of a training team: {dup}/{len(Xte)} = {dup/len(Xte):.3f}")
d1 = np.array([np.sort(hamming(Xte[i],Xtr))[0] for i in range(len(Xte))])
print(f"  held-out NN dist: mean {d1.mean():.2f}  <=3: {(d1<=3).mean():.3f}  <=8: {(d1<=8).mean():.3f}")
# also duplicated species-sets
sp = [tuple(sorted(r[c] for c in range(0,COLS,NF))) for r in X]
print("  unique species-6-sets:", len(set(sp)), "of", len(sp))

print("\n[D] scatter regime: species-conditioned copy baseline (the slotcopy analogue)")
# slot pool: every (species -> list of slot vectors) from TRAIN only
pool = defaultdict(list)
for r in Xtr:
    for s in range(NSLOT):
        v = r[s*NF:(s+1)*NF]
        pool[int(v[0])].append(v)
allslots = [r[s*NF:(s+1)*NF] for r in Xtr for s in range(NSLOT)]

def op_condcopy(X_src, cols, rng):
    Xo = X_src.copy()
    for b in range(Xo.shape[0]):
        for c in sorted(cols[b]):
            s, f = c//NF, c%NF
            if f == 0:                      # species field -> paste a whole random slot
                Xo[b, s*NF:(s+1)*NF] = allslots[rng.integers(0,len(allslots))]
            else:                           # copy this field from a same-species slot
                cand = pool.get(int(Xo[b, s*NF]))
                if not cand: continue
                Xo[b, c] = cand[rng.integers(0,len(cand))][f]
    return Xo

NPROP, K = 256, 3
src_i = rng.integers(0,len(Xtr),NPROP); X_src = Xtr[src_i]
cols = [list(rng.choice(COLS,K,replace=False)) for _ in range(NPROP)]
for name, A in [("uniform", op_uniform(X_src,cols,V,rng)),
                ("marginal", op_marginal(X_src,cols,V,rng,{k:np.array([r[c] for r in Xtr for c in range(COLS) if V.key(c)==k]) for k in V.sizes})),
                ("condcopy", op_condcopy(X_src,cols,rng))]:
    e = evaluate(V,L,A,X_src,Xtr)
    print(f"  {name:9s} legal {e['legal']:.3f}  changed {e['changed']:.3f}  legal&novel {e['legal_and_novel']:.3f}  viol {e['violations']}")

print("\n  same, slot regime (8 fields of one slot):")
slot_of = rng.integers(0,NSLOT,NPROP)
cols8 = [list(range(s*NF,(s+1)*NF)) for s in slot_of]
A = op_condcopy(X_src, cols8, rng)
e = evaluate(V,L,A,X_src,Xtr); print(f"  condcopy  legal {e['legal']:.3f} legal&novel {e['legal_and_novel']:.3f}")
