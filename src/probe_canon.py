"""Is the nearest-neighbour distance a slot-permutation artifact?
encode.py:8-10 says teams are canonicalised (slots sorted by species) 'before any
distance is computed'. pilot.py:172-173 computes distances on the RAW edited grid,
which is no longer sorted by species after any species-changing edit."""
import numpy as np
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import memorization_stats, hamming, set_seed, COLS
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V=Vocab(teams); X=np.stack([V.encode(t) for t in teams])
idx=rng.permutation(len(X)); nt=int(len(X)*.15); Xte,Xtr=X[idx[:nt]],X[idx[nt:]]

def recanon(R):
    out=R.copy()
    for b in range(len(R)):
        sl=R[b].reshape(NSLOT,NF)
        out[b]=sl[np.argsort(sl[:,0], kind="stable")].reshape(-1)
    return out

print("CALIBRATION (as the pilot reports it):")
m=memorization_stats(Xte,Xtr); print("  held-out teams        ", {k:round(v,3) for k,v in m.items()})
Xr=np.stack([[rng.integers(1,V.sizes[V.key(c)]) for c in range(COLS)] for _ in range(256)])
m=memorization_stats(recanon(Xr),Xtr); print("  uniform random teams  ", {k:round(v,3) for k,v in m.items()})

print("\nTHE CONTROL THE PILOT NEVER RAN: a REAL team whose slots are randomly permuted")
perm=[]
for b in range(len(Xte)):
    sl=Xte[b].reshape(NSLOT,NF); perm.append(sl[rng.permutation(NSLOT)].reshape(-1))
perm=np.array(perm)
m=memorization_stats(perm,Xtr)
print("  held-out, slots PERMUTED, not re-canonicalised:", {k:round(v,3) for k,v in m.items()})
m=memorization_stats(recanon(perm),Xtr)
print("  same rows re-canonicalised                    :", {k:round(v,3) for k,v in m.items()})

print("\nsanity: TRAIN teams (memorisation upper bound), permuted vs canonical")
sub=Xtr[rng.integers(0,len(Xtr),256)]
p2=np.array([r.reshape(NSLOT,NF)[rng.permutation(NSLOT)].reshape(-1) for r in sub])
print("  train, permuted   :", {k:round(v,3) for k,v in memorization_stats(p2,Xtr).items()})
print("  train, canonical  :", {k:round(v,3) for k,v in memorization_stats(sub,Xtr).items()})
