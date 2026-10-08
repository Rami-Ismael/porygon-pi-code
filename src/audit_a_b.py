"""Model-free audits: (A) memorisation-by-construction, (B) canonicalisation distance
inflation, (C) a species-conditioned copy baseline for the scatter regime."""
import numpy as np, random
from collections import defaultdict, Counter
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import op_uniform, op_marginal, evaluate, memorization_stats, hamming, COLS

rng = np.random.default_rng(0)
teams,_ = load_corpus(); teams=[t for t in teams if len(t)==6]
V,L = Vocab(teams), Legality(teams)
X = np.stack([V.encode(t) for t in teams])
idx = rng.permutation(len(X)); ntest=int(len(X)*0.15)
Xte, Xtr = X[idx[:ntest]], X[idx[ntest:]]
print("vocab sizes:", V.sizes)

# ---------- (A) memorisation is 1.0 for the identity move ----------
print("\n[A] memorisation criterion on local moves")
print("  identity (source teams themselves):", memorization_stats(Xtr[:256], Xtr))
# null: change k fields at random, uniform
for k in [1,3,8,16,24,48]:
    cols=[list(rng.choice(COLS,k,replace=False)) for _ in range(256)]
    src=Xtr[rng.integers(0,len(Xtr),256)]
    Xu=op_uniform(src,cols,V,rng)
    m=memorization_stats(Xu,Xtr)
    print(f"  uniform k={k:2d}: memorised {m['memorized']:.3f}  nn {m['mean_nn_dist']:.2f}")

# ---------- (B) canonicalisation ----------
print("\n[B] species-sort canonicalisation vs a changed species")
def recanon(A):
    B=A.reshape(-1,NSLOT,NF).copy()
    out=np.empty_like(B)
    for i in range(B.shape[0]):
        order=np.argsort(B[i,:,0],kind="stable")
        out[i]=B[i,order]
    return out.reshape(A.shape)
SP=[c for c in range(COLS) if c%NF==0]
spec_pool=np.unique(Xtr[:,SP])
raws,cans,srcraw=[],[],[]
N=400
for i in range(N):
    b=int(rng.integers(0,len(Xtr))); s=int(rng.integers(0,NSLOT))
    A=Xtr[b].copy()
    new=int(spec_pool[rng.integers(0,len(spec_pool))])
    while new==A[s*NF]: new=int(spec_pool[rng.integers(0,len(spec_pool))])
    A[s*NF]=new
    Ac=recanon(A[None,:])[0]
    raws.append(int(hamming(A,Xtr).min())); cans.append(int(hamming(Ac,Xtr).min()))
    srcraw.append(int((A!=Xtr[b]).sum()))
print(f"  1 species field changed, N={N}")
print(f"   pilot's raw NN Hamming     mean {np.mean(raws):.2f}  median {np.median(raws):.0f}  max {max(raws)}")
print(f"   re-canonicalised NN Hamming mean {np.mean(cans):.2f}  median {np.median(cans):.0f}  max {max(cans)}")
print(f"   raw distance to the SOURCE team (true edit size = 1): mean {np.mean(srcraw):.2f} max {max(srcraw)}")
print(f"   inflation factor on NN dist: {np.mean(raws)/max(np.mean(cans),1e-9):.2f}x")

# whole-slot replacement (slot regime analogue)
raws,cans,srcraw=[],[],[]
for i in range(N):
    b=int(rng.integers(0,len(Xtr))); s=int(rng.integers(0,NSLOT))
    b2=int(rng.integers(0,len(Xtr))); s2=int(rng.integers(0,NSLOT))
    A=Xtr[b].copy(); A[s*NF:(s+1)*NF]=Xtr[b2][s2*NF:(s2+1)*NF]
    Ac=recanon(A[None,:])[0]
    raws.append(int(hamming(A,Xtr).min())); cans.append(int(hamming(Ac,Xtr).min()))
    srcraw.append(int((A!=Xtr[b]).sum()))
print(f"  whole slot pasted (slotcopy), N={N}")
print(f"   raw NN mean {np.mean(raws):.2f}   canon NN mean {np.mean(cans):.2f}  inflation {np.mean(raws)/np.mean(cans):.2f}x")
print(f"   raw dist to source mean {np.mean(srcraw):.2f} (true edit <= 8)")

# how far apart are real teams from each other, canon vs shuffled
sh=Xtr.copy()
perm_sh=np.empty_like(sh)
for i in range(len(sh)):
    o=rng.permutation(NSLOT); perm_sh[i]=sh[i].reshape(NSLOT,NF)[o].reshape(-1)
d_can=np.mean([np.sort(hamming(Xte[i],Xtr))[0] for i in range(len(Xte))])
d_sh =np.mean([np.sort(hamming(perm_sh_i,Xtr))[0] for perm_sh_i in
               [Xte[i].reshape(NSLOT,NF)[rng.permutation(NSLOT)].reshape(-1) for i in range(len(Xte))]])
print(f"  held-out real teams: NN dist canonical {d_can:.2f} vs slot-shuffled {d_sh:.2f}")
