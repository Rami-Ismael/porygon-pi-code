"""What does 2.94 nats held-out actually beat? Compare the model's held-out masked
cross-entropy against trivial predictors on the SAME held-out teams."""
import numpy as np
from collections import defaultdict, Counter
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from pilot import COLS
rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams])
idx=rng.permutation(len(X)); nt=int(len(X)*.15); Xte,Xtr=X[idx[:nt]],X[idx[nt:]]

FT=["species","ability","item","move","nature"]
def counts_by_fieldtype(A):
    c={k:Counter() for k in FT}
    for r in A:
        for col in range(COLS): c[V.key(col)][int(r[col])]+=1
    return c
ct=counts_by_fieldtype(Xtr)
# conditional-on-species counts (train)
cond={k:defaultdict(Counter) for k in FT}
for r in Xtr:
    for s in range(NSLOT):
        sp=int(r[s*NF])
        for j in range(NF):
            col=s*NF+j; k=V.key(col)
            if k!="species": cond[k][sp][int(r[col])]+=1
spc=Counter(int(r[s*NF]) for r in Xtr for s in range(NSLOT))

def nll(dist_counts, y, vocabsize, alpha=0.5):
    tot=sum(dist_counts.values())+alpha*vocabsize
    return -np.log((dist_counts.get(y,0)+alpha)/tot)

# score every field of every held-out team (i.e. mask rate 1.0-ish; the pilot's
# reported loss averages over random mask rates but the target set is the same fields)
u_nll=[]; c_nll=[]; unif=[]
for r in Xte:
    for s in range(NSLOT):
        sp=int(r[s*NF])
        for j in range(NF):
            col=s*NF+j; k=V.key(col); y=int(r[col]); Vs=V.sizes[k]
            unif.append(np.log(Vs-1))
            u_nll.append(nll(ct[k],y,Vs))
            if k=="species": c_nll.append(nll(spc,y,Vs))
            else: c_nll.append(nll(cond[k][sp],y,Vs))
print(f"held-out per-field cross-entropy (nats), all 48 fields:")
print(f"  uniform over vocab            {np.mean(unif):.3f}")
print(f"  field-type marginal (unigram) {np.mean(u_nll):.3f}")
print(f"  conditioned on the species    {np.mean(c_nll):.3f}   <-- a lookup table, no training")
print(f"  the pilot's model (reported)  2.937")

# --- fair version: replicate the pilot's mask distribution exactly (t~U(0,1) per team,
# i.i.d. Bernoulli(t) per column, >=1 masked), score ONLY masked columns, and let the
# species-conditioned lookup fall back to the marginal when the species is itself masked.
rr=np.random.default_rng(7)
tot_u=tot_c=n=0
for rep in range(8):
    for r in Xte:
        t=rr.random(); m=rr.random(COLS)<t
        if not m.any(): m[rr.integers(0,COLS)]=True
        for col in range(COLS):
            if not m[col]: continue
            k=V.key(col); y=int(r[col]); Vs=V.sizes[k]; s=col//NF; sp=int(r[s*NF])
            tot_u+=nll(ct[k],y,Vs)
            if k=="species" or m[s*NF]: tot_c+=nll(ct[k],y,Vs)     # species unknown -> marginal
            else: tot_c+=nll(cond[k][sp],y,Vs)
            n+=1
print(f"\nunder the pilot's own random-mask-rate protocol (8 reps, masked columns only):")
print(f"  field-type marginal            {tot_u/n:.3f}")
print(f"  species-conditioned lookup     {tot_c/n:.3f}")
print(f"  pilot model held-out (ep600)   2.937")

# --- the regime that C1 is about: only k=3 scattered columns masked, species visible
tot_u=tot_c=n=0; rows=[]
for rep in range(8):
    for r in Xte:
        cols=rr.choice(COLS,3,replace=False)
        for col in cols:
            k=V.key(col); y=int(r[col]); Vs=V.sizes[k]; s=col//NF; sp=int(r[s*NF])
            masked_sp = (s*NF) in cols
            tot_u+=nll(ct[k],y,Vs)
            tot_c+=nll(ct[k],y,Vs) if (k=="species" or masked_sp) else nll(cond[k][sp],y,Vs)
            n+=1
print(f"\nk=3 scatter mask (the regime C1 is about), masked columns only:")
print(f"  field-type marginal        {tot_u/n:.3f}")
print(f"  species-conditioned lookup {tot_c/n:.3f}")
