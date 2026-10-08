import json, numpy as np, random
from collections import defaultdict, Counter
from corpus import load_corpus, norm
from encode import Vocab, Legality, team_fields, FIELDS, NF, NSLOT
from pilot import evaluate, op_uniform, op_marginal, op_slotcopy, hamming, memorization_stats, COLS

rng = np.random.default_rng(0)
teams,_ = load_corpus(); teams=[t for t in teams if len(t)==6]
V,L = Vocab(teams), Legality(teams)
X = np.stack([V.encode(t) for t in teams])
idx = rng.permutation(len(X)); ntest=int(len(X)*0.15)
Xte, Xtr = X[idx[:ntest]], X[idx[ntest:]]
colvals=defaultdict(list)
for c in range(COLS): colvals[V.key(c)] += list(Xtr[:,c])
colvals={k:np.array(v) for k,v in colvals.items()}
print("train",Xtr.shape,"heldout",Xte.shape)

# ---- corpus redundancy (question c) ----
print("unique encoded rows in full corpus:", len(np.unique(X,axis=0)), "of", len(X))
sp = [tuple(sorted(r[c] for c in range(0,COLS,NF))) for r in X]
cnt = Counter(sp)
print("unique species-multisets:", len(cnt), "of", len(X), "| top group sizes:", [n for _,n in cnt.most_common(6)])
print("teams in a species-multiset group of size>1:", sum(n for n in cnt.values() if n>1))
# how many heldout teams share their species-multiset with a train team
sptr = set(tuple(sorted(r[c] for c in range(0,COLS,NF))) for r in Xtr)
share = sum(1 for r in Xte if tuple(sorted(r[c] for c in range(0,COLS,NF))) in sptr)
print("heldout teams whose 6 species already appear in train:", share, "/", len(Xte))

# ---- replay regime scatter exactly ----
nprop=256; k=3
src_i = rng.integers(0,len(Xtr),nprop); X_src = Xtr[src_i]
cols = [list(rng.choice(COLS,k,replace=False)) for _ in range(nprop)]
Xu = op_uniform(X_src, cols, V, rng)
Xg = op_marginal(X_src, cols, V, rng, colvals)
eu=evaluate(V,L,Xu,X_src,Xtr); eg=evaluate(V,L,Xg,X_src,Xtr)
print("REPLAY CHECK uniform legal %.4f (json 0.0546875)  marginal legal %.4f (json 0.05859375)"%(eu["legal"],eg["legal"]))

# ---- build species-conditional pools from TRAIN ONLY ----
# pool[key][species_id] -> array of vocab ids seen for that field with that species
pool = {kk: defaultdict(list) for kk in ["ability","item","move","nature"]}
slotbank = defaultdict(list)   # species_id -> list of full 8-field slot rows
for r in Xtr:
    for s in range(NSLOT):
        base=s*NF; spid=int(r[base])
        slotbank[spid].append(r[base:base+NF].copy())
        for j in range(1,NF):
            pool[V.key(base+j)][spid].append(int(r[base+j]))
pool={kk:{s:np.array(v) for s,v in d.items()} for kk,d in pool.items()}

def op_condmarginal(X_src, cols, rng2):
    """cheap, model-free: species from corpus marginal; every other field drawn from
    the corpus pool of that field FOR THE SPECIES CURRENTLY IN THAT SLOT."""
    Xo=X_src.copy()
    for b in range(Xo.shape[0]):
        cs=sorted(cols[b], key=lambda c: 0 if c%NF==0 else 1)   # species first
        for c in cs:
            s=c//NF; j=c%NF; kk=V.key(c)
            if j==0:
                p=colvals["species"]; Xo[b,c]=p[rng2.integers(0,len(p))]
            else:
                spid=int(Xo[b,s*NF]); p=pool[kk].get(spid)
                if p is None or len(p)==0: p=colvals[kk]
                Xo[b,c]=p[rng2.integers(0,len(p))]
    return Xo

def op_fieldcopy(X_src, cols, rng2):
    """even cheaper: copy the field value from a random corpus slot of the SAME species."""
    Xo=X_src.copy()
    for b in range(Xo.shape[0]):
        cs=sorted(cols[b], key=lambda c: 0 if c%NF==0 else 1)
        for c in cs:
            s=c//NF; j=c%NF
            if j==0:
                p=colvals["species"]; Xo[b,c]=p[rng2.integers(0,len(p))]
            else:
                spid=int(Xo[b,s*NF]); bank=slotbank.get(spid)
                if not bank:
                    p=colvals[V.key(c)]; Xo[b,c]=p[rng2.integers(0,len(p))]
                else:
                    Xo[b,c]=bank[rng2.integers(0,len(bank))][j]
    return Xo

def op_slotcopy_scatter(X_src, cols, rng2):
    """scatter analogue of slotcopy: for each masked column, paste the whole donor
    slot's value for that field from a random corpus slot in the SAME slot index."""
    Xo=X_src.copy()
    for b in range(Xo.shape[0]):
        for c in cols[b]:
            s=c//NF; j=c%NF
            donor=Xtr[rng2.integers(0,len(Xtr))]
            s2=rng2.integers(0,NSLOT)
            Xo[b,c]=donor[s2*NF+j]
    return Xo

r2=np.random.default_rng(999)
for name,fn in [("cond-marginal",op_condmarginal),("field-copy",op_fieldcopy),("slotcopy-scatter",op_slotcopy_scatter)]:
    Xa=fn(X_src,cols,r2)
    e=evaluate(V,L,Xa,X_src,Xtr); m=memorization_stats(Xa,Xtr)
    print("%-17s legal %.3f  novel %.3f  changed %.3f  legal&novel %.3f  unique %.3f  exact %.3f  viol %s"%(
        name,e["legal"],e["novel"],e["changed"],e["legal_and_novel"],e["unique"],m["exact_copy"],e["violations"]))

print("\n--- dedup-aware variants (reject a move already in the slot; <=8 tries) ---")
def op_fieldcopy_dd(X_src, cols, rng2):
    Xo=X_src.copy()
    for b in range(Xo.shape[0]):
        cs=sorted(cols[b], key=lambda c: 0 if c%NF==0 else 1)
        for c in cs:
            s=c//NF; j=c%NF
            if j==0:
                p=colvals["species"]; Xo[b,c]=p[rng2.integers(0,len(p))]; continue
            spid=int(Xo[b,s*NF]); bank=slotbank.get(spid)
            for _ in range(8):
                if bank: val=int(bank[rng2.integers(0,len(bank))][j])
                else:
                    p=colvals[V.key(c)]; val=int(p[rng2.integers(0,len(p))])
                if not (3<=j<=6): break
                others={int(Xo[b,s*NF+jj]) for jj in range(3,7) if jj!=j}
                if val not in others: break
            Xo[b,c]=val
    return Xo
r3=np.random.default_rng(999)
Xa=op_fieldcopy_dd(X_src,cols,r3); e=evaluate(V,L,Xa,X_src,Xtr); m=memorization_stats(Xa,Xtr)
print("field-copy+dedup  legal %.3f  novel %.3f  changed %.3f  legal&novel %.3f  exact %.3f  viol %s"%(
    e["legal"],e["novel"],e["changed"],e["legal_and_novel"],m["exact_copy"],e["violations"]))

print("\n--- legality among ACTUALLY-CHANGED proposals ---")
def legal_changed(Xa):
    ch=[(Xa[i]!=X_src[i]).any() for i in range(len(Xa))]
    lg=[L.legal([V.decode_field(c,int(Xa[i,c])) for c in range(COLS)]) for i in range(len(Xa))]
    n=sum(ch); return sum(1 for a,b in zip(ch,lg) if a and b)/max(n,1), n
for nm,Xa in [("uniform",Xu),("marginal",Xg),("field-copy+dedup",Xa)]:
    v,n=legal_changed(Xa); print("  %-17s legal|changed %.3f (n=%d)"%(nm,v,n))
print("  model (from json)  legal|changed %.3f  (legal .28516 - exact .11328) / changed .88672"%((0.28515625-0.11328125)/0.88671875))

print("\n--- C3: is NN distance 38.3 evidence of off-manifold, or of slot-order mismatch? ---")
def perm_rows(A,r):
    B=A.reshape(len(A),NSLOT,NF).copy()
    for i in range(len(B)): B[i]=B[i][r.permutation(NSLOT)]
    return B.reshape(len(A),NSLOT*NF)
r4=np.random.default_rng(7)
print("  held-out teams, CANONICAL order   : mean NN dist %.1f  memorised %.3f"%(
    memorization_stats(Xte,Xtr)["mean_nn_dist"], memorization_stats(Xte,Xtr)["memorized"]))
Xtep=perm_rows(Xte,r4)
print("  held-out teams, PERMUTED slots    : mean NN dist %.1f  memorised %.3f"%(
    memorization_stats(Xtep,Xtr)["mean_nn_dist"], memorization_stats(Xtep,Xtr)["memorized"]))
Xtrp=perm_rows(Xtr,r4)
print("  TRAIN teams themselves, PERMUTED  : mean NN dist %.1f  exact %.3f"%(
    memorization_stats(Xtrp,Xtr)["mean_nn_dist"], memorization_stats(Xtrp,Xtr)["exact_copy"]))
