import numpy as np, time, json
from collections import Counter
from corpus import load_corpus, norm
from encode import Vocab, Legality, team_fields, FIELDS, NF, NSLOT
from pilot import evaluate, op_uniform, op_marginal, op_slotcopy, set_seed, COLS, decode_fields
set_seed(0); rng = np.random.default_rng(0)
teams,_ = load_corpus(); teams=[t for t in teams if len(t)==6]
V,L = Vocab(teams), Legality(teams)
X = np.stack([V.encode(t) for t in teams])
idx = rng.permutation(len(X)); ntest=int(len(X)*0.15)
Xte,Xtr = X[idx[:ntest]], X[idx[ntest:]]
print("teams",len(X),"train",len(Xtr),"heldout",len(Xte))

# 0) does the corpus pass its own widened oracle?
bad = [(i, L.violations(decode_fields(V,X[i]))) for i in range(len(X))]
bad = [b for b in bad if b[1]]
print("corpus teams failing widened oracle:", len(bad), "of", len(X), Counter(v for _,vs in bad for v in vs))

# stat points present?
sp = [sum(s["evs"].values()) for t in teams for s in t]
print("stat-point totals per slot: min",min(sp),"max",max(sp),"mean",round(np.mean(sp),1),
      "distinct",len(set(sp)), "counter-top", Counter(sp).most_common(5))
nz = [tuple(sorted(s["evs"].items())) for t in teams for s in t]
print("distinct EV spreads in corpus:", len(set(nz)), "of", len(nz), "slots")

N=256
# reproduce slot regime sources exactly as pilot does (rng state differs, but protocol same)
src_i = rng.integers(0,len(Xtr),N); X_src = Xtr[src_i]
slot_of = rng.integers(0,NSLOT,N)
cols = [list(range(s*NF,(s+1)*NF)) for s in slot_of]

def report(name, Xa):
    e = evaluate(V,L,Xa,X_src,Xtr)
    print(f"  {name:22s} legal {e['legal']:.3f}  novel {e['novel']:.3f}  changed {e['changed']:.3f} "
          f" legal&novel {e['legal_and_novel']:.3f} unique {e['unique']:.3f}  viol {e['violations']}")
    return e

print("\n== SLOT regime, model-free operators ==")
report("identity(no-op)", X_src.copy())
report("slotcopy", op_slotcopy(X_src, slot_of, Xtr, rng))
# degenerate: always paste the SAME slot (the most common species' first occurrence)
spec_counts = Counter(int(Xtr[b,s*NF]) for b in range(len(Xtr)) for s in range(NSLOT))
top_spec = spec_counts.most_common(1)[0][0]
donor=None
for b in range(len(Xtr)):
    for s in range(NSLOT):
        if int(Xtr[b,s*NF])==top_spec: donor = Xtr[b,s*NF:(s+1)*NF].copy(); break
    if donor is not None: break
print("  (degenerate donor species =", V.decode_field(0,top_spec), ", count", spec_counts[top_spec],")")
Xd = X_src.copy()
for b in range(N):
    s=slot_of[b]; Xd[b,s*NF:(s+1)*NF]=donor
report("paste-ONE-fixed-slot", Xd)
# degenerate 2: paste a slot from a *held-out* team (unseen)
Xh = X_src.copy()
for b in range(N):
    s=slot_of[b]; src=Xte[rng.integers(0,len(Xte))]; s2=rng.integers(0,NSLOT)
    Xh[b,s*NF:(s+1)*NF]=src[s2*NF:(s2+1)*NF]
report("slotcopy-from-HELDOUT", Xh)
report("uniform", op_uniform(X_src, cols, V, rng))
colvals={}
from collections import defaultdict
cv=defaultdict(list)
for c in range(COLS): cv[V.key(c)] += list(Xtr[:,c])
colvals={k:np.array(v) for k,v in cv.items()}
report("marginal", op_marginal(X_src, cols, V, rng, colvals))

# 3) rejection-sampling cost: how long does a legality check take?
Xu = op_uniform(X_src, cols, V, rng)
t0=time.time()
for _ in range(10):
    for i in range(N): L.violations(decode_fields(V,Xu[i]))
t_check=(time.time()-t0)/(10*N)
t0=time.time()
for _ in range(10): op_marginal(X_src, cols, V, rng, colvals)
t_marg=(time.time()-t0)/(10*N)
print(f"\nlegality check {t_check*1e6:.0f} us/proposal ; marginal draw {t_marg*1e6:.0f} us/proposal")
