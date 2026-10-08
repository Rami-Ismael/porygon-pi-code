import numpy as np, torch, random
from collections import Counter
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from model import MaskedFieldModel
from pilot import COLS, evaluate, memorization_stats, hamming, decode_fields
DEV="mps" if torch.backends.mps.is_available() else "cpu"
rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
Z=np.load("/tmp/vgc-pilot/arrays_s0.npz")
Xtr,Xte=Z["Xtr"],Z["Xte"]

def recanon(A):
    B=A.reshape(-1,NSLOT,NF).copy(); out=np.empty_like(B)
    for i in range(B.shape[0]):
        out[i]=B[i][np.argsort(B[i,:,0],kind="stable")]
    return out.reshape(A.shape)

print("=== 1. no-op / free-legality accounting (scatter) ===")
for arm in ["model","uniform","marginal"]:
    A=Z["scatter_"+arm]; S=Z["scatter_src"]
    noop=np.mean([(A[i]==S[i]).all() for i in range(len(A))])
    e=evaluate(V,L,A,S,Xtr)
    changed_mask=[not (A[i]==S[i]).all() for i in range(len(A))]
    legal_given_changed=np.mean([not L.violations(decode_fields(V,A[i])) for i in range(len(A)) if changed_mask[i]])
    print(f" {arm:9s} legal {e['legal']:.3f}  no-op {noop:.3f}  legal|changed {legal_given_changed:.3f}  legal&novel {e['legal_and_novel']:.3f}")

print("\n=== 2. unconditional: canonicalisation of the distance metric ===")
U=Z["uncond"]
m_raw=memorization_stats(U,Xtr); m_can=memorization_stats(recanon(U),Xtr)
print(f" model uncond  raw NN {m_raw['mean_nn_dist']:.2f} mem {m_raw['memorized']:.3f}"
      f"   | re-canonicalised NN {m_can['mean_nn_dist']:.2f} mem {m_can['memorized']:.3f}")
# reference points, all measured the SAME way
sh=np.stack([Xte[i].reshape(NSLOT,NF)[rng.permutation(NSLOT)].reshape(-1) for i in range(len(Xte))])
print(f" real held-out canonical      NN {memorization_stats(Xte,Xtr)['mean_nn_dist']:.2f} mem {memorization_stats(Xte,Xtr)['memorized']:.3f}")
print(f" real held-out slot-shuffled  NN {memorization_stats(sh,Xtr)['mean_nn_dist']:.2f} mem {memorization_stats(sh,Xtr)['memorized']:.3f}")
print(f" real held-out shuffled->recanon NN {memorization_stats(recanon(sh),Xtr)['mean_nn_dist']:.2f}")
R=np.stack([[rng.integers(1,V.sizes[V.key(c)]) for c in range(COLS)] for _ in range(256)])
print(f" uniform-random teams         NN {memorization_stats(R,Xtr)['mean_nn_dist']:.2f}")
print(f" uniform-random recanon       NN {memorization_stats(recanon(R),Xtr)['mean_nn_dist']:.2f}")

print("\n=== 3. unconditional: per-slot legality (is 0.000 just p^6?) ===")
ok=0; tot=0; percount=Counter()
for i in range(len(U)):
    f=decode_fields(V,U[i]); nbad=0
    for s in range(NSLOT):
        sub=f[:]  # build a 6x copy of this slot's team to isolate? use violations on slot only
    v=L.violations(f); percount[len(v)]+=1
print(" violations-per-team histogram (model uncond):",sorted(percount.items())[:8])
# per-slot: reuse Legality by making a fake team of 6 distinct copies is unsafe; count slot-level
def slot_bad(f,s):
    sl=f[s*NF:(s+1)*NF]; sp,ab,it=sl[0],sl[1],sl[2]; ms=[m for m in sl[3:7] if m]
    from corpus import dex_entry
    if dex_entry(sp) is None: return True
    if ab not in L.abils_for(sp): return True
    if len(set(ms))!=len(ms) or not(1<=len(ms)<=4): return True
    lm=L.moves_for(sp)
    return any(m not in lm for m in ms)
for nm,A in [("uncond",U),("slot_model",Z["slot_model"]),("slot_slotcopy",Z["slot_slotcopy"])]:
    fs=[decode_fields(V,A[i]) for i in range(len(A))]
    p=np.mean([[not slot_bad(f,s) for s in range(NSLOT)] for f in fs])
    print(f" {nm:14s} per-slot legal {p:.3f}  -> p^6={p**6:.4f}  actual team legal {evaluate(V,L,A,A,Xtr)['legal']:.4f}")

print("\n=== 4. species-clause failures when a species column is resampled ===")
SPCOLS=set(range(0,COLS,NF))
for arm in ["model","uniform","marginal"]:
    A=Z["scatter_"+arm]; C=Z["scatter_cols"]
    hit=[i for i in range(len(A)) if set(int(c) for c in C[i]) & SPCOLS]
    bad=sum(1 for i in hit if "speciesclause" in L.violations(decode_fields(V,A[i])))
    print(f"  {arm:9s} touched a species column in {len(hit)}/{len(A)}; duplicated a species in {bad}/{len(hit)} = {bad/max(len(hit),1):.3f}")
for nm in ["slot_model","slot_slotcopy","uncond"]:
    A=Z[nm]
    bad=sum(1 for i in range(len(A)) if "speciesclause" in L.violations(decode_fields(V,A[i])))
    print(f"  {nm:14s} species-clause violation rate {bad/len(A):.3f}")
