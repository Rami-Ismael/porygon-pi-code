"""Constraints the widened oracle (encode.py:85-115) never checks."""
import numpy as np
from collections import Counter, defaultdict
from corpus import load_corpus, norm, dex_entry, _items
from encode import Vocab, Legality, NF, NSLOT
from pilot import set_seed, op_slotcopy, COLS, decode_fields
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]

def is_mega(it):
    e=_items.get(it) or next((v for k,v in _items.items() if norm(v.get("name",k))==it), None)
    return bool(e and (e.get("megaStone") or e.get("megaEvolves")))
def is_zcrystal(it):
    e=_items.get(it) or next((v for k,v in _items.items() if norm(v.get("name",k))==it), None)
    return bool(e and e.get("zMove"))

# ITEM CLAUSE? do corpus teams ever repeat an item?
dupitem=0; megacount=Counter(); zcount=Counter()
for t in teams:
    its=[norm(s["item"]) for s in t if s["item"]]
    if len(set(its))!=len(its): dupitem+=1
    megacount[sum(is_mega(i) for i in its)]+=1
    zcount[sum(is_zcrystal(i) for i in its)]+=1
print("corpus teams with a repeated item:",dupitem,"of",len(teams),"-> Item Clause",
      "NOT in force" if dupitem else "APPARENTLY in force")
print("mega stones per corpus team:",dict(sorted(megacount.items())))
print("z-crystals per corpus team :",dict(sorted(zcount.items())))

# now: how often does slotcopy break a rule the oracle does not check?
N=1024
X_src=Xtr[rng.integers(0,len(Xtr),N)]; so=rng.integers(0,NSLOT,N)
Xa=op_slotcopy(X_src,so,Xtr,rng)
def team_items(row):
    return [V.decode_field(s*NF+2,int(row[s*NF+2])) for s in range(NSLOT)]
bad_mega=bad_dup=bad_z=0
for i in range(N):
    its=[x for x in team_items(Xa[i]) if x]
    if sum(is_mega(x) for x in its)>1: bad_mega+=1
    if sum(is_zcrystal(x) for x in its)>1: bad_z+=1
    if len(set(its))!=len(its): bad_dup+=1
print(f"\nslotcopy proposals (n={N}) that the oracle calls legal but that break:")
print(f"  >1 mega stone : {bad_mega/N:.3f}")
print(f"  >1 z-crystal  : {bad_z/N:.3f}")
print(f"  repeated item : {bad_dup/N:.3f}")
oracle_legal=np.mean([not L.violations(decode_fields(V,Xa[i])) for i in range(N)])
print(f"  oracle legal  : {oracle_legal:.3f}")
