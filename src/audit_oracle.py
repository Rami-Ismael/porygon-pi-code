import numpy as np, json
from collections import Counter
from corpus import load_corpus, norm, dex_entry
from encode import Vocab, Legality, NF, NSLOT, team_fields
from pilot import COLS, op_slotcopy, evaluate, hamming
rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams])
idx=rng.permutation(len(X)); nt=int(len(X)*.15); Xte,Xtr=X[idx[:nt]],X[idx[nt:]]

# what is in the corpus: restricted legendaries?
RESTRICTED={"mewtwo","lugia","hooh","kyogre","groudon","rayquaza","dialga","palkia","giratina",
 "reshiram","zekrom","kyurem","xerneas","yveltal","zygarde","cosmog","cosmoem","solgaleo","lunala",
 "necrozma","zacian","zamazenta","eternatus","calyrex","koraidon","miraidon","terapagos","dialgaorigin"}
cnt=Counter()
for t in teams:
    n=0
    for s in t:
        e=dex_entry(s["species"]); b=norm(e.get("baseSpecies",e.get("name",s["species"]))) if e else norm(s["species"])
        if b in RESTRICTED: n+=1
    cnt[n]+=1
print("restricted-count histogram over corpus teams:", sorted(cnt.items()))

def nrestr(fields):
    n=0
    for i in range(NSLOT):
        sp=fields[i*NF]; e=dex_entry(sp)
        b=norm(e.get("baseSpecies",e.get("name",sp))) if e else sp
        if b in RESTRICTED: n+=1
    return n
def dec(row): return [V.decode_field(c,int(row[c])) for c in range(COLS)]

maxr=max(k for k,v in cnt.items() if v>0)
# slotcopy arm: how many exceed the corpus-attested restricted cap?
src_i=rng.integers(0,len(Xtr),256); X_src=Xtr[src_i]
slot_of=rng.integers(0,NSLOT,256)
A=op_slotcopy(X_src,slot_of,Xtr,rng)
over=sum(1 for i in range(256) if nrestr(dec(A[i]))>maxr)
e=evaluate(V,L,A,X_src,Xtr)
print(f"slotcopy: oracle legal {e['legal']:.3f}; proposals exceeding corpus max restricted ({maxr}): {over/256:.3f}")
print(f"  -> slotcopy legality if the restricted cap were enforced: <= {e['legal']-over/256:.3f}")

# does the oracle check anything about Stat Points / Tera at all?
import inspect, encode
src=inspect.getsource(encode.Legality.violations)
print("oracle checks tera?", "tera" in src, " stat points/EV?", "ev" in src.lower())
# item duplication (no Item Clause in VGC -> fine). Check ability legality widening effect
widened=sum(1 for sp in L.corpus_abils for a in L.corpus_abils[sp] if a not in (set()|{x for x in (__import__('corpus').legal_abilities(sp) or set())}))
print("ability values accepted ONLY because the corpus attests them:", widened)
wm=0
for sp in L.corpus_moves:
    base=__import__('corpus').legal_moves(sp) or set()
    wm+=len(L.corpus_moves[sp]-base)
print("(species,move) pairs accepted ONLY because the corpus attests them:", wm)
