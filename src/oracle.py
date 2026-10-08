import numpy as np
from collections import defaultdict, Counter
from corpus import load_corpus, norm, legal_moves, legal_abilities
from encode import Vocab, Legality, NF, NSLOT
from pilot import COLS, memorization_stats
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
L=Legality(teams); V=Vocab(teams)
add_m=add_a=tot_m=tot_a=0
for sp in L.corpus_moves:
    base=legal_moves(sp) or set(); tot_m+=len(L.corpus_moves[sp]); add_m+=len(L.corpus_moves[sp]-base)
    ba=legal_abilities(sp) or set(); tot_a+=len(L.corpus_abils[sp]); add_a+=len(L.corpus_abils[sp]-base if False else L.corpus_abils[sp]-ba)
print("corpus-widening of the oracle: moves added beyond real learnset %d / %d corpus (species,move) pairs"%(add_m,tot_m))
print("                              abilities added beyond dex        %d / %d"%(add_a,tot_a))
print("mean legal moves per corpus species: %.0f"%np.mean([len(L.moves_for(s)) for s in L.corpus_moves]))
print("move vocab size (corpus moves only): %d"%(V.sizes["move"]-1))
# chance a uniform corpus-vocab move is legal for a random corpus species
mv=[V.itos["move"][i] for i in range(1,V.sizes["move"])]
ps=[np.mean([m in L.moves_for(s) for m in mv]) for s in list(L.corpus_moves)[:60]]
print("P(random move from corpus move-vocab is legal for a species): %.3f"%np.mean(ps))
# duplicates across the split
rng=np.random.default_rng(0)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
tr=set(map(lambda r: r.tobytes(), Xtr))
dupe=sum(1 for r in Xte if r.tobytes() in tr)
print("\nheld-out teams that are EXACT duplicates of a train team: %d / %d"%(dupe,len(Xte)))
ms=memorization_stats(Xte,Xtr)
d=[np.min((Xte[i][None,:]!=Xtr).sum(1)) for i in range(len(Xte))]
srt=np.sort(np.array(d))
print("held-out NN-dist distribution:", dict(Counter(d).most_common(8)))
sptr=set(tuple(sorted(r[c] for c in range(0,COLS,NF))) for r in Xtr)
memflag=[]
for i in range(len(Xte)):
    dd=np.sort((Xte[i][None,:]!=Xtr).sum(1)); memflag.append(dd[1]>0 and dd[0]<dd[1]/3)
sh=[tuple(sorted(Xte[i][c] for c in range(0,COLS,NF))) in sptr for i in range(len(Xte))]
print("held-out 'memorised' count: %d ; of those, how many share their 6 species with a train team: %d"%(sum(memflag),sum(1 for a,b in zip(memflag,sh) if a and b)))
