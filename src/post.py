import numpy as np, torch, random, json
from collections import defaultdict
from corpus import load_corpus
from encode import Vocab, Legality, team_fields, NF, NSLOT
from model import MaskedFieldModel
from pilot import evaluate, memorization_stats, op_uniform, op_marginal, COLS, set_seed
DEV="mps" if torch.backends.mps.is_available() else "cpu"
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
print("vocab sizes:",V.sizes)
print("corpus teams failing the widened oracle:",sum(0 if L.legal(team_fields(t)) else 1 for t in teams),"of",len(teams))
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
colvals=defaultdict(list)
for c in range(COLS): colvals[V.key(c)]+=list(Xtr[:,c])
colvals={k:np.array(v) for k,v in colvals.items()}
m=MaskedFieldModel(V).to(DEV); m.load_state_dict(torch.load("/tmp/vgc-pilot/ck.pt")["sd"]); m.eval()
src_i=rng.integers(0,len(Xtr),256); X_src=Xtr[src_i]
cols=[list(rng.choice(COLS,3,replace=False)) for _ in range(256)]
_=op_uniform(X_src,cols,V,rng); _=op_marginal(X_src,cols,V,rng,colvals)

@torch.no_grad()
def seq(model,x,cols_,gen=None):
    x=x.clone()
    for b in range(x.shape[0]):
        cs=list(cols_[b]); random.shuffle(cs)
        for c in cs: x[b,c]=0
        for c in cs:
            h=model(x[b:b+1]); lg=model.logits_at(h,c)[0]; lg[0]=-1e9
            x[b,c]=torch.multinomial(torch.softmax(lg,-1),1,generator=gen).item()
    return x
@torch.no_grad()
def par(model,x,cols_,gen=None):
    """one forward pass, all masked fields sampled independently"""
    x=x.clone()
    for b in range(x.shape[0]):
        for c in cols_[b]: x[b,c]=0
    h=model(x)
    for b in range(x.shape[0]):
        for c in cols_[b]:
            lg=model.logits_at(h,c)[b:b+1][0]; lg=lg.clone(); lg[0]=-1e9
            x[b,c]=torch.multinomial(torch.softmax(lg,-1),1,generator=gen).item()
    return x
g=torch.Generator(device=DEV); g.manual_seed(0)
Xm=seq(m,torch.tensor(X_src,device=DEV),cols,g).cpu().numpy()
g2=torch.Generator(device=DEV); g2.manual_seed(0)
Xp=par(m,torch.tensor(X_src,device=DEV),cols,g2).cpu().numpy()
for nm,A in [("model sequential(Gibbs)",Xm),("model parallel(1 fwd pass)",Xp)]:
    e=evaluate(V,L,A,X_src,Xtr); ms=memorization_stats(A,Xtr)
    ch=[(A[i]!=X_src[i]).any() for i in range(len(A))]
    lg=[L.legal([V.decode_field(c,int(A[i,c])) for c in range(COLS)]) for i in range(len(A))]
    lc=sum(1 for a,b in zip(ch,lg) if a and b)/max(sum(ch),1)
    print("%-27s legal %.3f  changed %.3f  legal|changed %.3f  legal&novel %.3f  exact %.3f"%(nm,e["legal"],e["changed"],lc,e["legal_and_novel"],ms["exact_copy"]))
print("(results_s0.json model scatter: legal 0.285 changed 0.887 legal&novel 0.172 exact 0.113)")
