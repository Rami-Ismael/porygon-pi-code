import numpy as np, torch, random
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from model import MaskedFieldModel
from pilot import evaluate, memorization_stats, COLS, set_seed
DEV="mps" if torch.backends.mps.is_available() else "cpu"
set_seed(0); rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
X=np.stack([V.encode(t) for t in teams]); idx=rng.permutation(len(X)); nt=int(len(X)*.15)
Xte,Xtr=X[idx[:nt]],X[idx[nt:]]
m=MaskedFieldModel(V).to(DEV); m.load_state_dict(torch.load("/tmp/vgc-pilot/ck.pt")["sd"]); m.eval()

@torch.no_grad()
def gen(n, temp, order, seed):
    g=torch.Generator(device=DEV); g.manual_seed(seed); random.seed(seed)
    x=torch.zeros(n,COLS,dtype=torch.long,device=DEV)
    for b in range(n):
        left=list(range(COLS))
        if order=="random": random.shuffle(left)
        while left:
            h=m(x[b:b+1])
            if order=="confidence":
                best,bc=None,None
                for c in left:
                    lg=m.logits_at(h,c)[0].clone(); lg[0]=-1e9
                    p=torch.softmax(lg/max(temp,1e-6),-1); mx=float(p.max())
                    if best is None or mx>best: best,bc,bp=mx,c,p
                c,p=bc,bp
            else:
                c=left[0]; lg=m.logits_at(h,c)[0].clone(); lg[0]=-1e9; p=torch.softmax(lg/max(temp,1e-6),-1)
            x[b,c]= int(torch.argmax(p)) if temp<=0.01 else int(torch.multinomial(p,1,generator=g))
            left.remove(c)
    return x.cpu().numpy()

def canon(A):
    B=A.reshape(len(A),NSLOT,NF).copy()
    for i in range(len(B)):
        B[i]=B[i][np.argsort([V.itos["species"][int(B[i,s,0])] for s in range(NSLOT)])]
    return B.reshape(len(A),NSLOT*NF)

print("unconditional generation, n=64 each (pilot.py used random order, T=1.0):")
for temp,order in [(1.0,"random"),(0.7,"random"),(0.4,"random"),(1.0,"confidence"),(0.5,"confidence"),(0.0,"confidence")]:
    A=gen(64,temp,order,11)
    e=evaluate(V,L,A,np.full_like(A,-1),Xtr); r=memorization_stats(A,Xtr); c=memorization_stats(canon(A),Xtr)
    print("  T=%.1f %-11s legal %.3f  unique %.3f | NNdist raw %.1f canon %.1f | memorised raw %.3f canon %.3f | exact canon %.3f"%(
        temp,order,e["legal"],e["unique"],r["mean_nn_dist"],c["mean_nn_dist"],r["memorized"],c["memorized"],c["exact_copy"]))
