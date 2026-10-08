"""Is 'model loses 0.023 vs 0.887' a property of the model, or of the sampler
(temperature 1.0 + random column order)?"""
import numpy as np, torch, random
from corpus import load_corpus
from encode import Vocab, Legality, NF, NSLOT
from model import MaskedFieldModel
from pilot import COLS, evaluate
DEV="mps" if torch.backends.mps.is_available() else "cpu"
rng=np.random.default_rng(0)
teams,_=load_corpus(); teams=[t for t in teams if len(t)==6]
V,L=Vocab(teams),Legality(teams)
Z=np.load("/tmp/vgc-pilot/arrays_s0.npz"); Xtr=Z["Xtr"]
model=MaskedFieldModel(V).to(DEV); model.load_state_dict(torch.load("/tmp/vgc-pilot/model_s0.pt")); model.eval()

@torch.no_grad()
def resample(x, cols_to_mask, temp=1.0, order="random", greedy=False, gen=None):
    x=x.clone()
    for b in range(x.shape[0]):
        cols=list(cols_to_mask[b])
        if order=="random": random.shuffle(cols)
        elif order=="speciesfirst":
            sp=[c for c in cols if c%NF==0]; rest=[c for c in cols if c%NF!=0]
            random.shuffle(rest); cols=sp+rest
        elif order=="lefttoright": cols=sorted(cols)
        for c in cols: x[b,c]=0
        for c in cols:
            h=model(x[b:b+1]); lg=model.logits_at(h,c)[0]/temp; lg[0]=-1e9
            if greedy: x[b,c]=int(lg.argmax())
            else: x[b,c]=torch.multinomial(torch.softmax(lg,-1),1,generator=gen).item()
    return x

for regime,cols,src in [("slot",Z["slot_cols"],Z["slot_src"]),("scatter",Z["scatter_cols"],Z["scatter_src"])]:
    cols=[list(r) for r in cols]
    print(f"\n== {regime} regime, 256 proposals from the pilot's own sources ==")
    for tag,kw in [("pilot (rand order, T=1)",dict(order="random",temp=1.0)),
                   ("species-first, T=1",dict(order="speciesfirst",temp=1.0)),
                   ("left-to-right, T=1",dict(order="lefttoright",temp=1.0)),
                   ("rand order, T=0.5",dict(order="random",temp=0.5)),
                   ("species-first, T=0.5",dict(order="speciesfirst",temp=0.5)),
                   ("species-first, T=0.25",dict(order="speciesfirst",temp=0.25)),
                   ("species-first, greedy",dict(order="speciesfirst",greedy=True))]:
        random.seed(0); g=torch.Generator(device=DEV); g.manual_seed(0)
        A=resample(torch.tensor(src,device=DEV),cols,gen=g,**kw).cpu().numpy()
        e=evaluate(V,L,A,src,Xtr)
        print(f"  {tag:26s} legal {e['legal']:.3f}  changed {e['changed']:.3f}  legal&novel {e['legal_and_novel']:.3f}  unique {e['unique']:.3f}")
