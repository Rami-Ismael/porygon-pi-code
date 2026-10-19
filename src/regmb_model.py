"""Inference loader for the full-regulation M-B diffusion model.

Legacy experiments keep their explicitly pinned checkpoints. New M-B callers
should use model() here; token order and legality come from the saved checkpoint.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from encode import Vocab
import lossdown as L
import asked_vs_got as X
from checkpoints import load_checkpoint

MEMBERS = ['F1_final_medium_noheldout_regmb_v2','G2_family_matrix_noprtrain_noheldout_regmb_v2']


def model(members=None):
    members=members or MEMBERS
    checkpoint=load_checkpoint(L.CK/(members[0]+'.pt'),map_location='cpu')
    if not checkpoint.get('regulation'):
        raise ValueError('full-regulation loader requires a versioned checkpoint')
    V=Vocab.from_regulation(checkpoint['regulation'])
    corpus,_,_,look,spreads=X.setup()
    look.update(V.regulation['names'])
    tables=L.Tables(V,corpus)
    functions=[]
    for member in members:
        net,cfg,_=L.load_net(member,V)
        functions.append(L.logprobs_fn(net,tables,tuple(cfg['rules'].split(','))))
    return V,tables,L.ensemble_fn(functions),look,spreads
