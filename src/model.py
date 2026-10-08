"""The minimum model the literature review identified as delivering the 'guided
local move': a masked-field predictor p(field | all other fields), trained
BERT-style with a random mask rate (this is exactly MDLM's objective, which the
paper itself calls 'a weighted average of masked language modeling losses').

No diffusion machinery, no score network, no continuous relaxation.
"""
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from encode import FIELDS, NF, NSLOT

class MaskedFieldModel(nn.Module):
    def __init__(self, vocab, d=192, nhead=6, nlayer=4, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.cols = NSLOT * NF
        # one embedding table per field type, one output head per field type
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in vocab.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in vocab.sizes.items()})
        self.pos = nn.Parameter(torch.randn(self.cols, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer)
        self.ln = nn.LayerNorm(d)
        self.keys = [vocab.key(c) for c in range(self.cols)]

    def forward(self, x):
        B = x.shape[0]
        h = torch.stack([self.emb[self.keys[c]](x[:, c]) for c in range(self.cols)], 1)
        h = self.tr(h + self.pos.unsqueeze(0))
        h = self.ln(h)
        return h  # B, cols, d

    def logits_at(self, h, col):
        return self.head[self.keys[col]](h[:, col])

    def loss(self, x, mask):
        """Cross-entropy on masked columns only."""
        xin = x.clone()
        xin[mask] = 0  # 0 == [MASK] in every field vocabulary
        h = self(xin)
        tot, n = 0.0, 0
        for c in range(self.cols):
            m = mask[:, c]
            if m.any():
                lg = self.logits_at(h, c)[m]
                tot = tot + F.cross_entropy(lg, x[m, c], reduction="sum")
                n += int(m.sum())
        return tot / max(n, 1)

def random_mask(x, gen=None):
    """MDLM-style random mask rate: t ~ U(0,1) per team, then mask each field w.p. t
    (at least one field always masked)."""
    B, C = x.shape
    t = torch.rand(B, 1, generator=gen, device=x.device)
    m = torch.rand(B, C, generator=gen, device=x.device) < t
    empty = ~m.any(1)
    if empty.any():
        idx = torch.randint(0, C, (int(empty.sum()),), generator=gen, device=x.device)
        m[empty, idx] = True
    return m

def permute_slots(x):
    """A team is a set: permuting slots is a symmetry, used as augmentation."""
    B = x.shape[0]
    out = x.view(B, NSLOT, NF).clone()
    for i in range(B):
        out[i] = out[i][torch.randperm(NSLOT)]
    return out.view(B, NSLOT * NF)

@torch.no_grad()
def resample_fields(model, x, cols_to_mask, temp=1.0, gen=None):
    """The local move: mask the given fields of a real team and resample them from
    the model, one field at a time in random order, conditioning on everything
    already filled in (Wang & Cho 2019 Gibbs order; also MaskGIT-style iterative
    unmasking). Returns a new team grid."""
    x = x.clone()
    for b in range(x.shape[0]):
        cols = list(cols_to_mask[b])
        random.shuffle(cols)
        for c in cols:
            x[b, c] = 0
        for c in cols:
            h = model(x[b:b+1])
            lg = model.logits_at(h, c)[0] / temp
            lg[0] = -1e9  # never emit [MASK]
            p = torch.softmax(lg, -1)
            x[b, c] = torch.multinomial(p, 1, generator=gen).item()
    return x
