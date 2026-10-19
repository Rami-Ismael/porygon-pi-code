"""Drill check 2026-10-17: pin species slot 1 uniformly at random, model fills the rest (ask 0.5, guidance 2). Decode only."""
import sys, math
from collections import Counter
import numpy as np, torch
sys.path.insert(0, "/tmp/vgc-pilot/src")
import lossdown as L, lossdown_battle as LB, sampler_battle as SB, activesearch as A
import diffusion as D
from hps_generate import Validator
SB._setup(); V, tables, f = SB.V, SB.tables, SB.f
SP = L.COLS_BY_KEY["species"]; NS = SB.SIZES["species"]; names = V.itos["species"]

@torch.no_grad()
def sample(n, pin_first):
    w = torch.full((n,), 0.5, device=L.DEV); wn = torch.full((n,), float("nan"), device=L.DEV)
    x = torch.zeros(n, L.COLS, dtype=torch.long, device=L.DEV)
    order = list(D.ORDER)
    if pin_first:
        x[:, SP[0]] = torch.randint(1, NS, (n,), device=L.DEV); order.remove(SP[0])
    for step, c in enumerate(order):
        t = torch.full((n,), 1.0 - (step + pin_first) / L.COLS, device=L.DEV)
        k = L.KEY_OF_COL[c]; j = L.COLS_BY_KEY[k].index(c)
        lc = 2.0 * f(x, t, w)[k][:, j] - 1.0 * f(x, t, wn)[k][:, j]
        ok = tables.allowed(x)[k][:, j]; ok[~ok.any(1)] = True; ok[:, 0] = False
        x[:, c] = torch.multinomial(torch.softmax(lc.masked_fill(~ok, -float("inf")), -1), 1).squeeze(1)
    return x

