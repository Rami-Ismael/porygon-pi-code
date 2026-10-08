"""Per-SLOT legality as a classifier-free condition.

Two global legal/illegal tokens failed: legality is a conjunction of ~36 local
constraints, and one team-level token spread its signal over 84 fields. Here each
candidate carries its own token (illegal / legal / null), added to that slot's 14
positions. Training labels are per slot from the exact oracle; clause violations
(duplicate species/item) mark both slots involved. At inference every slot is set
to legal=1 and amplified with guidance.

Pre-registered expectation: a large per-field lift (this is the fair version of
the idea); whole-team validity unconstrained somewhere well above zero but well
below constrained decoding; no win-rate change on top of constraints.
"""
import argparse, os, random, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from corpus import load_corpus, norm
from encode import Legality, NSLOT
import diffusion as D
import spreaddiffusion as S
from spreaddiffusion import SpreadVocab, SpreadConstraints, ORDER2, NF2, COLS2, row_to_slots
from wrdiffusion import load_labelled, wr_bin, NBIN, WNULL
import legalcls as LC
import legaldiffusion as LD

DEV = D.DEV
CKPT = "/tmp/vgc-pilot/slotlegal.pt"
SL_ILLEGAL, SL_LEGAL, SL_NULL = 0, 1, 2

def slot_labels(row, C):
    """Six per-slot legality labels from the exact rules."""
    g = np.asarray(row).reshape(NSLOT, NF2); lab = np.ones(NSLOT, dtype=int)
    bases = [C.base_of[int(s[0])] for s in g]; items = [int(s[2]) for s in g]
    for i, s in enumerate(g):
        si = int(s[0])
        if si in C.mv_ok and any(int(s[j]) > 0 and not bool(C.mv_ok[si][int(s[j])]) for j in range(3, 7)): lab[i] = 0
        if si in C.ab_ok and int(s[1]) > 0 and not bool(C.ab_ok[si][int(s[1])]): lab[i] = 0
        if np.clip(s[8:] - 1, 0, None).sum() > S.BUDGET: lab[i] = 0
        it = items[i]
        if it in LC.MEGA_STONE_OF if hasattr(LC, "MEGA_STONE_OF") else False: pass
        if bases.count(bases[i]) > 1: lab[i] = 0
        if it > 0 and items.count(it) > 1: lab[i] = 0
    return lab

class SlotLegalDiffusion(LD.LegalDiffusion):
    def __init__(self, vocab, **kw):
        super().__init__(vocab, **kw)
        d = self.pos.shape[1]; self.sleg = nn.Embedding(3, d)
        self.slot_of_col = torch.tensor([c // NF2 for c in range(COLS2)])
    def forward(self, x, t, y, w, g, sl):
        h = torch.stack([self.emb[self.keys[c]](x[:, c]) for c in range(COLS2)], 1)
        h = h + self.pos.unsqueeze(0) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        h = h + (self.style(y) + self.wr(w) + self.leg(g)).unsqueeze(1)
        h = h + self.sleg(sl)[:, self.slot_of_col.to(x.device)]          # [B, COLS2, d] per-slot token
        return self.ln(self.tr(h))
    def loss(self, x, y, w, g, sl, p_uncond=0.15):
        B = x.shape[0]; t = torch.rand(B, device=x.device)
        m = torch.rand(B, COLS2, device=x.device) < t.view(-1, 1)
        empty = ~m.any(1)
        if empty.any(): m[empty, torch.randint(0, COLS2, (int(empty.sum()),), device=x.device)] = True
        yy = y.clone(); yy[torch.rand(B, device=x.device) < p_uncond] = D.NULL
        ww = w.clone(); ww[torch.rand(B, device=x.device) < p_uncond] = WNULL
        gg = torch.full_like(g, LD.LEG_NULL)                               # global token unused here
        ss = sl.clone(); ss[torch.rand(B, NSLOT, device=x.device) < p_uncond] = SL_NULL
        xin = x.clone(); xin[m] = 0
        h = self(xin, t, yy, ww, gg, ss); wgt = 1.0 / t.clamp(min=1e-3); tot, n = 0., 0
        for k, cols in self.cols_by_key.items():
            lg = self.head[k](h[:, cols]); mk = m[:, cols]
            if mk.any():
                ce = F.cross_entropy(lg[mk], x[:, cols][mk], reduction="none")
                tot = tot + (wgt.view(-1, 1).expand(-1, len(cols))[mk] * ce).sum(); n += int(mk.sum())
        return tot / max(n, 1)

def train(epochs=60, seed=0, batch=128):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ["LD_HEAVY"] = "1"; LD.HEAVY = True
    V, L, C, corpus, teams, X, Y, W, G = LD.build_data(seed, n_synth=30000)
    SL = np.stack([slot_labels(r, C) for r in X])
    print(f"data: {len(X):,} rows · slot-legal {SL.mean():.1%} · rows fully legal {(SL.min(1)==1).mean():.1%}", flush=True)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nte = int(len(X) * .05); tr, te = idx[nte:], idx[:nte]
    m = SlotLegalDiffusion(V).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01)
    steps = epochs * ((len(tr) + batch - 1) // batch); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    T = lambda a, i: torch.tensor(a[i], device=DEV)
    xtr, ytr, wtr, gtr, str_ = T(X, tr), T(Y, tr), T(W, tr), T(G, tr), T(SL, tr)
    xte, yte, wte, gte, ste = T(X, te), T(Y, te), T(W, te), T(G, te), T(SL, te)
    print(f"train {len(tr):,} / held-out {len(te):,} · {steps:,} steps · {DEV}", flush=True); t0 = time.perf_counter()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(xtr.shape[0], device=DEV); tot = nb = 0
        for i in range(0, xtr.shape[0], batch):
            b = perm[i:i + batch]; pr = torch.randperm(NSLOT, device=DEV)     # permute slots, and their labels with them
            xb = xtr[b].view(-1, NSLOT, NF2)[:, pr].reshape(-1, COLS2); sb = str_[b][:, pr]
            loss = m.loss(xb, ytr[b], wtr[b], gtr[b], sb); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step(); sch.step(); tot += float(loss); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            m.eval()
            with torch.no_grad(): vl = float(np.mean([float(m.loss(xte, yte, wte, gte, ste)) for _ in range(4)]))
            print(f"  ep{ep+1:3d} train {tot/nb:6.2f}  held-out {vl:6.2f}  ({(time.perf_counter()-t0)/(ep+1):.0f}s/epoch)", flush=True)
            torch.save({"sd": m.state_dict(), "itos": V.itos}, CKPT)
    torch.save({"sd": m.state_dict(), "itos": V.itos}, CKPT); print("saved", CKPT, flush=True)

def load_model():
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    teams, wrs = load_labelled(); V = SpreadVocab(corpus + teams); L = Legality(corpus)
    ck = torch.load(CKPT, map_location=DEV); m = SlotLegalDiffusion(V).to(DEV); m.load_state_dict(ck["sd"]); m.eval(); return m, V, L, corpus, teams

@torch.no_grad()
def sample(model, C, n, wbin=5, g_wr=2.0, sleg=SL_NULL, g_leg=1.0, style="none", constrained=True, temp=1.0, device=DEV):
    ys = D.STYLES.index(style) if style in D.STYLES else D.NULL
    y = torch.full((n,), ys, device=device, dtype=torch.long); yN = torch.full_like(y, D.NULL)
    w = torch.full((n,), wbin, device=device, dtype=torch.long); wN = torch.full_like(w, WNULL)
    gN = torch.full((n,), LD.LEG_NULL, device=device, dtype=torch.long)
    sl = torch.full((n, NSLOT), sleg, device=device, dtype=torch.long); slN = torch.full_like(sl, SL_NULL)
    x = torch.zeros(n, COLS2, dtype=torch.long, device=device)
    for step, c in enumerate(ORDER2):
        t_now = 1.0 - step / len(ORDER2); tt = torch.full((n,), t_now, device=device)
        lu = F.log_softmax(model.logits(model(x, tt, yN, wN, gN, slN), c), -1); l = lu.clone()
        if g_wr != 0: l = l + g_wr * (F.log_softmax(model.logits(model(x, tt, y, w, gN, slN), c), -1) - lu)
        if sleg != SL_NULL and g_leg != 0: l = l + g_leg * (F.log_softmax(model.logits(model(x, tt, yN, wN, gN, sl), c), -1) - lu)
        for b in range(n):
            lb = l[b].clone()
            if constrained: ok = C.mask_for(c, x[b], device); lb[~ok] = -1e9
            else: lb[0] = -1e9
            x[b, c] = torch.multinomial(torch.softmax(lb / temp, -1), 1).item()
    return x

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=60); a = ap.parse_args(); train(a.epochs)
