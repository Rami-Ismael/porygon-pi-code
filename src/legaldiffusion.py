"""Masked diffusion over the full team (Stat Points included) with THREE
classifier-free conditions: playstyle, win-rate bin, and LEGALITY.

Legality as a condition token is what "classifier-free guidance for legal teams"
means: the model is trained on legal teams (token 1) and on manufactured illegal
ones (token 0, built by enumerating the rule tables), so at inference you can
condition on legal=1 and amplify it — no separate classifier, no rule oracle in
the loop. Whether that beats or adds to exact constrained decoding is the question
the four-arm evaluation answers.

Speed fix vs spreaddiffusion.py: one output head per FIELD TYPE (6 heads) instead of
one per column (84), so the loss is 6 cross-entropies instead of 84.
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from corpus import load_corpus, norm, STATS
from encode import Legality, NSLOT
import diffusion as D
import spreaddiffusion as S
from spreaddiffusion import SpreadVocab, SpreadConstraints, ORDER2, NF2, COLS2, row_to_slots
from wrdiffusion import load_labelled, wr_bin, NBIN, WNULL
import legalcls as LC

DEV = D.DEV
import os
CKPT = os.environ.get("LD_CKPT", "/tmp/vgc-pilot/legaldiffusion.pt")
HEAVY = os.environ.get("LD_HEAVY", "0") == "1"
LEG_ILLEGAL, LEG_LEGAL, LEG_NULL = 0, 1, 2
KEYS = ["species", "ability", "item", "move", "nature", "stat"]

class LegalDiffusion(nn.Module):
    def __init__(self, vocab, d=192, nhead=6, nlayer=4, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in vocab.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in vocab.sizes.items()})
        self.pos = nn.Parameter(torch.randn(COLS2, d) * .02)
        self.style = nn.Embedding(D.NULL + 1, d); self.wr = nn.Embedding(NBIN + 1, d); self.leg = nn.Embedding(3, d)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer); self.ln = nn.LayerNorm(d)
        self.keys = [vocab.key(c) for c in range(COLS2)]
        self.cols_by_key = {k: [c for c in range(COLS2) if self.keys[c] == k] for k in KEYS}
    def forward(self, x, t, y, w, g):
        h = torch.stack([self.emb[self.keys[c]](x[:, c]) for c in range(COLS2)], 1)
        h = h + self.pos.unsqueeze(0) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        h = h + (self.style(y) + self.wr(w) + self.leg(g)).unsqueeze(1)
        return self.ln(self.tr(h))
    def logits(self, h, c): return self.head[self.keys[c]](h[:, c])
    def loss(self, x, y, w, g, p_uncond=0.15):
        B = x.shape[0]; t = torch.rand(B, device=x.device)
        m = torch.rand(B, COLS2, device=x.device) < t.view(-1, 1)
        empty = ~m.any(1)
        if empty.any(): m[empty, torch.randint(0, COLS2, (int(empty.sum()),), device=x.device)] = True
        yy = y.clone(); yy[torch.rand(B, device=x.device) < p_uncond] = D.NULL
        ww = w.clone(); ww[torch.rand(B, device=x.device) < p_uncond] = WNULL
        gg = g.clone(); gg[torch.rand(B, device=x.device) < p_uncond] = LEG_NULL
        xin = x.clone(); xin[m] = 0
        h = self(xin, t, yy, ww, gg); wgt = 1.0 / t.clamp(min=1e-3)
        tot, n = 0., 0
        for k, cols in self.cols_by_key.items():                       # 6 heads, not 84
            hk = h[:, cols]; lg = self.head[k](hk)                       # [B, ncols, V_k]
            mk = m[:, cols]
            if mk.any():
                ce = F.cross_entropy(lg[mk], x[:, cols][mk], reduction="none")
                wk = wgt.view(-1, 1).expand(-1, len(cols))[mk]
                tot = tot + (wk * ce).sum(); n += int(mk.sum())
        return tot / max(n, 1)

def build_data(seed=0, n_synth=20000):
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    teams, wrs = load_labelled()
    V = SpreadVocab(corpus + teams); L = Legality(corpus); C = SpreadConstraints(V, L)
    rng = np.random.default_rng(seed)
    Xl = np.stack([V.encode(t) for t in teams]); Yl = np.array([D.style_of(t) for t in teams]); Wl = np.array([wr_bin(w) for w in wrs])
    Xc = np.stack([V.encode(t) for t in corpus]); Yc = np.array([D.style_of(t) for t in corpus]); Wc = np.full(len(corpus), WNULL)
    base = np.concatenate([Xl, Xc]); Xs, Gs = LC.synth_rows(V, L, C, base, n_synth, rng)
    if HEAVY:
        # pile 3-8 rule-targeted corruptions onto every illegal row so 'illegal' is a learnable class
        for i in np.where(Gs == 0)[0]:
            r = Xs[i]
            for _ in range(int(rng.integers(3, 9))): r = LC.corrupt(r, V, rng, C)
            Xs[i] = r; Gs[i] = int(LC.is_legal(V, L, r))
    X = np.concatenate([Xl, Xc, Xs]); Y = np.concatenate([Yl, Yc, np.zeros(len(Xs), dtype=int)])
    W = np.concatenate([Wl, Wc, np.full(len(Xs), WNULL)]); G = np.concatenate([np.ones(len(Xl) + len(Xc), dtype=int), Gs])
    return V, L, C, corpus, teams, X, Y, W, G

def train(epochs=60, seed=0, batch=128):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    V, L, C, corpus, teams, X, Y, W, G = build_data(seed)
    print(f"data: {len(X):,} rows · legal {G.mean():.1%} · with win-rate label {(W != WNULL).mean():.1%}", flush=True)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nte = int(len(X) * .05); tr, te = idx[nte:], idx[:nte]
    m = LegalDiffusion(V).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01)
    steps = epochs * ((len(tr) + batch - 1) // batch); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    T = lambda a, i: torch.tensor(a[i], device=DEV)
    xtr, ytr, wtr, gtr = T(X, tr), T(Y, tr), T(W, tr), T(G, tr); xte, yte, wte, gte = T(X, te), T(Y, te), T(W, te), T(G, te)
    print(f"train {len(tr):,} / held-out {len(te):,} · {sum(p.numel() for p in m.parameters()):,} params · {steps:,} steps · {DEV}", flush=True)
    t0 = time.perf_counter()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(xtr.shape[0], device=DEV); tot = nb = 0
        for i in range(0, xtr.shape[0], batch):
            b = perm[i:i + batch]; xb = xtr[b].view(-1, NSLOT, NF2)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, COLS2)
            loss = m.loss(xb, ytr[b], wtr[b], gtr[b]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step(); sch.step(); tot += float(loss); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            m.eval()
            with torch.no_grad(): vl = float(np.mean([float(m.loss(xte, yte, wte, gte)) for _ in range(4)]))
            print(f"  ep{ep+1:3d} train {tot/nb:6.2f}  held-out {vl:6.2f}  ({(time.perf_counter()-t0)/(ep+1):.0f}s/epoch)", flush=True)
            torch.save({"sd": m.state_dict(), "itos": V.itos}, CKPT)
    torch.save({"sd": m.state_dict(), "itos": V.itos}, CKPT); print("saved", CKPT, flush=True)

def load_model():
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    teams, wrs = load_labelled(); V = SpreadVocab(corpus + teams); L = Legality(corpus)
    ck = torch.load(CKPT, map_location=DEV); assert ck["itos"]["species"] == V.itos["species"]
    m = LegalDiffusion(V).to(DEV); m.load_state_dict(ck["sd"]); m.eval(); return m, V, L, corpus, teams

@torch.no_grad()
def sample(model, C, n, wbin=5, g_wr=2.0, leg=LEG_NULL, g_leg=1.0, style="none", constrained=True, temp=1.0, device=DEV):
    """Composable classifier-free guidance:
       l = l_u + g_wr * (l_wr - l_u) + g_leg * (l_leg - l_u), each a log-softmax; then
       (optionally) exact rule masks, then sample. leg=LEG_NULL disables the legality term."""
    ys = D.STYLES.index(style) if style in D.STYLES else D.NULL
    y = torch.full((n,), ys, device=device, dtype=torch.long); yN = torch.full_like(y, D.NULL)
    w = torch.full((n,), wbin, device=device, dtype=torch.long); wN = torch.full_like(w, WNULL)
    g = torch.full((n,), leg, device=device, dtype=torch.long); gN = torch.full_like(g, LEG_NULL)
    x = torch.zeros(n, COLS2, dtype=torch.long, device=device)
    for step, c in enumerate(ORDER2):
        t_now = 1.0 - step / len(ORDER2); tt = torch.full((n,), t_now, device=device)
        lu = F.log_softmax(model.logits(model(x, tt, yN, wN, gN), c), -1)
        l = lu.clone()
        if g_wr != 0:
            lw = F.log_softmax(model.logits(model(x, tt, y, w, gN), c), -1); l = l + g_wr * (lw - lu)
        if leg != LEG_NULL and g_leg != 0:
            lg = F.log_softmax(model.logits(model(x, tt, yN, wN, g), c), -1); l = l + g_leg * (lg - lu)
        for b in range(n):
            lb = l[b].clone()
            if constrained: ok = C.mask_for(c, x[b], device); lb[~ok] = -1e9
            else: lb[0] = -1e9
            x[b, c] = torch.multinomial(torch.softmax(lb / temp, -1), 1).item()
    return x

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "speed"]); ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    if a.cmd == "train": train(a.epochs)
    else:
        V, L, C, corpus, teams, X, Y, W, G = build_data(); m = LegalDiffusion(V).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4); T = lambda a: torch.tensor(a[:128], device=DEV)
        xb, yb, wb, gb = T(X), T(Y), T(W), T(G); t0 = time.perf_counter()
        for _ in range(20):
            loss = m.loss(xb, yb, wb, gb); opt.zero_grad(); loss.backward(); opt.step()
        dt = (time.perf_counter() - t0) / 20
        print(f"data {len(X):,} rows · {dt*1000:.0f} ms/step at batch 128 · epoch ≈ {dt*len(X)/128:.0f}s · 60 epochs ≈ {60*dt*len(X)/128/60:.0f} min")
