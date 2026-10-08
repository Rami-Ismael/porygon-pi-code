"""Masked diffusion over teams, conditioned on WIN RATE as well as playstyle.

This is the model the research goal actually asked for: add "win rate against the
meta" as a conditioning feature, then at inference ask for the top bin. It became
trainable only once the labelling campaign produced 4,850 (team, win-rate) pairs —
the corpus alone carries no such labels.

Conditioning: two tokens, playstyle (rule-derived) and win-rate bin (6 bins,
bin 5 = [0.5, 1.0]). Both are dropped independently with p=0.15 during training so
classifier-free guidance can amplify either at inference. Decoding is constrained
(species first, then ability/item, then moves) so every sample is rule-legal;
Showdown's validator is still the final gate.

  train   : python wrdiffusion.py train [epochs]
  sample  : python wrdiffusion.py sample --bin 5 --guidance 2 --n 32
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from corpus import parse_team, load_corpus, norm
from encode import Vocab, Legality, NF, NSLOT
import diffusion as D

DEV = D.DEV
COLS = D.COLS
CKPT = "/tmp/vgc-pilot/wrdiffusion.pt"
EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.01]
NBIN = 6
WNULL = NBIN                       # dropped-condition token for win rate

def wr_bin(w):
    for i in range(NBIN):
        if EDGES[i] <= w < EDGES[i + 1]: return i
    return NBIN - 1

def load_labelled():
    lab = {}
    for f in ["/tmp/vgc-pilot/ladder_labels.json", "/tmp/vgc-pilot/strata_labels.json"]:
        for p, v in json.load(open(f)).items(): lab[p] = v["win_rate"]
    teams, wrs = [], []
    for p, w in lab.items():
        t = parse_team(p)
        if t and len(t) == 6: teams.append(t); wrs.append(w)
    return teams, np.array(wrs)

class TeamDiffusionWR(nn.Module):
    def __init__(self, vocab, d=192, nhead=6, nlayer=4, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in vocab.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in vocab.sizes.items()})
        self.pos = nn.Parameter(torch.randn(COLS, d) * .02)
        self.style = nn.Embedding(D.NULL + 1, d)
        self.wr = nn.Embedding(NBIN + 1, d)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer)
        self.ln = nn.LayerNorm(d)
        self.keys = [vocab.key(c) for c in range(COLS)]

    def forward(self, x, t, y, w):
        h = torch.stack([self.emb[self.keys[c]](x[:, c]) for c in range(COLS)], 1)
        h = h + self.pos.unsqueeze(0) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        h = h + self.style(y).unsqueeze(1) + self.wr(w).unsqueeze(1)
        return self.ln(self.tr(h))

    def logits(self, h, c):
        return self.head[self.keys[c]](h[:, c])

    def loss(self, x, y, w, p_uncond=0.15):
        B = x.shape[0]
        t = torch.rand(B, device=x.device)
        m = torch.rand(B, COLS, device=x.device) < t.view(-1, 1)
        empty = ~m.any(1)
        if empty.any():
            m[empty, torch.randint(0, COLS, (int(empty.sum()),), device=x.device)] = True
        yy = y.clone(); yy[torch.rand(B, device=x.device) < p_uncond] = D.NULL
        ww = w.clone(); ww[torch.rand(B, device=x.device) < p_uncond] = WNULL
        xin = x.clone(); xin[m] = 0
        h = self(xin, t, yy, ww)
        tot, n = 0., 0
        for c in range(COLS):
            mc = m[:, c]
            if mc.any():
                wgt = 1.0 / t[mc].clamp(min=1e-3)
                ce = F.cross_entropy(self.logits(h, c)[mc], x[mc, c], reduction="none")
                tot = tot + (wgt * ce).sum(); n += int(mc.sum())
        return tot / max(n, 1)

def build_vocab():
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    teams, wrs = load_labelled()
    V = Vocab(corpus + teams)          # union: 0.68% of labelled fields are outside the corpus vocab
    L = Legality(corpus)
    return V, L, corpus, teams, wrs

def train(epochs=300, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    V, L, corpus, teams, wrs = build_vocab()
    X = np.stack([V.encode(t) for t in teams])
    Y = np.array([D.style_of(t) for t in teams])
    W = np.array([wr_bin(w) for w in wrs])
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nte = int(len(X) * .1)
    tr, te = idx[nte:], idx[:nte]
    m = TeamDiffusionWR(V).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xtr, ytr, wtr = (torch.tensor(a[tr], device=DEV) for a in (X, Y, W))
    xte, yte, wte = (torch.tensor(a[te], device=DEV) for a in (X, Y, W))
    print(f"train {len(tr)} / held-out {len(te)} labelled teams · "
          f"{sum(p.numel() for p in m.parameters()):,} params · {DEV}", flush=True)
    print("win-rate bins in training set:", np.bincount(W[tr], minlength=NBIN).tolist(), flush=True)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(xtr.shape[0], device=DEV); tot = nb = 0
        for i in range(0, xtr.shape[0], 64):
            b = perm[i:i + 64]
            xb = xtr[b].view(-1, NSLOT, NF)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, COLS)
            loss = m.loss(xb, ytr[b], wtr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step(); tot += float(loss); nb += 1
        sch.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            m.eval()
            with torch.no_grad():
                vl = float(np.mean([float(m.loss(xte, yte, wte)) for _ in range(6)]))
            print(f"  ep{ep+1:4d} train {tot/nb:7.2f}  held-out {vl:7.2f}", flush=True)
            torch.save({"sd": m.state_dict(), "itos": V.itos}, CKPT)
    torch.save({"sd": m.state_dict(), "itos": V.itos}, CKPT)
    print("saved", CKPT, flush=True)

def load_model():
    V, L, corpus, teams, wrs = build_vocab()
    ck = torch.load(CKPT, map_location=DEV)
    assert ck["itos"]["species"] == V.itos["species"], "vocab drift: rebuild"
    m = TeamDiffusionWR(V).to(DEV); m.load_state_dict(ck["sd"]); m.eval()
    return m, V, L, corpus, teams

@torch.no_grad()
def sample_constrained(model, C, n, wbin, style="none", guidance=1.0, temp=1.0, pin=None,
                       top_p=None, device=DEV):
    """Constrained decoding with classifier-free guidance on the (style, win-rate) pair.

    `top_p` adds nucleus filtering AFTER the feasibility mask, so it never resurrects an
    illegal value. Measured 2026-08-27 (`src/schemes.py`): temp 0.7 + top_p 0.9 scores
    0.260 against the top-50 pool at 97% Showdown-valid, versus 0.174 at 82% for the
    temp-1.0 default -- +8.7 points and higher validity, for free, at inference time.
    """
    y = torch.full((n,), D.STYLES.index(style) if style in D.STYLES else D.NULL, device=device, dtype=torch.long)
    w = torch.full((n,), wbin, device=device, dtype=torch.long)
    yN = torch.full_like(y, D.NULL); wN = torch.full_like(w, WNULL)
    x = torch.zeros(n, COLS, dtype=torch.long, device=device)
    todo = set(range(COLS))
    if pin:
        for c, v in pin.items(): x[:, c] = v; todo.discard(c)
    seq = [c for c in D.ORDER if c in todo]
    for step, c in enumerate(seq):
        t_now = 1.0 - step / max(len(seq), 1)
        tt = torch.full((n,), t_now, device=device)
        lc = F.log_softmax(model.logits(model(x, tt, y, w), c), -1)
        if guidance != 1.0:
            lu = F.log_softmax(model.logits(model(x, tt, yN, wN), c), -1)
            lc = guidance * lc + (1 - guidance) * lu
        for b in range(n):
            ok = C.mask_for(c, x[b], device)
            l = lc[b].clone(); l[~ok] = -1e9
            p = torch.softmax(l / temp, -1)
            if top_p is not None and 0 < top_p < 1:
                srt, idx = torch.sort(p, descending=True)
                cut = torch.cumsum(srt, -1) - srt > top_p     # keep the token that crosses top_p
                srt = srt.masked_fill(cut, 0.0)
                if float(srt.sum()) > 0:
                    p = torch.zeros_like(p).scatter_(0, idx, srt)
                    p = p / p.sum()
            x[b, c] = torch.multinomial(p, 1).item()
    return x

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sample"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bin", type=int, default=5)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--style", default="none")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.epochs)
    else:
        m, V, L, corpus, teams = load_model(); C = D.Constraints(V, L)
        x = sample_constrained(m, C, a.n, a.bin, a.style, a.guidance).cpu().numpy()
        ok = sum(L.legal([V.decode_field(c, int(r[c])) for c in range(COLS)]) for r in x)
        print(f"bin={a.bin} guidance={a.guidance} -> rule-legal {ok}/{a.n}")
        for r in x[:2]:
            print("  ---"); [print("   ", line) for line in D.to_text(V, r, corpus + teams)]
