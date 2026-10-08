"""Masked diffusion trained on the 100,000-team HPS dataset, conditioned on win rate.

The 100k teams from hierarchical product sampling (`teams/hps_reg_mb_100k.jsonl`)
are the training corpus; 2,000 of them carry a win-rate label measured against the
top-50 meta pool (24 battles each, BC policy both sides). The win-rate bin is a
conditioning token trained with classifier-free dropout; unlabelled teams always
carry the null token, so they train the unconditional distribution — a
semi-supervised D-CFG setup. At inference, ask for the top bin with guidance > 1.

Same MDLM objective, architecture, and constrained decoding as diffusion.py; the
only change is the conditioning token (win-rate bin instead of playstyle) and the
training corpus (100k HPS teams instead of 545 meta teams).

  train  : hpsdiffusion.py train --epochs 30
  sample : hpsdiffusion.py sample --bin 5 --guidance 2 --n 32
"""
import argparse, json, random
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from corpus import parse_team_text, norm, STATS
from encode import Vocab, Legality, NF, NSLOT
import diffusion as D

DEV = D.DEV
COLS = D.COLS
CKPT = "/tmp/vgc-pilot/hpsdiffusion.pt"
JSONL = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/teams/hps_reg_mb_100k.jsonl"
LABELS = "/tmp/vgc-pilot/hps_labels.json"
MANIFEST = "/tmp/vgc-pilot/hps_label_manifest.json"
# Bins are raw WINS out of 24 vs the top-50 pool, capped: {0} {1} {2} {>=3}.
# Chosen from the measured 2026-08-29 label distribution (1457/401/95/47) —
# win-RATE bins were useless because uniform-legal teams score mean 0.015,
# median 0.000 against the meta: nothing ever reached the 0.3 bin.
NBIN = 4
WNULL = NBIN                                    # null / dropped-condition token
TOPBIN = NBIN - 1

def wins_bin(wins):
    return min(int(wins), TOPBIN)

def load_hps():
    """All 100k teams, plus a wins bin per team (WNULL if unlabelled)."""
    lines = open(JSONL).readlines()
    teams = [parse_team_text(json.loads(l)["team"]) for l in lines]
    W = np.full(len(teams), WNULL, dtype=np.int64)
    try:
        man = json.load(open(MANIFEST)); lab = json.load(open(LABELS))
        for p, meta in man.items():
            if p in lab: W[meta["jsonl_line"]] = wins_bin(lab[p]["wins"])
        print(f"labels attached: {int((W != WNULL).sum())} of {len(teams)}")
    except FileNotFoundError:
        print("no labels found; training fully unconditional")
    return teams, W

class TeamDiffusionHPS(nn.Module):
    """diffusion.TeamDiffusion with a win-rate-bin token in place of playstyle."""
    def __init__(self, vocab, d=192, nhead=6, nlayer=4, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in vocab.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in vocab.sizes.items()})
        self.pos = nn.Parameter(torch.randn(COLS, d) * .02)
        self.wemb = nn.Embedding(NBIN + 1, d)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer)
        self.ln = nn.LayerNorm(d)
        self.keys = [vocab.key(c) for c in range(COLS)]
        # group columns by vocabulary so embeddings/heads run as 5 batched ops,
        # not 48 separate kernel launches (48 launches = 4x slower epochs on MPS)
        self.cols_by_key = {}
        for c, k in enumerate(self.keys):
            self.cols_by_key.setdefault(k, []).append(c)

    def forward(self, x, t, w):
        B, d = x.shape[0], self.pos.shape[1]
        h = torch.empty(B, COLS, d, device=x.device)
        for k, cols in self.cols_by_key.items():
            h[:, cols] = self.emb[k](x[:, cols])
        h = h + self.pos.unsqueeze(0)
        h = h + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        h = h + self.wemb(w).unsqueeze(1)
        return self.ln(self.tr(h))

    def logits(self, h, c):
        return self.head[self.keys[c]](h[:, c])

    def loss(self, x, w, p_uncond=0.15):
        B = x.shape[0]
        t = torch.rand(B, device=x.device)
        m = torch.rand(B, COLS, device=x.device) < t.view(-1, 1)
        empty = ~m.any(1)
        if empty.any():
            m[empty, torch.randint(0, COLS, (int(empty.sum()),), device=x.device)] = True
        ww = w.clone()
        ww[torch.rand(B, device=x.device) < p_uncond] = WNULL
        xin = x.clone(); xin[m] = 0
        h = self(xin, t, ww)
        wt_all = (1.0 / t.clamp(min=1e-3))
        tot, n = 0., 0
        for k, cols in self.cols_by_key.items():
            mk = m[:, cols]
            if not mk.any(): continue
            lg = self.head[k](h[:, cols])              # B x nc x vocab
            ce = F.cross_entropy(lg[mk], x[:, cols][mk], reduction="none")
            wt = wt_all.view(-1, 1).expand(-1, len(cols))[mk]
            tot = tot + (wt * ce).sum(); n += int(mk.sum())
        return tot / max(n, 1)

def build_vocab(teams):
    return Vocab(teams), Legality(teams)

def train(epochs=30, seed=0, batch=256, init_from=""):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    teams, W = load_hps()
    V, L = build_vocab(teams)
    X = np.stack([V.encode(t) for t in teams])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); nte = 2000
    tr, te = idx[nte:], idx[:nte]
    model = TeamDiffusionHPS(V).to(DEV)
    if init_from:
        sd = torch.load(init_from, map_location=DEV)["sd"]
        own = model.state_dict()
        drop = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
        for k in drop: del sd[k]
        model.load_state_dict(sd, strict=False)
        print(f"resumed weights from {init_from}" +
              (f" (reinitialised {drop} on shape change)" if drop else ""), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xtr = torch.tensor(X[tr], device=DEV); wtr = torch.tensor(W[tr], device=DEV)
    xte = torch.tensor(X[te], device=DEV); wte = torch.tensor(W[te], device=DEV)
    print(f"train {len(tr)} / held-out {len(te)}, vocab sizes {V.sizes}, "
          f"{sum(p.numel() for p in model.parameters()):,} params, {DEV}", flush=True)
    print("labelled bins in train:", np.bincount(W[tr], minlength=NBIN+1).tolist(), flush=True)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr.shape[0], device=DEV)
        tot = nb = 0
        for i in range(0, xtr.shape[0], batch):
            b = perm[i:i+batch]
            xb = xtr[b].view(-1, NSLOT, NF)
            perm_s = torch.rand(xb.shape[0], NSLOT, device=xb.device).argsort(1)
            xb = torch.gather(xb, 1, perm_s.unsqueeze(-1).expand(-1, -1, NF)).reshape(-1, COLS)
            loss = model.loss(xb, wtr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            tot += float(loss); nb += 1
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([float(model.loss(xte[:512], wte[:512])) for _ in range(2)]))
        print(f"  ep{ep+1:3d} train {tot/nb:8.3f}  held-out {vl:8.3f}", flush=True)
    torch.save({"sd": model.state_dict()}, CKPT)
    print("saved", CKPT)

def load_model(V):
    m = TeamDiffusionHPS(V).to(DEV)
    m.load_state_dict(torch.load(CKPT, map_location=DEV)["sd"]); m.eval()
    return m

@torch.no_grad()
def _step_logits(model, x, t, w, guidance):
    B = x.shape[0]
    tt = torch.full((B,), t, device=x.device)
    hc = model(x, tt, w)
    if guidance == 1.0 or (w == WNULL).all():
        return [model.logits(hc, c) for c in range(COLS)]
    hu = model(x, tt, torch.full_like(w, WNULL))
    out = []
    for c in range(COLS):
        lc = F.log_softmax(model.logits(hc, c), -1)
        lu = F.log_softmax(model.logits(hu, c), -1)
        out.append(guidance * lc + (1 - guidance) * lu)
    return out

@torch.no_grad()
def sample_constrained(model, C, n, wbin, guidance=1.0, temp=1.0, device=DEV):
    """Dependency-order constrained decode, conditioned on win-rate bin `wbin`
    (WNULL = unconditional)."""
    w = torch.full((n,), wbin, device=device, dtype=torch.long)
    x = torch.zeros(n, COLS, dtype=torch.long, device=device)
    seq = list(D.ORDER)
    for step, c in enumerate(seq):
        t_now = 1.0 - step / max(len(seq), 1)
        lg = _step_logits(model, x, t_now, w, guidance)[c]
        for b in range(n):
            ok = C.mask_for(c, x[b], device)
            l = lg[b].clone(); l[~ok] = -1e9
            p = torch.softmax(l / temp, -1)
            x[b, c] = torch.multinomial(p, 1).item()
    return x

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sample"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--init_from", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin", type=int, default=WNULL)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.epochs, a.seed, init_from=a.init_from)
    else:
        teams, _ = load_hps()
        V, L = build_vocab(teams)
        model = load_model(V)
        C = D.Constraints(V, L)
        x = sample_constrained(model, C, a.n, a.bin, a.guidance, a.temp).cpu().numpy()
        for i in range(a.n):
            print([V.decode_field(c, int(x[i][c])) for c in range(COLS)][:NF])

if __name__ == "__main__":
    cli()
