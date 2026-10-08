"""Masked diffusion over the FULL team, Stat Points included, conditioned on
playstyle and win-rate bin.

Each candidate is 14 fields: species, ability, item, 4 moves, nature, and the six
Stat Point values (HP/Atk/Def/SpA/SpD/Spe, each 0..32, summing to at most 66).
Earlier models borrowed the spread from a corpus slot of the same species; this
one generates it, so a novel spread is reachable.

Constrained decoding enforces every rule during sampling: species/item clauses,
learnsets (authoritative table), ability ownership, mega-stone pairing, and the
Stat Point budget (a value that would push the slot over 66 is masked out).
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from corpus import parse_team, load_corpus, norm, STATS, dex_entry, MEGA_STONE_OF
from encode import Vocab, Legality, canon, NSLOT
import diffusion as D
from wrdiffusion import load_labelled, wr_bin, NBIN, WNULL, EDGES

DEV = D.DEV
NF2 = 14; COLS2 = NSLOT * NF2
NSTAT = 33                       # values 0..32 -> tokens 1..33; 0 is [MASK]
BUDGET = 66
CKPT = "/tmp/vgc-pilot/spreaddiffusion.pt"

class SpreadVocab:
    def __init__(self, teams):
        self.base = Vocab(teams)
        self.itos = dict(self.base.itos); self.stoi = dict(self.base.stoi)
        self.itos["stat"] = ["[MASK]"] + [str(v) for v in range(NSTAT)]
        self.stoi["stat"] = {v: i for i, v in enumerate(self.itos["stat"])}
        self.sizes = {k: len(v) for k, v in self.itos.items()}
    def key(self, col):
        j = col % NF2
        return "stat" if j >= 8 else self.base.key(j)
    def encode(self, team):
        out = []
        for s in canon(team):
            ms = sorted(s["moves"], key=norm) + [""] * (4 - len(s["moves"]))
            vals = [norm(s["species"]), norm(s["ability"]), norm(s["item"]),
                    norm(ms[0]), norm(ms[1]), norm(ms[2]), norm(ms[3]), norm(s["nature"])]
            row = [self.stoi[self.base.key(j)].get(v, 0) for j, v in enumerate(vals)]
            row += [min(32, max(0, int(s["evs"].get(st, 0)))) + 1 for st in STATS]
            out += row
        return np.array(out, dtype=np.int64)
    def decode_field(self, col, idx):
        return self.itos[self.key(col)][idx]

class SpreadDiffusion(nn.Module):
    def __init__(self, vocab, d=192, nhead=6, nlayer=4, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in vocab.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in vocab.sizes.items()})
        self.pos = nn.Parameter(torch.randn(COLS2, d) * .02)
        self.style = nn.Embedding(D.NULL + 1, d); self.wr = nn.Embedding(NBIN + 1, d)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer); self.ln = nn.LayerNorm(d)
        self.keys = [vocab.key(c) for c in range(COLS2)]
    def forward(self, x, t, y, w):
        h = torch.stack([self.emb[self.keys[c]](x[:, c]) for c in range(COLS2)], 1)
        h = h + self.pos.unsqueeze(0) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        h = h + self.style(y).unsqueeze(1) + self.wr(w).unsqueeze(1)
        return self.ln(self.tr(h))
    def logits(self, h, c): return self.head[self.keys[c]](h[:, c])
    def loss(self, x, y, w, p_uncond=0.15):
        B = x.shape[0]; t = torch.rand(B, device=x.device)
        m = torch.rand(B, COLS2, device=x.device) < t.view(-1, 1)
        empty = ~m.any(1)
        if empty.any(): m[empty, torch.randint(0, COLS2, (int(empty.sum()),), device=x.device)] = True
        yy = y.clone(); yy[torch.rand(B, device=x.device) < p_uncond] = D.NULL
        ww = w.clone(); ww[torch.rand(B, device=x.device) < p_uncond] = WNULL
        xin = x.clone(); xin[m] = 0
        h = self(xin, t, yy, ww); tot, n = 0., 0
        for c in range(COLS2):
            mc = m[:, c]
            if mc.any():
                wgt = 1.0 / t[mc].clamp(min=1e-3)
                ce = F.cross_entropy(self.logits(h, c)[mc], x[mc, c], reduction="none")
                tot = tot + (wgt * ce).sum(); n += int(mc.sum())
        return tot / max(n, 1)

class SpreadConstraints:
    """Rule masks for the 14-field grid, including the Stat Point budget."""
    def __init__(self, V, L):
        self.V, self.L = V, L
        self.species_vals = V.itos["species"]; self.move_vals = V.itos["move"]
        self.item_vals = V.itos["item"]; self.abil_vals = V.itos["ability"]
        self.mv_ok, self.ab_ok, self.base_of = {}, {}, {}
        for si, sp in enumerate(self.species_vals):
            e = dex_entry(sp) if si else None
            self.base_of[si] = norm(e.get("baseSpecies", e.get("name", sp))) if e else sp
            if si == 0: continue
            lm = L.moves_for(sp); la = L.abils_for(sp)
            self.mv_ok[si] = torch.tensor([j > 0 and m in lm for j, m in enumerate(self.move_vals)])
            self.ab_ok[si] = torch.tensor([j > 0 and a in la for j, a in enumerate(self.abil_vals)])
    def mask_for(self, col, row, device):
        k = self.V.key(col); slot = col // NF2; j = col % NF2; n = self.V.sizes[k]
        ok = torch.ones(n, dtype=torch.bool); ok[0] = False
        if j >= 8:                                           # Stat Point budget
            spent = sum(max(0, int(row[slot*NF2 + jj]) - 1) for jj in range(8, 14) if jj != j and int(row[slot*NF2 + jj]) != 0)
            cap = max(0, min(32, BUDGET - spent))
            ok[cap + 2:] = False                             # tokens above value `cap`
            return ok.to(device)
        if j == 0:
            used = {self.base_of[int(row[s2*NF2])] for s2 in range(NSLOT) if s2 != slot and int(row[s2*NF2]) != 0}
            if used:
                for vi in range(1, n):
                    if self.base_of[vi] in used: ok[vi] = False
        elif j == 2:
            for s2 in range(NSLOT):
                if s2 != slot and int(row[s2*NF2+2]) != 0: ok[int(row[s2*NF2+2])] = False
            si = int(row[slot*NF2])
            if si != 0:
                base = self.base_of[si]; is_mega = "mega" in self.species_vals[si]
                for vi in range(1, n):
                    st = self.item_vals[vi]
                    if st in MEGA_STONE_OF and (is_mega or MEGA_STONE_OF[st] != base): ok[vi] = False
        elif j == 1:
            si = int(row[slot*NF2])
            if si != 0 and si in self.ab_ok: ok &= self.ab_ok[si]
        elif 3 <= j <= 6:
            si = int(row[slot*NF2])
            if si != 0 and si in self.mv_ok: ok &= self.mv_ok[si]
            for j2 in range(3, 7):
                if j2 != j and int(row[slot*NF2+j2]) != 0: ok[int(row[slot*NF2+j2])] = False
        if not ok.any():
            if 3 <= j <= 6:
                si = int(row[slot*NF2])
                if si != 0 and si in self.mv_ok: ok = self.mv_ok[si].clone()
            if not ok.any(): ok[0] = True
        return ok.to(device)

ORDER2 = ([c for c in range(COLS2) if c % NF2 == 0] + [c for c in range(COLS2) if c % NF2 in (1, 2)] +
          [c for c in range(COLS2) if 3 <= c % NF2 <= 6] + [c for c in range(COLS2) if c % NF2 == 7] +
          [c for c in range(COLS2) if c % NF2 >= 8])

def build():
    corpus, _ = load_corpus(); corpus = [t for t in corpus if len(t) == 6]
    teams, wrs = load_labelled()
    V = SpreadVocab(corpus + teams); L = Legality(corpus)
    return V, L, corpus, teams, wrs

def train(epochs=200, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    V, L, corpus, teams, wrs = build()
    X = np.stack([V.encode(t) for t in teams]); Y = np.array([D.style_of(t) for t in teams]); W = np.array([wr_bin(w) for w in wrs])
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nte = int(len(X) * .1); tr, te = idx[nte:], idx[:nte]
    m = SpreadDiffusion(V).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xtr, ytr, wtr = (torch.tensor(a[tr], device=DEV) for a in (X, Y, W)); xte, yte, wte = (torch.tensor(a[te], device=DEV) for a in (X, Y, W))
    print(f"train {len(tr)} / held-out {len(te)} · {COLS2} columns · {sum(p.numel() for p in m.parameters()):,} params · {DEV}", flush=True)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(xtr.shape[0], device=DEV); tot = nb = 0
        for i in range(0, xtr.shape[0], 64):
            b = perm[i:i+64]; xb = xtr[b].view(-1, NSLOT, NF2)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, COLS2)
            loss = m.loss(xb, ytr[b], wtr[b]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step(); tot += float(loss); nb += 1
        sch.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            m.eval()
            with torch.no_grad(): vl = float(np.mean([float(m.loss(xte, yte, wte)) for _ in range(6)]))
            print(f"  ep{ep+1:4d} train {tot/nb:7.2f}  held-out {vl:7.2f}", flush=True)
            torch.save({"sd": m.state_dict(), "itos": V.itos}, CKPT)
    torch.save({"sd": m.state_dict(), "itos": V.itos}, CKPT); print("saved", CKPT, flush=True)

def load_model():
    V, L, corpus, teams, wrs = build()
    ck = torch.load(CKPT, map_location=DEV); assert ck["itos"]["species"] == V.itos["species"]
    m = SpreadDiffusion(V).to(DEV); m.load_state_dict(ck["sd"]); m.eval(); return m, V, L, corpus, teams

@torch.no_grad()
def sample(model, C, n, wbin, style="none", guidance=1.0, temp=1.0, constrained=True,
           cls=None, cls_scale=0.0, device=DEV):
    """Constrained decoding with CFG on (style, win-rate); optional legality-classifier
    guidance (D-CBG): add cls_scale * log p(legal | x with column c := v) to each value's logit."""
    y = torch.full((n,), D.STYLES.index(style) if style in D.STYLES else D.NULL, device=device, dtype=torch.long)
    w = torch.full((n,), wbin, device=device, dtype=torch.long); yN = torch.full_like(y, D.NULL); wN = torch.full_like(w, WNULL)
    x = torch.zeros(n, COLS2, dtype=torch.long, device=device)
    seq = list(ORDER2)
    for step, c in enumerate(seq):
        t_now = 1.0 - step / max(len(seq), 1); tt = torch.full((n,), t_now, device=device)
        lc = F.log_softmax(model.logits(model(x, tt, y, w), c), -1)
        if guidance != 1.0:
            lu = F.log_softmax(model.logits(model(x, tt, yN, wN), c), -1); lc = guidance * lc + (1 - guidance) * lu
        if cls is not None and cls_scale > 0:
            lc = lc + cls_scale * cls.value_logprobs(x, c, t_now)     # [n, vocab] log p(legal | x, x_c := v)
        for b in range(n):
            l = lc[b].clone()
            if constrained: ok = C.mask_for(c, x[b], device); l[~ok] = -1e9
            else: l[0] = -1e9
            x[b, c] = torch.multinomial(torch.softmax(l / temp, -1), 1).item()
    return x

def row_to_slots(V, row, look):
    out = []
    for i in range(NSLOT):
        b = i * NF2; g = lambda j: V.decode_field(b + j, int(row[b + j]))
        mv = [look.get(g(3+j), g(3+j)) for j in range(4) if g(3+j) and g(3+j) != "[MASK]"]
        evs = {st: (int(row[b + 8 + k]) - 1 if int(row[b + 8 + k]) > 0 else 0) for k, st in enumerate(STATS)}
        out.append(dict(species=look.get(g(0), g(0)), item=look.get(g(2), g(2)), ability=look.get(g(1), g(1)),
                        nature=look.get(g(7), g(7)), moves=mv, evs=evs))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "smoke"]); ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "train": train(a.epochs)
    else:
        V, L, corpus, teams, wrs = build()
        X = np.stack([V.encode(t) for t in teams[:50]])
        print("grid", X.shape, "| stat tokens range", X[:, [c for c in range(COLS2) if c % NF2 >= 8]].min(), X[:, [c for c in range(COLS2) if c % NF2 >= 8]].max())
        sums = [(X[i].reshape(NSLOT, NF2)[:, 8:] - 1).sum(1) for i in range(5)]
        print("per-slot Stat Point sums (first 5 teams):", [s.tolist() for s in sums])
        C = SpreadConstraints(V, L); row = torch.zeros(COLS2, dtype=torch.long); row[8:13] = torch.tensor([33, 33, 1, 1, 1])  # 32+32+0+0+0 spent
        ok = C.mask_for(13, row, "cpu"); print("budget mask: with 64 spent, max allowed value =", int(ok.nonzero().max()) - 1)
