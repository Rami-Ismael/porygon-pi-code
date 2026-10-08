"""The complete search unit the research goal specified.

Earlier models covered 8 categorical fields per candidate (species, ability, item,
4 moves, nature) and BORROWED Stat Points from the corpus. That was a real gap: the
goal asks to "sample full VGC teams (species, ability, item, moves, Stat Points)".

Here the spread is a 9th field: one categorical over the distinct spreads observed
in the corpus. Two reasons this is the right encoding rather than six integers:
  - every value is a real, legal spread, so the 66-point budget and the 32-per-stat
    cap hold by construction instead of needing a repair step;
  - 4,116 of 4,152 corpus candidates spend exactly 66, so the budget is not a free
    variable in practice - what varies is the allocation, which is what a categorical
    over observed allocations captures.
Any spread is legal on any candidate, so no cross-field constraint is needed.

Tera type is deliberately absent: the `champions` mod deletes teraType, so it is not
a decision variable in this regulation.
"""
import argparse, json, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from corpus import load_corpus, norm, STATS, dex_entry, MEGA_STONE_OF
from encode import Vocab as BaseVocab, Legality, canon, NSLOT
import diffusion as D

NF2 = 9                       # 8 categorical + 1 spread
COLS2 = NSLOT * NF2
DEV = D.DEV
CKPT2 = "/tmp/vgc-pilot/fullteam.pt"

def spread_key(evs):
    return "|".join(f"{s}:{evs.get(s,0)}" for s in STATS)

def key_to_evs(k):
    return {p.split(":")[0]: int(p.split(":")[1]) for p in k.split("|")}

class FullVocab:
    """8 categorical field vocabularies (reused) plus a spread vocabulary."""
    def __init__(self, teams):
        self.base = BaseVocab(teams)
        spreads = sorted({spread_key(s["evs"]) for t in teams for s in t})
        self.itos = dict(self.base.itos)
        self.itos["spread"] = ["[MASK]"] + spreads
        self.stoi = dict(self.base.stoi)
        self.stoi["spread"] = {v: i for i, v in enumerate(self.itos["spread"])}
        self.sizes = {k: len(v) for k, v in self.itos.items()}
    def key(self, col):
        j = col % NF2
        if j == 8: return "spread"
        return self.base.key(j)          # base indexes by field within a slot
    def encode(self, team):
        out = []
        for s in canon(team):
            ms = sorted(s["moves"], key=norm) + [""] * (4 - len(s["moves"]))
            vals = [norm(s["species"]), norm(s["ability"]), norm(s["item"]),
                    norm(ms[0]), norm(ms[1]), norm(ms[2]), norm(ms[3]), norm(s["nature"])]
            row = [self.stoi[self.base.key(j)].get(v, 0) for j, v in enumerate(vals)]
            row.append(self.stoi["spread"].get(spread_key(s["evs"]), 0))
            out += row
        return np.array(out, dtype=np.int64)
    def decode_field(self, col, idx):
        return self.itos[self.key(col)][idx]

class FullTeamDiffusion(nn.Module):
    def __init__(self, vocab, d=192, nhead=6, nlayer=4, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in vocab.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in vocab.sizes.items()})
        self.pos = nn.Parameter(torch.randn(COLS2, d) * .02)
        self.style = nn.Embedding(D.NULL + 1, d)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4*d, dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer)
        self.ln = nn.LayerNorm(d)
        self.keys = [vocab.key(c) for c in range(COLS2)]
    def forward(self, x, t, y):
        h = torch.stack([self.emb[self.keys[c]](x[:, c]) for c in range(COLS2)], 1)
        h = h + self.pos.unsqueeze(0) + self.tproj(t.view(-1,1).float()).unsqueeze(1) \
              + self.style(y).unsqueeze(1)
        return self.ln(self.tr(h))
    def logits(self, h, c):
        return self.head[self.keys[c]](h[:, c])
    def loss(self, x, y, p_uncond=0.15):
        B = x.shape[0]
        t = torch.rand(B, device=x.device)
        m = torch.rand(B, COLS2, device=x.device) < t.view(-1,1)
        empty = ~m.any(1)
        if empty.any():
            m[empty, torch.randint(0, COLS2, (int(empty.sum()),), device=x.device)] = True
        yy = y.clone(); yy[torch.rand(B, device=x.device) < p_uncond] = D.NULL
        xin = x.clone(); xin[m] = 0
        h = self(xin, t, yy)
        tot = n = 0
        for c in range(COLS2):
            mc = m[:, c]
            if mc.any():
                w = 1.0 / t[mc].clamp(min=1e-3)
                ce = F.cross_entropy(self.logits(h, c)[mc], x[mc, c], reduction="none")
                tot = tot + (w*ce).sum(); n += int(mc.sum())
        return tot / max(n, 1)

class FullConstraints(D.Constraints):
    """Same rules, re-indexed for 9 fields per slot. The spread column is
    unconstrained: every value in its vocabulary is a legal spread."""
    def __init__(self, V, L):
        self.V, self.L = V, L
        self.species_vals = V.itos["species"]; self.move_vals = V.itos["move"]
        self.item_vals = V.itos["item"];       self.abil_vals = V.itos["ability"]
        self.mv_ok, self.ab_ok, self.base_of = {}, {}, {}
        for si, sp in enumerate(self.species_vals):
            e = dex_entry(sp) if si else None
            self.base_of[si] = norm(e.get("baseSpecies", e.get("name", sp))) if e else sp
            if si == 0: continue
            lm = L.moves_for(sp); la = L.abils_for(sp)
            self.mv_ok[si] = torch.tensor([j > 0 and m in lm for j, m in enumerate(self.move_vals)])
            self.ab_ok[si] = torch.tensor([j > 0 and a in la for j, a in enumerate(self.abil_vals)])
    def mask_for(self, col, row, device):
        k = self.V.key(col); slot = col // NF2; j = col % NF2
        n = self.V.sizes[k]
        ok = torch.ones(n, dtype=torch.bool); ok[0] = False
        if j == 8:                                   # spread: always legal
            return ok.to(device)
        if j == 0:
            used = {self.base_of[int(row[s2*NF2])] for s2 in range(NSLOT)
                    if s2 != slot and int(row[s2*NF2]) != 0}
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
                    if st in MEGA_STONE_OF and (is_mega or MEGA_STONE_OF[st] != base):
                        ok[vi] = False
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

ORDER2 = ([c for c in range(COLS2) if c % NF2 == 0] +
          [c for c in range(COLS2) if c % NF2 in (1, 2)] +
          [c for c in range(COLS2) if 3 <= c % NF2 <= 6] +
          [c for c in range(COLS2) if c % NF2 in (7, 8)])

@torch.no_grad()
def sample_full(model, C, n, style="none", guidance=1.0, temp=1.0, pin=None, device=DEV):
    y = torch.full((n,), D.STYLES.index(style) if style in D.STYLES else D.NULL,
                   device=device, dtype=torch.long)
    x = torch.zeros(n, COLS2, dtype=torch.long, device=device)
    todo = set(range(COLS2))
    if pin:
        for c, v in pin.items(): x[:, c] = v; todo.discard(c)
    seq = [c for c in ORDER2 if c in todo]
    for step, c in enumerate(seq):
        t_now = 1.0 - step / max(len(seq), 1)
        tt = torch.full((n,), t_now, device=device)
        hc = model(x, tt, y)
        lg = model.logits(hc, c)
        if guidance != 1.0:
            hu = model(x, tt, torch.full_like(y, D.NULL))
            lg = guidance*F.log_softmax(lg, -1) + (1-guidance)*F.log_softmax(model.logits(hu, c), -1)
        for b in range(n):
            ok = C.mask_for(c, x[b], device)
            l = lg[b].clone(); l[~ok] = -1e9
            p = torch.softmax(l/temp, -1)
            x[b, c] = torch.multinomial(p, 1).item()
    return x

def row_to_paste(V, row):
    from propose import slot_to_text
    slots = []
    for i in range(NSLOT):
        b = i*NF2
        g = lambda j: V.decode_field(b+j, int(row[b+j]))
        mv = [g(3+j) for j in range(4) if g(3+j) and g(3+j) != "[MASK]"]
        sk = g(8)
        slots.append(dict(species=g(0), item=g(2), ability=g(1), nature=g(7),
                          moves=mv, evs=key_to_evs(sk) if sk != "[MASK]" else {s:0 for s in STATS}))
    return slots

def train_full(epochs=800, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
    V = FullVocab(teams); L = Legality(teams)
    X = np.stack([V.encode(t) for t in teams])
    Y = np.array([D.style_of(t) for t in teams])
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X)); nte = int(len(X)*.15)
    Xtr, Ytr, Xte, Yte = X[idx[nte:]], Y[idx[nte:]], X[idx[:nte]], Y[idx[:nte]]
    m = FullTeamDiffusion(V).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xtr = torch.tensor(Xtr, device=DEV); ytr = torch.tensor(Ytr, device=DEV)
    xte = torch.tensor(Xte, device=DEV); yte = torch.tensor(Yte, device=DEV)
    print(f"train {len(Xtr)} / held-out {len(Xte)}, {sum(p.numel() for p in m.parameters()):,} params")
    for ep in range(epochs):
        m.train(); perm = torch.randperm(xtr.shape[0], device=DEV); tot = nb = 0
        for i in range(0, xtr.shape[0], 64):
            b = perm[i:i+64]
            xb = xtr[b].view(-1, NSLOT, NF2)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, COLS2)
            loss = m.loss(xb, ytr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.); opt.step(); tot += float(loss); nb += 1
        sch.step()
        if (ep+1) % 200 == 0 or ep == 0:
            m.eval()
            with torch.no_grad(): vl = float(np.mean([float(m.loss(xte, yte)) for _ in range(6)]))
            print(f"  ep{ep+1:4d} train {tot/nb:8.2f} held-out {vl:8.2f}", flush=True)
    torch.save({"sd": m.state_dict()}, CKPT2); print("saved", CKPT2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_full(int(sys.argv[2]) if len(sys.argv) > 2 else 800)
