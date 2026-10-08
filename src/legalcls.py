"""A legality classifier for classifier guidance.

p(legal | partially masked team, t): trained on the same noised inputs the diffusion
model sees, so it can score candidate values mid-decoding (D-CBG, Schiff et al.
ICLR 2025). Positives are real/labelled teams; negatives are those teams with 1-5
fields corrupted, labelled by the rule oracle (learnsets, abilities, clauses,
mega pairing, Stat Point budget). Some corruptions stay legal — those are kept as
positives, which gives the classifier hard cases near the boundary.

Exposes value_logprobs(x, c, t) -> [n, |vocab_c|] = log p(legal | x with x_c := v),
which the sampler adds to the diffusion logits scaled by cls_scale.
"""
import argparse, random, sys
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from encode import NSLOT
import diffusion as D
import spreaddiffusion as S

DEV = D.DEV
CKPT = "/tmp/vgc-pilot/legalcls.pt"
CAT_COLS = [c for c in range(S.COLS2) if c % S.NF2 < 8]
STAT_COLS = [c for c in range(S.COLS2) if c % S.NF2 >= 8]

def is_legal(V, L, row):
    cat = [V.decode_field(c, int(row[c])) for c in CAT_COLS]
    if not L.legal(cat): return False
    g = np.asarray(row).reshape(NSLOT, S.NF2)[:, 8:]
    return bool(((g - 1).clip(min=0).sum(1) <= S.BUDGET).all())

def corrupt(row, V, rng, C=None):
    """Half the time: random field noise (label decided by the oracle).
    Half the time: a RULE-TARGETED edit whose label is known from the authoritative
    table — an illegal move/ability for that species, a duplicated species or item, an
    over-budget stat — or a legal move swap. This teaches the classifier the table
    itself, sampled over all cells rather than only the 16.5% the corpus shows."""
    r = row.copy()
    if C is None or rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        for c in rng.choice(CAT_COLS, k, replace=False):
            r[c] = int(rng.integers(1, V.sizes[V.key(int(c))]))
        if rng.random() < 0.3:
            c = int(rng.choice(STAT_COLS)); r[c] = int(rng.integers(20, S.NSTAT)) + 1
        return r
    slot = int(rng.integers(0, NSLOT)); b = slot * S.NF2; si = int(r[b])
    kind = rng.integers(0, 5)
    if kind == 0 and si in C.mv_ok:                       # illegal move for this species
        bad = (~C.mv_ok[si]).nonzero().flatten(); bad = bad[bad > 0]
        if len(bad): r[b + 3 + int(rng.integers(0, 4))] = int(bad[rng.integers(0, len(bad))])
    elif kind == 1 and si in C.ab_ok:                     # ability the species cannot have
        bad = (~C.ab_ok[si]).nonzero().flatten(); bad = bad[bad > 0]
        if len(bad): r[b + 1] = int(bad[rng.integers(0, len(bad))])
    elif kind == 2:                                       # duplicate species (Species Clause)
        other = int(rng.integers(0, NSLOT)); r[b] = r[other * S.NF2] if other != slot else r[b]
    elif kind == 3:                                       # duplicate item (Item Clause)
        other = int(rng.integers(0, NSLOT)); r[b + 2] = r[other * S.NF2 + 2] if other != slot else r[b + 2]
    else:                                                 # a LEGAL move swap (hard positive)
        if si in C.mv_ok:
            good = C.mv_ok[si].nonzero().flatten()
            present = set(int(r[b + 3 + j]) for j in range(4))
            good = [int(g) for g in good if int(g) not in present]
            if good: r[b + 3 + int(rng.integers(0, 4))] = good[rng.integers(0, len(good))]
    return r

class LegalClassifier(nn.Module):
    def __init__(self, vocab, d=128, nhead=4, nlayer=3):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in vocab.sizes.items()})
        self.pos = nn.Parameter(torch.randn(S.COLS2, d) * .02)
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer); self.ln = nn.LayerNorm(d); self.out = nn.Linear(2 * d, 1)
        self.keys = [vocab.key(c) for c in range(S.COLS2)]
    def forward(self, x, t):
        h = torch.stack([self.emb[self.keys[c]](x[:, c]) for c in range(S.COLS2)], 1)
        h = h + self.pos.unsqueeze(0) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        h = self.ln(self.tr(h))
        return self.out(torch.cat([h.mean(1), h.max(1).values], -1)).squeeze(-1)   # logit of legal
    @torch.no_grad()
    def value_logprobs(self, x, c, t_now, chunk=512):
        """For every sample and every candidate value v of column c: log p(legal | x, x_c := v)."""
        n = x.shape[0]; nv = self.vocab.sizes[self.keys[c]]
        out = torch.zeros(n, nv, device=x.device)
        for b in range(n):
            xb = x[b].unsqueeze(0).repeat(nv, 1); xb[:, c] = torch.arange(nv, device=x.device)
            tt = torch.full((nv,), t_now, device=x.device)
            lp = torch.cat([F.logsigmoid(self(xb[i:i+chunk], tt[i:i+chunk])) for i in range(0, nv, chunk)])
            out[b] = lp
        return out

def synth_rows(V, L, C, base_rows, n, rng):
    """Manufacture n labelled rows that ENUMERATE the rule tables.

    Each row starts from a real team, then one slot gets a targeted edit whose label
    is known from the authoritative tables: moves drawn uniformly over that species'
    legal set (positive) or with exactly one move from its illegal set (negative);
    an ability the species cannot have; a duplicated species or item; an over-budget
    stat. Cells are sampled uniformly, so the classifier sees the whole table instead
    of the ~16% of it the corpus reveals."""
    X, Y = [], []
    nv_move = V.sizes["move"]
    for _ in range(n):
        r = base_rows[rng.integers(0, len(base_rows))].copy()
        slot = int(rng.integers(0, NSLOT)); b = slot * S.NF2; si = int(r[b])
        kind = rng.integers(0, 8)
        if kind <= 3 and si in C.mv_ok:                    # moves: 2 legal-set rows, 2 illegal-move rows
            good = C.mv_ok[si].nonzero().flatten().tolist(); bad = (~C.mv_ok[si]).nonzero().flatten().tolist()
            bad = [v for v in bad if v > 0]
            if len(good) >= 4 and bad:
                pick = list(rng.choice(good, 4, replace=False))
                if kind >= 2: pick[int(rng.integers(0, 4))] = int(bad[rng.integers(0, len(bad))]); lab = 0
                else: lab = 1
                for j in range(4): r[b + 3 + j] = int(pick[j])
                X.append(r); Y.append(lab); continue
        if kind == 4 and si in C.ab_ok:                    # ability
            good = C.ab_ok[si].nonzero().flatten().tolist(); bad = [v for v in (~C.ab_ok[si]).nonzero().flatten().tolist() if v > 0]
            if good and bad:
                if rng.random() < 0.5: r[b + 1] = int(good[rng.integers(0, len(good))]); lab = 1
                else: r[b + 1] = int(bad[rng.integers(0, len(bad))]); lab = 0
                X.append(r); Y.append(lab); continue
        if kind == 5:                                      # duplicate species
            o = int(rng.integers(0, NSLOT))
            if o != slot: r[b] = r[o * S.NF2]; X.append(r); Y.append(0); continue
        if kind == 6:                                      # duplicate item
            o = int(rng.integers(0, NSLOT))
            if o != slot and r[o * S.NF2 + 2] != 0: r[b + 2] = r[o * S.NF2 + 2]; X.append(r); Y.append(0); continue
        c = int(rng.choice(STAT_COLS)); r[c] = int(rng.integers(24, S.NSTAT)) + 1   # over-budget stat
        X.append(r); Y.append(int(is_legal(V, L, r)))
    return np.stack(X), np.array(Y)

def train(epochs=40, seed=0, n_rows=200_000, device=None, max_steps=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = device or DEV
    V, L, corpus, teams, wrs = S.build(); C = S.SpreadConstraints(V, L)
    rng = np.random.default_rng(seed)
    base = np.stack([V.encode(t) for t in corpus + teams])
    Xs, Ys = synth_rows(V, L, C, base, n_rows, rng)
    X = np.concatenate([base, Xs]); Yl = np.concatenate([np.ones(len(base), dtype=int), Ys])
    print(f"classifier data: {len(X):,} rows, {Yl.mean():.1%} legal · device {dev}", flush=True)
    idx = rng.permutation(len(X)); nte = min(4000, int(len(X) * .05)); tr, te = idx[nte:], idx[:nte]
    m = LegalClassifier(V).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=.01)
    xtr, ytr = torch.tensor(X[tr], device=dev), torch.tensor(Yl[tr], device=dev, dtype=torch.float)
    xte, yte = torch.tensor(X[te], device=dev), torch.tensor(Yl[te], device=dev, dtype=torch.float)
    steps_per_epoch = (len(tr) + 255) // 256; total = epochs * steps_per_epoch if max_steps is None else max_steps
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    def noised(xb):
        t = torch.rand(xb.shape[0], device=dev) ** 3; mk = torch.rand_like(xb, dtype=torch.float) < t.view(-1, 1)
        xn = xb.clone(); xn[mk] = 0; return xn, t
    step = 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(xtr.shape[0], device=dev); tot = nb = 0
        for i in range(0, xtr.shape[0], 256):
            b = perm[i:i+256]; xn, t = noised(xtr[b])
            loss = F.binary_cross_entropy_with_logits(m(xn, t), ytr[b]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            tot += float(loss); nb += 1; step += 1
            if step % 250 == 0 or step == total:
                m.eval()
                with torch.no_grad():
                    accs = []
                    for tval in (0.0, 0.3):
                        tt = torch.full((xte.shape[0],), tval, device=dev); mk = torch.rand_like(xte, dtype=torch.float) < tval
                        xn2 = xte.clone(); xn2[mk] = 0; accs.append(float(((m(xn2, tt) > 0).float() == yte).float().mean()))
                print(f"  step {step:5d}  loss {tot/nb:.3f}  held-out acc @t=0 {accs[0]:.3f}  @0.3 {accs[1]:.3f}", flush=True); m.train()
            if step >= total: break
        if step >= total: break
    torch.save({"sd": m.state_dict()}, CKPT); print("saved", CKPT, flush=True)

def load(V):
    m = LegalClassifier(V).to(DEV); m.load_state_dict(torch.load(CKPT, map_location=DEV)["sd"]); m.eval(); return m

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--device", default=None); ap.add_argument("--max-steps", type=int, default=None); a = ap.parse_args()
    train(a.epochs, n_rows=a.rows, device=a.device, max_steps=a.max_steps)
