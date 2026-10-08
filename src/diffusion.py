"""A discrete (masked) diffusion model over VGC teams — the thing the research goal
proposed, built on the Champions VGC 2026 Reg M-B corpus.

This is masked diffusion in the MDLM sense (Sahoo et al., NeurIPS 2024): the forward
process masks each field independently with probability 1 - alpha_t; the reverse
process progressively unmasks. Unmasked fields are carried over unchanged, which is
what makes hard slot-pinning free.

Two properties from the research goal are implemented and testable here:
  (1) LOCAL MOVE   - noise a real team partway (mask k fields) and denoise it back,
                     the discrete analogue of SDEdit. See `edit`.
  (2) GUIDANCE     - classifier-free guidance on a rule-derived playstyle label,
                     using the D-CFG form of Schiff et al. (ICLR 2025):
                     p_cond^gamma * p_uncond^(1-gamma), renormalised per field.
Hard constraints ("must contain Garchomp") are NOT guidance: pin the slot and the
carry-over property preserves it for free.

Usage:
  python diffusion.py train
  python diffusion.py sample --style trickroom --guidance 3 --n 8
  python diffusion.py sample --pin Garchomp --n 8
  python diffusion.py edit --k 8 --n 8
  python diffusion.py evaltest
"""
import argparse, json, math, random
from collections import Counter
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from corpus import load_corpus, norm, STATS
from encode import Vocab, Legality, team_fields, canon, FIELDS, NF, NSLOT

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
COLS = NSLOT * NF
CKPT = "/tmp/vgc-pilot/diffusion.pt"

# ---- playstyle labels, derived by rule (no hand labelling) ------------------
def has_move(t, m): return any(norm(m) == norm(x) for s in t for x in s["moves"])
def has_abil(t, a): return any(norm(a) == norm(s["ability"]) for s in t)
def has_item(t, i): return any(norm(i) == norm(s["item"]) for s in t)
RULES = {
    "trickroom": lambda t: has_move(t, "Trick Room"),
    "rain":      lambda t: has_abil(t, "Drizzle") or has_item(t, "Damp Rock"),
    "sun":       lambda t: has_abil(t, "Drought") or has_item(t, "Heat Rock"),
    "sand":      lambda t: has_abil(t, "Sand Stream") or has_item(t, "Smooth Rock"),
    "snow":      lambda t: has_abil(t, "Snow Warning") or has_item(t, "Icy Rock"),
    "tailwind":  lambda t: has_move(t, "Tailwind"),
}
STYLES = ["none"] + list(RULES)            # index 0 == "none"
NULL = len(STYLES)                          # the dropped-conditioning token

def style_of(t):
    for i, k in enumerate(RULES, start=1):
        if RULES[k](t): return i
    return 0

class TeamDiffusion(nn.Module):
    """Predicts the clean value of every field from a partially masked team."""
    def __init__(self, vocab, d=192, nhead=6, nlayer=4, dropout=0.1):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in vocab.sizes.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in vocab.sizes.items()})
        self.pos = nn.Parameter(torch.randn(COLS, d) * .02)
        self.style = nn.Embedding(NULL + 1, d)       # +1 for the null token
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer)
        self.ln = nn.LayerNorm(d)
        self.keys = [vocab.key(c) for c in range(COLS)]

    def forward(self, x, t, y):
        h = torch.stack([self.emb[self.keys[c]](x[:, c]) for c in range(COLS)], 1)
        h = h + self.pos.unsqueeze(0)
        h = h + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        h = h + self.style(y).unsqueeze(1)
        return self.ln(self.tr(h))

    def logits(self, h, c):
        return self.head[self.keys[c]](h[:, c])

    def loss(self, x, y, p_uncond=0.15):
        B = x.shape[0]
        t = torch.rand(B, device=x.device)                 # alpha_t = 1 - t
        m = torch.rand(B, COLS, device=x.device) < t.view(-1, 1)
        empty = ~m.any(1)
        if empty.any():
            m[empty, torch.randint(0, COLS, (int(empty.sum()),), device=x.device)] = True
        yy = y.clone()
        drop = torch.rand(B, device=x.device) < p_uncond   # classifier-free dropout
        yy[drop] = NULL
        xin = x.clone(); xin[m] = 0                        # 0 == [MASK]
        h = self(xin, t, yy)
        tot, n = 0., 0
        for c in range(COLS):
            mc = m[:, c]
            if mc.any():
                # MDLM weighting: divide by the masking rate t
                w = (1.0 / t[mc].clamp(min=1e-3))
                ce = F.cross_entropy(self.logits(h, c)[mc], x[mc, c], reduction="none")
                tot = tot + (w * ce).sum(); n += int(mc.sum())
        return tot / max(n, 1)

@torch.no_grad()
def _step_logits(model, x, t, y, guidance):
    """D-CFG: gamma on the conditional, (1-gamma) on the unconditional, per field."""
    B = x.shape[0]
    tt = torch.full((B,), t, device=x.device)
    hc = model(x, tt, y)
    if guidance == 1.0 or (y == NULL).all():
        return [model.logits(hc, c) for c in range(COLS)]
    hu = model(x, tt, torch.full_like(y, NULL))
    out = []
    for c in range(COLS):
        lc = F.log_softmax(model.logits(hc, c), -1)
        lu = F.log_softmax(model.logits(hu, c), -1)
        out.append(guidance * lc + (1 - guidance) * lu)     # renormalised by softmax later
    return out

@torch.no_grad()
def sample(model, n, style="none", guidance=1.0, steps=48, pin=None, V=None,
           x0=None, mask=None, temp=1.0, device=DEV):
    """Ancestral masked-diffusion sampling.

    pin  : dict {column index: value index} held fixed (carry-over unmasking)
    x0   : optional starting team; mask says which columns to regenerate (the local move)
    """
    y = torch.full((n,), STYLES.index(style) if style in STYLES else NULL,
                   device=device, dtype=torch.long)
    if x0 is None:
        x = torch.zeros(n, COLS, dtype=torch.long, device=device)
        masked = torch.ones(n, COLS, dtype=torch.bool, device=device)
    else:
        x = x0.clone().to(device)
        masked = mask.clone().to(device)
        x[masked] = 0
    if pin:
        for c, v in pin.items():
            x[:, c] = v; masked[:, c] = False
    for i in range(steps):
        t_now = 1.0 - i / steps
        t_next = 1.0 - (i + 1) / steps
        if not masked.any(): break
        lg = _step_logits(model, x, t_now, y, guidance)
        # probability a masked field is revealed at this step
        p_rev = 0.0 if t_now <= 0 else (t_now - t_next) / t_now
        for c in range(COLS):
            mc = masked[:, c]
            if not mc.any(): continue
            reveal = mc & (torch.rand(n, device=device) < p_rev)
            if i == steps - 1: reveal = mc            # reveal everything on the last step
            if not reveal.any(): continue
            p = torch.softmax(lg[c][reveal] / temp, -1)
            p[:, 0] = 0                               # never emit [MASK]
            p = p / p.sum(-1, keepdim=True)
            x[reveal, c] = torch.multinomial(p, 1).squeeze(-1)
            masked[:, c] = mc & ~reveal
    return x

def load_all():
    teams, names = load_corpus()
    keep = [(t, n) for t, n in zip(teams, names) if len(t) == 6]
    teams = [t for t, _ in keep]
    V = Vocab(teams); L = Legality(teams)
    X = np.stack([V.encode(t) for t in teams])
    Y = np.array([style_of(t) for t in teams])
    return teams, V, L, X, Y

def train(epochs=800, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    teams, V, L, X, Y = load_all()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); nte = int(len(X) * .15)
    Xtr, Ytr = X[idx[nte:]], Y[idx[nte:]]
    Xte, Yte = X[idx[:nte]], Y[idx[:nte]]
    model = TeamDiffusion(V).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xtr = torch.tensor(Xtr, device=DEV); ytr = torch.tensor(Ytr, device=DEV)
    xte = torch.tensor(Xte, device=DEV); yte = torch.tensor(Yte, device=DEV)
    print(f"train {len(Xtr)} / held-out {len(Xte)} teams, {sum(p.numel() for p in model.parameters()):,} params, {DEV}")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(xtr.shape[0], device=DEV)
        tot = nb = 0
        for i in range(0, xtr.shape[0], 64):
            b = perm[i:i+64]
            xb = xtr[b].view(-1, NSLOT, NF)
            xb = torch.stack([r[torch.randperm(NSLOT)] for r in xb]).view(-1, COLS)
            loss = model.loss(xb, ytr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            tot += float(loss); nb += 1
        sch.step()
        if (ep + 1) % 100 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                vl = float(np.mean([float(model.loss(xte, yte)) for _ in range(6)]))
            print(f"  ep{ep+1:4d} train {tot/nb:8.2f}  held-out {vl:8.2f}")
    torch.save({"sd": model.state_dict()}, CKPT)
    print("saved", CKPT)

def to_text(V, row, teams):
    """Render a sampled grid as a readable team (species/item/ability/moves)."""
    look = {}
    for t in teams:
        for s in t:
            look[norm(s["species"])] = s["species"]; look[norm(s["ability"])] = s["ability"]
            if s["item"]: look[norm(s["item"])] = s["item"]
            for m in s["moves"]: look[norm(m)] = m
    out = []
    for i in range(NSLOT):
        b = i * NF
        g = lambda j: look.get(V.decode_field(b+j, int(row[b+j])), V.decode_field(b+j, int(row[b+j])))
        mv = [g(3+j) for j in range(4) if g(3+j)]
        out.append(f"{g(0)} @ {g(2)} | {g(1)} | {', '.join(mv)}")
    return out

# =====================================================================
# Constrained decoding.
#
# The adversarial review of the earlier masked-field pilot established that
# unconstrained temperature-1 ancestral decoding produces 0/256 legal teams,
# while the SAME weights decoded against the legality oracle produce 256/256.
# The failure was the decoder, not the model. So constrained decoding is the
# default here: at each reveal we zero the probability of any value that would
# violate a rule given what is already decided.
#
# This is not a hack bolted on top - it is the discrete analogue of projecting
# each denoising step back onto the feasible set, and it is exactly what makes
# a generator usable as a proposal distribution: an illegal team cannot be
# battled at all, so illegal mass is wasted mass.
# =====================================================================
from corpus import dex_entry, legal_moves as _sd_moves, legal_abilities as _sd_abils, MEGA_STONE_OF

class Constraints:
    """Per-column feasible-value masks, conditioned on what is already revealed."""
    def __init__(self, V, L):
        self.V, self.L = V, L
        self.species_vals = V.itos["species"]
        self.move_vals = V.itos["move"]
        self.item_vals = V.itos["item"]
        self.abil_vals = V.itos["ability"]
        # species -> boolean masks over the move / ability vocabularies
        self.mv_ok, self.ab_ok = {}, {}
        self.base_of = {}
        for si, sp in enumerate(self.species_vals):
            e = dex_entry(sp) if si else None
            self.base_of[si] = norm(e.get("baseSpecies", e.get("name", sp))) if e else sp
        for si, sp in enumerate(self.species_vals):
            if si == 0: continue
            lm = L.moves_for(sp); la = L.abils_for(sp)
            self.mv_ok[si] = torch.tensor([j > 0 and m in lm for j, m in enumerate(self.move_vals)])
            self.ab_ok[si] = torch.tensor([j > 0 and a in la for j, a in enumerate(self.abil_vals)])

    def mask_for(self, col, row, device):
        """Feasible mask for `col` given the partially decoded team `row`
        (0 marks a still-masked field)."""
        k = self.V.key(col); slot = col // NF; j = col % NF
        n = self.V.sizes[k]
        ok = torch.ones(n, dtype=torch.bool)
        ok[0] = False                                   # never emit [MASK]
        if j == 0:                                      # species: Species Clause
            # Showdown compares BASE species, so Maushold and Maushold-Four collide.
            used = set()
            for s2 in range(NSLOT):
                if s2 != slot and int(row[s2*NF]) != 0:
                    used.add(self.base_of[int(row[s2*NF])])
            if used:
                for vi in range(1, n):
                    if self.base_of[vi] in used: ok[vi] = False
        elif j == 2:                                    # item: Item Clause + mega stones
            for s2 in range(NSLOT):
                if s2 != slot and int(row[s2*NF+2]) != 0:
                    ok[int(row[s2*NF+2])] = False
            si = int(row[slot*NF])
            if si != 0:
                base = self.base_of[si]
                sp_is_mega = "mega" in self.species_vals[si]
                for vi in range(1, n):
                    st = self.item_vals[vi]
                    if st in MEGA_STONE_OF:
                        # a stone belongs on the BASE forme; Showdown rejects
                        # "Blastoise-Mega @ Blastoisinite" ("please fix its item")
                        if sp_is_mega or MEGA_STONE_OF[st] != base:
                            ok[vi] = False
        elif j == 1:                                    # ability: must belong to species
            si = int(row[slot*NF])
            if si != 0 and si in self.ab_ok: ok &= self.ab_ok[si]
        elif 3 <= j <= 6:                               # move: learnset + no duplicates
            si = int(row[slot*NF])
            if si != 0 and si in self.mv_ok: ok &= self.mv_ok[si]
            for j2 in range(3, 7):
                if j2 != j and int(row[slot*NF+j2]) != 0:
                    ok[int(row[slot*NF+j2])] = False
        if not ok.any():
            # Relax the duplicate-move rule rather than emit [MASK], which Showdown
            # parses as a move literally named "mask".
            if 3 <= j <= 6:
                si = int(row[slot*NF])
                if si != 0 and si in self.mv_ok:
                    ok = self.mv_ok[si].clone()
            if not ok.any():
                ok[0] = True
        return ok.to(device)

# reveal species first, then ability/item, then moves, then nature: constraints
# only bind once the species they depend on is known
ORDER = ([c for c in range(COLS) if c % NF == 0] +
         [c for c in range(COLS) if c % NF in (1, 2)] +
         [c for c in range(COLS) if 3 <= c % NF <= 6] +
         [c for c in range(COLS) if c % NF == 7])

@torch.no_grad()
def sample_constrained(model, C, n, style="none", guidance=1.0, temp=1.0,
                       pin=None, x0=None, mask=None, device=DEV):
    """Decode in dependency order, projecting each step onto the feasible set."""
    y = torch.full((n,), STYLES.index(style) if style in STYLES else NULL,
                   device=device, dtype=torch.long)
    if x0 is None:
        x = torch.zeros(n, COLS, dtype=torch.long, device=device)
        todo = set(range(COLS))
    else:
        x = x0.clone().to(device)
        todo = {c for c in range(COLS) if bool(mask[0, c])}
        x[:, sorted(todo)] = 0
    if pin:
        for c, v in pin.items():
            x[:, c] = v; todo.discard(c)
    seq = [c for c in ORDER if c in todo]
    for step, c in enumerate(seq):
        t_now = 1.0 - step / max(len(seq), 1)
        lg = _step_logits(model, x, t_now, y, guidance)[c]
        for b in range(n):
            ok = C.mask_for(c, x[b], device)
            l = lg[b].clone(); l[~ok] = -1e9
            p = torch.softmax(l / temp, -1)
            x[b, c] = torch.multinomial(p, 1).item()
    return x

def _load(V):
    m = TeamDiffusion(V).to(DEV)
    m.load_state_dict(torch.load(CKPT, map_location=DEV)["sd"]); m.eval()
    return m

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sample", "edit", "evaltest"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--style", default="none")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--pin", default=None, help="species to force onto the team")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--unconstrained", action="store_true")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.epochs); return
    teams, V, L, X, Y = load_all()
    model = _load(V); C = Constraints(V, L)
    pin = None
    if a.pin:
        si = V.stoi["species"].get(norm(a.pin))
        if si is None: raise SystemExit(f"{a.pin!r} not in the corpus vocabulary")
        pin = {0: si}
    if a.cmd == "sample":
        f = (lambda: sample(model, a.n, a.style, a.guidance, pin=pin, temp=a.temp)) \
            if a.unconstrained else \
            (lambda: sample_constrained(model, C, a.n, a.style, a.guidance, a.temp, pin=pin))
        x = f().cpu().numpy()
        ok = sum(L.legal([V.decode_field(c, int(r[c])) for c in range(COLS)]) for r in x)
        print(f"style={a.style} guidance={a.guidance} temp={a.temp} "
              f"constrained={not a.unconstrained} -> legal {ok}/{a.n}")
        for r in x[:min(a.n, 3)]:
            print("  --- team ---")
            for line in to_text(V, r, teams): print("   ", line)
    elif a.cmd == "edit":
        rng = np.random.default_rng(0)
        src = X[rng.integers(0, len(X), a.n)]
        m = np.zeros((a.n, COLS), bool)
        for b in range(a.n):
            m[b, rng.choice(COLS, a.k, replace=False)] = True
        x = sample_constrained(model, C, a.n, a.style, a.guidance, a.temp,
                               x0=torch.tensor(src), mask=torch.tensor(m)).cpu().numpy()
        ok = sum(L.legal([V.decode_field(c, int(r[c])) for c in range(COLS)]) for r in x)
        ch = float(np.mean((x != src).sum(1)))
        print(f"local move k={a.k}: legal {ok}/{a.n}, mean fields changed {ch:.2f}")
    elif a.cmd == "evaltest":
        evaltest(model, C, V, L, X, Y, teams)

if __name__ == "__main__":
    cli()
