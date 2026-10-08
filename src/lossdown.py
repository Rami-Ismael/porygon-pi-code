"""Loss-down retrain of the team diffusion model (2026-10-07).

Goal (owner): retrain on everything we own -- the matchup matrix plus every team the loops
wrote -- and push the loss below every earlier run, using techniques from current papers.

ONE BENCHMARK, never trained on: asked_vs_got's held-out 500 teams (loss_diagnosis split).
Every model -- old checkpoints and new runs -- is scored on the SAME masks (common random
numbers: same slot shuffles, same t, same mask draws), so differences are the model's.

Three numbers per model, all on those 500 teams, conditioned on their win-rate labels:
  legacy  : the number every earlier training log printed ("held-out cond ~4.9"):
            sum over masked fields of CE/t, divided by the number of masked fields (pooled)
  elbo    : the proper MDLM/MD4 bound on -log p(team), in nats per field:
            E_t[ sum_masked (w_c/t) CE_c ] / 48   (w_c = 1 for the uniform schedule)
  ce      : plain cross-entropy per masked field at the same masks

Techniques under test (papers in docs/loss-down.md):
  data      146,994 distinct teams (lossdata.py) vs the 14,736 the old model saw
  legal     structured output: logits restricted to values the visible fields allow
            (species<->ability/moves/item, Species/Item Clause, canonical move order)
  notime    time-independent network (RADD, Ou et al. ICLR 2025)
  strat     stratified t across the batch (MD4 Algorithm 1)
  sched     per-field-type polynomial schedule alpha_c = 1 - t^w_c (GenMD4 / learned order)
  ema       exponential moving average of weights
  tags      source tag (model / real / fill / hps) as metadata conditioning
  size      width / depth / epochs

    python lossdown.py baseline                 # old checkpoints on the benchmark
    python lossdown.py run NAME key=val ...     # train one config, score on val + test
    python lossdown.py table                    # every result so far
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import copy, json, math, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/tmp/vgc-pilot/src")
import asked_vs_got as X
import diffusion as D
import hpsdiffusion as H
import activesearch as A
from corpus import norm, MEGA_STONE_OF, dex_entry
from encode import NF, NSLOT

DEV, COLS = D.DEV, D.COLS
RES = X.RES
DATA = RES / "lossdata.npz"
OUT = RES / "lossdown.json"
CK = Path.home() / ".local/share/vgc-pilot-runtime/lossdown"     # checkpoints outside iCloud
KEYS = ["species", "ability", "item", "move", "nature"]
NAN = float("nan")


# ---------------------------------------------------------------- legality tables (structured output)
class Tables:
    """Boolean compatibility tables on the device. allowed(xin) gives, for every column, the
    values consistent with the fields that are visible (0 = masked). Never excludes a value a
    legal team can hold -- check() verifies that on every team before it is used."""
    def __init__(self, V, corpus):
        C = D.Constraints(V, A.Legality(corpus))
        self.V = V
        sp, ab, it, mv = (V.itos[k] for k in ("species", "ability", "item", "move"))
        nsp, nab, nit, nmv = len(sp), len(ab), len(it), len(mv)
        AB = torch.ones(nsp, nab, dtype=torch.bool); MV = torch.ones(nsp, nmv, dtype=torch.bool)
        IT = torch.ones(nsp, nit, dtype=torch.bool)
        self.EMV = V.stoi["move"].get(""); self.EIT = V.stoi["item"].get("")
        for si in range(1, nsp):
            AB[si] = C.ab_ok[si]; MV[si] = C.mv_ok[si]
            if self.EMV is not None: MV[si, self.EMV] = True
            e = dex_entry(sp[si]) or {}
            mega = norm(e.get("forme", "")).startswith("mega")          # not a substring test: "meganium"
            own = norm(e.get("requiredItem", "")) if mega else ""
            for vi, st in enumerate(it):
                if st in MEGA_STONE_OF:
                    # the data (Showdown-validated HPS teams included) holds Mega formes with their
                    # own stone; a base forme takes a stone whose mega changes from it
                    IT[si, vi] = (st == own) if mega else MEGA_STONE_OF[st] in (C.base_of[si], sp[si])
        AB[0] = MV[0] = IT[0] = False
        bases = sorted({C.base_of[si] for si in range(1, nsp)})
        BASE = torch.tensor([0] + [1 + bases.index(C.base_of[si]) for si in range(1, nsp)])
        self.AB, self.MV, self.IT, self.BASE = (t.to(DEV) for t in (AB, MV, IT, BASE))
        self.nbase = len(bases) + 1
        self.sizes = dict(species=nsp, ability=nab, item=nit, move=nmv, nature=V.sizes["nature"])

    def allowed(self, xin, rules=("clause", "compat", "order")):
        B = xin.shape[0]; x = xin.view(B, NSLOT, NF)
        sp, ab, it, mv = x[..., 0], x[..., 1], x[..., 2], x[..., 3:7]
        vsp, vab, vit, vmv = sp != 0, ab != 0, it != 0, mv != 0
        T = lambda n: torch.ones(B, NSLOT, n, dtype=torch.bool, device=xin.device)
        osp, oab, oit, ona = T(self.sizes["species"]), T(self.sizes["ability"]), T(self.sizes["item"]), T(self.sizes["nature"])
        omv = torch.ones(B, NSLOT, 4, self.sizes["move"], dtype=torch.bool, device=xin.device)
        if "clause" in rules:
            # Species Clause on base species; Item Clause on non-empty items, against the OTHER slots
            bo = F.one_hot(torch.where(vsp, self.BASE[sp], 0), self.nbase); bo[..., 0] = 0
            other = (bo.sum(1, keepdim=True) - bo) > 0
            osp &= ~torch.gather(other, 2, self.BASE.view(1, 1, -1).expand(B, NSLOT, -1))
            io = F.one_hot(it, self.sizes["item"]); io[..., 0] = 0
            if self.EIT is not None: io[..., self.EIT] = 0
            oit &= ~((io.sum(1, keepdim=True) - io) > 0)
        if "compat" in rules or "abil" in rules:
            oab &= torch.where(vsp.unsqueeze(-1), self.AB[sp], True)
            osp &= torch.where(vab.unsqueeze(-1), self.AB.T[ab], True)
        if "compat" in rules or "learn" in rules:
            omv &= torch.where(vsp.unsqueeze(-1), self.MV[sp], True).unsqueeze(2)
            for j in range(4):
                osp &= torch.where(vmv[..., j].unsqueeze(-1), self.MV.T[mv[..., j]], True)
        if "compat" in rules or "stone" in rules:
            oit &= torch.where(vsp.unsqueeze(-1), self.IT[sp], True)
            osp &= torch.where(vit.unsqueeze(-1), self.IT.T[it], True)
        if "order" in rules and self.EMV is not None:
            # moves are stored sorted (vocabulary order) with '' padding last
            idx = torch.arange(self.sizes["move"], device=xin.device).view(1, 1, -1); E = self.EMV
            for j in range(4):
                for j2 in range(4):
                    if j2 == j: continue
                    v = mv[..., j2].unsqueeze(-1); vis = vmv[..., j2].unsqueeze(-1); ne = v != E
                    if j2 < j:
                        ok = torch.where(ne, (idx > v) | (idx == E), idx == E)
                    else:
                        ok = torch.where(ne, (idx < v) & (idx != E), torch.ones_like(idx, dtype=torch.bool))
                    omv[:, :, j] &= torch.where(vis, ok, True)
        elif "order" in rules or "clause" in rules:
            for j in range(4):                                         # at least no duplicate non-empty moves
                for j2 in range(4):
                    if j2 == j: continue
                    o = F.one_hot(mv[..., j2], self.sizes["move"]).bool(); o[..., 0] = False
                    if self.EMV is not None: o[..., self.EMV] = False
                    omv[:, :, j] &= ~o
        for o in (osp, oab, oit, ona): o[..., 0] = False
        omv[..., 0] = False
        return dict(species=osp, ability=oab, item=oit, move=omv.reshape(B, NSLOT * 4, -1), nature=ona)

    @torch.no_grad()
    def check(self, X, rules=("clause", "compat", "order"), batch=4096, reps=3, seed=0):
        """Fraction of teams whose true value is excluded anywhere, under random masks."""
        g = torch.Generator().manual_seed(seed)
        bad = torch.zeros(len(X), dtype=torch.bool)
        for _ in range(reps):
            for i in range(0, len(X), batch):
                x = torch.as_tensor(X[i:i + batch], device=DEV)
                t = torch.rand(len(x), 1, generator=g).to(DEV)
                m = torch.rand(len(x), COLS, generator=g).to(DEV) < t
                xin = x.masked_fill(m, 0)
                al = self.allowed(xin, rules)
                for k, cols in COLS_BY_KEY.items():
                    tv = x[:, cols]
                    ok = torch.gather(al[k], 2, tv.unsqueeze(-1)).squeeze(-1)
                    bad[i:i + batch] |= (~ok).any(1).cpu()
        return bad.numpy()


def _cols_by_key():
    V = None
    out = {}
    for c in range(COLS):
        k = D.FIELDS[c % NF]; k = "move" if k.startswith("m") and k != "nature" else k
        out.setdefault(k, []).append(c)
    return out
COLS_BY_KEY = _cols_by_key()
KEY_OF_COL = {c: k for k, cs in COLS_BY_KEY.items() for c in cs}


# ---------------------------------------------------------------- the model
class Net(nn.Module):
    """Transformer denoiser over the 48-field grid. Parameter names match TeamDiffusionHPS /
    ContinuousWR so the old checkpoints warm-start it."""
    def __init__(self, V, d=192, nhead=6, nlayer=4, dropout=0.1, time_cond=True, ntag=0, nfreq=8, prime=1, mrow=0):
        super().__init__()
        self.prime = prime
        self.mrow = mrow
        if mrow:                # the team's matchup-matrix row: win rate vs each top-50 column, NaN = untested
            self.mproj = nn.Sequential(nn.Linear(2 * mrow, d), nn.SiLU(), nn.Linear(d, d))
            self.mnull = nn.Parameter(torch.zeros(d))
        self.emb = nn.ModuleDict({k: nn.Embedding(v, d) for k, v in V.sizes.items()})
        if prime == 2:          # Prime (Chao et al., NeurIPS 2025): one d/2 embedding per sub-token, concatenated
            self.base = {k: math.ceil(math.sqrt(v - 1)) for k, v in V.sizes.items()}
            self.emb1 = nn.ModuleDict({k: nn.Embedding(b + 1, d // 2) for k, b in self.base.items()})
            self.emb2 = nn.ModuleDict({k: nn.Embedding(b + 1, d - d // 2) for k, b in self.base.items()})
        self.head = nn.ModuleDict({k: nn.Linear(d, v) for k, v in V.sizes.items()})
        self.pos = nn.Parameter(torch.randn(COLS, d) * .02)
        self.time_cond = time_cond
        self.tproj = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("freqs", (2.0 ** torch.arange(nfreq)) * math.pi)
        self.wproj = nn.Sequential(nn.Linear(2 * nfreq + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.wnull = nn.Parameter(torch.zeros(d))
        self.ntag = ntag
        if ntag: self.tag = nn.Embedding(ntag, d)
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, nlayer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def wvec(self, w):
        null = torch.isnan(w); z = torch.nan_to_num(w, 0.0).view(-1, 1)
        v = self.wproj(torch.cat([z, torch.sin(z * self.freqs), torch.cos(z * self.freqs)], 1))
        return torch.where(null.view(-1, 1), self.wnull.expand_as(v), v)

    def mvec(self, r):
        has = ~torch.isnan(r)
        v = self.mproj(torch.cat([torch.nan_to_num(r, 0.0), has.float()], 1))
        return torch.where(has.any(1, keepdim=True), v, self.mnull.expand_as(v))

    def forward(self, x, t, w, tag=None):
        B, d = x.shape[0], self.pos.shape[1]
        if self.mrow:                                           # w = [scalar win rate, row...]
            r = w[:, 1:] if w.dim() == 2 else torch.full((B, self.mrow), NAN, device=x.device)
            w = w[:, 0] if w.dim() == 2 else w
        h = torch.empty(B, COLS, d, device=x.device)
        for k, cols in COLS_BY_KEY.items():
            if self.prime == 2:                                         # x: [B, 48, 2] digits, 0 = masked
                h[:, cols] = torch.cat([self.emb1[k](x[:, cols, 0]), self.emb2[k](x[:, cols, 1])], -1)
            else:
                h[:, cols] = self.emb[k](x[:, cols])
        h = h + self.pos.unsqueeze(0) + self.wvec(w).unsqueeze(1)
        if self.time_cond: h = h + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        if self.ntag and tag is not None: h = h + self.tag(tag).unsqueeze(1)
        if self.mrow: h = h + self.mvec(r).unsqueeze(1)
        return self.ln(self.tr(h))

    def logits(self, h):
        return {k: self.head[k](h[:, cols]) for k, cols in COLS_BY_KEY.items()}


def logprobs_fn(net, tables=None, rules=("clause", "compat", "order"), tag_id=None):
    """(xin, t, w) -> {key: log-probs [B, ncols, V]} with optional structured-output masking."""
    def f(xin, t, w):
        tag = None if tag_id is None else torch.full((xin.shape[0],), tag_id, device=xin.device, dtype=torch.long)
        lg = net.logits(net(xin, t, w, tag)) if isinstance(net, Net) else _old_logits(net, xin, t, w)
        if tables is not None:
            al = tables.allowed(xin, rules)
            lg = {k: v.masked_fill(~al[k], -1e4) for k, v in lg.items()}
        return {k: F.log_softmax(v.float(), -1) for k, v in lg.items()}
    return f


def _old_logits(m, xin, t, w):
    if isinstance(m, X.ContinuousWR): h = m(xin, t, w)
    else: h = m(xin, t, torch.full((xin.shape[0],), H.WNULL, device=xin.device, dtype=torch.long))
    return {k: m.head[k](h[:, cols]) for k, cols in COLS_BY_KEY.items()}


# ---------------------------------------------------------------- the benchmark
@torch.no_grad()
def evaluate(f, x, w, reps=16, seed=12345, sched=None, batch=500):
    """Common-random-number ELBO on fixed teams. sched: per-column exponent tensor (None = uniform).
    t is stratified per team across reps; slot order shuffled per rep, as in training."""
    sw = torch.ones(COLS) if sched is None else sched.cpu()
    g = torch.Generator().manual_seed(seed)
    N = len(x)
    xt_all = torch.as_tensor(x); wt_all = torch.as_tensor(w)
    acc = dict(legacy_num=0.0, legacy_den=0, elbo=0.0, ce=0.0, n=0)
    per_key = {k: [0.0, 0] for k in KEYS}
    per_team = torch.zeros(N)
    for r in range(reps):
        perm = torch.rand(N, NSLOT, generator=g).argsort(1)
        tt = ((r + torch.rand(N, generator=g)) / reps).clamp(min=1e-4)
        u = torch.rand(N, COLS, generator=g)
        fallback = torch.randint(0, COLS, (N,), generator=g)
        for i in range(0, N, batch):
            sl = slice(i, i + batch)
            xb = torch.gather(xt_all[sl].view(-1, NSLOT, NF), 1, perm[sl].unsqueeze(-1).expand(-1, -1, NF)).reshape(-1, COLS)
            t = tt[sl]
            mraw = u[sl] < t.view(-1, 1) ** sw.view(1, -1)
            m = mraw.clone(); empty = ~m.any(1); m[empty, fallback[sl][empty]] = True
            xb, m, mraw, t, wb = xb.to(DEV), m.to(DEV), mraw.to(DEV), t.to(DEV), wt_all[sl].to(DEV)
            lp = f(xb.masked_fill(m, 0), t, wb)
            ce0 = torch.zeros_like(xb, dtype=torch.float32)
            for k, cols in COLS_BY_KEY.items():
                ce0[:, cols] = -torch.gather(lp[k], 2, xb[:, cols].unsqueeze(-1)).squeeze(-1)
            ce = ce0 * m
            wcol = sw.to(DEV).view(1, -1) / t.view(-1, 1)
            # legacy: the old training-log convention (a draw with nothing masked is forced to mask one field)
            acc["legacy_num"] += float((ce / t.view(-1, 1)).sum()); acc["legacy_den"] += int(m.sum())
            # elbo: the proper bound -- a draw with nothing masked contributes 0
            e = (ce0 * mraw * wcol).sum(1) / COLS
            acc["elbo"] += float(e.sum()); per_team[sl] += e.cpu()
            acc["ce"] += float(ce.sum()); acc["n"] += int(m.sum())
            for k, cols in COLS_BY_KEY.items():
                per_key[k][0] += float(ce[:, cols].sum()); per_key[k][1] += int(m[:, cols].sum())
    out = dict(legacy=acc["legacy_num"] / acc["legacy_den"], elbo=acc["elbo"] / (N * reps), ce=acc["ce"] / acc["n"])
    out.update({f"ce_{k}": v[0] / max(v[1], 1) for k, v in per_key.items()})
    pt = (per_team / reps).numpy()
    out["elbo_se_teams"] = float(pt.std(ddof=1) / math.sqrt(N))
    if DEV == "mps": torch.mps.empty_cache()           # three runs at once filled 24 GB and swapped (2026-10-07)
    return out, pt


def load_data():
    d = np.load(DATA)
    return {k: d[k] for k in d.files}


FAM = ["real", "oldpool", "loops08", "temperature", "cemrev", "redund", "matrix", "fill", "hps"]
NTAG = 6
def tag_of(S):
    """Source tag per team: 0 real, 1 the August-September loops (the test's own family),
    2 temperature campaign, 3 cem / reverse-score / redundancy / matrix runs, 4 random fill, 5 HPS."""
    b = lambda f: ((S >> FAM.index(f)) & 1).astype(bool)
    tag = np.full(len(S), 3, np.int64)
    tag[b("hps")] = 5; tag[b("fill")] = 4
    tag[b("cemrev") | b("redund") | b("matrix")] = 3
    tag[b("temperature")] = 2
    tag[b("oldpool") | b("loops08")] = 1
    tag[b("real")] = 0
    return tag
TEST_TAG = 1


def splits(d, nval=2000, seed=7):
    """val = 2,000 teams from the test's own family (the loops that produced the test 500)."""
    rng = np.random.default_rng(seed)
    cand = np.where(tag_of(d["S"]) == TEST_TAG)[0]
    val = rng.choice(cand, nval, replace=False)
    tr = np.setdiff1d(np.arange(len(d["X"])), val)
    return tr, val


# ---------------------------------------------------------------- training
DEFAULT = dict(d=192, nlayer=4, nhead=6, dropout=0.1, time_cond=1, tags=0, legal=1, rules="clause,compat,order",
               strat=1, sched="1,1,1,1,1", objective="elbo", epochs=10, batch=256, lr=5e-4, wd=0.01,
               warmup=500, ema=0.999, p_uncond=0.15, data="all", warm="", seed=0, final=0, maxmin=0,
               sources="model,real,fill,hps", upw=1, force=0, prime=1, mrow=0, moveorder="alpha")


def sched_tensor(s):
    w = dict(zip(KEYS, [float(v) for v in s.split(",")]))
    return torch.tensor([w[KEY_OF_COL[c]] for c in range(COLS)], dtype=torch.float32)


def matrix_holdout(d, n=1000, seed=11):
    """1,000 teams with >= 40 tested top-50 columns, never trained on by the matrix-conditioned run."""
    full = np.where((d["M"][:, :, 1] > 0).sum(1) >= 40)[0]
    return np.sort(np.random.default_rng(seed).choice(full, n, replace=False))


def cond_matrix(d, idx):
    """[scalar win rate, win rate vs each top-50 column (NaN = untested)]."""
    M = d["M"][idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(M[:, :, 1] > 0, M[:, :, 0] / np.maximum(M[:, :, 1], 1), np.nan).astype(np.float32)
    return np.concatenate([d["W"][idx][:, None], r], 1)


# ---------------------------------------------------------------- canonical move order by popularity
def apply_moveorder(d, V, tables):
    """Re-index the move vocabulary by training frequency (most common = 1, '' padding last) and re-sort
    each slot's moves in that order. A bijection on teams: with the order mask a model gives zero mass to
    non-canonical sequences, so -log p(sequence) is still -log p(team). Returns new (d, tables)."""
    Vm = V.sizes["move"]; cols = COLS_BY_KEY["move"]; E = V.stoi["move"].get("")
    cnt = np.bincount(d["X"][:, cols].ravel(), minlength=Vm)
    nonempty = [v for v in range(1, Vm) if v != E]
    new = np.zeros(Vm, np.int64)
    for i, v in enumerate(sorted(nonempty, key=lambda v: (-cnt[v], v))): new[v] = 1 + i
    if E is not None: new[E] = Vm - 1
    def tx(X):
        X = X.copy(); m = new[X.reshape(-1, NSLOT, NF)[:, :, 3:7]]
        X.reshape(-1, NSLOT, NF)[:, :, 3:7] = np.sort(m, -1)
        return X
    d = dict(d); d["X"] = tx(d["X"]); d["xte"] = tx(d["xte"])
    if tables is not None:
        t2 = copy.copy(tables); inv = torch.as_tensor(np.argsort(new), device=DEV)     # new index -> old index
        t2.MV = tables.MV[:, inv]; t2.EMV = None if E is None else int(new[E])
        tables = t2
    return d, tables


def train_run(name, cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    corpus, V, *_ = X.setup()
    d = load_data()
    tr, val = splits(d)
    if cfg["final"]: tr = np.arange(len(d["X"]))                     # fold val back in for the final model
    mhold = matrix_holdout(d) if cfg["mrow"] else np.array([], np.int64)
    tr = np.setdiff1d(tr, mhold)
    keep_src = [["model", "real", "fill", "hps"].index(s) for s in cfg["sources"].split(",")]
    tr = tr[np.isin(d["T"][tr], keep_src)]
    TAG = tag_of(d["S"])
    if cfg["data"] == "old":                                          # the old model's pool (+ corpus)
        tr = tr[(((d["S"][tr] >> FAM.index("oldpool")) & 1) == 1) | (TAG[tr] == 0)]
    elif cfg["data"] == "family":                                     # the test's family (+ corpus)
        tr = tr[(TAG[tr] == TEST_TAG) | (TAG[tr] == 0)]
    elif cfg["data"] == "family+matrix":                              # + every team with a matchup-matrix row
        tr = tr[(TAG[tr] == TEST_TAG) | (TAG[tr] == 0) | (d["M"][tr, :, 1].sum(1) > 0)]
    if cfg["upw"] > 1:                                                # oversample the test's family
        fam = tr[TAG[tr] == TEST_TAG]
        tr = np.concatenate([tr] + [fam] * (cfg["upw"] - 1))
    tables = Tables(V, corpus) if cfg["legal"] else None
    rules = tuple(cfg["rules"].split(","))
    if cfg["moveorder"] == "freq": d, tables = apply_moveorder(d, V, tables)
    if tables is not None:
        bad = tables.check(d["X"][tr], rules)
        print(f"structured output excludes a true value in {bad.sum()} of {len(tr)} training teams -> dropped", flush=True)
        tr = tr[~bad]
        badv = tables.check(d["xte"], rules, reps=8)
        assert badv.sum() == 0, f"{badv.sum()} TEST teams violate the tables; refusing"
    xtr = torch.as_tensor(d["X"][tr], device=DEV)
    wtr = torch.as_tensor(cond_matrix(d, tr) if cfg["mrow"] else d["W"][tr], device=DEV)
    ttr = torch.as_tensor(TAG[tr], device=DEV)
    net = Net(V, d=cfg["d"], nhead=cfg["nhead"], nlayer=cfg["nlayer"], dropout=cfg["dropout"],
              time_cond=bool(cfg["time_cond"]), ntag=NTAG if cfg["tags"] else 0, prime=cfg["prime"],
              mrow=d["M"].shape[1] if cfg["mrow"] else 0).to(DEV)
    digits = Digits(V, d["X"][tr]) if cfg["prime"] == 2 else None
    if cfg["warm"]:
        wp = CK / cfg["warm"] if (CK / cfg["warm"]).exists() else RES / cfg["warm"]
        sd = torch.load(wp, map_location=DEV)["sd"]
        own = net.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        net.load_state_dict(sd, strict=False); print(f"warm start from {cfg['warm']}: {len(sd)} tensors", flush=True)
    ema = copy.deepcopy(net).eval() if cfg["ema"] else None
    sched = sched_tensor(cfg["sched"]).to(DEV)
    steps_per_ep = math.ceil(len(tr) / cfg["batch"]); total = steps_per_ep * cfg["epochs"]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98))
    lr_at = lambda s: cfg["lr"] * min(1.0, (s + 1) / cfg["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    f_tr = logprobs_fn(net, tables, rules, tag_id=TEST_TAG if cfg["tags"] else None)
    print(f"[{name}] {len(tr)} teams · {sum(p.numel() for p in net.parameters()):,} params · {total} steps · {DEV}", flush=True)
    t0 = time.perf_counter(); step = 0; log = []
    for ep in range(cfg["epochs"]):
        net.train(); perm = torch.randperm(len(tr), device=DEV); run = 0.0
        for i in range(0, len(tr), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            loss = batch_loss(net, tables, rules, xtr[b], wtr[b], ttr[b] if cfg["tags"] else None, sched, cfg, digits)
            for gp in opt.param_groups: gp["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
            if ema is not None:
                with torch.no_grad():
                    dec = min(cfg["ema"], (1 + step) / (10 + step))
                    for pe, p in zip(ema.parameters(), net.parameters()): pe.lerp_(p, 1 - dec)
            run += float(loss.detach())
        net.eval()
        m = ema if ema is not None else net
        vx, vw = d["X"][val], (cond_matrix(d, val) if cfg["mrow"] else d["W"][val])
        if digits is not None:
            r, _ = evaluate_prime(m, digits, tables, rules, vx, vw, tag_id=TEST_TAG if cfg["tags"] else None, reps=4)
        else:
            r, _ = evaluate(logprobs_fn(m, tables, rules, tag_id=TEST_TAG if cfg["tags"] else None), vx, vw, reps=4)
        log.append(dict(ep=ep + 1, train=run / steps_per_ep, val_elbo=r["elbo"], val_legacy=r["legacy"], val_ce=r["ce"]))
        print(f"  ep{ep+1:3d} train {run/steps_per_ep:.4f} · val elbo {r['elbo']:.4f} legacy {r['legacy']:.4f} "
              f"ce {r['ce']:.4f} ({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    m = ema if ema is not None else net
    CK.mkdir(parents=True, exist_ok=True)
    torch.save({"sd": m.state_dict(), "cfg": cfg, "digits": None if digits is None else {k: v.cpu() for k, v in digits.D.items()}},
               CK / f"{name}.pt")
    res = score_model(m, tables, rules, cfg, d, val, digits)
    res.update(cfg=cfg, log=log, minutes=(time.perf_counter() - t0) / 60, n_train=int(len(tr)))
    save(name, res)
    return res


def batch_loss(net, tables, rules, x, w, tag, sched, cfg, digits=None):
    B = x.shape[0]
    ps = torch.rand(B, NSLOT, device=x.device).argsort(1)                     # slot order is a nuisance variable
    x = torch.gather(x.view(B, NSLOT, NF), 1, ps.unsqueeze(-1).expand(-1, -1, NF)).reshape(B, COLS)
    if cfg["strat"]:
        t = ((torch.rand(1, device=x.device) + torch.arange(B, device=x.device) / B) % 1.0)[torch.randperm(B, device=x.device)]
    else:
        t = torch.rand(B, device=x.device)
    t = t.clamp(min=1e-3)
    m = torch.rand(B, COLS, device=x.device) < t.view(-1, 1) ** sched.view(1, -1)
    empty = ~m.any(1)
    if cfg["force"] and empty.any(): m[empty, torch.randint(0, COLS, (int(empty.sum()),), device=x.device)] = True
    ww = w.clone(); ww[torch.rand(B, device=x.device) < cfg["p_uncond"]] = NAN
    if ww.dim() == 2:                                          # matrix row: drop it whole, or column by column
        ww[torch.rand(B, device=x.device) < 0.3, 1:] = NAN
        ww[:, 1:][torch.rand(B, ww.shape[1] - 1, device=x.device) < 0.2] = NAN
    if digits is not None:                                    # Prime: each sub-token masked on its own
        md = torch.rand(B, COLS, 2, device=x.device) < (t.view(-1, 1) ** sched.view(1, -1)).unsqueeze(-1)
        ce = prime_ce(net, digits, tables, rules, x, md, t, ww, tag)
        return ((ce * sched.view(1, -1) / t.view(-1, 1)).sum(1) / COLS).mean()
    xin = x.masked_fill(m, 0)
    lg = net.logits(net(xin, t, ww, tag))
    if tables is not None:
        al = tables.allowed(xin, rules)
        lg = {k: v.masked_fill(~al[k], -1e4) for k, v in lg.items()}
    ce = torch.zeros(B, COLS, device=x.device)
    for k, cols in COLS_BY_KEY.items():
        ce[:, cols] = F.cross_entropy(lg[k].float().transpose(1, 2), x[:, cols], reduction="none")
    ce = ce * m
    if cfg["objective"] == "elbo":
        return ((ce * sched.view(1, -1) / t.view(-1, 1)).sum(1) / COLS).mean()
    if cfg["objective"] == "legacy":
        return (ce / t.view(-1, 1)).sum() / m.sum().clamp(min=1)
    return ce.sum() / m.sum().clamp(min=1)                                                    # plain CE


def score_model(m, tables, rules, cfg, d, val, digits=None):
    tag = TEST_TAG if cfg.get("tags") else None
    if digits is not None:
        out = {}
        out["test_cond"], pt = evaluate_prime(m, digits, tables, rules, d["xte"], d["wte"], tag_id=tag)
        out["test_uncond"], _ = evaluate_prime(m, digits, tables, rules, d["xte"], np.full(len(d["xte"]), np.nan, np.float32), tag_id=tag)
        out["val_cond"], _ = evaluate_prime(m, digits, tables, rules, d["X"][val], d["W"][val], tag_id=tag, reps=8)
        out["per_team_test_elbo"] = pt.round(4).tolist()
        tc = out["test_cond"]
        print(f"  TEST (500, cond) elbo {tc['elbo']:.4f}±{tc['elbo_se_teams']:.4f} | uncond elbo {out['test_uncond']['elbo']:.4f}", flush=True)
        return out
    f = logprobs_fn(m, tables, rules, tag_id=tag)
    out = {}
    if cfg.get("mrow"):
        mh = matrix_holdout(d); C = cond_matrix(d, mh)
        nanrow = C.copy(); nanrow[:, 1:] = np.nan
        out["mhold_row"], a = evaluate(f, d["X"][mh], C)
        out["mhold_scalar"], b = evaluate(f, d["X"][mh], nanrow)
        out["mhold_uncond"], c = evaluate(f, d["X"][mh], np.full_like(C, np.nan))
        for k, (x1, x2) in dict(row_vs_scalar=(a, b), scalar_vs_uncond=(b, c)).items():
            dd = x1 - x2; out[k] = dict(diff=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))))
        print(f"  MATRIX HOLDOUT ({len(mh)}) elbo row {out['mhold_row']['elbo']:.4f} · scalar {out['mhold_scalar']['elbo']:.4f} "
              f"· uncond {out['mhold_uncond']['elbo']:.4f} | row-scalar {out['row_vs_scalar']['diff']:+.4f}±{out['row_vs_scalar']['se']:.4f}", flush=True)
        wte = np.concatenate([d["wte"][:, None], np.full((len(d["wte"]), d["M"].shape[1]), np.nan, np.float32)], 1)
        out["test_cond"], pt = evaluate(f, d["xte"], wte)
        out["test_uncond"], _ = evaluate(f, d["xte"], np.full_like(wte, np.nan))
        out["val_cond"], _ = evaluate(f, d["X"][val], cond_matrix(d, val), reps=8)
        out["per_team_test_elbo"] = pt.round(4).tolist()
        if cfg.get("sched", "1,1,1,1,1") != "1,1,1,1,1":
            sw = sched_tensor(cfg["sched"])
            out["test_cond_own_sched"], po = evaluate(f, d["xte"], wte, sched=sw)
            out["val_cond_own_sched"], _ = evaluate(f, d["X"][val], cond_matrix(d, val), reps=8, sched=sw)
            out["mhold_row_own"], a2 = evaluate(f, d["X"][mh], C, sched=sw)
            out["mhold_scalar_own"], b2 = evaluate(f, d["X"][mh], nanrow, sched=sw)
            dd = a2 - b2; out["row_vs_scalar_own"] = dict(diff=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))))
            out["per_team_test_elbo_own"] = po.round(4).tolist()
            print(f"  own schedule: test {out['test_cond_own_sched']['elbo']:.4f} · matrix holdout row-scalar "
                  f"{out['row_vs_scalar_own']['diff']:+.4f}±{out['row_vs_scalar_own']['se']:.4f}", flush=True)
        return out
    out["test_cond"], pt = evaluate(f, d["xte"], d["wte"])
    out["test_uncond"], _ = evaluate(f, d["xte"], np.full(len(d["xte"]), np.nan, np.float32))
    out["val_cond"], _ = evaluate(f, d["X"][val], d["W"][val], reps=8)
    if cfg.get("sched", "1,1,1,1,1") != "1,1,1,1,1":
        out["test_cond_own_sched"], po = evaluate(f, d["xte"], d["wte"], sched=sched_tensor(cfg["sched"]))
        out["val_cond_own_sched"], _ = evaluate(f, d["X"][val], d["W"][val], reps=8, sched=sched_tensor(cfg["sched"]))
        out["per_team_test_elbo_own"] = po.round(4).tolist()
    out["per_team_test_elbo"] = pt.round(4).tolist()
    tc = out["test_cond"]
    print(f"  TEST (500, cond) legacy {tc['legacy']:.4f} · elbo {tc['elbo']:.4f}±{tc['elbo_se_teams']:.4f} · ce {tc['ce']:.4f} "
          f"| uncond elbo {out['test_uncond']['elbo']:.4f}", flush=True)
    return out


def load_old_keys(V):
    """Encodings of the old model's training teams (asked_vs_got_data minus the test 500, plus corpus)."""
    import lossdata
    _, _, xo, _ = lossdata.test_split(V)
    keys = {r.tobytes() for r in xo}
    corpus, *_ = X.setup()
    keys |= {V.encode(t).tobytes() for t in corpus}
    return keys


def save(name, res):
    allr = json.load(open(OUT)) if OUT.exists() else {}
    allr[name] = res
    json.dump(allr, open(OUT, "w"), indent=1)


# ---------------------------------------------------------------- old checkpoints on the benchmark
def baseline():
    corpus, V, *_ = X.setup()
    d = load_data(); tr, val = splits(d)
    tables = Tables(V, corpus)
    olds = {"asked_vs_got.pt (2026-10-04, best so far)": ("cwr", RES / "asked_vs_got.pt"),
            "temperature_p0.pt (unconditional p0)": ("hps", RES / "temperature_p0.pt")}
    for name, (kind, path) in olds.items():
        m = (X.ContinuousWR(V) if kind == "cwr" else H.TeamDiffusionHPS(V)).to(DEV)
        m.load_state_dict(torch.load(path, map_location=DEV)["sd"]); m.eval()
        for masked in (False, True):
            f = logprobs_fn(m, tables if masked else None)
            out = {}
            out["test_cond"], pt = evaluate(f, d["xte"], d["wte"])
            out["test_uncond"], _ = evaluate(f, d["xte"], np.full(len(d["xte"]), np.nan, np.float32))
            out["val_cond"], _ = evaluate(f, d["X"][val], d["W"][val], reps=8)
            out["per_team_test_elbo"] = pt.round(4).tolist()
            key = f"OLD {name}" + (" + structured output at eval only" if masked else "")
            save(key, out)
            tc = out["test_cond"]
            print(f"{key}\n   legacy {tc['legacy']:.4f} · elbo {tc['elbo']:.4f}±{tc['elbo_se_teams']:.4f} · ce {tc['ce']:.4f} · "
                  f"uncond elbo {out['test_uncond']['elbo']:.4f} · val elbo {out['val_cond']['elbo']:.4f}", flush=True)


def table(ref="OLD asked_vs_got.pt (2026-10-04, best so far)"):
    """Every run on the test 500. elbo = uniform schedule; own = the run's own unmasking schedule
    (its training objective, a valid bound for the model sampled in that order). diff = paired
    per-team difference of the better of the two vs the old best, same masks."""
    allr = json.load(open(OUT))
    base = np.array(allr[ref]["per_team_test_elbo"])
    print(f"{'run':46s} {'legacy':>7s} {'elbo':>7s} {'own':>7s} {'best vs old (paired)':>21s} {'val':>7s}")
    for k, r in allr.items():
        if "test_cond" not in r: continue
        tc = r["test_cond"]; pt = np.array(r["per_team_test_elbo"])
        own = r.get("test_cond_own_sched", {}).get("elbo", float("nan"))
        if "per_team_test_elbo_own" in r and own < tc["elbo"]: pt = np.array(r["per_team_test_elbo_own"])
        dd = pt - base
        vo = r.get("val_cond_own_sched", {}).get("elbo", r["val_cond"]["elbo"])
        print(f"{k[:46]:46s} {tc['legacy']:7.4f} {tc['elbo']:7.4f} {own:7.4f} {dd.mean():+11.4f}±{dd.std(ddof=1)/np.sqrt(len(dd)):.4f} "
              f"{min(vo, r['val_cond']['elbo']):7.4f}")


# ---------------------------------------------------------------- Prime: partial masking (l = 2)
class Digits:
    """value -> two base-b sub-tokens. Codes follow frequency rank in the training data, so the
    first sub-token is a popularity tier (Prime uses the base-b code of the token id)."""
    def __init__(self, V, Xtrain):
        self.D, self.base = {}, {}
        for k, cols in COLS_BY_KEY.items():
            C = V.sizes[k]; b = math.ceil(math.sqrt(C - 1)); self.base[k] = b
            cnt = np.bincount(Xtrain[:, cols].ravel(), minlength=C)[1:]
            rank = np.empty(C - 1, np.int64); rank[np.argsort(-cnt, kind="stable")] = np.arange(C - 1)
            Dk = np.full((C, 2), -1, np.int64); Dk[1:, 0] = rank // b; Dk[1:, 1] = rank % b
            self.D[k] = torch.as_tensor(Dk, device=DEV)

    def split(self, x):
        """[B, 48] values -> [B, 48, 2] digit+1 (0 reserved for MASK)."""
        out = torch.zeros(*x.shape, 2, dtype=torch.long, device=x.device)
        for k, cols in COLS_BY_KEY.items():
            out[:, cols] = self.D[k][x[:, cols]] + 1
        return out


def prime_ce(net, digits, tables, rules, x, md, t, w, tag):
    """Per-field sum over masked sub-tokens of -log p(sub-token | visible), the Prime ELBO integrand.
    x [B,48] true values, md [B,48,2] True = sub-token masked."""
    xd = digits.split(x).masked_fill(md, 0)
    full = ~md.any(-1)
    xfield = x.masked_fill(~full, 0)                                 # legality sees only fully revealed fields
    lg = net.logits(net(xd, t, w, tag))
    al = tables.allowed(xfield, rules) if tables is not None else None
    ce = torch.zeros(x.shape[0], COLS, device=x.device)
    for k, cols in COLS_BY_KEY.items():
        Dk = digits.D[k]                                             # [C, 2]
        v = lg[k].float()
        ok = torch.ones_like(v, dtype=torch.bool) if al is None else al[k]
        for j in range(2):                                           # consistency with visible sub-tokens
            vis = ~md[:, cols, j]
            ok = ok & (~vis.unsqueeze(-1) | (Dk[:, j].view(1, 1, -1) == (xd[:, cols, j] - 1).unsqueeze(-1)))
        ok[..., 0] = False
        p = torch.softmax(v.masked_fill(~ok, -1e4), -1)
        tv = x[:, cols]
        for j in range(2):
            tj = Dk[tv, j]                                           # true digit
            pj = (p * (Dk[:, j].view(1, 1, -1) == tj.unsqueeze(-1))).sum(-1)
            ce[:, cols] += md[:, cols, j] * -torch.log(pj.clamp_min(1e-30))
    return ce


@torch.no_grad()
def evaluate_prime(net, digits, tables, rules, x, w, tag_id=None, reps=16, seed=12345, batch=500):
    """Same slot shuffles, t and first-sub-token draws as evaluate(); the second sub-token has its own stream."""
    g = torch.Generator().manual_seed(seed); g2 = torch.Generator().manual_seed(seed + 1)
    N = len(x); xt_all = torch.as_tensor(x); wt_all = torch.as_tensor(w)
    tot = 0.0; per_team = torch.zeros(N); per_key = {k: [0.0, 0] for k in KEYS}
    for r in range(reps):
        perm = torch.rand(N, NSLOT, generator=g).argsort(1)
        tt = ((r + torch.rand(N, generator=g)) / reps).clamp(min=1e-4)
        u = torch.rand(N, COLS, generator=g); torch.randint(0, COLS, (N,), generator=g)
        u2 = torch.rand(N, COLS, generator=g2)
        for i in range(0, N, batch):
            sl = slice(i, i + batch)
            xb = torch.gather(xt_all[sl].view(-1, NSLOT, NF), 1, perm[sl].unsqueeze(-1).expand(-1, -1, NF)).reshape(-1, COLS)
            t = tt[sl]
            md = torch.stack([u[sl] < t.view(-1, 1), u2[sl] < t.view(-1, 1)], -1)
            xb, md, t, wb = xb.to(DEV), md.to(DEV), t.to(DEV), wt_all[sl].to(DEV)
            tag = None if tag_id is None else torch.full((len(xb),), tag_id, device=DEV, dtype=torch.long)
            ce = prime_ce(net, digits, tables, rules, xb, md, t, wb, tag)
            e = (ce / t.view(-1, 1)).sum(1) / COLS
            tot += float(e.sum()); per_team[sl] += e.cpu()
            for k, cols in COLS_BY_KEY.items():
                per_key[k][0] += float((ce[:, cols] / t.view(-1, 1)).sum()); per_key[k][1] += len(cols) * len(xb)
    pt = (per_team / reps).numpy()
    out = dict(elbo=tot / (N * reps), elbo_se_teams=float(pt.std(ddof=1) / math.sqrt(N)), legacy=float("nan"), ce=float("nan"))
    out.update({f"elbo_{k}": v[0] / v[1] for k, v in per_key.items()})
    return out, pt


# ---------------------------------------------------------------- exact likelihood of the deployed decoder
@torch.no_grad()
def exact_nll(f, x, w, order=None, seed=12345, batch=500, perms=1):
    """-log p(team) for the sampler that reveals columns in a FIXED order (diffusion.ORDER by default:
    species, ability/item, moves, nature) -- exact, not a bound (DUEL, ICML 2026: deterministic
    unmasking admits exact likelihood). Same slot shuffles as evaluate()'s first rep. For a model with a
    time input, t = fraction still masked. nats per field."""
    order = list(D.ORDER) if order is None else order
    g = torch.Generator().manual_seed(seed)
    N = len(x); xt_all = torch.as_tensor(x); wt_all = torch.as_tensor(w)
    per_team = torch.zeros(N); per_key = {k: 0.0 for k in KEYS}
    for _ in range(perms):
        perm = torch.rand(N, NSLOT, generator=g).argsort(1)
        for i in range(0, N, batch):
            sl = slice(i, i + batch)
            xb = torch.gather(xt_all[sl].view(-1, NSLOT, NF), 1, perm[sl].unsqueeze(-1).expand(-1, -1, NF)).reshape(-1, COLS).to(DEV)
            wb = wt_all[sl].to(DEV); cur = torch.zeros_like(xb)
            for step, c in enumerate(order):
                t = torch.full((len(xb),), 1.0 - step / COLS, device=DEV)
                k = KEY_OF_COL[c]; j = COLS_BY_KEY[k].index(c)
                lp = f(cur, t, wb)[k][:, j]
                nl = -lp.gather(1, xb[:, c:c + 1]).squeeze(1)
                per_team[sl] += nl.cpu() / COLS; per_key[k] += float(nl.sum()) / COLS
                cur[:, c] = xb[:, c]
    pt = (per_team / perms).numpy()
    return dict(nll=float(pt.mean()), se_teams=float(pt.std(ddof=1) / math.sqrt(N)),
                **{f"nll_{k}": v / (N * perms) for k, v in per_key.items()}), pt


def exact(names):
    corpus, V, *_ = X.setup(); d = load_data(); tables = Tables(V, corpus)
    for name in names:
        net, cfg, _ = load_net(name, V)
        dd, tb = apply_moveorder(d, V, tables) if cfg["moveorder"] == "freq" else (d, tables)
        f = logprobs_fn(net, tb if cfg["legal"] else None, tuple(cfg["rules"].split(",")),
                        tag_id=TEST_TAG if cfg["tags"] else None)
        r, pt = exact_nll(f, dd["xte"], dd["wte"], perms=4)
        allr = json.load(open(OUT)); allr[name]["test_exact_order"] = r; allr[name]["per_team_test_exact"] = pt.round(4).tolist()
        json.dump(allr, open(OUT, "w"), indent=1)
        print(f"{name}: exact NLL in diffusion.ORDER {r['nll']:.4f} ± {r['se_teams']:.4f} (time input: {bool(cfg['time_cond'])}) "
              + " ".join(f"{k} {r['nll_'+k]:.3f}" for k in KEYS), flush=True)


# ---------------------------------------------------------------- ensembles of saved runs
def load_net(name, V):
    ck = torch.load(CK / f"{name}.pt", map_location=DEV)
    cfg = dict(DEFAULT); cfg.update(ck["cfg"])
    net = Net(V, d=cfg["d"], nhead=cfg["nhead"], nlayer=cfg["nlayer"], dropout=cfg["dropout"],
              time_cond=bool(cfg["time_cond"]), ntag=NTAG if cfg["tags"] else 0, prime=cfg["prime"],
              mrow=49 if cfg.get("mrow") else 0).to(DEV)
    net.load_state_dict(ck["sd"]); net.eval()
    digits = None
    if ck.get("digits"):
        digits = Digits.__new__(Digits); digits.D = {k: v.to(DEV) for k, v in ck["digits"].items()}
    return net, cfg, digits


def rescore(names):
    """Re-score saved runs with the current evaluator (keeps their training log)."""
    corpus, V, *_ = X.setup()
    d = load_data(); tr, val = splits(d)
    tables = Tables(V, corpus)
    allr = json.load(open(OUT)) if OUT.exists() else {}
    for name in names:
        net, cfg, digits = load_net(name, V)
        print(name, flush=True)
        dd, tb = apply_moveorder(d, V, tables) if cfg["moveorder"] == "freq" else (d, tables)
        res = score_model(net, tb if cfg["legal"] else None, tuple(cfg["rules"].split(",")), cfg, dd, val, digits)
        old = allr.get(name, {})
        res.update({k: old[k] for k in ("cfg", "log", "minutes", "n_train") if k in old})
        save(name, res)


def ensemble_fn(fs):
    """Mixture of the members' predictive distributions, field by field: log mean_k p_k."""
    def f(xin, t, w):
        outs = [g(xin, t, w) for g in fs]
        return {k: torch.logsumexp(torch.stack([o[k] for o in outs]), 0) - math.log(len(outs)) for k in outs[0]}
    return f


def ensemble(name, members):
    corpus, V, *_ = X.setup()
    d = load_data(); tr, val = splits(d)
    tables = Tables(V, corpus)
    fs = []
    orders = {load_net(m, V)[1]["moveorder"] for m in members}
    assert len(orders) == 1, "ensemble members must share a move order"
    if orders == {"freq"}: d, tables = apply_moveorder(d, V, tables)
    for mname in members:
        net, cfg, _ = load_net(mname, V)
        fs.append(logprobs_fn(net, tables if cfg["legal"] else None, tuple(cfg["rules"].split(",")),
                              tag_id=TEST_TAG if cfg["tags"] else None))
    f = ensemble_fn(fs)
    out = {}
    out["test_cond"], pt = evaluate(f, d["xte"], d["wte"])
    out["test_uncond"], _ = evaluate(f, d["xte"], np.full(len(d["xte"]), np.nan, np.float32))
    out["val_cond"], _ = evaluate(f, d["X"][val], d["W"][val], reps=8)
    out["per_team_test_elbo"] = pt.round(4).tolist(); out["members"] = members
    scheds = {load_net(m, V)[1]["sched"] for m in members}
    if len(scheds) == 1 and scheds != {"1,1,1,1,1"}:         # members share an unmasking order: score it too
        sw = sched_tensor(scheds.pop())
        out["test_cond_own_sched"], po = evaluate(f, d["xte"], d["wte"], sched=sw)
        out["val_cond_own_sched"], _ = evaluate(f, d["X"][val], d["W"][val], reps=8, sched=sw)
        out["per_team_test_elbo_own"] = po.round(4).tolist()
        print(f"  own schedule: test elbo {out['test_cond_own_sched']['elbo']:.4f}", flush=True)
    save(name, out)
    tc = out["test_cond"]
    print(f"{name}: legacy {tc['legacy']:.4f} · elbo {tc['elbo']:.4f} · ce {tc['ce']:.4f} · uncond {out['test_uncond']['elbo']:.4f} "
          f"· val {out['val_cond']['elbo']:.4f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "baseline": baseline()
    elif cmd == "table": table()
    elif cmd == "ens": ensemble(sys.argv[2], sys.argv[3:])
    elif cmd == "rescore": rescore(sys.argv[2:])
    elif cmd == "exact": exact(sys.argv[2:])
    elif cmd == "run":
        name = sys.argv[2]; cfg = dict(DEFAULT)
        for a in sys.argv[3:]:
            k, v = a.split("=", 1)
            cfg[k] = type(DEFAULT[k])(v) if not isinstance(DEFAULT[k], str) else v
        train_run(name, cfg)
