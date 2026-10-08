"""Experiment A — asked vs got. Pre-registration: docs/asked-vs-got.md.

Ask a continuous-win-rate diffusion model for y* = 0.0 … 0.9 at guidance 1/2/4,
battle what it writes, and plot the win rate asked for against the win rate got.

    python asked_vs_got.py data      # gather labelled teams (no battles)
    python asked_vs_got.py train     # fine-tune p0 with a continuous win-rate condition
    python asked_vs_got.py gen       # 64 Showdown-valid teams per cell
    python asked_vs_got.py battle    # 24 battles per team vs the top-50 (resumable per cell)
    python asked_vs_got.py report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, "/tmp/vgc-pilot/src")
import activesearch as A
import diffusion as D
import hpsdiffusion as H
import matchup_db as MDB
import pool
from corpus import parse_team_text
from encode import NF, NSLOT
import hps_generate
from hps_generate import Validator
# /tmp Showdown lost node_modules files to tmp cleanup (2026-10-04); durable validator copy, same commit 913da36
hps_generate.SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"

REPO = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot")
RES = REPO / "results"
OUT = RES / "asked_vs_got.json"
DATA = RES / "asked_vs_got_data.json"
CKPT = RES / "asked_vs_got.pt"
P0 = RES / "temperature_p0.pt"
WORK = Path("/tmp/vgc-pilot/askedgot")
SOURCES = ["activesearch", "dsame_experiment", "entropyloop", "gradguide", "gradloop"]
ASKED = [round(0.1 * i, 1) for i in range(10)]
GUIDE = [1.0, 2.0, 4.0]
N, BATTLES, SEED, EPOCHS = 64, 24, 2026, 30
PER_CELL, POLICY = 1, 1          # one battle per top-50 opponent; policy 1 = bc_100 both sides, Showdown 913da36
NAN = float("nan")


# ---------------------------------------------------------------- data
def _walk(o, acc):
    if isinstance(o, dict):
        if isinstance(o.get("pastes"), list) and isinstance(o.get("y"), list) and len(o["pastes"]) == len(o["y"]):
            acc.append(o)
        for v in o.values(): _walk(v, acc)

def gather():
    pooled = {}                                   # canon hash -> [paste, wins, battles]
    def add(p, y, n):
        c = MDB.canonical_paste(p); h = MDB.canon_hash(c)
        e = pooled.setdefault(h, [p, 0.0, 0])
        e[1] += y * n; e[2] += n
    for s in SOURCES:
        acc = []; _walk(json.load(open(RES / f"{s}.json")), acc)
        for o in acc:
            for p, y in zip(o["pastes"], o["y"]):
                if y is not None: add(p, float(y), BATTLES)
    con = MDB.connect()
    for r in MDB.labels(con, 1, min_coverage=40):          # policy 1 = bc_100 both sides, showdown 913da36
        (p,) = con.execute("SELECT paste FROM team WHERE team_id=?", (r["team_id"],)).fetchone()
        add(p, r["score"], r["battles"])
    rows = [dict(paste=p, y=w / n, battles=n) for p, w, n in pooled.values()]
    ys = np.array([r["y"] for r in rows])
    json.dump(dict(n=len(rows), max=float(ys.max()), quantiles=np.quantile(ys, [.5, .9, .99]).round(3).tolist(),
                   rows=rows), open(DATA, "w"))
    print(f"{len(rows)} distinct labelled teams · mean {ys.mean():.3f} · max {ys.max():.3f} · "
          f"≥0.6: {(ys >= .6).sum()} · ≥0.8: {(ys >= .8).sum()}")


# ---------------------------------------------------------------- model
class ContinuousWR(H.TeamDiffusionHPS):
    """TeamDiffusionHPS with a continuous win-rate condition; NaN = dropped (null)."""
    def __init__(self, V, nfreq=8, **kw):
        super().__init__(V, **kw)
        d = self.pos.shape[1]
        self.register_buffer("freqs", (2.0 ** torch.arange(nfreq)) * math.pi)
        self.wproj = nn.Sequential(nn.Linear(2 * nfreq + 1, d), nn.SiLU(), nn.Linear(d, d))
        self.wnull = nn.Parameter(torch.zeros(d))

    def wvec(self, w):
        null = torch.isnan(w); z = torch.nan_to_num(w, 0.0).view(-1, 1)
        v = self.wproj(torch.cat([z, torch.sin(z * self.freqs), torch.cos(z * self.freqs)], 1))
        return torch.where(null.view(-1, 1), self.wnull.expand_as(v), v)

    def forward(self, x, t, w):
        B, d = x.shape[0], self.pos.shape[1]
        h = torch.empty(B, D.COLS, d, device=x.device)
        for k, cols in self.cols_by_key.items():
            h[:, cols] = self.emb[k](x[:, cols])
        h = h + self.pos.unsqueeze(0) + self.tproj(t.view(-1, 1).float()).unsqueeze(1)
        h = h + self.wvec(w).unsqueeze(1)
        return self.ln(self.tr(h))

    def loss(self, x, w, p_uncond=0.15):
        ww = w.clone(); ww[torch.rand(x.shape[0], device=x.device) < p_uncond] = NAN
        return super().loss(x, ww, p_uncond=0.0)


def setup():
    cf = A.load_corpus_files(); corpus = [t for _, t in cf]
    V = A.Vocab(corpus); C = D.Constraints(V, A.Legality(corpus))
    look, spreads = A.decode_tables(corpus)
    return corpus, V, C, look, spreads


def train():
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    corpus, V, *_ = setup()
    rows = json.load(open(DATA))["rows"]
    X, W, drop = [], [], 0
    for r in rows:
        t = parse_team_text(r["paste"])
        if len(t) != 6: drop += 1; continue
        e = V.encode(t)
        if (e == 0).any(): drop += 1; continue     # a field outside the corpus vocabulary
        X.append(e); W.append(r["y"])
    X, W = np.stack(X), np.array(W, dtype=np.float32)
    Xc = np.stack([V.encode(t) for t in corpus]); Wc = np.full(len(Xc), np.nan, np.float32)
    idx = rng.permutation(len(X)); te, tr = idx[:500], idx[500:]
    xtr = torch.tensor(np.concatenate([X[tr], Xc]), device=D.DEV)
    wtr = torch.tensor(np.concatenate([W[tr], Wc]), device=D.DEV)
    xte, wte = torch.tensor(X[te], device=D.DEV), torch.tensor(W[te], device=D.DEV)
    m = ContinuousWR(V).to(D.DEV)
    miss, _ = m.load_state_dict(torch.load(P0, map_location=D.DEV)["sd"], strict=False)
    print(f"{len(tr)} labelled + {len(Xc)} corpus (null) · held-out {len(te)} · dropped {drop} · "
          f"new params {miss}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        m.train(); tl = A._epoch(m, opt, xtr, wtr, 128); sch.step(); m.eval()
        with torch.no_grad():
            torch.manual_seed(ep)
            lc = float(m.loss(xte, wte, p_uncond=0.0))
            torch.manual_seed(ep)
            lu = float(m.loss(xte, torch.full_like(wte, NAN), p_uncond=0.0))
        print(f"  ep{ep+1:3d} train {tl:7.3f} · held-out cond {lc:7.3f} uncond {lu:7.3f} "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
    torch.save({"sd": m.state_dict()}, CKPT)
    print("saved", CKPT)


def load(V):
    m = ContinuousWR(V).to(D.DEV); m.load_state_dict(torch.load(CKPT, map_location=D.DEV)["sd"]); m.eval()
    return m


@torch.no_grad()
def sample(m, C, n, ystar, g):
    """Dependency-order constrained decode, classifier-free guidance on the win-rate condition."""
    w = torch.full((n,), NAN if ystar is None else ystar, device=D.DEV)
    wn = torch.full((n,), NAN, device=D.DEV)
    x = torch.zeros(n, D.COLS, dtype=torch.long, device=D.DEV)
    for step, c in enumerate(D.ORDER):
        tt = torch.full((n,), 1.0 - step / len(D.ORDER), device=D.DEV)
        lc = F.log_softmax(m.logits(m(x, tt, w), c), -1)
        if ystar is not None and g != 1.0:
            lu = F.log_softmax(m.logits(m(x, tt, wn), c), -1)
            lc = g * lc + (1 - g) * lu
        for b in range(n):
            l = lc[b].clone(); l[~C.mask_for(c, x[b], D.DEV)] = -1e9
            x[b, c] = torch.multinomial(torch.softmax(l, -1), 1).item()
    return x


def cells():
    return [("uncond", None, 1.0)] + [(f"y{a:.1f}_g{g:g}", a, g) for g in GUIDE for a in ASKED]


def gen():
    corpus, V, C, look, spreads = setup(); m = load(V)
    out = json.load(open(OUT)) if OUT.exists() else {"cells": {}}
    val = Validator()
    try:
        for i, (tag, a, g) in enumerate(cells()):
            if tag in out["cells"]: continue
            torch.manual_seed(SEED + i); rng = np.random.default_rng(SEED + i)
            pastes, tried, t0 = [], 0, time.perf_counter()
            while len(pastes) < N and tried < N * 8:
                x = sample(m, C, 48, a, g).cpu().numpy()
                for r in x:
                    tried += 1
                    p = A.row_to_paste(V, r, look, spreads, rng)
                    if val(p) is None: pastes.append(p)
                    if len(pastes) >= N: break
            out["cells"][tag] = dict(asked=a, guidance=g, pastes=pastes, tried=tried, valid=len(pastes) / tried)
            json.dump(out, open(OUT, "w"))
            print(f"  {tag:12s} {len(pastes)}/{tried} valid ({len(pastes)/tried:.0%}) "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
    finally:
        val.close()


def battle():
    """Every battle goes into the matchup matrix (~/vgc-data/matchup_regmb.sqlite, policy 1):
    one battle per (generated team, top-50 column) cell via matchup_db.score_pastes."""
    # /tmp/vgc-pilot lost the vgc_bench package and Showdown node_modules to tmp cleanup;
    # runtime copies are the same commits (vgc-bench d79f953, Showdown 913da36)
    pool.REPO = str(Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2")
    MDB.SHOWDOWN = hps_generate.SHOWDOWN
    con = MDB.connect()
    out = json.load(open(OUT))
    todo = [t for t, c in out["cells"].items() if "y" not in c]
    for k in range(0, len(todo), 8):                       # 8 cells (~512 teams) per database call
        chunk = todo[k:k + 8]
        pastes = [p for t in chunk for p in out["cells"][t]["pastes"]]
        origins = [t for t in chunk for _ in out["cells"][t]["pastes"]]
        val = MDB.Validator()                               # score_pastes, inlined: per-cell origin, one battle call per chunk
        try:
            ids = [MDB.add_team(con, p, f"askedgot:{o}", val)[0] for p, o in zip(pastes, origins)]
        finally:
            val.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, POLICY, sorted(legal), MDB.set_members(con, "top50"), PER_CELL, seed=SEED,
                   note=f"asked_vs_got {chunk[0]}..{chunk[-1]}")
        i = 0
        for t in chunk:
            n = len(out["cells"][t]["pastes"]); sc = [MDB.score(con, POLICY, tid) if tid in legal else None for tid in ids[i:i + n]]
            out["cells"][t]["team_id"] = ids[i:i + n]
            out["cells"][t]["y"] = [s["score"] if s else None for s in sc]
            out["cells"][t]["battles"] = [s["battles"] if s else 0 for s in sc]
            i += n
        json.dump(out, open(OUT, "w"))
        print(f"  battled {chunk}", flush=True)


def fill():
    """Battle any (team, top-50) cell still missing -- a shard that hangs on exit is killed and its
    battles discarded -- then recompute every cell's scores from the matchup matrix."""
    pool.REPO = str(Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2")
    con = MDB.connect(); out = json.load(open(OUT))
    ids = sorted({t for c in out["cells"].values() for t in c.get("team_id", [])})
    legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
    n = MDB.ensure(con, POLICY, sorted(legal), MDB.set_members(con, "top50"), PER_CELL, seed=SEED, note="asked_vs_got fill")
    print(f"  fill: {n} battles for missing cells", flush=True)
    for c in out["cells"].values():
        if "team_id" not in c: continue
        sc = [MDB.score(con, POLICY, t) if t in legal else None for t in c["team_id"]]
        c["y"] = [s["score"] if s else None for s in sc]
        c["battles"] = [s["battles"] if s else 0 for s in sc]
    json.dump(out, open(OUT, "w"))


# ---------------------------------------------------------------- report
def report():
    out = json.load(open(OUT)); data = json.load(open(DATA))
    train_hash = {MDB.canon_hash(MDB.canonical_paste(r["paste"])) for r in data["rows"]}
    rng = np.random.default_rng(0); summ = {}
    for t, c in out["cells"].items():
        y = np.array([v for v in c["y"] if v is not None], float)
        sets = {tuple(sorted(s["species"] for s in parse_team_text(p))) for p in c["pastes"]}
        copies = sum(MDB.canon_hash(MDB.canonical_paste(p)) in train_hash for p in c["pastes"])
        summ[t] = dict(asked=c["asked"], guidance=c["guidance"], n=len(y), got=float(y.mean()),
                       se=float(y.std(ddof=1) / math.sqrt(len(y))), sd=float(y.std(ddof=1)),
                       valid=c["valid"], species_sets=len(sets), copies=copies)
    slopes = {}
    for g in GUIDE:
        cs = [c for c in out["cells"].values() if c["asked"] is not None and c["guidance"] == g]
        A_ = [np.array([v for v in c["y"] if v is not None]) for c in cs]
        asked = np.array([c["asked"] for c in cs])
        def fit(groups):
            xs = np.concatenate([np.full(len(gr), a) for a, gr in zip(asked, groups)]); ys = np.concatenate(groups)
            b, a0 = np.polyfit(xs, ys, 1); return b, a0
        b, a0 = fit(A_)
        boots = [fit([gr[rng.integers(0, len(gr), len(gr))] for gr in A_])[0] for _ in range(2000)]
        lo, hi = np.quantile(boots, [.025, .975])
        sd = float(np.mean([gr.std(ddof=1) for gr in A_]))
        step = b * 0.01
        n_step = math.ceil(2 * (1.96 + 0.84) ** 2 * sd ** 2 / step ** 2) if step > 0 else None
        slopes[f"{g:g}"] = dict(slope=float(b), intercept=float(a0), ci=[float(lo), float(hi)],
                                team_sd=sd, teams_per_arm_to_see_plus_0_01=n_step)
    out["summary"], out["slopes"] = summ, slopes
    out["training"] = dict(n=data["n"], max=data["max"], quantiles=data["quantiles"])
    json.dump(out, open(OUT, "w"))
    print(f"training labels: {data['n']} teams, max {data['max']:.3f}, median/p90/p99 {data['quantiles']}")
    print(f"{'cell':12s} {'asked':>5s} {'got':>6s} {'±se':>6s} {'valid':>6s} {'sets':>5s} {'copies':>6s}")
    for t, s in summ.items():
        a = "—" if s["asked"] is None else f"{s['asked']:.1f}"
        print(f"{t:12s} {a:>5s} {s['got']:6.3f} {s['se']:6.3f} {s['valid']:6.0%} {s['species_sets']:5d} {s['copies']:6d}")
    for g, s in slopes.items():
        print(f"w={g}: slope {s['slope']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] · intercept {s['intercept']:.3f} · "
              f"teams/arm to see +0.01: {s['teams_per_arm_to_see_plus_0_01']}")


if __name__ == "__main__":
    {"data": gather, "train": train, "gen": gen, "battle": battle, "fill": fill, "report": report}[sys.argv[1]]()
