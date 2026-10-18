"""Foundation swap: every one-shot generator method, on the old and the new model (2026-10-08).
Pre-registration: docs/foundation-swap.md.

    python foundation.py surrogate     # fit the shared ridge, pick the guidance strength
    python foundation.py gen           # 64 valid teams per (model, arm)
    python battle_supervisor.py foundation.py battle --then report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import lossdown_battle as LB
import asked_vs_got as X
import activesearch as A
import diffusion as D
import alignment as AL
import gradguide as GG
import streamliner as ST
import matchup_db as MDB
import hps_generate
from checkpoints import load_checkpoint
from hps_generate import Validator, slot_to_text
from corpus import parse_team_text, norm, STATS
from encode import NF, NSLOT

import os
SET = os.environ.get("FOUNDATION_SET", "")          # "" = the original old-vs-new run; "v3" = the regulation-vocabulary rerun
OUT = L.RES / ("foundation.json" if not SET else f"foundation_{SET}.json")
RIDGE = L.RES / "foundation_ridge_noheldout.npz"   # foundation_ridge.npz was fitted with ranks 51-100 in it (heldout.py)
MODELS = ["old", "new"] if not SET else [SET]
ARMS = ["lowtemp", "protect3", "surrogate", "guide", "repair", "stack"]
if SET: ARMS = ["base", "cfg"] + ARMS                # v3 has no earlier plain / ask-0.5 cells to reuse
SEED_ORDER = ["lowtemp", "protect3", "surrogate", "guide", "repair", "stack", "base", "cfg"]
# The members trained without ranks 51-100 (heldout.py). FOUNDATION_MEMBERS=legacy reproduces the earlier lineage,
# whose F1/G2 parents saw 50 held-out rows (8 labelled).
V3_MEMBERS = (["F1_final_medium_regmb_v3", "G2_family_matrix_noprtrain_regmb_v3"] if os.environ.get("FOUNDATION_MEMBERS") == "legacy"
              else ["F1_final_medium_noheldout_regmb_v3", "G2_family_matrix_noprtrain_noheldout_regmb_v3"])
REUSED = {"base": "uncond", "cfg": "y0.5_g2"}
N, SEED, POOL = 64, 2026, 512
NAN = float("nan")
PROTECT = {norm(m) for m in ST.PROTECT}
SPECIES_COLS = [c for c in range(L.COLS) if c % NF == 0]
LAM_OLD = 36.0                                   # gradguide naive@36, the old run's best guidance


# ---------------------------------------------------------------- the shared surrogate
def row_team(V, r):
    t = []
    for s in range(NSLOT):
        g = lambda j: V.decode_field(s * NF + j, int(r[s * NF + j]))
        t.append(dict(species=g(0), ability=g(1), item=g(2), nature=g(7),
                      moves=[g(3 + j) for j in range(4) if g(3 + j) not in ("", "[MASK]")]))
    return t


def fit_surrogate():
    corpus, V, *_ = X.setup()
    d = L.load_data()
    lab = np.where(np.isfinite(d["W"]))[0]
    teams = [row_team(V, r) for r in d["X"][lab]]; y = d["W"][lab].astype(float)
    Fe = A.Feats(teams + corpus)
    XtX = np.zeros((Fe.dim, Fe.dim)); Xty = np.zeros(Fe.dim)
    rng = np.random.default_rng(0); hold = rng.random(len(teams)) < 0.1
    for i in range(0, len(teams), 4000):
        Xb = Fe.mat(teams[i:i + 4000]); yb = y[i:i + 4000]; tr = ~hold[i:i + 4000]
        XtX += Xb[tr].T @ Xb[tr]; Xty += Xb[tr].T @ yb[tr]
    w = np.linalg.solve(XtX + np.diag(Fe.lam), Xty)
    ph = Fe.mat([t for t, h in zip(teams, hold) if h]) @ w
    rank = lambda a: np.argsort(np.argsort(a))                      # Spearman = Pearson on ranks (no scipy here)
    rho = float(np.corrcoef(rank(ph), rank(y[hold]))[0, 1])
    # guidance strength: match the species-tilt spread of the old naive@36 run
    lt, ly, _ = GG.all_labels(); Fo, wo = GG.fit_ridge(lt, ly, corpus)
    sd = lambda Fx, wx: np.std([wx[j] for (k, v), j in Fx.idx.items() if k == "species"])
    lam = LAM_OLD * sd(Fo, wo) / sd(Fe, w)
    keys = [f"{k}|{v}" for (k, v), j in sorted(Fe.idx.items(), key=lambda kv: kv[1])]
    pairs = [f"{a}|{b}" for (a, b), j in sorted(Fe.pairs.items(), key=lambda kv: kv[1])]
    np.savez(RIDGE, w=w, keys=np.array(keys), pairs=np.array(pairs), lam=lam, rho=rho, n=len(teams))
    print(f"ridge on {len(teams):,} labelled teams, dim {Fe.dim}: held-out Spearman {rho:+.3f} · "
          f"guidance lambda {lam:.1f} (old naive@36 with {len(lt)} labels)")


class Surrogate:
    def __init__(self, V):
        z = np.load(RIDGE)
        self.w, self.lam = z["w"], float(z["lam"])
        self.idx = {tuple(k.split("|", 1)): i for i, k in enumerate(z["keys"])}
        off = len(z["keys"])
        self.pairs = {tuple(p.split("|", 1)): off + i for i, p in enumerate(z["pairs"])}
        self.b = float(self.w[-1])
        # decoder-vocabulary views for guidance
        self.single = {}
        for k in ["species", "ability", "item", "move", "nature"]:
            v = np.zeros(len(V.itos[k]))
            for i, s in enumerate(V.itos[k]):
                j = self.idx.get((k, s))
                if j is not None: v[i] = self.w[j]
            self.single[k] = torch.tensor(v, dtype=torch.float32, device=L.DEV)
        S = len(V.itos["species"]); Wp = np.zeros((S, S)); sid = {s: i for i, s in enumerate(V.itos["species"])}
        for (a, b), j in self.pairs.items():
            if a in sid and b in sid: Wp[sid[a], sid[b]] = Wp[sid[b], sid[a]] = self.w[j]
        self.Wp = torch.tensor(Wp, dtype=torch.float32, device=L.DEV)

    def score(self, team):
        x = 0.0
        sps = sorted({norm(s["species"]) for s in team})
        for i, a in enumerate(sps):
            for b in sps[i + 1:]:
                j = self.pairs.get((a, b)); x += self.w[j] if j is not None else 0.0
        for s in team:
            for k, v in (("species", s["species"]), ("ability", s["ability"]), ("item", s["item"]), ("nature", s["nature"])):
                j = self.idx.get((k, norm(v))) if v else None; x += self.w[j] if j is not None else 0.0
            for m in s["moves"]:
                j = self.idx.get(("move", norm(m))); x += self.w[j] if j is not None else 0.0
        return x + self.b

    def tilt(self, V, x, c):
        """naive gradient of the ridge w.r.t. column c's one-hot, given what is decoded (gradguide 'naive')."""
        k = V.key(c); t = self.single[k].unsqueeze(0).expand(x.shape[0], -1).clone()
        if k == "species":
            for c2 in SPECIES_COLS:
                if c2 != c: t += self.Wp[x[:, c2]] * (x[:, c2] > 0).unsqueeze(1)
        return t


# ---------------------------------------------------------------- one sampler, two models
class Gen:
    def __init__(self, which):
        self.which = which
        self.corpus, self.V, self.C, self.look, self.spreads = X.setup()
        if which == "old":
            m = X.ContinuousWR(self.V).to(L.DEV); m.load_state_dict(load_checkpoint(X.CKPT, map_location=L.DEV)["sd"]); m.eval()
            self.m = m
        elif which == "v3":                                   # Codex regmb v3: full M-B vocabulary + v3 stone policy
            import regmb_model
            self.V, self.tables, self.f, self.look, self.spreads = regmb_model.model(V3_MEMBERS)
        else:
            _, self.tables, self.f, _, _ = LB.model()
        _, _, _, self.modes, self.natures = AL.build_tables(self.corpus)

    def logp(self, x, t, w, c):
        if self.which == "old":
            return F.log_softmax(self.m.logits(self.m(x, t, w), c).float(), -1)
        k = L.KEY_OF_COL[c]; return self.f(x, t, w)[k][:, L.COLS_BY_KEY[k].index(c)]

    def ok(self, x, c):
        if self.which == "old":
            return torch.stack([self.C.mask_for(c, x[b], L.DEV) for b in range(x.shape[0])])
        k = L.KEY_OF_COL[c]; o = self.tables.allowed(x)[k][:, L.COLS_BY_KEY[k].index(c)].clone()
        o[~o.any(1)] = True; o[:, 0] = False
        return o

    @torch.no_grad()
    def sample(self, n, ystar=None, g=1.0, temp=1.0, top_p=1.0, sur=None, lam=0.0, g_species=None, cads=None):
        """g_species: guidance weight on the six species columns (None = g). cads: noise scale s on the condition
        embedding, annealed gamma = step / (COLS - 1) (docs/guidance-timing-cads.md)."""
        w = torch.full((n,), NAN if ystar is None else ystar, device=L.DEV); wn = torch.full((n,), NAN, device=L.DEV)
        x = torch.zeros(n, L.COLS, dtype=torch.long, device=L.DEV)
        try:
            for step, c in enumerate(D.ORDER):
                t = torch.full((n,), 1.0 - step / L.COLS, device=L.DEV)
                L.COND_NOISE = None if cads is None else (cads, step / (L.COLS - 1))
                gc = g_species if (g_species is not None and c in SPECIES_COLS) else g
                lc = self.logp(x, t, w, c)
                if ystar is not None and gc != 1.0: lc = gc * lc + (1 - gc) * self.logp(x, t, wn, c)
                if sur is not None and lam > 0: lc = lc + lam * sur.tilt(self.V, x, c)
                lc = lc.masked_fill(~self.ok(x, c), -float("inf")) / temp
                p = torch.softmax(lc, -1)
                if top_p < 1.0:
                    ps, ix = p.sort(-1, descending=True); keep = ps.cumsum(-1) - ps < top_p
                    p = torch.zeros_like(p).scatter(1, ix, ps * keep); p = p / p.sum(-1, keepdim=True)
                x[:, c] = torch.multinomial(p, 1).squeeze(1)
        finally:
            L.COND_NOISE = None
        return x

    def to_paste(self, r, rng, repair=False):
        slots = []
        for s in range(NSLOT):
            g = lambda j: self.V.decode_field(s * NF + j, int(r[s * NF + j]))
            sp = g(0); pl = self.spreads.get(sp) or [hps_generate.sample_spread(rng)]   # new species: random legal spread (as v3)
            slot = dict(species=self.look.get(sp, sp), item=self.look.get(g(2), g(2)), ability=self.look.get(g(1), g(1)),
                        nature=self.look.get(g(7), g(7)), moves=[self.look.get(g(3 + j), g(3 + j)) for j in range(4)
                                                                 if g(3 + j) not in ("", "[MASK]")],
                        evs=pl[int(rng.integers(0, len(pl)))])
            if repair: slot["nature"] = AL.repair_nature(slot, self.modes, self.natures)
            slots.append(slot)
        return "\n\n".join(slot_to_text(s) for s in slots) + "\n", slots


def protect_count(slots):
    return sum(any(norm(m) in PROTECT for m in s["moves"]) for s in slots)


ARM_CFG = {
    "base":      dict(),
    "cfg":       dict(ystar=0.5, g=2.0),
    "lowtemp":   dict(temp=0.7, top_p=0.9),
    "protect3":  dict(filter="protect3"),
    "surrogate": dict(pick="surrogate"),
    "guide":     dict(guide=True),
    "repair":    dict(repair=True),
    "stack":     dict(temp=0.7, top_p=0.9, ystar=0.5, g=2.0, guide=True, repair=True, filter="protect3", pick="surrogate"),
}


def gen():
    out = json.load(open(OUT)) if OUT.exists() else {"cells": {}}
    val = Validator()
    try:
        for mi, which in enumerate(MODELS):
            G = Gen(which); sur = Surrogate(G.V)
            for ai, arm in enumerate(ARMS):
                tag = f"{which}:{arm}"
                if tag in out["cells"]: continue
                k = SEED_ORDER.index(arm)                    # seed by method name, so every model set shares them
                cfg = ARM_CFG[arm]; torch.manual_seed(SEED + 10 * k); rng = np.random.default_rng(SEED + 10 * k)
                want = POOL if cfg.get("pick") else N
                keep, tried, passed, t0 = [], 0, 0, time.perf_counter()
                while len(keep) < want and tried < want * 12:
                    x = G.sample(96, cfg.get("ystar"), cfg.get("g", 1.0), cfg.get("temp", 1.0), cfg.get("top_p", 1.0),
                                 sur if cfg.get("guide") else None, sur.lam if cfg.get("guide") else 0.0).cpu().numpy()
                    for r in x:
                        tried += 1
                        p, slots = G.to_paste(r, rng, cfg.get("repair", False))
                        if val(p) is not None: continue
                        passed += 1
                        if cfg.get("filter") == "protect3" and protect_count(slots) < 3: continue
                        keep.append((p, slots))
                        if len(keep) >= want: break
                if cfg.get("pick") == "surrogate":
                    sc = [sur.score(s) for _, s in keep]; order = np.argsort(sc)[::-1][:N]
                    picked = [keep[i] for i in order]; mu = [float(sc[i]) for i in order]
                else:
                    picked, mu = keep[:N], [float(sur.score(s)) for _, s in keep[:N]]
                out["cells"][tag] = dict(model=which, arm=arm, pastes=[p for p, _ in picked], mu=mu, tried=tried,
                                         valid=passed / max(tried, 1), kept=len(keep), lam=sur.lam if cfg.get("guide") else 0.0)
                json.dump(out, open(OUT, "w"))
                print(f"  {tag:18s} {len(picked)} teams · {passed}/{tried} valid ({passed/max(tried,1):.0%}) · kept {len(keep)} "
                      f"· surrogate mean {np.mean(mu):.3f} · {time.perf_counter()-t0:.0f}s", flush=True)
            del G
    finally:
        val.close()


def battle():
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    todo = [t for t, c in out["cells"].items() if "y" not in c]
    for k in range(0, len(todo), 4):
        chunk = todo[k:k + 4]
        pastes = [p for t in chunk for p in out["cells"][t]["pastes"]]
        origins = [t for t in chunk for _ in out["cells"][t]["pastes"]]
        v = MDB.Validator()
        try:
            ids = [MDB.add_team(con, p, f"foundation:{o}", v)[0] for p, o in zip(pastes, origins)]
        finally:
            v.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=SEED,
                   note=f"foundation {chunk[0]}..{chunk[-1]}")
        i = 0
        for t in chunk:
            n = len(out["cells"][t]["pastes"])
            sc = [MDB.score(con, LB.POLICY, tid) if tid in legal else None for tid in ids[i:i + n]]
            out["cells"][t].update(team_id=ids[i:i + n], y=[s["score"] if s else None for s in sc],
                                   battles=[s["battles"] if s else 0 for s in sc])
            i += n
        json.dump(out, open(OUT, "w"))
        print(f"  battled {chunk}", flush=True)


def _stats(y):
    y = np.array([v for v in y if v is not None], float)
    return float(y.mean()), float(y.std(ddof=1) / math.sqrt(len(y))), len(y)


def report():
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    reused = {"old": json.load(open(X.OUT))["cells"], "new": json.load(open(LB.OUT))["cells"]}
    known = {r.tobytes() for r in L.load_data()["X"]}
    corpus, V, *_ = X.setup()
    rows = {}
    for which in MODELS:
        for arm, cell in REUSED.items():
            c = reused[which][cell]; rows[(which, arm)] = dict(y=c["y"], pastes=c["pastes"], valid=c["valid"])
        for arm in ARMS:
            c = out["cells"].get(f"{which}:{arm}")
            if c and "team_id" in c:
                sc = [MDB.score(con, LB.POLICY, t) for t in c["team_id"]]
                c["y"] = [s["score"] if s and s["score"] is not None else None for s in sc]
                rows[(which, arm)] = dict(y=c["y"], pastes=c["pastes"], valid=c["valid"], mu=c.get("mu"))
    summ = {}
    for (which, arm), r in rows.items():
        m, se, n = _stats(r["y"])
        teams = [parse_team_text(p) for p in r["pastes"]]
        summ[f"{which}:{arm}"] = dict(model=which, arm=arm, got=m, se=se, n=n, valid=r["valid"],
                                      sets=len({tuple(sorted(s["species"] for s in t)) for t in teams}),
                                      copies=sum(V.encode(t).tobytes() in known for t in teams),
                                      mu=float(np.mean(r["mu"])) if r.get("mu") else None)
    out["summary"] = summ
    json.dump(out, open(OUT, "w"))
    print(f"{'arm':10s} {'old':>13s} {'new':>13s} {'new-old':>16s} {'gain old':>9s} {'gain new':>9s} {'interaction':>14s}")
    bo, bn = summ["old:base"], summ["new:base"]
    for arm in ["base", "cfg"] + ARMS:
        o, n_ = summ.get(f"old:{arm}"), summ.get(f"new:{arm}")
        if not o or not n_: continue
        d = n_["got"] - o["got"]; sd = math.hypot(n_["se"], o["se"])
        go, gn = o["got"] - bo["got"], n_["got"] - bn["got"]
        gi = gn - go; si = math.sqrt(n_["se"] ** 2 + bn["se"] ** 2 + o["se"] ** 2 + bo["se"] ** 2)
        print(f"{arm:10s} {o['got']:.3f}±{o['se']:.3f}  {n_['got']:.3f}±{n_['se']:.3f}  {d:+.3f} ({d/sd:+.1f}σ)  "
              f"{go:+9.3f} {gn:+9.3f}  {gi:+.3f} ({gi/si:+.1f}σ)")


def report_v3():
    """v3 cells next to the old and new models from the original run. Copies are checked by field
    NAMES (team_fields), never by encoding v3 teams with the corpus vocabulary (the silent-[MASK] bug)."""
    from encode import team_fields
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    prev = json.load(open(L.RES / "foundation.json"))["summary"]
    corpus, V, *_ = X.setup(); d = L.load_data()
    known = {tuple(V.decode_field(c, int(r[c])) for c in range(L.COLS)) for r in d["X"]}
    corpus_species = set(V.itos["species"])
    summ = {}
    for arm in ARMS:
        c = out["cells"].get(f"{SET}:{arm}")
        if not c or "team_id" not in c: continue
        sc = [MDB.score(con, LB.POLICY, t) for t in c["team_id"]]
        c["y"] = [s["score"] if s and s["score"] is not None else None for s in sc]
        m, se, n = _stats(c["y"])
        teams = [parse_team_text(p) for p in c["pastes"]]
        summ[f"{SET}:{arm}"] = dict(model=SET, arm=arm, got=m, se=se, n=n, valid=c["valid"],
                                    sets=len({tuple(sorted(norm(s["species"]) for s in t)) for t in teams}),
                                    copies=sum(tuple(team_fields(t)) in known for t in teams),
                                    new_species_teams=sum(any(norm(s["species"]) not in corpus_species for s in t) for t in teams),
                                    mu=float(np.mean(c["mu"])) if c.get("mu") else None)
    out["summary"] = summ
    json.dump(out, open(OUT, "w"))
    print(f"{'arm':10s} {'old':>7s} {'new':>7s} {'v3':>13s} {'v3 - new':>16s} {'valid':>6s} {'sets':>5s} {'new-species teams':>18s}")
    for arm in ARMS:
        v = summ.get(f"{SET}:{arm}")
        if not v: continue
        o, nw = prev.get(f"old:{arm}"), prev.get(f"new:{arm}")
        d_ = v["got"] - nw["got"]; sd = math.hypot(v["se"], nw["se"])
        print(f"{arm:10s} {o['got']:7.3f} {nw['got']:7.3f} {v['got']:.3f}±{v['se']:.3f} {d_:+.3f} ({d_/sd:+.1f}σ) "
              f"{v['valid']:6.0%} {v['sets']:5d} {v['new_species_teams']:12d}/{v['n']}")


if __name__ == "__main__":
    {"surrogate": fit_surrogate, "gen": gen, "battle": battle, "report": report_v3 if SET else report}[sys.argv[1]]()
