"""Diffusion-BBO with the paper's full target rule (2026-10-16). Pre-registration: docs/dbbo-faithful.md, "Full paper rule".

An owner-approved exception to the 2026-08-29 decision against uncertainty acquisition. Wu, Kuang, Niu, Ma & Yu,
arXiv 2407.00610, §5.2–5.3 (Eq. 7–8), §6, App. D.2. Everything not listed here is dbbo_faithful_precise unchanged:
D starts as the 1,105 precise-label teams; every battled team gets 4 battles per top-50 column (about 196); models from
random weights (condition dropout 0.15, learning rate 1e-3, batch 256, steps matched, best-validation checkpoint);
guidance 2; 4 rounds x 128 battled; the same finalist protocol.

Each round:
  1. Train M = 5 models on D, each from its own random seed.
  2. Candidates y = w * (best label in D), w in {0.6, 0.7, 0.8, 0.9, 1.0}.
  3. For every y and model, decode N_UNC = 64 teams and compute ||x~|| per team (definition below).
     Delta_epistemic(y) = variance across the 5 models of the mean ||x~||   (Eq. 7)
     Delta_aleatoric(y) = mean across the 5 models of the variance of ||x~||  (logged only)
  4. y* = argmax of log y - log Delta_epistemic(y) (§6 takes the logarithm of both terms). The raw y - Delta_epistemic
     choice is logged alongside.
  5. The 128 battled teams are decoded at y*, split equally across the 5 models (26, 26, 26, 25, 25).

x~ (the continuous stand-in for the paper's design vector). Hard one-hot teams all have the same norm, so the paper's
App. D.2 softening is applied with the decoder's own distribution: for each of the 48 fields, at the step it is decoded,
x~_field = 0.6 * one-hot(sampled token) + 0.4 * p_field, where p_field is the legality-masked, guided predictive distribution
before sampling. Spreading the remaining 0.4 uniformly (the paper's choice) would again give every team the same norm.
||x~|| is the L2 norm of the 48 concatenated field vectors.

    python battle_supervisor.py dbbo_paper.py run --then final --then report
    DBBO_SMOKE=1 python dbbo_paper.py run     # 1 round, small sizes; checks that Delta_epistemic varies across w
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("DBBO_PRECISE", "1")              # the w = 1 port's precise settings: D0, 196 battles, matched steps
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dbbo_faithful as DF
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import matchup_db as MDB
import heldout as HO
import diversity_loops as DL
import diffusion as D
from hps_generate import Validator

# DBBO_PAPER_FIXED_W=0.6/0.7/0.8/0.9/1.0: the ablations of the w choice (2026-10-16). Identical to the full rule (5 seed models,
# precise start, 196 battles, the same 26/26/26/25/25 split, same finalists) except that w is fixed and no uncertainty is computed.
FIXED_W = float(os.environ["DBBO_PAPER_FIXED_W"]) if os.environ.get("DBBO_PAPER_FIXED_W") else None
# DBBO_PAPER_ALLW=1 (2026-10-17): a VARIANT, not the paper's rule. No w is chosen: every round battles teams at all five asks
# (26/26/26/25/25 across w = 0.6 ... 1.0, each w's teams split across the 5 models). Delta_epistemic is still logged per w.
ALLW = bool(os.environ.get("DBBO_PAPER_ALLW"))
TAG = os.environ.get("DBBO_PAPER_TAG", "dbbo_allw" if ALLW else "dbbo_paper" if FIXED_W is None else f"dbbo_paper_w{int(round(FIXED_W * 10)):02d}")
M, WS, N_UNC, ROUNDS, N_BATTLE, SOFT = 5, [0.6, 0.7, 0.8, 0.9, 1.0], 64, 4, 128, 0.6
SEED = 9393
if DF.SMOKE: TAG += "_smoke"; N_UNC, ROUNDS, N_BATTLE = 8, 1, 10
OUT = L.RES / f"{TAG}.json"
DEV, NAN = L.DEV, float("nan")


def save(out): json.dump(out, open(OUT, "w"))


@torch.no_grad()
def decode(G, y, n, g=2.0):
    """The w = 1 port's decoder (Gen.sample: fixed order, guidance g, temperature 1) that also returns ||x~|| per team."""
    w = torch.full((n,), y, device=DEV); wn = torch.full((n,), NAN, device=DEV)
    x = torch.zeros(n, L.COLS, dtype=torch.long, device=DEV); sq = torch.zeros(n, device=DEV)
    for step, c in enumerate(D.ORDER):
        t = torch.full((n,), 1.0 - step / L.COLS, device=DEV)
        lc = g * G.logp(x, t, w, c) + (1 - g) * G.logp(x, t, wn, c)
        p = torch.softmax(lc.masked_fill(~G.ok(x, c), -float("inf")), -1)
        tok = torch.multinomial(p, 1).squeeze(1); x[:, c] = tok
        soft = (1 - SOFT) * p; soft[torch.arange(n, device=DEV), tok] += SOFT
        sq += (soft ** 2).sum(1)
    return x, torch.sqrt(sq)


def models(V, tables, rows, r):
    fs, infos = [], []
    for m in range(M):
        DF.TAG, DF.SEED = f"{TAG}_m{m}", SEED + 1000 * m      # own checkpoint and own random seed per member
        f, info = DF.train(V, tables, rows, r); fs.append(f); infos.append(info)
    return fs, infos


def choose_w(G, fs, best, r):
    rows = []
    for wv in WS:
        y = wv * best; means, varis = [], []
        for m, f in enumerate(fs):
            G.f = f; torch.manual_seed(SEED + 100 * r + 10 * m)
            _, nrm = decode(G, y, N_UNC); nrm = nrm.cpu().numpy()
            means.append(float(nrm.mean())); varis.append(float(nrm.var(ddof=1)))
        ep = float(np.var(means, ddof=1)); al = float(np.mean(varis))
        rows.append(dict(w=wv, y=y, delta_epistemic=ep, delta_aleatoric=al, member_means=means,
                         score_log=math.log(y) - math.log(max(ep, 1e-300)), score_raw=y - ep))
    pick = max(rows, key=lambda d: d["score_log"]); pick_raw = max(rows, key=lambda d: d["score_raw"])
    eps = [d["delta_epistemic"] for d in rows]
    varies = (max(eps) - min(eps)) > 1e-9 * max(max(eps), 1e-12) and max(eps) > 0
    return rows, pick, pick_raw, varies


def split(n, k): return [n // k + (1 if i < n % k else 0) for i in range(k)]


def generate(G, fs, y, rng, n=None):
    shares = split(N_BATTLE if n is None else n, M)
    val = Validator(); pastes, owner, tried = [], [], 0
    try:
        for m, (f, k) in enumerate(zip(fs, shares)):
            G.f = f; got = 0; t_m = 0
            while got < k and t_m < 8 * k:
                x, _ = decode(G, y, 96)
                for row in x.cpu().numpy():
                    t_m += 1; p, _ = G.to_paste(row, rng, repair=True)
                    if val(p) is None and not HO.is_heldout_text(p): pastes.append(p); owner.append(m); got += 1
                    if got >= k: break
            tried += t_m
    finally: val.close()
    return pastes, owner, tried


def run():
    LB._stack(); con = MDB.connect()
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(models=M, w_candidates=WS, n_uncertainty=N_UNC, rounds=ROUNDS,
        n_battle=N_BATTLE, soft_onehot=SOFT, x_tilde="0.6*onehot(sampled)+0.4*p_field, 48 fields concatenated, L2 norm",
        pick=("none: all five asks battled (VARIANT, not the paper's rule)" if ALLW else "argmax log y - log Delta_epistemic"
              if FIXED_W is None else f"fixed w = {FIXED_W}"), battles_per_column=DF.PER_CELL_PRECISE, seed=SEED,
        exception="owner-approved exception to the 2026-08-29 no-uncertainty-acquisition decision"), "rounds": {}}
    G = FD.Gen("v3"); V, tables = G.V, G.tables
    if "D0" not in out: out["D0"], out["D0_info"] = DF.initial_D_precise(V); save(out)
    Dset = list(out["D0"])
    for r in range(1, ROUNDS + 1):
        c = out["rounds"].get(str(r))
        if c and "y" in c:
            Dset += [dict(paste=p, y=v, src=f"r{r}") for p, v in zip(c["battled"], c["y"]) if v is not None]; continue
        t0 = time.perf_counter(); rng = np.random.default_rng(SEED + r)
        best = max(x["y"] for x in Dset)
        fs, infos = models(V, tables, Dset, r)
        if FIXED_W is not None:                              # ablation: no uncertainty score, w fixed
            cands, pick, pick_raw, varies = [], dict(w=FIXED_W, y=FIXED_W * best), dict(w=None), True
        else:
            cands, pick, pick_raw, varies = choose_w(G, fs, best, r)
        print(f"[r{r}] best label {best:.3f} · " + " · ".join(f"w {d['w']}: Δep {d['delta_epistemic']:.2e} Δal {d['delta_aleatoric']:.2e} "
              f"score {d['score_log']:.2f}" for d in cands) + f" → w {pick['w']} (raw rule: w {pick_raw['w']})", flush=True)
        if not varies and not ALLW:                          # in the variant Delta chooses nothing, so it need not vary
            out["stopped"] = dict(round=r, reason="Delta_epistemic does not vary across w; the rule collapses to w = 1"); save(out)
            raise SystemExit("STOP: Delta_epistemic does not vary across the five w values")
        y = pick["y"]
        if ALLW:                                             # variant: battle at every candidate ask, choose nothing
            pastes, owner, tried, wof, tried_w = [], [], 0, [], {}
            for wv, k in zip(WS, split(N_BATTLE, len(WS))):
                p_, o_, t_ = generate(G, fs, wv * best, rng, k)
                pastes += p_; owner += o_; tried += t_; wof += [wv] * len(p_); tried_w[str(wv)] = (len(p_), t_)
            y = None
        else:
            pastes, owner, tried = generate(G, fs, y, rng)
        v = MDB.Validator()
        try: ids = [MDB.add_team(con, p, f"{TAG}:r{r}", v)[0] for p in pastes]
        finally: v.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), DF.PER_CELL_PRECISE, seed=SEED + r, note=f"{TAG} r{r}")
        ys = [MDB.score(con, LB.POLICY, t)["score"] if t in legal else None for t in ids]
        c = dict(best_in_D=best, n_D=len(Dset), candidates=cands, w=None if ALLW else pick["w"], w_would_pick=pick["w"],
                 w_raw_rule=pick_raw["w"], ask=y,
                 train=[None if i is None else {k: i.get(k) for k in ("best_val", "n_train", "epochs", "skipped")} for i in infos],
                 battled=pastes, owner=owner, team_id=ids, y=ys, tried=tried, valid=len(pastes) / max(tried, 1),
                 stats=DL.bstats(ys, pastes), compositions=len({DL.comp_paste(p) for p in pastes if DL.comp_paste(p)}),
                 minutes=(time.perf_counter() - t0) / 60)
        if ALLW:
            c["w_of_team"] = wof
            c["per_w"] = {str(wv): dict(ask=wv * best, n=sum(1 for a, v in zip(wof, ys) if a == wv and v is not None),
                                         mean=float(np.mean([v for a, v in zip(wof, ys) if a == wv and v is not None] or [np.nan])),
                                         legal=tried_w[str(wv)][0] / max(tried_w[str(wv)][1], 1)) for wv in WS}
        out["rounds"][str(r)] = c; save(out)
        Dset += [dict(paste=p, y=v, src=f"r{r}") for p, v in zip(pastes, ys) if v is not None]
        s = c["stats"]
        if ALLW: print(f"[r{r}] per w: " + " · ".join(f"w {k}: ask {d['ask']:.3f} → {d['mean']:.3f} (legal {d['legal']:.2f})"
                                                     for k, d in c["per_w"].items()), flush=True)
        print(f"[r{r}] w {c['w']} · ask {y if y is None else round(y, 3)} · battled {s['n']}: win rate {s['mean']:.3f} ± {s['se']:.3f} · max {s['max']:.3f} · "
              f"line-ups {c['compositions']} · legal {c['valid']:.2f} · {c['minutes']:.1f} min", flush=True)


def check():
    """Battle-free smoke: train the 5 members on D0 (1 epoch under DBBO_SMOKE) and show Delta across the five w values."""
    G = FD.Gen("v3"); V, tables = G.V, G.tables
    D0, info = DF.initial_D_precise(V); best = max(x["y"] for x in D0)
    fs, _ = models(V, tables, D0, 0)
    cands, pick, pick_raw, varies = choose_w(G, fs, best, 0)
    for d in cands:
        print(f"w {d['w']}: y {d['y']:.3f} · Δepistemic {d['delta_epistemic']:.3e} · Δaleatoric {d['delta_aleatoric']:.3e} · "
              f"member means {[round(v, 4) for v in d['member_means']]} · log score {d['score_log']:.2f} · raw score {d['score_raw']:.4f}")
    print(f"pick (log rule) w {pick['w']} · pick (raw rule) w {pick_raw['w']} · Δepistemic varies across w: {varies}")


def final():
    DF.OUT, DF.TAG = OUT, TAG
    DF.final()


def report():
    out = json.load(open(OUT)); rng = np.random.default_rng(0)
    print("win rate = wins / battles vs the top-50 meta teams, behaviour-cloned policy (policy 1) on both sides")
    for r, c in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])):
        s = c["stats"]
        if c.get("per_w"):
            print(f"round {r} (all five asks; the rule would have picked w {c['w_would_pick']}): " + " · ".join(
                f"w {k}: ask {d['ask']:.3f} → {d['mean']:.3f}, legal {d['legal']:.2f}" for k, d in c["per_w"].items()))
            continue
        print(f"round {r}: w {c['w']} (raw rule w {c['w_raw_rule']}) · ask {c['ask']:.3f} (best in D {c['best_in_D']:.3f}) · battled "
              f"{s['mean']:.3f} ± {s['se']:.3f} · max {s['max']:.3f} · legal {c['valid']:.2f} · line-ups {c['compositions']}")
        print("    " + " · ".join(f"w {d['w']}: Δep {d['delta_epistemic']:.2e} Δal {d['delta_aleatoric']:.2e}" for d in c["candidates"]))
    if out.get("heldout"):
        f = np.array([x["fresh"] for x in out["final"]]); h = np.array([x["heldout"] for x in out["heldout"]])
        print(f"finalists ({len(f)} x 192): vs top-50 {f.mean():.3f} (best {f.max():.3f}) · vs ranks 51-100 {h.mean():.3f}")
        refs = [("cem_v3_noheldout", "cem_v3_noheldout.json"), ("w = 1 port, raw labels", "dbbo_faithful.json"),
                ("w = 1 port, precise labels", "dbbo_faithful_precise.json")]
        if FIXED_W is not None or ALLW: refs.insert(0, ("full rule, uncertainty-picked w (dbbo_paper)", "dbbo_paper.json"))
        for name, file in refs:
            if not (L.RES / file).exists(): continue
            o = json.load(open(L.RES / file))
            if not o.get("heldout"): continue
            bf = np.array([x["fresh"] for x in o["final"]]); bh = np.array([x["heldout"] for x in o["heldout"]])
            d = lambda a, b: [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean() for _ in range(10000)]
            c1, c2 = np.quantile(d(f, bf), [.025, .975]), np.quantile(d(h, bh), [.025, .975])
            print(f"minus {name} ({bf.mean():.3f} / {bh.mean():.3f}): {f.mean() - bf.mean():+.3f} [{c1[0]:+.3f}, {c1[1]:+.3f}] · "
                  f"{h.mean() - bh.mean():+.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}]")


if __name__ == "__main__":
    {"run": run, "check": check, "final": final, "report": report}[sys.argv[1]]()
