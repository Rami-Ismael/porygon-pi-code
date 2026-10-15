"""Guidance timing and condition noise: two cheap diversity arms on the generation-0 proposer (2026-10-15).
Pre-registration: docs/guidance-timing-cads.md.

  control   foundation.Gen.sample as cem_v3 generation 0: ask 0.5, guidance 2, temperature 1, ridge tilt, repair
  interval  guidance 1 on the six species columns, 2 on the other 42 (Gen.sample g_species=1.0)
  cads      CADS noise on the win-rate condition embedding, s = 0.1 relative, annealed over the 48 steps
Per (arm, seed 1-3): first 512 validator-accepted proposals; the 64 highest by ridge score battled, one battle
per top-50 column, behaviour-cloning policy both sides (policy 1), origin 'gtc:<arm>_s<seed>'.
Win rate = wins / battles against the top-50 meta teams.

    python guidance_timing_cads.py gen
    python battle_supervisor.py guidance_timing_cads.py battle --then report
    GTC_SMOKE=1 python guidance_timing_cads.py gen       # tiny plumbing check, own file
    GTC_TAG=species15 python guidance_timing_cads.py gen # follow-up arm (2026-10-15, second section of the doc):
                                                         # species guidance 1.5, fresh control, seed base 7100,
                                                         # results/guidance_timing_cads_species15.json, origin gtc15
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import matchup_db as MDB
import heldout as HO
from hps_generate import Validator
from corpus import parse_team_text, norm
from encode import team_fields

ARMS = ["control", "interval", "cads"]
SEEDS = [1, 2, 3]
PROPOSE, BATTLE, CUT, BATCH = 512, 64, 0.531, 96
ASK, G, TEMP, S_CADS = 0.5, 2.0, 1.0, 0.1
SEED0 = 7000
SMOKE = bool(os.environ.get("GTC_SMOKE"))
OUT = L.RES / ("guidance_timing_cads_smoke.json" if SMOKE else "guidance_timing_cads.json")
ORIGIN = "gtcsmoke" if SMOKE else "gtc"
if SMOKE: PROPOSE, BATTLE, SEEDS, BATCH = 24, 4, [1], 24
ARM_KW = {"control": {}, "interval": dict(g_species=1.0), "cads": dict(cads=S_CADS), "species15": dict(g_species=1.5)}
TAG = os.environ.get("GTC_TAG", "")
if TAG == "species15":          # the follow-up arm gets its own file, origin and seed base; the first run's file is never touched
    ARMS, SEED0 = ["control", "species15"], 7100
    OUT = L.RES / ("guidance_timing_cads_species15_smoke.json" if SMOKE else "guidance_timing_cads_species15.json")
    ORIGIN = "gtc15smoke" if SMOKE else "gtc15"
elif TAG: raise SystemExit(f"unknown GTC_TAG {TAG!r}")


def save(out): json.dump(out, open(OUT, "w"))
def lineup(slots): return tuple(sorted(norm(s["species"]) for s in slots))


# ---------------------------------------------------------------- diversity of the 512
def vendi(lineups):
    """Vendi score (Friedman & Dieng, TMLR 2023) with K_ij = shared species / 6."""
    vocab = {s: i for i, s in enumerate(sorted({s for l in lineups for s in l}))}
    B = np.zeros((len(lineups), len(vocab)))
    for i, l in enumerate(lineups):
        for s in l: B[i, vocab[s]] += 1 / math.sqrt(6)
    lam = np.linalg.eigvalsh(B @ B.T / len(lineups)); lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def diversity(stream):
    """stream: list of (paste, slots) in sampling order, the first PROPOSE accepted."""
    ls = [lineup(s) for _, s in stream]; cnt = Counter(ls); n = len(ls)
    H = -sum(c / n * math.log(c / n) for c in cnt.values())
    keys = {tuple(team_fields(s)) for _, s in stream}
    return dict(accepted=n, lineups=len(cnt), effective=math.exp(H), largest_share=max(cnt.values()) / n,
                vendi=vendi(ls), unique_48=len(keys) / n, species_used=len({s for l in ls for s in l}),
                top_lineups=[dict(species=list(l), count=c) for l, c in cnt.most_common(3)])


# ---------------------------------------------------------------- proposing
def propose(G_, sur, seed, kw):
    torch.manual_seed(SEED0 + seed); rng = np.random.default_rng(SEED0 + seed)
    val = Validator(); kept, tried, t0 = [], 0, time.perf_counter()
    try:
        while len(kept) < PROPOSE and tried < 8 * PROPOSE:
            for r in G_.sample(BATCH, ASK, G, TEMP, 1.0, sur, sur.lam, **kw).cpu().numpy():
                tried += 1
                p, slots = G_.to_paste(r, rng, repair=True)
                if val(p) is None and not HO.is_heldout_text(p): kept.append((p, slots))
                if len(kept) >= PROPOSE: break
    finally:
        val.close()
    return kept, tried, time.perf_counter() - t0


def gen():
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(arms=ARMS, seeds=SEEDS, propose=PROPOSE, battle=BATTLE,
                     ask=ASK, guidance=G, temp=TEMP, s_cads=S_CADS, seed0=SEED0, members=FD.V3_MEMBERS,
                     ridge=str(FD.RIDGE.name), kernel="shared species / 6"), "cells": {}}
    G_ = FD.Gen("v3"); sur = FD.Surrogate(G_.V)
    for seed in SEEDS:
        for arm in ARMS:
            tag = f"{arm}_s{seed}"
            if tag in out["cells"]: continue
            kept, tried, secs = propose(G_, sur, seed, ARM_KW[arm])
            mu = [float(sur.score(s)) for _, s in kept]
            order = [int(i) for i in np.argsort(mu)[::-1][:BATTLE]]
            d = diversity(kept)
            out["cells"][tag] = dict(arm=arm, seed=seed, proposals=[p for p, _ in kept], mu=mu, tried=tried,
                                     valid=len(kept) / max(tried, 1), seconds=secs, div=d,
                                     battled=[kept[i][0] for i in order], battled_mu=[mu[i] for i in order],
                                     battled_lineups=len({lineup(kept[i][1]) for i in order}))
            save(out)
            print(f"  {tag:12s} {len(kept)}/{tried} valid · line-ups {d['lineups']}/{d['accepted']} eff {d['effective']:.0f} "
                  f"largest {d['largest_share']:.1%} vendi {d['vendi']:.1f} unique48 {d['unique_48']:.2f} · "
                  f"battled line-ups {out['cells'][tag]['battled_lineups']} · {secs:.0f}s", flush=True)


# ---------------------------------------------------------------- battling
def battle():
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    for tag, c in out["cells"].items():
        if "y" in c: continue
        v = MDB.Validator()
        try: ids = [MDB.add_team(con, p, f"{ORIGIN}:{tag}", v)[0] for p in c["battled"]]
        finally: v.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=SEED0 + c["seed"], note=f"{ORIGIN} {tag}")
        sc = [MDB.score(con, LB.POLICY, t) if t in legal else None for t in ids]
        c["team_id"] = ids; c["y"] = [s["score"] if s else None for s in sc]
        y = np.array([a for a in c["y"] if a is not None], float)
        c["stats"] = dict(mean=float(y.mean()), se=float(y.std(ddof=1) / math.sqrt(len(y))), max=float(y.max()),
                          above_cut=int((y >= CUT).sum()), n=len(y))
        save(out)
        print(f"  {tag:12s} battled {len(y)}: mean {y.mean():.3f} ± {c['stats']['se']:.3f} · max {y.max():.3f} · ≥{CUT}: {c['stats']['above_cut']}", flush=True)


# ---------------------------------------------------------------- report and the rule
def report():
    out = json.load(open(OUT)); cfg = out["config"]; rng = np.random.default_rng(0)
    get = lambda a, s: out["cells"].get(f"{a}_s{s}")
    ys = lambda c: np.array([v for v in c["y"] if v is not None], float)
    boot = lambda y: y[rng.integers(0, len(y), len(y))].mean()
    print(f"{'cell':12s} {'line-ups':>8s} {'eff':>5s} {'share':>6s} {'vendi':>6s} {'uniq48':>6s} {'battled lu':>10s} {'win rate':>14s} {'max':>5s} {'≥cut':>4s}")
    summ = {}
    for a in cfg["arms"]:
        for s in cfg["seeds"]:
            c = get(a, s)
            if not c: continue
            d = c["div"]; line = f"{a + '_s' + str(s):12s} {d['lineups']:8d} {d['effective']:5.0f} {d['largest_share']:6.1%} {d['vendi']:6.1f} {d['unique_48']:6.2f} {c['battled_lineups']:10d}"
            if "y" in c:
                y = ys(c); b = [boot(y) for _ in range(4000)]; lo, hi = np.quantile(b, [.025, .975])
                line += f" {y.mean():.3f} [{lo:.3f},{hi:.3f}] {y.max():5.3f} {c['stats']['above_cut']:4d}"
                c["stats"]["ci95"] = [float(lo), float(hi)]
            print(line)
    print("\nmean over seeds")
    for a in cfg["arms"]:
        cs = [get(a, s) for s in cfg["seeds"]]
        if not all(cs): continue
        summ[a] = dict(lineups=[c["div"]["lineups"] for c in cs], vendi=[c["div"]["vendi"] for c in cs],
                       largest_share=[c["div"]["largest_share"] for c in cs], battled_lineups=[c["battled_lineups"] for c in cs],
                       win=[c["stats"]["mean"] for c in cs if "stats" in c])
        print(f"{a:9s} line-ups {np.mean(summ[a]['lineups']):6.1f} · vendi {np.mean(summ[a]['vendi']):5.1f} · share {np.mean(summ[a]['largest_share']):.1%} · "
              f"battled line-ups {np.mean(summ[a]['battled_lineups']):.1f}" + (f" · win {np.mean(summ[a]['win']):.3f}" if summ[a]["win"] else ""))
    print("\ndecision rule: line-ups > control in every seed AND bootstrap 95% lower bound of (arm - control) win rate > -0.03")
    for a in cfg["arms"]:
        if a == "control": continue
        cs = [(get(a, s), get("control", s)) for s in cfg["seeds"]]
        if not all(x and y and "y" in x and "y" in y for x, y in cs): print(f"{a:9s} not battled yet"); continue
        d_lu = [x["div"]["lineups"] - y["div"]["lineups"] for x, y in cs]
        d_vendi = [x["div"]["vendi"] - y["div"]["vendi"] for x, y in cs]
        ya, yc = [ys(x) for x, _ in cs], [ys(y) for _, y in cs]
        point = float(np.mean([p.mean() - q.mean() for p, q in zip(ya, yc)]))
        b = [np.mean([boot(p) - boot(q) for p, q in zip(ya, yc)]) for _ in range(4000)]
        lo, hi = np.quantile(b, [.025, .975])
        ok = all(x > 0 for x in d_lu) and lo > -0.03
        summ[a].update(lineups_vs_control=d_lu, vendi_vs_control=d_vendi, win_diff=point, ci=[float(lo), float(hi)], passes=bool(ok))
        print(f"{a:9s} line-ups vs control {d_lu} · vendi {[round(v, 1) for v in d_vendi]} · win {point:+.3f} [{lo:+.3f}, {hi:+.3f}] · {'PASSES' if ok else 'fails'}")
    out["summary"] = summ; save(out)


if __name__ == "__main__":
    {"gen": gen, "battle": battle, "report": report}[sys.argv[1]]()
