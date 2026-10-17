"""Flat-prior retrain, generation-0 measurement (2026-10-17). Pre-registration: docs/flat-prior-retrain.md.

Arms are two checkpoint pairs sampled by the same sampler (lossdown_battle.sample, diffusion.ORDER,
guidance 2 on the win-rate condition, temperature 1):
  control   lossdown_battle.MEMBERS, the clean mixture
  fambal    the same configs retrained with lossdown's fambal=1 (every tag_of family at equal
            expected share of each epoch's draws)
Cells: arm x asked {0.5, 0.7} x sampling seed {1, 2, 3}; the first 64 validator-accepted teams per
cell (held-out ranks 51-100 refused), 49 battles each vs the top-50 meta teams, behaviour-cloning
policy both sides (policy 1), origin 'flatprior:<cell>'. Confirm stage as sampler_qd_run.
Win rate = wins / battles against the top-50 meta teams.

    python flat_prior_run.py gen
    python battle_supervisor.py flat_prior_run.py battle --then confirm --then report
FLAT_SMOKE=1: 4 teams, seed 1, asked 0.5, own file and origin; not a result.
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import lossdown_battle as LB
import asked_vs_got as X
import activesearch as A
import matchup_db as MDB
import heldout as HO
import sampler_qd as Q
from hps_generate import Validator
from corpus import parse_team_text

SMOKE = bool(os.environ.get("FLAT_SMOKE"))
OUT = L.RES / ("flat_prior_run_smoke.json" if SMOKE else "flat_prior_run.json")
ARMS = {"control": list(LB.MEMBERS), "fambal": [m + "_fambal" for m in LB.MEMBERS]}
ASKED, SEEDS = ([0.5], [1]) if SMOKE else ([0.5, 0.7], [1, 2, 3])
G, N, NB, POLICY, PER_CELL = 2.0, (4 if SMOKE else 64), 96, 1, 1
SEED, ORIGIN = 1717, ("flatprior_smoke" if SMOKE else "flatprior")
CONFIRM_NOTE = ORIGIN + " confirm"
NAN = float("nan")


def cells():
    return [(f"{arm}_y{a:.1f}_s{s}", arm, a, s) for s in SEEDS for a in ASKED for arm in ARMS]


def mixture(members, V, tables):
    fs = []
    for m in members:
        net, cfg, _ = L.load_net(m, V)
        fs.append(L.logprobs_fn(net, tables, tuple(cfg["rules"].split(","))))
    return L.ensemble_fn(fs)


def save(out):
    json.dump(out, open(OUT, "w"))


# ---------------------------------------------------------------- stages
def gen():
    corpus, V, C, look, spreads = X.setup()
    tables = L.Tables(V, corpus)
    arms = {a: m for a, m in ARMS.items() if all((L.CK / f"{x}.pt").exists() for x in m)}
    fs = {a: mixture(m, V, tables) for a, m in arms.items()}
    out = json.load(open(OUT)) if OUT.exists() else {"arms": ARMS, "guidance": G, "n": N, "cells": {}}
    val = Validator()
    try:
        for tag, arm, a, s in cells():
            if tag in out["cells"] or arm not in fs: continue
            base = SEED + 100 * s + int(round(a * 10))            # the same state for both arms of a (asked, seed)
            rng = np.random.default_rng(base)
            pastes, rows, tried, b, held, t0 = [], [], 0, 0, 0, time.perf_counter()
            while len(pastes) < N and tried < N * 8:
                torch.manual_seed(base * 1000 + b); b += 1
                x = LB.sample(fs[arm], tables, NB, a, G).cpu().numpy()
                assert (x != 0).all(), "a [MASK] left the sampler"
                if L.DEV == "mps": torch.mps.empty_cache()
                for r in x:
                    tried += 1
                    p = A.row_to_paste(V, r, look, spreads, rng)
                    if val(p) is None:
                        if HO.is_heldout_text(p): held += 1; continue
                        pastes.append(p); rows.append(r.tolist())
                    if len(pastes) >= N: break
            out["cells"][tag] = dict(arm=arm, asked=a, seed=s, guidance=G, pastes=pastes, rows=rows, tried=tried,
                                     valid=len(pastes) / tried, heldout_refused=held, batches=b,
                                     decode_seconds=time.perf_counter() - t0)
            save(out)
            print(f"  {tag:18s} {len(pastes)}/{tried} valid ({len(pastes)/tried:.0%}) held-out refused {held} "
                  f"{b} batches {time.perf_counter()-t0:.0f}s", flush=True)
    finally:
        val.close()


def battle():
    """sampler_battle.battle() with this run's origin. Only missing cells are battled."""
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    todo = [t for t, c in out["cells"].items() if "y" not in c]
    for k in range(0, len(todo), 4):
        chunk = todo[k:k + 4]
        pastes = [p for t in chunk for p in out["cells"][t]["pastes"]]
        origins = [t for t in chunk for _ in out["cells"][t]["pastes"]]
        val = MDB.Validator()
        try:
            ids = [MDB.add_team(con, p, f"{ORIGIN}:{o}", val)[0] for p, o in zip(pastes, origins)]
        finally:
            val.close()
        legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
        MDB.ensure(con, POLICY, sorted(legal), MDB.set_members(con, "top50"), PER_CELL, seed=SEED,
                   note=f"{ORIGIN} {chunk[0]}..{chunk[-1]}")
        i = 0
        for t in chunk:
            n = len(out["cells"][t]["pastes"])
            sc = [MDB.score(con, POLICY, tid) if tid in legal else None for tid in ids[i:i + n]]
            out["cells"][t].update(team_id=ids[i:i + n], y=[s["score"] if s else None for s in sc],
                                   battles=[s["battles"] if s else 0 for s in sc])
            i += n
        save(out)
        print(f"  battled {chunk}", flush=True)


# confirm stage: sampler_qd_run.winners / fresh / confirm, keyed by cell instead of arm
def winners(out):
    """{cell: [(lineup, team_id, y)]}: the best team of every line-up in the cell that cleared the bar."""
    res = {}
    for tag, c in out["cells"].items():
        if "y" not in c: continue
        best = {}
        for p, t, y in zip(c["pastes"], c["team_id"], c["y"]):
            if y is None: continue
            l = Q.describe(p)[0]
            if l not in best or y > best[l][1]: best[l] = (t, y)
        res[tag] = [(l, t, y) for l, (t, y) in best.items() if y >= Q.BAR]
    return res


def fresh(con, team, note):
    """Win rate over the battles recorded under `note` only."""
    cols = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'")); cols.pop(team, None)
    rows = con.execute("SELECT c.lo, c.hi, SUM(c.lo_wins), SUM(c.hi_wins), SUM(c.battles) FROM contribution c JOIN batch b "
                       "USING (batch_id) WHERE b.policy_id=? AND b.note=? AND (c.lo=? OR c.hi=?) GROUP BY c.lo, c.hi",
                       (POLICY, note, team, team)).fetchall()
    w = n = 0
    for lo, hi, lw, hw, k in rows:
        c = hi if lo == team else lo
        if c in cols and k: w += lw if lo == team else hw; n += k
    return (w / n if n else None), n


def confirm():
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT))
    win = winners(out)
    ids = sorted({t for v in win.values() for _, t, _ in v})
    cols = MDB.set_members(con, "top50")
    by_target = {}
    for t in ids:                                  # ensure() only tops up a deficit: target = have + 1
        if out.get("confirm", {}).get(str(t)) is not None: continue
        have = max([MDB.battles_done(con, POLICY, t, c) for c in cols if c != t] or [0])
        by_target.setdefault(have + 1, []).append(t)
    for target, ts in sorted(by_target.items()):
        MDB.ensure(con, POLICY, ts, cols, target, seed=9191, note=CONFIRM_NOTE)
    conf = out.get("confirm", {})
    for t in ids:
        if conf.get(str(t)) is None: conf[str(t)] = dict(zip(("fresh", "fresh_battles"), fresh(con, t, CONFIRM_NOTE)))
    out["confirm"] = conf
    save(out)
    print(f"confirmed {len(ids)} teams", flush=True)


# ---------------------------------------------------------------- report
def species_entropy(pastes):
    cnt = Counter(s for p in pastes for s in Q.describe(p)[0]); n = sum(cnt.values())
    return float(math.exp(-sum(c / n * math.log(c / n) for c in cnt.values())))


def report():
    LB._stack()
    con = MDB.connect(); out = json.load(open(OUT)); rng = np.random.default_rng(0)
    corpus, V, *_ = X.setup()
    known = {r.tobytes() for r in L.load_data()["X"]}
    real = {V.encode(t).tobytes() for t in corpus}
    conf = out.get("confirm", {}); win = winners(out)
    boot = lambda y: y[rng.integers(0, len(y), len(y))].mean()
    summ = {}
    print(f"{'cell':18s} {'line-ups':>8s} {'expH':>5s} {'share':>6s} {'winning':>7s} {'QD':>6s} {'win rate':>21s} {'best':>5s} {'≥bar':>4s} {'conf':>4s} {'copies tr/real':>14s}")
    for tag, c in out["cells"].items():
        if "team_id" in c:                                  # re-read: a supervisor restart may have filled cells
            sc = [MDB.score(con, POLICY, tid) for tid in c["team_id"]]
            c["y"] = [s["score"] if s and s["score"] is not None else None for s in sc]
            c["battles"] = [s["battles"] if s else 0 for s in sc]
        teams = [(*Q.describe(p), y) for p, y in zip(c["pastes"], c.get("y", [None] * len(c["pastes"])))]
        lines = Counter(l for l, _, _ in teams)
        parsed = [parse_team_text(p) for p in c["pastes"]]
        s = dict(arm=c["arm"], asked=c["asked"], seed=c["seed"], n=len(c["pastes"]), lineups=len(lines),
                 largest_share=max(lines.values()) / len(c["pastes"]), exp_entropy=species_entropy(c["pastes"]),
                 valid=c["valid"], heldout_refused=c.get("heldout_refused", 0),
                 copies_train=sum(V.encode(t).tobytes() in known for t in parsed),
                 copies_real=sum(V.encode(t).tobytes() in real for t in parsed))
        line = f"{tag:18s} {s['lineups']:8d} {s['exp_entropy']:5.1f} {s['largest_share']:6.1%}"
        scored = [t for t in teams if t[2] is not None]
        if scored:
            q = Q.qd(scored); y = np.array([t[2] for t in scored], float)
            b = [boot(y) for _ in range(4000)]; lo, hi = np.quantile(b, [.025, .975])
            fr = [conf[str(t)]["fresh"] for _, t, _ in win.get(tag, []) if conf.get(str(t), {}).get("fresh") is not None]
            s.update(winning=q["winning"], qd=q["qd"], mean=float(y.mean()), ci95=[float(lo), float(hi)], best=float(y.max()),
                     above_bar=int((y >= Q.BAR).sum()), battles=int(sum(c["battles"])),
                     confirmed=sum(v >= Q.BAR for v in fr), confirm_fresh=fr,
                     best_fresh=max(fr) if fr else None)
            if fr:
                bt = max(win[tag], key=lambda w: conf[str(w[1])]["fresh"] if conf.get(str(w[1]), {}).get("fresh") is not None else -1)
                s["best_fresh_lineup"] = list(bt[0]); s["best_fresh_team_id"] = bt[1]
            line += (f" {q['winning']:7d} {q['qd']:6.1f} {y.mean():.3f} [{lo:.3f},{hi:.3f}] {y.max():5.3f} {s['above_bar']:4d} "
                     f"{s['confirmed']:4d}")
        line += f" {s['copies_train']:6d}/{s['copies_real']:<6d}"
        print(line); summ[tag] = s
    out["summary"] = summ
    arms = {}
    print("\nmean over cells")
    for arm in ARMS:
        cs = [summ[t] for t, _, _, _ in cells() if t in summ and "mean" in summ[t]]
        cs = [s for s in cs if s["arm"] == arm]
        if not cs: continue
        arms[arm] = dict(cells=len(cs), lineups=[s["lineups"] for s in cs], exp_entropy=[s["exp_entropy"] for s in cs],
                         winning=sum(s["winning"] for s in cs), qd=sum(s["qd"] for s in cs),
                         win=float(np.mean([s["mean"] for s in cs])), confirmed=sum(s["confirmed"] for s in cs),
                         best=max(s["best"] for s in cs), copies_real=sum(s["copies_real"] for s in cs))
        bf = [(s["best_fresh"], s.get("best_fresh_lineup"), s.get("best_fresh_team_id")) for s in cs if s.get("best_fresh") is not None]
        if bf:
            v, l, t = max(bf, key=lambda z: z[0]); arms[arm].update(best_fresh=v, best_fresh_lineup=l, best_fresh_team_id=t)
        a = arms[arm]
        print(f"{arm:9s} cells {a['cells']} · line-ups {np.mean(a['lineups']):.1f} · exp(H) {np.mean(a['exp_entropy']):.1f} · winning {a['winning']} · "
              f"QD {a['qd']:.1f} · win {a['win']:.3f} · confirmed {a['confirmed']} · best {a['best']:.3f} · best fresh "
              f"{a.get('best_fresh', float('nan')):.3f} {a.get('best_fresh_lineup')}")
    print("\nrule: (a) line-ups > control in every cell AND (b) bootstrap 95% lower bound of (fambal - control) win rate > -0.03 "
          "AND (c) confirmed winners not below control")
    if all(a in arms for a in ("control", "fambal")):
        pairs = [(summ[f"fambal_y{a:.1f}_s{s}"], summ[f"control_y{a:.1f}_s{s}"]) for s in SEEDS for a in ASKED
                 if f"fambal_y{a:.1f}_s{s}" in summ and f"control_y{a:.1f}_s{s}" in summ]
        d_lu = [x["lineups"] - y["lineups"] for x, y in pairs]
        ys = lambda tag: np.array([v for v in out["cells"][tag]["y"] if v is not None], float)
        ya = [ys(f"fambal_y{a:.1f}_s{s}") for s in SEEDS for a in ASKED]; yc = [ys(f"control_y{a:.1f}_s{s}") for s in SEEDS for a in ASKED]
        point = float(np.mean([p.mean() - q.mean() for p, q in zip(ya, yc)]))
        b = [np.mean([boot(p) - boot(q) for p, q in zip(ya, yc)]) for _ in range(4000)]
        lo, hi = np.quantile(b, [.025, .975])
        ca, cb = arms["fambal"]["confirmed"], arms["control"]["confirmed"]
        ok = (all(x > 0 for x in d_lu), lo > -0.03, ca >= cb)
        out["verdict"] = dict(lineups_vs_control=d_lu, win_diff=point, ci95=[float(lo), float(hi)],
                              confirmed=[ca, cb], clauses=[bool(x) for x in ok], passes=bool(all(ok)))
        print(f"fambal: line-ups vs control per cell {d_lu} ({'all above' if ok[0] else 'NOT all above'}) · win {point:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] ({'clears' if ok[1] else 'fails'} -0.03) · confirmed {ca} vs {cb} ({'ok' if ok[2] else 'below'}) "
              f"-> {'PASSES' if all(ok) else 'FAILS'}")
    out["arms_summary"] = arms
    save(out)


if __name__ == "__main__":
    {"gen": gen, "battle": battle, "confirm": confirm, "report": report}[sys.argv[1]]()
