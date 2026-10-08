"""Multimodality as distinct hill-climb endpoints, one protocol on both sides.

Protocol (fixed 2026-10-01, before any VGC battle):
  start          : 24 random points (benchmarks) / 6 real meta teams, chosen with seed 5 (VGC)
  move           : first improvement; propose one random single edit (benchmarks: one coordinate to another
                   value; VGC: legality-conditioned operators A-F, Showdown-validated)
  accept         : deterministic f: strictly better. Noisy f: better by more than 1.96 x SE of the difference
                   (benchmark noise SE from 4 replicate evaluations at the start; VGC: binomial SE of the two
                   team scores)
  stop           : 40 consecutive rejected proposals, or 120 proposals
  report         : distinct endpoints (canonical identity), endpoint value spread; benchmarks also: share of
                   endpoints that are true local optima under an exhaustive single-edit check
VGC evaluation   : matchup_db, 4 battles per cell vs the 49 distinct top-50 teams (196 battles per team),
                   cached, so revisited teams cost nothing.
Run: python scripts/multimodality.py bench | vgc
"""
import copy, importlib.util, json, math, random, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "results"
OUT = RES / "multimodality.json"
STARTS, PATIENCE, CAP = 24, 40, 120
def load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def save(key, val):
    d = json.load(open(OUT)) if OUT.exists() else {}
    d[key] = val; json.dump(d, open(OUT, "w"), indent=1)

def bench():
    BF = load("bf", "bench_features.py")
    res = {}
    for name in ["contamination", "ising", "pest_combo", "pest_bounce", "maxsat60", "labs50", "ackley53"]:
        f, card, noisy = BF.TASKS[name]()
        rng = np.random.default_rng(5); sims = iter(range(10 ** 8))
        ev = lambda x: f(x, np.random.default_rng(3_000_000 + next(sims)))
        x0 = BF.rand_point(card, rng)
        sd = float(np.std([ev(x0) for _ in range(4)], ddof=1)) if noisy else 0.0
        thr = 1.96 * math.sqrt(2) * sd
        ends, vals, props = [], [], []
        for _ in range(STARTS):
            x = BF.rand_point(card, rng); fx = ev(x); fails = n = 0
            while fails < PATIENCE and n < CAP:
                n += 1; y = BF.edit(x, card, rng); fy = ev(y)
                if fy < fx - thr: x, fx, fails = y, fy, 0
                else: fails += 1
            ends.append(x); vals.append(fx); props.append(n)
        key = lambda x: tuple(np.round(np.asarray(x, float), 6))
        distinct = {key(x) for x in ends}
        def local_opt(x):                                   # exhaustive single-edit check (continuous coords skipped)
            fx = np.mean([ev(x) for _ in range(4)]) if noisy else ev(x)
            for j, c in enumerate(card):
                if c == 0: continue
                for v in range(c):
                    if v == x[j]: continue
                    y = x.copy(); y[j] = v
                    fy = np.mean([ev(y) for _ in range(4)]) if noisy else ev(y)
                    if fy < fx - (1.96 * math.sqrt(2) * sd / 2 if noisy else 0): return False
            return True
        lo = [local_opt(x) for x in ends]
        res[name] = dict(starts=STARTS, distinct_endpoints=len(distinct), true_local_optima_share=float(np.mean(lo)),
                         endpoint_value_min=float(min(vals)), endpoint_value_max=float(max(vals)),
                         mean_proposals=float(np.mean(props)), noise_sd=sd)
        print(name, res[name], flush=True)
    save("bench", res)

def vgc():
    sys.path.insert(0, "/tmp/vgc-pilot/src"); sys.path.append(str(RES.parent / "src"))
    import matchup_db as MDB
    from corpus import parse_team_text
    dsx = load("dsx", "dsame_experiment.py")
    mut = dsx.mutation_module(5); mut["rng"] = np.random.default_rng(5); rng = random.Random(5)
    con = MDB.connect(); pid = MDB.get_policy(con); val = MDB.Validator()
    top = MDB.set_members(con, "top50"); starts = rng.sample(top, 6)
    def score(t):
        MDB.ensure(con, pid, [t], top, 4, seed=5, note="multimodality climb")
        s = MDB.score(con, pid, t, "top50"); return s["score"], s["se"]
    res = []
    for st in starts:
        paste = con.execute("SELECT paste FROM team WHERE team_id=?", (st,)).fetchone()[0]
        cur, team = st, parse_team_text(paste); fx, sx = score(cur); fails = n = acc = 0
        while fails < PATIENCE and n < CAP:
            n += 1; child = copy.deepcopy(team)
            op = rng.choice([o for o in mut["COND"] if o[0] in "ABCDEF"])
            if mut["COND"][op](child): continue
            txt = mut["team_to_text"](child)
            if val(txt) is not None: continue
            with con: cid, _ = MDB.add_team(con, txt, "multimodality", val)
            if cid == cur: continue
            fy, sy = score(cid)
            if fy > fx + 1.96 * math.sqrt(sx ** 2 + sy ** 2): cur, team, fx, sx, fails, acc = cid, parse_team_text(txt), fy, sy, 0, acc + 1
            else: fails += 1
        sp = sorted(s["species"] for s in team)
        res.append(dict(start=st, end=cur, end_score=fx, proposals=n, accepted=acc, species=sp))
        print(res[-1], flush=True); save("vgc_partial", res)
    out = dict(starts=len(res), distinct_endpoints=len({r["end"] for r in res}),
               distinct_species_sets=len({tuple(r["species"]) for r in res}), endpoints=res)
    save("vgc", out); val.close(); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    {"bench": bench, "vgc": vgc}[sys.argv[1]]()
