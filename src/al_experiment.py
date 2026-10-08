"""Budget-matched test: does active learning beat random labelling for the
win-rate surrogate?

Both arms spend exactly the same battle budget labelling teams from the same
pool, then fit the same ridge (field indicators + species-pair indicators, the
feature family that scored held-out R2 0.60 on the old labels). They differ only
in WHICH teams they label:
  random : one draw of 500 teams, uniform over the pool
  active : 100 random seed, then 4 rounds x 100 picked by the current ridge -
           80 by predictive-variance with a greedy max-min diversity rule,
           20 by highest predicted win rate (so the top tail gets labels too)

Judged on a shared held-out set of 300 teams spanning the quality range
(real / j slots corrupted for j in 1,2,3,5 / fully uniform), 24 battles each.
Metrics: Spearman rank correlation, R2, and the measured win rate of the
top-20 teams each surrogate ranks first ("top-20 yield").

Battle cost: 7,200 held-out + 2 x 12,000 arms = 31,200.
"""
import itertools, json, os, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import parse_team, parse_team_text, norm, STATS
import hps_generate as HG
import pool
import top50

ROOT = Path("/tmp/vgc-pilot/al_exp")
RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/al_experiment.json"
JSONL = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/teams/hps_reg_mb_100k.jsonl"
BATTLES = 24
SEED_N, ROUND_N, ROUNDS = 100, 100, 4          # active arm: 100 + 4x100 = 500
RAND_N = SEED_N + ROUND_N * ROUNDS             # random arm: 500

# ---------------------------------------------------------------- team assembly
def corrupt(team, j, T, rng, val):
    """Replace j random slots of a real team with uniform-legal slots; Showdown-gate."""
    for _ in range(30):
        keep_idx = sorted(rng.choice(6, 6 - j, replace=False))
        kept = [team[i] for i in keep_idx]
        used_base, used_item = set(), set()
        for s in kept:
            sp = norm(s["species"])
            used_base.add(T["base_of"].get(sp, sp))
            if s["item"]: used_item.add(norm(s["item"]))
        slots = list(kept)
        tries = 0
        while len(slots) < 6 and tries < 100:
            tries += 1
            new = HG.sample_slot(T, rng, used_base, used_item)
            if new is not None: slots.append(new)
        if len(slots) < 6: continue
        txt = "\n\n".join(HG.slot_to_text(s) for s in slots) + "\n"
        if val(txt) is None: return txt
    return None

def build_teams():
    rng = np.random.default_rng(11)
    T = HG.build_tables()
    val = HG.Validator()
    corpus_files = sorted(Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb").glob("MB*.txt"))
    order = [corpus_files[i] for i in rng.permutation(len(corpus_files))]
    labelled_lines = {m["jsonl_line"] for m in
                      json.load(open("/tmp/vgc-pilot/hps_label_manifest.json")).values()}
    hps_lines = open(JSONL).readlines()
    free = [i for i in range(len(hps_lines)) if i not in labelled_lines]
    free = [free[i] for i in rng.permutation(len(free))]
    fi = iter(free)

    def uniform_txt():
        return json.loads(hps_lines[next(fi)])["team"]

    spec = {"holdout": dict(real=50, j1=50, j2=50, j3=50, j5=50, uniform=50),
            "pool":    dict(real=500, j1=700, j2=700, j3=700, j5=700, uniform=700)}
    src = iter(order)                              # disjoint draws, "real" strata only
    parsed_corpus = [t for t in (parse_team(f) for f in corpus_files) if len(t) == 6]
    out = {}
    for part, mix in spec.items():
        d = ROOT / part
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.txt"): f.unlink()
        files, meta = [], {}
        for stratum, n in mix.items():
            made = 0
            while made < n:
                if stratum == "real":
                    txt = next(src).read_text()
                elif stratum == "uniform":
                    txt = uniform_txt()
                else:
                    j = int(stratum[1])
                    base = parsed_corpus[rng.integers(0, len(parsed_corpus))]
                    txt = corrupt(base, j, T, rng, val)
                    if txt is None: continue
                p = d / f"{stratum}_{made:04d}.txt"
                p.write_text(txt)
                files.append(str(p)); meta[str(p)] = stratum; made += 1
        out[part] = (files, meta)
        print(f"{part}: {len(files)} teams", flush=True)
    val.close()
    json.dump({p: m for p, (f, m) in out.items()}, open(ROOT / "strata.json", "w"))
    return out["holdout"][0], out["pool"][0], {**out["holdout"][1], **out["pool"][1]}

# ---------------------------------------------------------------- ridge machinery
LAM_S, LAM_P, MIN_PAIR = 10.0, 30.0, 8

class Feats:
    """Fixed feature map built from the whole pool+holdout (arm-independent):
    indicators for every species/ability/item/move/nature value, plus
    species-pair indicators for pairs seen >= MIN_PAIR times."""
    def __init__(self, all_files):
        vals = {k: set() for k in ["species", "ability", "item", "move", "nature"]}
        pc = Counter()
        self.parsed = {}
        for f in all_files:
            t = parse_team(f); self.parsed[f] = t
            sps = sorted({norm(s["species"]) for s in t})
            for a, b in itertools.combinations(sps, 2): pc[(a, b)] += 1
            for s in t:
                vals["species"].add(norm(s["species"])); vals["ability"].add(norm(s["ability"]))
                if s["item"]: vals["item"].add(norm(s["item"]))
                if s["nature"]: vals["nature"].add(norm(s["nature"]))
                for m in s["moves"]: vals["move"].add(norm(m))
        self.idx, off = {}, 0
        for k in vals:
            for v in sorted(vals[k]):
                self.idx[(k, v)] = off; off += 1
        self.n_single = off
        self.pairs = {p: off + i for i, p in enumerate(
            sorted(p for p, c in pc.items() if c >= MIN_PAIR))}
        self.dim = off + len(self.pairs) + 1
        self.lam = np.concatenate([np.full(self.n_single, LAM_S),
                                   np.full(len(self.pairs), LAM_P), [0.0]])

    def vec(self, f):
        x = np.zeros(self.dim); x[-1] = 1.0
        t = self.parsed[f]
        sps = sorted({norm(s["species"]) for s in t})
        for a, b in itertools.combinations(sps, 2):
            if (a, b) in self.pairs: x[self.pairs[(a, b)]] = 1.0
        for s in t:
            for k, v in [("species", s["species"]), ("ability", s["ability"]),
                         ("item", s["item"]), ("nature", s["nature"])]:
                if v and (k, norm(v)) in self.idx: x[self.idx[(k, norm(v))]] = 1.0
            for m in s["moves"]:
                if ("move", norm(m)) in self.idx: x[self.idx[("move", norm(m))]] = 1.0
        return x

def fit(F, X, y):
    A = X.T @ X + np.diag(F.lam)
    Ainv = np.linalg.inv(A)
    w = Ainv @ (X.T @ y)
    return w, Ainv

# ---------------------------------------------------------------- labelling
def label(files, opp, tag):
    print(f"  labelling {len(files)} teams ({tag})", flush=True)
    res = pool.score([(f, opp) for f in files], battles=BATTLES, conc=50)
    return {f: res[f]["win_rate"] for f in files if f in res}

# ---------------------------------------------------------------- active arm
def active_pick(F, Xp, pool_files, labelled, w, Ainv, n_unc=80, n_top=20):
    free = [i for i, f in enumerate(pool_files) if f not in labelled]
    mu = Xp[free] @ w
    var = np.einsum("ij,jk,ik->i", Xp[free], Ainv, Xp[free])
    picked = []
    # 20 by predicted mean (top tail gets labels)
    for i in np.argsort(-mu):
        if len(picked) >= n_top: break
        picked.append(free[i])
    # 80 by predictive variance with greedy max-min diversity on the feature vector
    var_map = {free[k]: var[k] for k in range(len(free))}
    cand = sorted((i for i in free if i not in set(picked)),
                  key=lambda i: -var_map[i])[:400]
    chosen = []
    for i in cand:
        if len(chosen) >= n_unc: break
        if chosen:
            d = min(np.abs(Xp[i, :F.n_single] - Xp[j, :F.n_single]).sum() for j in chosen[-20:])
            if d < 8: continue                     # near-duplicate of a fresh pick
        chosen.append(i)
    picked += chosen[:n_unc]
    # top up if diversity rule starved the batch
    for i in cand:
        if len(picked) >= n_unc + n_top: break
        if i not in set(picked): picked.append(i)
    return [pool_files[i] for i in picked[:n_unc + n_top]]

# ---------------------------------------------------------------- metrics
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))

def evaluate(name, F, w, hold_files, Xh, yh):
    pred = Xh @ w
    r2 = 1 - ((yh - pred)**2).sum() / ((yh - yh.mean())**2).sum()
    top = np.argsort(-pred)[:20]
    m = {"spearman": spearman(pred, yh), "r2": float(r2),
         "top20_yield": float(yh[top].mean()),
         "best_possible_top20": float(np.sort(yh)[-20:].mean())}
    print(f"  {name}: spearman {m['spearman']:.3f}  r2 {m['r2']:.3f}  "
          f"top-20 yield {m['top20_yield']:.3f} (ceiling {m['best_possible_top20']:.3f})", flush=True)
    return m

# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    random.seed(0)
    hold_files, pool_files, strata = build_teams()
    F = Feats(hold_files + pool_files)
    print(f"features: {F.n_single} singles + {len(F.pairs)} species pairs", flush=True)
    Xh = np.stack([F.vec(f) for f in hold_files])
    Xp = np.stack([F.vec(f) for f in pool_files])

    opp = top50.files()                                  # all 50 incl. featured/; raises if one is missing

    yh_map = label(hold_files, opp, "held-out")
    hold_files = [f for f in hold_files if f in yh_map]
    Xh = np.stack([F.vec(f) for f in hold_files])
    yh = np.array([yh_map[f] for f in hold_files])
    print(f"held-out spread: mean {yh.mean():.3f}  min {yh.min():.3f}  max {yh.max():.3f}", flush=True)

    rng = np.random.default_rng(21)

    # ----- random arm
    rand_files = [pool_files[i] for i in rng.choice(len(pool_files), RAND_N, replace=False)]
    rand_lab = label(rand_files, opp, "random arm, 500")
    Xr = np.stack([F.vec(f) for f in rand_lab])
    wr_, _ = fit(F, Xr, np.array(list(rand_lab.values())))

    # ----- active arm
    seed_files = [pool_files[i] for i in rng.choice(len(pool_files), SEED_N, replace=False)]
    act_lab = label(seed_files, opp, "active arm, seed 100")
    rounds_meta = []
    for r in range(ROUNDS):
        Xa = np.stack([F.vec(f) for f in act_lab])
        w, Ainv = fit(F, Xa, np.array(list(act_lab.values())))
        batch = active_pick(F, Xp, pool_files, act_lab, w, Ainv)
        rounds_meta.append({"round": r + 1,
                            "picked_strata": dict(Counter(strata[f] for f in batch))})
        print(f"  round {r+1} picks: {rounds_meta[-1]['picked_strata']}", flush=True)
        act_lab.update(label(batch, opp, f"active round {r+1}, {len(batch)}"))
    Xa = np.stack([F.vec(f) for f in act_lab])
    wa, _ = fit(F, Xa, np.array(list(act_lab.values())))

    print(f"\nbudget check: random {len(rand_lab)*BATTLES}  active {len(act_lab)*BATTLES} battles", flush=True)
    res = {"random": evaluate("random", F, wr_, hold_files, Xh, yh),
           "active": evaluate("active", F, wa, hold_files, Xh, yh)}
    out = {"metrics": res, "rounds": rounds_meta,
           "budget": {"battles_per_arm": RAND_N * BATTLES, "holdout_battles": len(yh) * BATTLES},
           "holdout": {"mean": float(yh.mean()),
                       "by_stratum": {s: float(np.mean([yh_map[f] for f in hold_files
                                                        if strata[f] == s]))
                                      for s in ["real", "j1", "j2", "j3", "j5", "uniform"]}},
           "labels": {"random": {f: rand_lab[f] for f in rand_lab},
                      "active": {f: act_lab[f] for f in act_lab},
                      "holdout": yh_map},
           "strata": strata}
    json.dump(out, open(RESULTS, "w"), indent=1)
    print(f"wrote {RESULTS}  ({(time.time()-t0)/60:.0f} min total)", flush=True)

if __name__ == "__main__":
    main()
