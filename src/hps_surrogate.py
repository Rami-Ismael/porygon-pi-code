"""Surrogate on the 100k HPS dataset itself: can team features predict the
win rate of uniform-legal teams, and does active learning help there?

Uses the 2,000 already-labelled HPS teams (results/hps_labels.json, 24 battles
each vs the top-50 meta pool) — no new battles. Three questions:

  1. fit    : ridge (field indicators + species-pair indicators, the family
              from al_experiment.py) on 1,500 HPS teams, judged on 500 held out.
  2. active : simulated budget-matched active learning inside the 1,500 —
              random 500 labels vs 100 seed + 4x100 picked by predictive
              variance + diversity + top-mean (labels revealed from the file).
  3. transfer: the ladder-pool surrogate (al_experiment random arm, 500 labels,
              team files still in /tmp/vgc-pilot/al_exp) predicting the 2,000
              HPS teams — does quality-spread training transfer down here?

Labels are near-degenerate (mean 0.015, 73% exact zeros), so alongside
Spearman/R2 we report AUC for "won at least 1 of 24" and top-20 yield.
Writes results/hps_surrogate.json.
"""
import itertools, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot")
sys.path.insert(0, str(REPO / "src"))
from corpus import parse_team_text, norm

JSONL = REPO / "teams/hps_reg_mb_100k.jsonl"
RESULTS = REPO / "results/hps_surrogate.json"
LAM_S, LAM_P, MIN_PAIR = 10.0, 30.0, 8
SEED_N, ROUND_N, ROUNDS = 100, 100, 4

# ---------------------------------------------------------------- data
def load_hps():
    man = json.load(open(REPO / "results/hps_label_manifest.json"))
    lab = json.load(open(REPO / "results/hps_labels.json"))
    lines = open(JSONL).readlines()
    teams, y, wins = [], [], []
    for f, m in man.items():
        if f not in lab: continue
        t = parse_team_text(json.loads(lines[m["jsonl_line"]])["team"])
        if len(t) != 6: continue
        teams.append(t); y.append(lab[f]["win_rate"]); wins.append(lab[f]["wins"])
    return teams, np.array(y), np.array(wins)

# ---------------------------------------------------------------- features
class Feats:
    """Same family as al_experiment.Feats, keyed by parsed team index."""
    def __init__(self, parsed_teams, min_pair=MIN_PAIR):
        vals = {k: set() for k in ["species", "ability", "item", "move", "nature"]}
        pc = Counter()
        for t in parsed_teams:
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
            sorted(p for p, c in pc.items() if c >= min_pair))}
        self.dim = off + len(self.pairs) + 1
        self.lam = np.concatenate([np.full(self.n_single, LAM_S),
                                   np.full(len(self.pairs), LAM_P), [0.0]])

    def vec(self, t):
        x = np.zeros(self.dim); x[-1] = 1.0
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
    return Ainv @ (X.T @ y), Ainv

# ---------------------------------------------------------------- metrics
def ranks(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]: j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r

def spearman(a, b):
    ra, rb = ranks(a) - ranks(a).mean(), ranks(b) - ranks(b).mean()
    return float((ra * rb).sum() / np.sqrt((ra**2).sum() * (rb**2).sum()))

def auc(score, pos):
    r = ranks(score)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    return float((r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))

def evaluate(pred, y, wins):
    top = np.argsort(-pred)[:20]
    return {"spearman": spearman(pred, y),
            "r2": float(1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum()),
            "auc_win1": auc(pred, wins >= 1),
            "top20_yield": float(y[top].mean()),
            "pool_mean": float(y.mean()),
            "best_possible_top20": float(np.sort(y)[-20:].mean())}

# ---------------------------------------------------------------- active (simulated)
def active_pick(F, Xp, labelled, w, Ainv, n_unc=80, n_top=20):
    free = [i for i in range(len(Xp)) if i not in labelled]
    mu = Xp[free] @ w
    var = np.einsum("ij,jk,ik->i", Xp[free], Ainv, Xp[free])
    picked = [free[i] for i in np.argsort(-mu)[:n_top]]
    cand = [free[i] for i in np.argsort(-var)][:400]
    chosen = []
    for i in cand:
        if len(chosen) >= n_unc: break
        if i in picked: continue
        if chosen:
            d = min(np.abs(Xp[i, :F.n_single] - Xp[j, :F.n_single]).sum() for j in chosen[-20:])
            if d < 8: continue
        chosen.append(i)
    picked += chosen
    for i in cand:
        if len(picked) >= n_unc + n_top: break
        if i not in picked: picked.append(i)
    return picked[:n_unc + n_top]

# ---------------------------------------------------------------- main
def main():
    teams, y, wins = load_hps()
    print(f"HPS labelled teams: {len(teams)}  mean wr {y.mean():.3f}  "
          f"zeros {(wins==0).mean():.0%}  max {y.max():.3f}", flush=True)
    F = Feats(teams)
    print(f"features: {F.n_single} singles + {len(F.pairs)} species pairs", flush=True)
    X = np.stack([F.vec(t) for t in teams])

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(teams))
    hold, pool_idx = perm[:500], perm[500:]
    Xh, yh, wh = X[hold], y[hold], wins[hold]
    Xp, yp = X[pool_idx], y[pool_idx]

    out = {"n": len(teams), "label_mean": float(y.mean()),
           "zeros_frac": float((wins == 0).mean()),
           "features": {"singles": F.n_single, "pairs": len(F.pairs)}}

    # 1. plain fit, all 1,500 pool labels
    w_all, _ = fit(F, Xp, yp)
    out["fit_1500"] = evaluate(Xh @ w_all, yh, wh)
    print("fit 1500:", out["fit_1500"], flush=True)

    # 2. budget-matched arms inside the pool (labels revealed from file)
    rand = rng.choice(len(Xp), SEED_N + ROUND_N * ROUNDS, replace=False)
    w_r, _ = fit(F, Xp[rand], yp[rand])
    out["random_500"] = evaluate(Xh @ w_r, yh, wh)
    print("random 500:", out["random_500"], flush=True)

    labelled = set(rng.choice(len(Xp), SEED_N, replace=False).tolist())
    rounds = []
    for r in range(ROUNDS):
        li = sorted(labelled)
        w, Ainv = fit(F, Xp[li], yp[li])
        batch = active_pick(F, Xp, labelled, w, Ainv)
        rounds.append({"round": r + 1, "picked_mean_wr": float(yp[batch].mean())})
        labelled |= set(batch)
    li = sorted(labelled)
    w_a, _ = fit(F, Xp[li], yp[li])
    out["active_500"] = evaluate(Xh @ w_a, yh, wh)
    out["active_rounds"] = rounds
    print("active 500:", out["active_500"], flush=True)

    # bootstrap difference active - random (spearman on holdout)
    pr, pa = Xh @ w_r, Xh @ w_a
    diffs = []
    for _ in range(2000):
        b = rng.integers(0, len(yh), len(yh))
        diffs.append(spearman(pa[b], yh[b]) - spearman(pr[b], yh[b]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out["active_minus_random_spearman"] = {"mean": float(np.mean(diffs)),
                                           "ci95": [float(lo), float(hi)]}

    # 3. feature story: fit on all 2,000, top coefficients
    w_full, _ = fit(F, X, y)
    names = [None] * F.n_single
    for (k, v), i in F.idx.items(): names[i] = f"{k}:{v}"
    coef = w_full[:F.n_single]
    top_pos = [(names[i], round(float(coef[i]), 4)) for i in np.argsort(-coef)[:15]]
    top_neg = [(names[i], round(float(coef[i]), 4)) for i in np.argsort(coef)[:15]]
    out["top_features"] = {"positive": top_pos, "negative": top_neg}
    print("top +:", top_pos[:6], flush=True)
    print("top -:", top_neg[:6], flush=True)

    # 4. transfer: ladder-pool surrogate -> HPS teams
    al = json.load(open(REPO / "results/al_experiment.json"))
    lad_files = [f for f in al["labels"]["random"] if os.path.exists(f)]
    if len(lad_files) >= 400:
        lad_teams = [parse_team_text(open(f).read()) for f in lad_files]
        lad_y = np.array([al["labels"]["random"][f] for f in lad_files])
        FJ = Feats(lad_teams + teams)
        XL = np.stack([FJ.vec(t) for t in lad_teams])
        XH = np.stack([FJ.vec(t) for t in teams])
        w_t, _ = fit(FJ, XL, lad_y)
        out["transfer_ladder_to_hps"] = evaluate(XH @ w_t, y, wins)
        out["transfer_ladder_to_hps"]["n_train"] = len(lad_files)
        print("transfer:", out["transfer_ladder_to_hps"], flush=True)
    else:
        out["transfer_ladder_to_hps"] = None
        print("transfer skipped: ladder team files gone", flush=True)

    json.dump(out, open(RESULTS, "w"), indent=1)
    print(f"wrote {RESULTS}", flush=True)

if __name__ == "__main__":
    main()
