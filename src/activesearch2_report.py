"""Read results/activesearch2.json and judge the two levers separately.

  lever 1 (elite CUT vs elite FRACTION): does holding the cut stop the generation-2-4
          decay that run 1 measured?  Read `cut` against `active`, and watch the
          elite cut itself: run 1's loosened 0.458 -> 0.375.

  lever 2 (MARGIN vs binary WIN as the score): two reads.  The end-to-end one is
          `margin` vs `active` and `both` vs `cut`.  The decisive one is battle-free
          and needs no arm at all: cross-validate two ridges on the pooled labelled
          teams, one trained on win rate and one on margin, and score BOTH against
          held-out WIN RATE.  If margin is the better training target, the
          margin-trained ridge ranks win rate better.

Objective f is expected win rate throughout; margin is only ever a surrogate signal.
Bootstrap intervals are over teams, 20,000 resamples.
"""
import itertools, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import parse_team, parse_team_text
import activesearch as A

RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/activesearch2.json"
RUN1 = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/activesearch.json"
B = 20_000
ARM_ORDER = ["active", "cut", "margin", "both"]


def boot_diff(a, b, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
                  for _ in range(B)])
    return a.mean() - b.mean(), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ci(d, lo, hi):
    return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("" if lo <= 0 <= hi else "  *")


def cv_target_test(teams, win, margin, folds=5, seed=0):
    """Which target trains the better ranker of WIN RATE? Battle-free."""
    F = A.Feats(teams)
    X = F.mat(teams)
    win, margin = np.asarray(win, float), np.asarray(margin, float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(teams))
    pred = {"win_rate": np.zeros(len(teams)), "margin": np.zeros(len(teams))}
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        for tgt, y in (("win_rate", win), ("margin", margin)):
            w = np.linalg.solve(X[tr].T @ X[tr] + np.diag(F.lam), X[tr].T @ y[tr])
            pred[tgt][te] = X[te] @ w
    out = {}
    for tgt in pred:
        p = pred[tgt]
        r2 = 1 - ((win - p) ** 2).sum() / ((win - win.mean()) ** 2).sum()
        top = np.argsort(-p)[:20]
        out[tgt] = dict(spearman=A.spearman(p, win),
                        r2_vs_win=float(r2) if tgt == "win_rate" else None,
                        top20_win=float(win[top].mean()))
    return out


def main():
    out = json.load(open(RESULTS))
    G, cfg = out["gens"], out["config"]
    gens = cfg["gens"]
    aw = np.array([v["win_rate"] for v in out["anchor"].values()], float)
    am = np.array([v["margin"] for v in out["anchor"].values()], float)
    rng = np.random.default_rng(0)

    n_lab = (1 + 4 * gens) * cfg["battle"]
    print(f"budget: {cfg['anchor_n']} anchor + {n_lab} labelled proposals × "
          f"{cfg['battles']} battles = {(cfg['anchor_n'] + n_lab) * cfg['battles']:,} battles")
    print(f"real corpus teams (n={len(aw)}): win rate {aw.mean():.4f} "
          f"(median {np.median(aw):.3f})  ·  margin {am.mean():.4f} "
          f"(median {np.median(am):.3f})")
    print(f"fixed cuts used by the `cut`/`both` arms: win_rate {np.median(aw):.3f}, "
          f"margin {np.median(am):.3f}\n")

    def wr(tag):
        return [v["win_rate"] for v in G[tag]["lab"]] if tag in G else []

    print(f"{'arm-gen':12s} {'win':>8s} {'se':>7s} {'margin':>7s} {'p90':>6s} {'max':>6s} "
          f"{'≥.458':>6s} {'elites':>7s} {'cut':>6s} {'prop':>5s} {'copy':>6s} {'NN':>5s}")
    order = ["gen0"] + [f"{a}_g{g}" for a in ARM_ORDER for g in range(1, gens + 1)]
    for tag in order:
        if tag not in G:
            continue
        s = G[tag]["stats"]; p = s.get("proposal") or {}
        print(f"{tag:12s} {s['mean']:8.4f} {s['se'] or 0:7.4f} "
              f"{s.get('mean_margin', 0):7.4f} {s['p90']:6.3f} {s['max']:6.3f} "
              f"{s['frac_ge_real_median']:6.1%} {s.get('elite_n', 0):7d} "
              f"{s.get('elite_cut', 0):6.3f} {s.get('elite_proposals', 0):5d} "
              f"{p.get('copy_rate', 0):6.1%} {p.get('nn_hamming', 0):5.1f}")

    print("\nLEVER 1 — fixed elite cut vs fixed elite fraction")
    print("  the cut itself, per generation (run 1 loosened 0.458 → 0.375):")
    for arm in ARM_ORDER:
        cuts = [G[f"{arm}_g{g}"]["stats"].get("elite_cut") for g in range(1, gens + 1)
                if f"{arm}_g{g}" in G]
        ns = [G[f"{arm}_g{g}"]["stats"].get("elite_n") for g in range(1, gens + 1)
              if f"{arm}_g{g}" in G]
        print(f"    {arm:8s} cut " + " → ".join(f"{c:.3f}" for c in cuts) +
              "   elites " + " → ".join(str(n) for n in ns))
    for a, b in (("cut", "active"), ("both", "margin")):
        for g in range(1, gens + 1):
            x, y = wr(f"{a}_g{g}"), wr(f"{b}_g{g}")
            if x and y:
                print(f"    g{g}  {a} − {b} = {ci(*boot_diff(x, y, rng))}")
        x = sum((wr(f"{a}_g{g}") for g in range(1, gens + 1)), [])
        y = sum((wr(f"{b}_g{g}") for g in range(1, gens + 1)), [])
        if x and y:
            print(f"    pooled  {a} − {b} = {ci(*boot_diff(x, y, rng))}")
        xl = sum((wr(f"{a}_g{g}") for g in (3, 4)), [])
        yl = sum((wr(f"{b}_g{g}") for g in (3, 4)), [])
        if xl and yl:
            print(f"    late (g3+g4, where run 1 decayed)  {a} − {b} = "
                  f"{ci(*boot_diff(xl, yl, rng))}")

    print("\nLEVER 2 — margin vs binary win as the score")
    for a, b in (("margin", "active"), ("both", "cut")):
        x = sum((wr(f"{a}_g{g}") for g in range(1, gens + 1)), [])
        y = sum((wr(f"{b}_g{g}") for g in range(1, gens + 1)), [])
        if x and y:
            print(f"    pooled  {a} − {b} = {ci(*boot_diff(x, y, rng))}")

    # pooled labelled set: anchor + every batch
    teams = [parse_team(f) for f in out["anchor"]]
    win = list(aw); marg = list(am)
    pteams, pwin, pmarg = [], [], []
    for tag in order:
        if tag not in G:
            continue
        ts = [parse_team_text(p) for p in G[tag]["pastes"]]
        teams += ts; pteams += ts
        win += [v["win_rate"] for v in G[tag]["lab"]]
        marg += [v["margin"] for v in G[tag]["lab"]]
        pwin += [v["win_rate"] for v in G[tag]["lab"]]
        pmarg += [v["margin"] for v in G[tag]["lab"]]
    print(f"\n    margin tracks win rate at Spearman {A.spearman(marg, win):+.3f} "
          f"over all {len(win)} labelled teams "
          f"({A.spearman(pmarg, pwin):+.3f} over the {len(pwin)} proposals alone)")

    for name, ts, w_, m_ in (("all labelled (real + proposals)", teams, win, marg),
                             ("proposals only", pteams, pwin, pmarg)):
        r = cv_target_test(ts, w_, m_)
        print(f"\n    5-fold CV ridge, {name} (n={len(ts)}) — both scored against "
              f"held-out WIN RATE:")
        for tgt in ("win_rate", "margin"):
            v = r[tgt]
            extra = "" if v["r2_vs_win"] is None else f"  R² {v['r2_vs_win']:+.3f}"
            print(f"      trained on {tgt:9s} Spearman {v['spearman']:+.3f}"
                  f"{extra}   top-20 mean win {v['top20_win']:.3f}")
        d = r["margin"]["spearman"] - r["win_rate"]["spearman"]
        print(f"      margin − win rate as training target: {d:+.3f} Spearman")

    print("\n2x2 CELL MEANS (win rate, pooled over 4 generations)")
    print(f"    {'':10s} {'frac':>10s} {'cut':>10s}")
    for key, row in (("win_rate", ("active", "cut")), ("margin", ("margin", "both"))):
        vals = []
        for arm in row:
            v = sum((wr(f"{arm}_g{g}") for g in range(1, gens + 1)), [])
            vals.append(np.mean(v) if v else float("nan"))
        print(f"    {key:10s} {vals[0]:10.4f} {vals[1]:10.4f}")

    print("\nLEVEL & PROVENANCE")
    allw = np.array(pwin, float)
    print(f"  {len(allw)} labelled proposals · best {allw.max():.3f} · "
          f"{(allw >= 0.458).sum()} ≥ real median · {(allw >= aw.mean()).sum()} ≥ real mean")
    best = max((t for t in order if t in G),
               key=lambda t: G[t]["stats"]["mean"])
    print(f"  best arm-generation: {best} at {G[best]['stats']['mean']:.4f} "
          f"vs real {aw.mean():.4f}")
    nn = [(G[t]['stats'].get('proposal') or {}).get('nn_hamming') for t in order if t in G]
    cp = [(G[t]['stats'].get('proposal') or {}).get('copy_rate') for t in order if t in G]
    nn = [x for x in nn if x is not None]; cp = [x for x in cp if x is not None]
    print(f"  copy rate {min(cp):.1%}–{max(cp):.1%} · NN-Hamming {min(nn):.1f}–{max(nn):.1f} "
          f"(real held-out 21.0, uniform-random 44.7)")

    if Path(RUN1).exists():
        r1 = json.load(open(RUN1))["gens"]
        a1 = [np.mean(r1[f"active_g{g}"]["y"]) for g in range(1, 5) if f"active_g{g}" in r1]
        print(f"\n  run 1 reference — same configuration as this run's `active` arm: "
              + " / ".join(f"{v:.3f}" for v in a1))


if __name__ == "__main__":
    main()
