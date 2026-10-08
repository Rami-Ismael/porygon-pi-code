"""Is there matchup structure worth conditioning on?

Every win rate so far is averaged over the whole top-50 pool, so the model has never
been told WHO it is fighting. Before building a counter-team-conditioned generator,
check the premise: does a team's win rate actually depend on the opponent, beyond
noise? If teams are simply good or bad against everyone, there is nothing to
condition on and the richer input cannot help.

Battles 300 teams (spanning the label range) against 10 individual meta teams,
24 battles each, and decomposes the variance:
    team effect      -- some teams are better than others
    opponent effect  -- some meta teams are harder than others
    interaction      -- THIS is the counter-team signal
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/tmp/vgc-pilot/src")
import pool

def main():
    nT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    nO = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    nB = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    lab = {}
    for f in ["/tmp/vgc-pilot/ladder_labels.json", "/tmp/vgc-pilot/strata_labels.json"]:
        for p, v in json.load(open(f)).items(): lab[p] = v["win_rate"]
    rng = np.random.default_rng(0)
    # stratify the teams across the label range so both good and bad teams are represented
    items = sorted(lab.items()); w = np.array([v for _, v in items])
    bins = np.clip((w * 5).astype(int), 0, 4); chosen = []
    for b in range(5):
        ix = np.where(bins == b)[0]
        chosen += [items[i][0] for i in rng.choice(ix, min(nT // 5, len(ix)), replace=False)]
    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb"); opp = []
    for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
        for c in (root / f"{i}.txt", root / "featured" / f"{i}.txt"):
            if c.exists(): opp.append(str(c)); break
    opp = opp[:nO]
    print(f"{len(chosen)} teams x {len(opp)} opponents x {nB} battles = {len(chosen)*len(opp)*nB:,} battles", flush=True)
    M = np.zeros((len(chosen), len(opp))); t0 = time.perf_counter()
    for j, o in enumerate(opp):
        res = pool.score([(t, [o]) for t in chosen], battles=nB, conc=50, quiet=True)
        for i, t in enumerate(chosen): M[i, j] = res[t]["win_rate"]
        print(f"  opponent {j+1}/{len(opp)} done ({(time.perf_counter()-t0)/60:.0f} min)", flush=True)
    json.dump({"teams": chosen, "opponents": [Path(o).stem for o in opp], "matrix": M.tolist()},
              open("/tmp/vgc-pilot/matchup.json", "w"))
    gm = M.mean(); te = M.mean(1) - gm; oe = M.mean(0) - gm
    inter = M - gm - te[:, None] - oe[None, :]
    noise = (M * (1 - M) / nB).mean()          # binomial variance per cell
    print(f"\ngrand mean {gm:.3f}")
    print(f"variance from TEAM      {te.var():.5f}  ({te.var()/M.var():.0%} of total)")
    print(f"variance from OPPONENT  {oe.var():.5f}  ({oe.var()/M.var():.0%})")
    print(f"variance from MATCHUP   {inter.var():.5f}  ({inter.var()/M.var():.0%})  <- the counter-team signal")
    print(f"  expected from battle noise alone: {noise:.5f}")
    print(f"  matchup signal beyond noise: {max(0.0, inter.var()-noise):.5f}  "
          f"({'REAL - counter-conditioning has something to learn' if inter.var() > 1.5*noise else 'NOT distinguishable from noise'})")
    best = M.argmax(0)
    print(f"\nis the best counter opponent-specific? distinct best-teams across {len(opp)} opponents: {len(set(best.tolist()))}")
    print(f"  a team that is best vs one opponent ranks (median) {np.median([np.argsort(-M[:,j]).tolist().index(best[j]) for j in range(len(opp))]):.0f} vs others")
    print("MATCHUP_DONE", flush=True)

if __name__ == "__main__":
    main()
