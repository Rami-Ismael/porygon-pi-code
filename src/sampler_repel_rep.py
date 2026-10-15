"""Replication: does species repulsion on top of "free species, focused sets" confirm more winners? (2026-10-15)

The one-shot run (docs/species-repulsion.md) failed its rule, but its secondary read showed repel_settemp05
confirming 8 of 14 winning line-ups on fresh battles against 2 of 12 for settemp05 (+6 ± 3.2, one seed per
cell). The owner asked for a three-seed replication at the two asked values where the lift appeared.

Cells: {settemp05, repel_settemp05} x asked {0.5, 0.7} x seeds {1, 2, 3} = 12 cells, 64 Showdown-valid teams
each, guidance 2, alpha 256 (from results/sampler_repel_check.json), one battle per top-50 column (49 battles
per team), policy 1, origin 'repelrep:<cell>'. The "seed" is the sampling seed only (the model is fixed).
Win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy both sides.

Decision rule (docs/species-repulsion-replication.md, fixed before any battle):
  primary    confirmed winners, pooled over the 6 cells per arm: best team of every winning line-up
             (search >= 0.531) re-battled on 49 fresh battles, confirmed = fresh >= 0.531.
             REPLICATED if repel_settemp05 minus settemp05 > 2 Poisson standard errors (sqrt of the sum)
             AND the difference is >= 0 in at least 2 of 3 seeds.
  guards     mean win rate over the 6 cells, repel minus settemp05, 95% lower bound > -0.03;
             winning line-ups (pooled) not fewer than settemp05's.
  secondary  distinct line-ups per 64 by cell, QD-score (sampler_qd.py) with jackknife SE.
Anything short of REPLICATED closes the lead.

    python sampler_repel_rep.py gen
    python battle_supervisor.py sampler_repel_rep.py battle --then confirm --then report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import sampler_battle as SB
import sampler_why as SW
import sampler_qd as Q
import sampler_repel as SR

OUT = L.RES / "sampler_repel_rep.json"
ARMS = ["settemp05", "repel_settemp05"]
ASKED, SEEDS = [0.5, 0.7], [1, 2, 3]


def cells():
    return [(f"{arm}_s{s}_y{a:.1f}", arm, a) for arm in ARMS for s in SEEDS for a in ASKED]


def _route():
    SB.OUT, SB.ARMS, SB.ASKED, SB.SEED, SB.ORIGIN = OUT, ARMS, ASKED, 7373, "repelrep"
    SB.cells = cells
    a = SR._alpha()
    SB.FN.update(settemp05=lambda n: SW.arm_today_temp(n, 0.5, species_tau=1.0),
                 repel_settemp05=lambda n: SR.arm_repel(n, a, tau=0.5, species_tau=1.0))
    SR.OUT, SR.CONFIRM_NOTE = OUT, "repelrep confirm"


def _seed_of(tag):
    import re
    return int(re.search(r"_s(\d+)_y", tag).group(1))


def report():
    SB.report()                                       # per-cell table; out["arms"] = mean over the 6 cells per arm
    out = json.load(open(OUT))
    conf = out.get("confirm", {})
    pooled, by_seed = {}, {}
    for tag, c in out["cells"].items():
        if "y" not in c: continue
        ts = [(*Q.describe(p), y) for p, y in zip(c["pastes"], c["y"]) if y is not None]
        pooled.setdefault(c["arm"], []).extend(ts)
        by_seed.setdefault((c["arm"], _seed_of(tag)), []).extend(ts)
    summ = {}
    Q.table("REPLICATION, pooled 6 cells per arm (64 teams x asked 0.5/0.7 x seeds 1-3; 49 battles each)", pooled, "settemp05", summ)
    # confirmed winners: pooled (winners() picks the best team per line-up across all of an arm's cells) and per seed
    win = SR.winners(out)
    confirmed = {}
    for arm, v in win.items():
        f = [conf[str(t)]["fresh"] for _, t, _ in v if conf.get(str(t), {}).get("fresh") is not None]
        confirmed[arm] = dict(winning=len(v), confirmed=int(sum(x >= Q.BAR for x in f)),
                              fresh_mean=float(np.mean(f)) if f else None)
    per_seed = {}
    for s in SEEDS:
        sub = {"cells": {t: c for t, c in out["cells"].items() if _seed_of(t) == s}}
        for arm, v in SR.winners(sub).items():
            f = [conf[str(t)]["fresh"] for _, t, _ in v if conf.get(str(t), {}).get("fresh") is not None]
            per_seed[f"{arm}_s{s}"] = dict(winning=len(v), confirmed=int(sum(x >= Q.BAR for x in f)))
    print("\nconfirmed winners (best team per winning line-up, 49 fresh battles)")
    for arm, d in confirmed.items():
        print(f"{arm:16s} winning {d['winning']:3d}  confirmed {d['confirmed']:3d}  fresh mean {d['fresh_mean'] if d['fresh_mean'] is not None else float('nan'):.3f}  "
              + "  ".join(f"s{s}: {per_seed.get(f'{arm}_s{s}', {}).get('confirmed', 0)}/{per_seed.get(f'{arm}_s{s}', {}).get('winning', 0)}" for s in SEEDS))
    a, b = confirmed.get("repel_settemp05", {}).get("confirmed", 0), confirmed.get("settemp05", {}).get("confirmed", 0)
    se = math.sqrt(a + b) if a + b else float("nan")
    seeds_ok = sum(per_seed.get(f"repel_settemp05_s{s}", {}).get("confirmed", 0) >= per_seed.get(f"settemp05_s{s}", {}).get("confirmed", 0) for s in SEEDS)
    primary = (a - b) > 2 * se if a + b else False
    arms = out["arms"]
    d = arms["repel_settemp05"]["mean"] - arms["settemp05"]["mean"]
    dse = math.hypot(arms["repel_settemp05"]["se"], arms["settemp05"]["se"])
    strength = d - 1.96 * dse > -0.03
    lineups_ok = summ["repel_settemp05"]["winning"] >= summ["settemp05"]["winning"]
    verdict = dict(confirmed_repel=a, confirmed_settemp05=b, diff=a - b, poisson_se=se, seeds_not_below=seeds_ok,
                   primary=bool(primary and seeds_ok >= 2), strength_diff=d, strength_lower=d - 1.96 * dse, strength=strength,
                   winning_not_fewer=lineups_ok, replicated=bool(primary and seeds_ok >= 2 and strength and lineups_ok))
    print(f"\nREPLICATION VERDICT: confirmed {a} vs {b} ({a - b:+d} ± {se:.1f}), seeds not below {seeds_ok}/3, "
          f"strength {d:+.3f} lower {d - 1.96 * dse:+.3f} ({'ok' if strength else 'FAIL'}), winning not fewer {lineups_ok} "
          f"-> {'REPLICATED' if verdict['replicated'] else 'NOT REPLICATED'}")
    out["summary_qd"], out["confirmed"], out["per_seed"], out["verdict"], out["alpha"] = summ, confirmed, per_seed, verdict, SR._alpha()
    json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    st = sys.argv[1]
    _route()
    if st == "gen": SB.gen("--smoke" in sys.argv)
    elif st == "battle": SB.battle()
    elif st == "confirm": SR.confirm()
    elif st == "report": report()
