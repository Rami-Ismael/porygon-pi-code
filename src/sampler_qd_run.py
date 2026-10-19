"""Samplers vs quality-diversity on the clean (no held-out) F1 + G2 mixture (2026-10-14).

Owner's question: does a different sampler find a WIDE range of different teams that each beat the
top-50 meta teams? sampler_qd.py answered it on the 2026-10-11 runs, but those used the F1/G2
checkpoints that were moved to _contaminated_heldout/ on 2026-10-14, so entropy-bounded unmasking
(the one listed sampler never battled) cannot be compared with them. All arms run again here, on
the clean mixture (lossdown_battle.MEMBERS), in one protocol:
  today      one field per pass in diffusion.ORDER (lossdown_battle.sample)
  planning   Path Planning, self-planned (sampler_battle.arm_planning)
  remask     remasking loop, species never re-masked (sampler_battle.arm_remask)
  settemp05  today's order, temperature 0.5 on the sets, 1 on the species (sampler_why) -- the best
             QD arm on the 2026-10-11 data
  eb_conf1   entropy-bounded unmasking (sampler_eb), lowest-entropy fields first, gamma 1 nat:
             near-sequential (~45 passes per row), the paper's small-gamma regime
  eb_conf4   the same at gamma 4 nats: ~20 passes per row, several fields per model call
  eb_order4  entropy-bounded unmasking along today's order, gamma 4: ~15 passes per row
  (gamma 0.5 was dropped after a decode-only check: a team carries ~42 nats, so 0.5 reveals one field
  per pass almost everywhere and reduces to an adaptive order)
Protocol = sampler_battle.py: asked {0.1 .. 0.9}, guidance 2, the first 64 Showdown-valid teams per
cell, one battle per top-50 column (49 battles), policy 1, origin 'samplerqd:<cell>'.
Win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy both sides.

Quality-diversity (niche = species line-up, bar 0.531 = a real top-50 team vs the other 49):
  winning line-ups   distinct line-ups whose best team scores >= 0.531
  QD-score           sum over line-ups of the best team's win rate
Decision rule, written before any battle: each arm minus today, jackknife standard errors (sampler_qd.py);
beyond 2 standard errors is a difference.

Confirm stage (winner's curse): a team picked because it cleared 0.531 on 49 battles is biased up.
The best team of every winning line-up gets one more battle per top-50 column (49 fresh battles);
"confirmed" = the fresh battles alone score >= 0.531. A team at exactly the bar passes half the time,
so confirmed counts are conservative; they are compared between arms, not read as absolute.

    python sampler_qd_run.py gen
    python battle_supervisor.py sampler_qd_run.py battle --then confirm --then report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lossdown as L
import lossdown_battle as LB
import matchup_db as MDB
import sampler_battle as SB
import sampler_why as SW
import sampler_eb as SE
import sampler_qd as Q
import pool

# Own servers (servers-own copies, started 2026-10-14 on 8150-8156): the shared 8123-8129 servers had run
# ~24 h for another job and finished one battle job in 90 s across seven shards. Patched here, so
# matchup_db.run_jobs (which calls pool.live_ports()) sends this run's battles only to these ports.
OWN_PORTS = list(range(8150, 8157))
_live = pool.live_ports
pool.live_ports = lambda ports=OWN_PORTS: _live(ports)

OUT = L.RES / "sampler_qd_run.json"
ARMS = ["today", "planning", "remask", "settemp05", "eb_conf1", "eb_conf4", "eb_order4"]
CONFIRM_NOTE = "samplerqd confirm"


def _route():
    SB.OUT, SB.ARMS, SB.SEED, SB.ORIGIN = OUT, ARMS, 7272, "samplerqd"
    SB.FN.update(settemp05=lambda n: SW.arm_today_temp(n, 0.5, species_tau=1.0),
                 eb_conf1=lambda n: SE.arm_eb(n, False, 1.0), eb_conf4=lambda n: SE.arm_eb(n, False, 4.0),
                 eb_order4=lambda n: SE.arm_eb(n, True, 4.0))


def winners(out):
    """{arm: [(lineup, team_id, y)]}: the best team of every line-up that cleared the bar."""
    res = {}
    for c in out["cells"].values():
        if "y" not in c: continue
        for p, t, y in zip(c["pastes"], c["team_id"], c["y"]):
            if y is None: continue
            l = Q.describe(p)[0]
            cur = res.setdefault(c["arm"], {}).get(l)
            if cur is None or y > cur[1]: res[c["arm"]][l] = (t, y)
    return {a: [(l, t, y) for l, (t, y) in d.items() if y >= Q.BAR] for a, d in res.items()}


def fresh(con, team, note):
    """Win rate over the battles recorded under `note` only (dsame_experiment.fresh_score)."""
    cols = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'")); cols.pop(team, None)
    rows = con.execute("SELECT c.lo, c.hi, SUM(c.lo_wins), SUM(c.hi_wins), SUM(c.battles) FROM contribution c JOIN batch b "
                       "USING (batch_id) WHERE b.policy_id=? AND b.note=? AND (c.lo=? OR c.hi=?) GROUP BY c.lo, c.hi",
                       (SB.POLICY, note, team, team)).fetchall()
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
        have = max([MDB.battles_done(con, SB.POLICY, t, c) for c in cols if c != t] or [0])
        if out.get("confirm", {}).get(str(t)) is None:
            by_target.setdefault(have + 1, []).append(t)
    for target, ts in sorted(by_target.items()):
        MDB.ensure(con, SB.POLICY, ts, cols, target, seed=9191, note=CONFIRM_NOTE)
    out["confirm"] = {str(t): dict(zip(("fresh", "fresh_battles"), fresh(con, t, CONFIRM_NOTE))) for t in ids}
    json.dump(out, open(OUT, "w"))
    print(f"confirmed {len(ids)} teams", flush=True)


def report():
    out = json.load(open(OUT))
    arms = {}
    for c in out["cells"].values():
        if "y" not in c: continue
        arms.setdefault(c["arm"], []).extend((*Q.describe(p), y) for p, y in zip(c["pastes"], c["y"]) if y is not None)
    summ = {}
    Q.table("CLEAN MODEL, ONE-SHOT (64 teams x asked 0.1/0.3/0.5/0.7/0.9 per arm; 49 battles each)", arms, "today", summ)
    conf = out.get("confirm", {})
    if conf:
        print("\nconfirmation: best team of each winning line-up, 49 fresh battles")
        print(f"{'arm':11s} {'winning':>7s} {'confirmed':>9s} {'mean search':>11s} {'mean fresh':>10s}")
        for a, v in winners(out).items():
            f = [conf[str(t)]["fresh"] for _, t, _ in v if conf.get(str(t), {}).get("fresh") is not None]
            k = sum(x >= Q.BAR for x in f)
            summ[a]["confirmed"] = k; summ[a]["confirm_fresh_mean"] = float(np.mean(f)) if f else None
            print(f"{a:11s} {len(v):7d} {k:9d} {np.mean([y for *_, y in v]) if v else float('nan'):11.3f} "
                  f"{np.mean(f) if f else float('nan'):10.3f}")
        b = summ["today"]["confirmed"]
        for a, s in summ.items():
            if a != "today":
                se = math.sqrt(s["confirmed"] + b)                    # Poisson counts
                print(f"  {a:11s} confirmed {s['confirmed'] - b:+d} ± {se:.1f} vs today")
    out["summary_qd"] = summ
    json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    st = sys.argv[1]
    _route()
    if st == "gen":
        SB.gen()
        for a, v in SE.PASSES.items(): print(f"{a}: model calls per batch mean {np.mean(v):.1f} (today: 48)")
    elif st == "battle": SB.battle()
    elif st == "confirm": confirm()
    elif st == "report": report()
