"""Do samplers raise a quality-diversity score? Re-reads stored battles only (2026-10-14).

Question (owner, 2026-10-14): can a different sampler (Path Planning, the remasking loop, entropy-
bounded unmasking, ...) find a WIDE range of different teams that each beat the meta threat pool?

The 2026-10-11 runs (sampler_battle.json, sampler_why.json, cem_planning.json) scored strength and
counted line-ups separately. This script joins the two into quality-diversity numbers. No new battles.

Niche = species line-up (six species, sorted). For a set of battled teams:
  winning line-ups   distinct line-ups whose best team scores >= 0.531 (a real top-50 team's win rate
                     against the other 49) -- the headline number
  QD-score           sum over line-ups of the best team's win rate
  ratio coverage     occupied cells of the static offense/bulk axis (DSA-ME bin edges, 12 cells)
Win rate = wins / battles against the top-50 meta teams, behaviour-cloned policy both sides,
49 battles per team in the one-shot runs.

Equal budget: one-shot arms compare 320 teams each (64 per asked value x 5 asked values);
search loops compare the 128 battled teams of generations 1-3 (384 teams) on seed 1.
Uncertainty: leave-one-team-out jackknife standard error; arms are independent samples, so the
standard error of a difference is the root sum of squares. Beyond 2 standard errors = a difference.

    /tmp/vgc-pilot/.venv/bin/python sampler_qd.py
"""
import json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
import matchup_db as MDB
from corpus import parse_team_text, dex_entry, norm

RES = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results")
BAR, GRID = 0.531, 12
EDGES = json.load(open(RES / "dsame_experiment.json"))["pilot"]["edges"]["ratio"]
NAT = {"adamant": ("atk", "spa"), "bold": ("def", "atk"), "brave": ("atk", "spe"), "calm": ("spd", "atk"),
       "careful": ("spd", "spa"), "gentle": ("spd", "def"), "hasty": ("spe", "def"), "impish": ("def", "spa"),
       "jolly": ("spe", "spa"), "lax": ("def", "spd"), "lonely": ("atk", "def"), "mild": ("spa", "def"),
       "modest": ("spa", "atk"), "naive": ("spe", "spd"), "naughty": ("atk", "spd"), "quiet": ("spa", "spe"),
       "rash": ("spa", "spd"), "relaxed": ("def", "spe"), "sassy": ("spd", "spe"), "timid": ("spe", "atk")}
SPK = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


def entry(species):
    """dex entry; a cosmetic forme (Vivillon-Polar, Alcremie-Matcha-Cream) is its base species."""
    e = dex_entry(species)
    return dex_entry(e["baseSpecies"]) if e.get("isCosmeticForme") else e


def ob_ratio(team):
    """dsame_experiment.ob_ratio: mean over slots of ln(max(Atk,SpA)) - 1/2 ln(HP*(Def+SpD)/2)."""
    vals = []
    for s in team:
        b = entry(s["species"])["baseStats"]
        up, dn = NAT.get(norm(s.get("nature") or ""), (None, None))
        st = {}
        for k in SPK:
            sp = int((s.get("evs") or {}).get(SPK[k], 0))
            st[k] = b[k] + sp + 75 if k == "hp" else math.floor((b[k] + sp + 20) * (1.1 if k == up else 0.9 if k == dn else 1.0))
        vals.append(math.log(max(st["atk"], st["spa"])) - 0.5 * math.log(st["hp"] * (st["def"] + st["spd"]) / 2))
    return float(np.mean(vals))


def describe(paste):
    tm = parse_team_text(paste)
    return tuple(sorted(norm(entry(s["species"])["name"]) for s in tm)), int(np.clip(np.digitize(ob_ratio(tm), EDGES), 0, GRID - 1))


def qd(teams):
    """teams = [(lineup, ratio_cell, y)] -> dict of the three numbers."""
    best = {}
    for l, _, y in teams: best[l] = max(best.get(l, -1.0), y)
    return dict(n=len(teams), lineups=len(best), winning=sum(v >= BAR for v in best.values()),
                qd=float(sum(best.values())), mean=float(np.mean([t[2] for t in teams])),
                top=float(max(t[2] for t in teams)), ratio_cells=len({c for *_, c in teams}),
                winning_teams=sum(t[2] >= BAR for t in teams))


def jack(teams):
    """Leave-one-team-out jackknife standard errors of winning line-ups and QD-score.
    (A with-replacement bootstrap duplicates teams and so undercounts distinct line-ups.)"""
    n, full = len(teams), qd(teams)
    se = {}
    for k in ("winning", "qd", "lineups"):
        th = np.array([qd(teams[:i] + teams[i + 1:])[k] for i in range(n)], float)
        se[k] = float(math.sqrt((n - 1) / n * ((th - th.mean()) ** 2).sum()))
    return se


def one_shot():
    arms = {}
    for f in ("sampler_battle.json", "sampler_why.json"):
        d = json.load(open(RES / f))
        for tag, c in d["cells"].items():
            ts = [(*describe(p), y) for p, y in zip(c["pastes"], c["y"]) if y is not None]
            arms.setdefault(c["arm"], {})[c["asked"]] = ts
    return arms


def loops(con):
    arms = {}
    dm = json.load(open(RES / "diversity_methods_all.json"))["gens"]
    cp = json.load(open(RES / "cem_planning.json"))["gens"]
    src = {"control": dm, "taboo": dm, "plan": cp, "plan_taboo": cp, "sets": cp, "sets_taboo": cp}
    for arm, gens in src.items():
        cells_ = []
        for g in (1, 2, 3):
            r = gens[f"{arm}_s1_g{g}"]
            ids = r["team_id"]
            paste = dict(con.execute(f"SELECT team_id, paste FROM team WHERE team_id IN ({','.join('?' * len(ids))})", ids))
            cells_.append([(*describe(paste[t]), y) for t, y in zip(ids, r["y"]) if y is not None])
        arms[arm] = cells_
    return arms


def table(title, arms, base, out):
    """arms = {arm: team list}; prints and stores each arm plus its difference from `base`."""
    print(f"\n{title}\n{'arm':11s} {'mean':>6s} {'top':>5s} {'line-ups':>8s} {'winning line-ups':>17s} {'QD-score':>13s} {'ratio cells':>11s}")
    for arm, ts in arms.items():
        s = qd(ts); s["se"] = jack(ts); out[arm] = s
        print(f"{arm:11s} {s['mean']:6.3f} {s['top']:5.2f} {s['lineups']:8d} {s['winning']:8d} ± {s['se']['winning']:4.1f}"
              f" {s['qd']:7.1f} ± {s['se']['qd']:3.1f} {s['ratio_cells']:11d}")
    print(f"difference from {base} (± standard error)")
    for arm, s in out.items():
        if arm == base: continue
        b = out[base]
        s["vs_base"] = {k: dict(diff=s[k] - b[k], se=math.hypot(s["se"][k], b["se"][k])) for k in ("winning", "qd", "lineups")}
        v = s["vs_base"]
        print(f"{arm:11s} winning line-ups {v['winning']['diff']:+4d} ± {v['winning']['se']:4.1f}   "
              f"QD-score {v['qd']['diff']:+6.1f} ± {v['qd']['se']:4.1f}   line-ups {v['lineups']['diff']:+4d} ± {v['lineups']['se']:4.1f}")


def main():
    out = {"bar": BAR, "one_shot": {}, "one_shot_by_ask": {}, "loops": {}}
    os_ = one_shot()
    table("ONE-SHOT (320 teams per arm = 64 x asked 0.1/0.3/0.5/0.7/0.9; 49 battles each)",
          {arm: [t for a in sorted(by) for t in by[a]] for arm, by in os_.items()}, "today", out["one_shot"])
    print("\nper asked value: winning line-ups / line-ups / mean win rate")
    for arm, by in os_.items():
        out["one_shot_by_ask"][arm] = {str(a): qd(by[a]) for a in sorted(by)}
        print(f"{arm:11s} " + "  ".join(f"{a}:{s['winning']:2d}/{s['lineups']:2d}/{s['mean']:.2f}"
                                        for a, s in out["one_shot_by_ask"][arm].items()))
    con = MDB.connect()
    table("SEARCH LOOPS (seed 1, generations 1-3, 128 battled per generation = 384 teams)",
          {arm: [t for c in cells_ for t in c] for arm, cells_ in loops(con).items()}, "control", out["loops"])
    json.dump(out, open(RES / "sampler_qd.json", "w"), indent=1)


if __name__ == "__main__":
    main()
