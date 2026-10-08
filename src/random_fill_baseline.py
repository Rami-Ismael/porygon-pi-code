"""Random-fill baseline (owner, "What is the baseline.md" → Random Search, 2026-10-04).

Blank part of a real team, fill each blank with a UNIFORMLY RANDOM value Showdown accepts given the
rest of the team, battle it against the top-50 meta teams, and compare with the untouched original.
This is the floor any completion method (diffusion, LLM) has to beat; report a method as the share
of the random → original gap it closes.

Starting teams: 50 legal VGCPastes teams NOT in the top-50 set (seeded), so the starting teams and
the opponents are disjoint. Tasks = the owner's list (2026-10-04), one seeded mask per team per
(task, k): Stat Points k=1–6, items 1–6, moves 1–24, whole Pokémon 1–6, abilities 1–6,
Stat Alignment 1–6, mixed parts (items/moves/abilities/alignments/Stat Points) 1–40. Random values come from the project vocabulary filtered by the decoder's
legality constraints (learnsets, Species/Item Clause); a whole Pokémon also gets a random legal
Stat Point spread (66 points, at most 32 per stat). Every fill is checked by Showdown's validator,
redrawn up to 50 times. Everything outside the mask is kept exactly.

Every battle goes into the matchup matrix: one per top-50 opponent (49), policy 1.

    python random_fill_baseline.py build    # masks + random fills + validation, no battles
    python random_fill_baseline.py battle   # originals + fills vs top-50, into the matrix
    python random_fill_baseline.py report
"""
import json, random, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hps_generate
hps_generate.SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"
import activesearch as A
import diffusion as D
import matchup_db as MDB
from corpus import parse_team_text, norm, STATS
from encode import NF, NSLOT, canon
from hps_generate import Validator, slot_to_text

A.pool.REPO = str(Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2")
MDB.SHOWDOWN = hps_generate.SHOWDOWN
REPO = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot")
TASKS_OUT = REPO / "results/random_fill_tasks.json"
OUT = REPO / "results/random_fill_baseline.json"
N_TEAMS, REPS, SEED, POLICY = 50, 1, 20261004, 1
TASKS = ([("stats", k) for k in range(1, 7)] + [("item", k) for k in range(1, 7)] + [("move", k) for k in range(1, 25)]
         + [("pokemon", k) for k in range(1, 7)] + [("ability", k) for k in range(1, 7)] + [("alignment", k) for k in range(1, 7)]
         + [("mixed", k) for k in range(1, 41)])
PARTS = ([("item", s) for s in range(NSLOT)] + [("move", s, m) for s in range(NSLOT) for m in range(4)]
         + [("ability", s) for s in range(NSLOT)] + [("alignment", s) for s in range(NSLOT)] + [("stats", s) for s in range(NSLOT)])
FIELD = {"species": 0, "ability": 1, "item": 2, "nature": 7}


def random_spread(rng):
    sp = {s: 0 for s in STATS}; left = 66
    while left:
        s = rng.choice([s for s in STATS if sp[s] < 32]); sp[s] += 1; left -= 1
    return sp


def start_teams(con):
    top = set(MDB.set_members(con, "top50"))
    rows = con.execute("SELECT team_id, paste FROM team WHERE origin='vgcpastes' AND legal=1 ORDER BY team_id").fetchall()
    rows = [(t, p) for t, p in rows if t not in top and len(parse_team_text(p)) == 6]
    pick = random.Random(SEED).sample(rows, N_TEAMS)
    return [(t, p) for t, p in sorted(pick)]


def part_cols(part):
    kind, s = part[0], part[1]
    return {"item": [s * NF + 2], "ability": [s * NF + 1], "alignment": [s * NF + 7], "stats": [],
            "move": [s * NF + 3 + (part[2] if len(part) > 2 else 0)], "pokemon": [s * NF + j for j in range(NF)]}[kind]


def make_mask(task, k, rng):
    """cols: grid columns to refill; slots: Pokémon whose Stat Points are redrawn."""
    if task == "pokemon":
        parts = [("pokemon", s) for s in rng.sample(range(NSLOT), k)]
    elif task == "move":
        parts = [("move", s, m) for s, m in rng.sample([(s, m) for s in range(NSLOT) for m in range(4)], k)]
    elif task == "mixed":
        parts = [tuple(p) for p in rng.sample(PARTS, k)]
    else:
        parts = [(task, s) for s in rng.sample(range(NSLOT), k)]
    return {"parts": parts, "cols": sorted(c for p in parts for c in part_cols(p)),
            "slots": sorted({p[1] for p in parts if p[0] in ("pokemon", "stats")})}


def fill(team, mask, V, C, look, rng):
    """team in canonical order (encode.canon), moves sorted as in the grid. Returns a new team."""
    row = V.encode(team).copy()
    for c in mask["cols"]: row[c] = 0
    rt = __import__("torch").tensor(row)
    for c in [c for c in D.ORDER if c in mask["cols"]]:
        ok = C.mask_for(c, rt, "cpu").nonzero().flatten().tolist()
        if c % NF in (3, 4, 5, 6):                      # no duplicate move within a slot
            have = {int(rt[(c // NF) * NF + j]) for j in range(3, 7)}
            ok = [v for v in ok if v not in have] or ok
        rt[c] = rng.choice(ok)
    new = [dict(s, moves=sorted(s["moves"], key=norm)) for s in team]
    name = lambda c: look.get(V.decode_field(c, int(rt[c])), V.decode_field(c, int(rt[c])))
    for c in mask["cols"]:
        s, j = divmod(c, NF)
        if j == 0: new[s]["species"] = name(c)
        elif j == 1: new[s]["ability"] = name(c)
        elif j == 2: new[s]["item"] = name(c)
        elif j == 7: new[s]["nature"] = name(c)
        else:
            mv = list(new[s]["moves"]) + [""] * 4; mv[j - 3] = name(c); new[s]["moves"] = [m for m in mv[:4] if m]
    for s in mask["slots"]:
        new[s]["evs"] = random_spread(rng)
    return new


def to_paste(team):
    return "\n\n".join(slot_to_text(dict(species=s["species"], item=s["item"], ability=s["ability"],
                                         nature=s["nature"], moves=s["moves"], evs=s["evs"])) for s in team) + "\n"


def build():
    con = MDB.connect()
    cf = A.load_corpus_files(); corpus = [t for _, t in cf]
    V = A.Vocab(corpus); C = D.Constraints(V, A.Legality(corpus)); look, _ = A.decode_tables(corpus)
    val = Validator(); tasks, fills, fails = [], [], 0
    try:
        for tid, paste in start_teams(con):
            team = canon(parse_team_text(paste))
            if (V.encode(team) == 0).any(): continue                # a field outside the vocabulary
            for task, k in TASKS:
                for rep in range(REPS):
                    rng = random.Random(f"{SEED}:{tid}:{task}:{k}:{rep}")
                    mask = make_mask(task, k, rng)
                    tasks.append(dict(start_team=tid, task=task, k=k, rep=rep, mask=mask, team_paste=to_paste(team)))
                    for attempt in range(50):
                        p = to_paste(fill(team, mask, V, C, look, rng))
                        err = val(p)
                        if err is None: break
                    else:
                        fails += 1; p = None
                    fills.append(dict(start_team=tid, task=task, k=k, rep=rep, paste=p, attempts=attempt + 1))
    finally:
        val.close()
    json.dump(dict(seed=SEED, tasks=tasks, note="shared masks: every completion method fills these"), open(TASKS_OUT, "w"))
    json.dump(dict(fills=fills, failed=fails), open(OUT, "w"))
    n_start = len({t["start_team"] for t in tasks})
    print(f"{n_start} starting teams · {len(fills)} random fills · {fails} could not be made legal in 50 draws · "
          f"mean draws {np.mean([f['attempts'] for f in fills]):.2f}")


def battle():
    con = MDB.connect(); out = json.load(open(OUT)); val = Validator()
    try:
        starts = sorted({f["start_team"] for f in out["fills"]})
        with con:
            for f in out["fills"]:
                if f["paste"] and "team_id" not in f:
                    f["team_id"] = MDB.add_team(con, f["paste"], f"randomfill:{f['task']}{f['k']}", val)[0]
    finally:
        val.close()
    ids = sorted(set(starts) | {f["team_id"] for f in out["fills"] if f.get("team_id")})
    legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
    MDB.ensure(con, POLICY, sorted(legal), MDB.set_members(con, "top50"), 1, note="random_fill_baseline")
    sc = lambda t: MDB.score(con, POLICY, t)["score"] if t in legal else None
    out["original"] = {str(t): sc(t) for t in starts}
    for f in out["fills"]:
        f["score"] = sc(f["team_id"]) if f.get("team_id") else None
    json.dump(out, open(OUT, "w"))
    print("battled", len(legal), "teams")


def report():
    out = json.load(open(OUT)); rng = np.random.default_rng(0); res = {}
    orig = {int(k): v for k, v in out["original"].items()}
    print(f"originals: {len(orig)} real teams, mean win rate {np.mean(list(orig.values())):.3f}")
    print(f"{'task':12s} {'n':>4s} {'filled':>7s} {'change':>8s} {'95% CI':>18s} {'worse':>6s} {'better':>7s}")
    for task, k in TASKS:
        fs = [f for f in out["fills"] if f["task"] == task and f["k"] == k and f.get("score") is not None]
        by = {}
        for f in fs: by.setdefault(f["start_team"], []).append(f["score"] - orig[f["start_team"]])
        teams = list(by)
        d = np.array([np.mean(by[t]) for t in teams])
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        lo, hi = np.quantile(boots, [.025, .975])
        alld = np.array([x for t in teams for x in by[t]])
        res[f"{task}{k}"] = dict(n=len(fs), filled=float(np.mean([f["score"] for f in fs])), change=float(d.mean()),
                                 ci=[float(lo), float(hi)], share_worse=float((alld < 0).mean()), share_better=float((alld > 0).mean()))
        r = res[f"{task}{k}"]
        print(f"{task + ' ×' + str(k):12s} {r['n']:4d} {r['filled']:7.3f} {r['change']:+8.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"{r['share_worse']:6.0%} {r['share_better']:7.0%}")
    out["summary"] = res; json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    {"build": build, "battle": battle, "report": report}[sys.argv[1]]()
