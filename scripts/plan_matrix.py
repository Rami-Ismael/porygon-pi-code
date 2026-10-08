"""Plan-vs-plan matchup matrix for one pair of teams: every one of the 90 x 90 = 8,100 team-preview
plan pairs, battled with team preview forced and the behaviour-cloning policy on both sides after it.

Stored in the matchup database (~/vgc-data/matchup_regmb.sqlite) in its own tables, never in `cell`:
the team-pair cells hold battles where the policy chose its own four, these hold battles where the
four were forced, so the two must not be summed.

  plan_cell(policy_id, lo, lo_plan, hi, hi_plan, lo_wins, hi_wins, ties, battles)   lo < hi, as in `cell`
  plan_contribution(batch_id, ...)   what each run added (provenance, replicates)

Deficit-only, like matchup_db.ensure: asking for 24 per cell after 8 battles only the missing 16.
A plan is "lead+lead/back+back" in species ids, each pair sorted.

  python scripts/plan_matrix.py --rows MB522 --cols MB763 --rounds 8,16,24
"""
import argparse, itertools, json, os, socket, sqlite3, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import matchup_db as mdb

RUNTIME = Path.home() / ".local/share/vgc-pilot-runtime"
VGCBENCH = RUNTIME / "unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2"
PY = RUNTIME / "venv/bin/python"
POLICY_ZIP = RUNTIME / "downloads/bc_100.zip"       # sha256 57f5edcab415cf6c = policy_id 1 in the database
PORTS = [8123, 8124, 8125, 8126, 8127, 8128, 8129]
WORK = Path("/tmp/vgc-pilot/plan_matrix"); WORK.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS plan_cell (
  policy_id INTEGER NOT NULL REFERENCES policy,
  lo INTEGER NOT NULL REFERENCES team, lo_plan TEXT NOT NULL,
  hi INTEGER NOT NULL REFERENCES team, hi_plan TEXT NOT NULL,
  lo_wins INTEGER NOT NULL, hi_wins INTEGER NOT NULL, ties INTEGER NOT NULL, battles INTEGER NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (policy_id, lo, lo_plan, hi, hi_plan),
  CHECK (lo < hi), CHECK (battles = lo_wins + hi_wins + ties)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS plan_contribution (
  batch_id INTEGER NOT NULL REFERENCES batch, lo INTEGER NOT NULL, lo_plan TEXT NOT NULL,
  hi INTEGER NOT NULL, hi_plan TEXT NOT NULL,
  lo_wins INTEGER NOT NULL, hi_wins INTEGER NOT NULL, ties INTEGER NOT NULL, battles INTEGER NOT NULL,
  PRIMARY KEY (batch_id, lo, lo_plan, hi, hi_plan)
) WITHOUT ROWID;
"""


def team_id(con, source_id):
    r = con.execute("SELECT team_id FROM team_source WHERE source_id=? OR source_id LIKE ?",
                    (source_id, f"%/{source_id}")).fetchone()
    if not r: raise SystemExit(f"{source_id} is not in the database")
    return r[0]


def species_of(con, tid):
    return con.execute("SELECT species_key FROM team WHERE team_id=?", (tid,)).fetchone()[0].split(",")


def plans_of(species):
    out = []
    for lead in itertools.combinations(sorted(species), 2):
        rest = [s for s in sorted(species) if s not in lead]
        for back in itertools.combinations(rest, 2):
            out.append(f"{'+'.join(lead)}/{'+'.join(back)}")
    assert len(out) == 90
    return out


def done(con, pid, lo, hi):
    d = defaultdict(int)
    for lp, hp, n in con.execute("SELECT lo_plan, hi_plan, battles FROM plan_cell WHERE policy_id=? AND lo=? AND hi=?",
                                 (pid, lo, hi)):
        d[(lp, hp)] = n
    return d


def live_ports():
    up = []
    for p in PORTS:
        s = socket.socket(); s.settimeout(0.25)
        try: s.connect(("127.0.0.1", p)); up.append(p)
        except OSError: pass
        finally: s.close()
    return up


def run_round(con, pid, lo, hi, target, conc=50):
    lo_plans, hi_plans = plans_of(species_of(con, lo)), plans_of(species_of(con, hi))
    have = done(con, pid, lo, hi)
    jobs = []
    for lp in lo_plans:
        need = {hp: target - have[(lp, hp)] for hp in hi_plans if have[(lp, hp)] < target}
        if not need: continue
        sched = []
        for k in range(max(need.values())):          # interleave column plans so a crash leaves cells even
            sched += [hp for hp, n in need.items() if n > k]
        jobs.append({"row_file": mdb.paste_file(con, lo), "col_file": mdb.paste_file(con, hi),
                     "row_plan": lp, "schedule": sched})
    if not jobs:
        print(f"  {target}/cell: nothing to do", flush=True); return 0
    ports = live_ports()
    if not ports: raise SystemExit("no Showdown server is listening on 8123-8129")
    shards = [[] for _ in ports]
    for i, j in enumerate(jobs): shards[i % len(ports)].append(j)
    tag = f"{os.getpid()}_{int(time.time())}"
    procs, t0 = [], time.perf_counter()
    for i, (port, sh) in enumerate(zip(ports, shards)):
        if not sh: continue
        spec, out = WORK / f"plan_{tag}_{i}.json", WORK / f"plan_{tag}_{i}.out.json"
        json.dump({"jobs": sh, "conc": conc, "port": port, "policy": str(POLICY_ZIP)}, open(spec, "w"))
        log = open(WORK / f"plan_{tag}_{i}.log", "w")
        procs.append((subprocess.Popen([str(PY), str(REPO / "src/plan_shard.py"), str(spec), str(out)], cwd=VGCBENCH,
                                       env=dict(os.environ, PYTHONPATH=str(VGCBENCH)), stdout=log, stderr=log), out))
    total_sched = sum(len(j["schedule"]) for j in jobs)
    print(f"  {target}/cell: {len(jobs)} row plans, {total_sched:,} battles scheduled on {len(ports)} servers", flush=True)
    while any(p.poll() is None for p, _ in procs):
        time.sleep(30)
        n = sum(sum(r["finished"] for r in json.load(open(o))) for _, o in procs if Path(o).exists())
        dt = time.perf_counter() - t0
        print(f"    {n:,}/{total_sched:,} battles · {dt/60:.1f} min · {n/max(dt,1e-9):.1f}/s", flush=True)
    results, lost, viol = [], 0, 0
    for p, o in procs:
        if not Path(o).exists():
            print(f"  WARNING shard {o.name} wrote nothing (exit {p.returncode})", flush=True); continue
        for r in json.load(open(o)):
            lost += r["unattributed"]; viol += r.get("plan_violations", 0)
            for hp, (w, l, t) in r["by_col"].items():
                results.append((r["row_plan"], hp, w, l, t))
    secs = time.perf_counter() - t0
    n = record(con, pid, lo, hi, results, secs, f"plan-matrix {target}/cell", lost)
    print(f"  recorded {n:,} battles · {lost} unattributed · {viol} plan violations · {secs/60:.1f} min · {n/max(secs,1e-9):.1f}/s", flush=True)
    return n


def record(con, pid, lo, hi, results, secs, note, lost):
    total = sum(w + l + t for *_, w, l, t in results)
    with con:
        bid = con.execute("INSERT INTO batch (policy_id,started_at,seconds,seed,battles,unattributed,note) "
                          "VALUES (?,datetime('now'),?,?,?,?,?)", (pid, secs, None, total, lost, note)).lastrowid
        for lp, hp, w, l, t in results:
            k = w + l + t
            if not k: continue
            con.execute("INSERT INTO plan_contribution VALUES (?,?,?,?,?,?,?,?,?)", (bid, lo, lp, hi, hp, w, l, t, k))
            con.execute("INSERT INTO plan_cell VALUES (?,?,?,?,?,?,?,?,?,datetime('now')) "
                        "ON CONFLICT (policy_id,lo,lo_plan,hi,hi_plan) DO UPDATE SET lo_wins=lo_wins+excluded.lo_wins, "
                        "hi_wins=hi_wins+excluded.hi_wins, ties=ties+excluded.ties, battles=battles+excluded.battles, "
                        "updated_at=excluded.updated_at", (pid, lo, lp, hi, hp, w, l, t, k))
    return total


def export(con, pid, lo, hi, names, path):
    lo_plans, hi_plans = plans_of(species_of(con, lo)), plans_of(species_of(con, hi))
    cells = {(a, b): (w, n) for a, b, w, n in con.execute(
        "SELECT lo_plan, hi_plan, lo_wins, battles FROM plan_cell WHERE policy_id=? AND lo=? AND hi=?", (pid, lo, hi))}
    out = {"row_team": names[0], "col_team": names[1], "policy_id": pid,
           "note": "row team's wins; team preview forced to each plan, behaviour-cloning policy both sides after it",
           "row_plans": lo_plans, "col_plans": hi_plans,
           "wins": [[cells.get((a, b), (0, 0))[0] for b in hi_plans] for a in lo_plans],
           "battles": [[cells.get((a, b), (0, 0))[1] for b in hi_plans] for a in lo_plans]}
    Path(path).write_text(json.dumps(out)); print(f"  exported {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="MB522"); ap.add_argument("--cols", default="MB763")
    ap.add_argument("--rounds", default="8,16,24", help="cumulative battles per cell after each round")
    ap.add_argument("--conc", type=int, default=50)
    a = ap.parse_args()
    con = mdb.connect(); con.executescript(SCHEMA); con.commit()
    r, c = team_id(con, a.rows), team_id(con, a.cols)
    if r > c: r, c, names = c, r, (a.cols, a.rows)
    else: names = (a.rows, a.cols)
    pid = con.execute("SELECT policy_id FROM policy WHERE row_policy='bc_100.zip:57f5edcab415cf6c' "
                      "AND simulator='showdown:913da36'").fetchone()[0]
    print(f"{names[0]} (team {r}) vs {names[1]} (team {c}), policy {pid}", flush=True)
    out = REPO / "results" / f"plan_matrix_{names[0]}_{names[1]}.json"
    for target in [int(x) for x in a.rounds.split(",")]:
        run_round(con, pid, r, c, target, a.conc)
        export(con, pid, r, c, names, out)


if __name__ == "__main__":
    main()
