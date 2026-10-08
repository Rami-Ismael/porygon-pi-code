"""Sparse matchup matrix for one VGC format, stored in SQLite.

Rows and columns are legal teams. A cell holds the battle record of one pair of teams
under one fixed battle-policy configuration: wins for each side, ties, battles. A pair
that has never been battled has no row at all, so "unknown" is never confused with 0.

Purpose
  (a) a cache of the objective: battles already run for a pair are never run again;
      asking for more battles only tops a cell up to the requested count. A team's
      score is its row averaged over the columns of a team set (the top-50 meta).
  (b) training labels for the surrogate model (per cell, and per team over a set).

Design: docs/matchup-matrix.md.  Run:  python src/matchup_db.py --help
"""
import argparse, hashlib, json, math, os, random, sqlite3, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DB = Path(os.environ.get("MATCHUP_DB", Path.home() / "vgc-data" / "matchup_regmb.sqlite"))
PASTES = Path(os.environ.get("MATCHUP_PASTES", Path.home() / "vgc-data" / "pastes"))
FORMAT = "gen9championsvgc2026regmb"
POOL_DIR = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")          # VGCPastes Reg M-B pool (+ featured/)
TOP50 = REPO / "results" / "top50_evs.json"
SHOWDOWN = Path("/tmp/vgc-pilot/vgc-bench/pokemon-showdown")
POLICY_FILE = Path("/tmp/bc_100.zip")                             # behaviour-cloning policy, both sides
STATS = ["HP", "Atk", "Def", "SpA", "SpD", "Spe"]
NEUTRAL = {"bashful", "docile", "hardy", "quirky", "serious"}

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS team (
  team_id     INTEGER PRIMARY KEY,
  format_id   TEXT NOT NULL,
  canon_hash  TEXT NOT NULL,              -- sha256 of the canonical paste
  paste       TEXT NOT NULL,              -- canonical paste: the exact text that is battled
  species_key TEXT NOT NULL,              -- sorted species ids; identifies the opponent in a battle
  base_key    TEXT NOT NULL,              -- same with base species (fallback attribution)
  legal       INTEGER,                    -- Showdown TeamValidator on the canonical paste: 1 / 0 / NULL unchecked
  legal_error TEXT,
  origin      TEXT NOT NULL,              -- 'vgcpastes', 'search:<run>', ...
  created_at  TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE (format_id, canon_hash)
);
CREATE TABLE IF NOT EXISTS team_source (  -- every file or record that maps to a team (duplicates kept)
  source TEXT NOT NULL, source_id TEXT NOT NULL, team_id INTEGER NOT NULL REFERENCES team,
  path TEXT, event TEXT, placement TEXT, date TEXT,
  PRIMARY KEY (source, source_id)
);
CREATE TABLE IF NOT EXISTS team_set (     -- named column sets, e.g. 'top50'
  set_name TEXT NOT NULL, team_id INTEGER NOT NULL REFERENCES team, weight REAL NOT NULL DEFAULT 1,
  PRIMARY KEY (set_name, team_id)
);
CREATE TABLE IF NOT EXISTS policy (       -- what produced a cell; cells never mix configurations
  policy_id INTEGER PRIMARY KEY, format_id TEXT NOT NULL,
  row_policy TEXT NOT NULL, col_policy TEXT NOT NULL, simulator TEXT NOT NULL, note TEXT,
  UNIQUE (format_id, row_policy, col_policy, simulator)
);
CREATE TABLE IF NOT EXISTS cell (         -- one unordered pair; lo < hi; absent = never battled
  policy_id INTEGER NOT NULL REFERENCES policy,
  lo INTEGER NOT NULL REFERENCES team, hi INTEGER NOT NULL REFERENCES team,
  lo_wins INTEGER NOT NULL, hi_wins INTEGER NOT NULL, ties INTEGER NOT NULL, battles INTEGER NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (policy_id, lo, hi),
  CHECK (lo < hi), CHECK (battles = lo_wins + hi_wins + ties)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS cell_hi ON cell (policy_id, hi);
CREATE TABLE IF NOT EXISTS batch (        -- provenance of every battle run
  batch_id INTEGER PRIMARY KEY, policy_id INTEGER NOT NULL REFERENCES policy,
  started_at TEXT NOT NULL, seconds REAL, seed INTEGER, battles INTEGER NOT NULL DEFAULT 0,
  unattributed INTEGER NOT NULL DEFAULT 0, note TEXT
);
CREATE TABLE IF NOT EXISTS contribution ( -- what each batch added to each cell (audit / replicates)
  batch_id INTEGER NOT NULL REFERENCES batch, lo INTEGER NOT NULL, hi INTEGER NOT NULL,
  lo_wins INTEGER NOT NULL, hi_wins INTEGER NOT NULL, ties INTEGER NOT NULL, battles INTEGER NOT NULL,
  PRIMARY KEY (batch_id, lo, hi)
) WITHOUT ROWID;
CREATE VIEW IF NOT EXISTS matchup AS      -- directed view: row team's record against column team
  SELECT policy_id, lo AS row_team, hi AS col_team, lo_wins AS wins, hi_wins AS losses, ties, battles,
         CAST(lo_wins AS REAL) / battles AS win_rate FROM cell WHERE battles > 0
  UNION ALL
  SELECT policy_id, hi, lo, hi_wins, lo_wins, ties, battles,
         CAST(hi_wins AS REAL) / battles FROM cell WHERE battles > 0;
"""

# ---------------------------------------------------------------- canonical team identity
def _corpus():
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import corpus
    return corpus

def _norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())

def canonical_slots(paste):
    """Parse a Showdown paste into canonical slots. Order of slots and of moves, nicknames,
    gender, level and neutral-nature spelling do not change the team; everything a player
    chooses (species, ability, item, moves, Stat Points, alignment) does."""
    slots, parsed = [], []
    for block in paste.strip().split("\n\n"):        # one block per Pokemon: corpus.parse_team_text merges an
        parsed += _corpus().parse_team_text(block)    # item-less slot into the slot before it (MB77, MB502, MB661)
    for s in parsed:
        nat = s.get("nature") or "Serious"
        if _norm(nat) in NEUTRAL: nat = "Serious"
        evs = {k: int(s["evs"].get(k, 0)) for k in STATS}
        slots.append({"species": s["species"].strip(), "item": (s.get("item") or "").strip(),
                      "ability": s["ability"].strip(), "nature": nat.strip(),
                      "evs": evs, "moves": sorted(m.strip() for m in s["moves"])})
    key = lambda x: (_norm(x["species"]), _norm(x["item"]), _norm(x["ability"]))
    return sorted(slots, key=key)

def slot_text(s):
    head = s["species"] + (f" @ {s['item']}" if s["item"] else "")
    lines = [head, f"Ability: {s['ability']}", "Level: 50"]
    ev = " / ".join(f"{s['evs'][k]} {k}" for k in STATS if s["evs"].get(k))
    if ev: lines.append(f"EVs: {ev}")
    lines.append(f"{s['nature']} Nature")
    lines += [f"- {m}" for m in s["moves"]]
    return "\n".join(lines)

def canonical_paste(paste):
    return "\n\n".join(slot_text(s) for s in canonical_slots(paste)) + "\n"

def canon_hash(canon):
    return hashlib.sha256(canon.encode()).hexdigest()

def species_keys(canon):
    c = _corpus(); full, base = [], []
    for s in canonical_slots(canon):
        full.append(_norm(s["species"]))
        e = c.dex_entry(s["species"])
        base.append(_norm(e.get("baseSpecies", s["species"])) if e else _norm(s["species"]))
    return ",".join(sorted(full)), ",".join(sorted(base))

class Validator:
    """Showdown's own TeamValidator through the project's batch script."""
    def __init__(self):
        self.p = subprocess.Popen(["node", "validate-teams-batch.js"], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, cwd=str(SHOWDOWN))
    def __call__(self, text, fmt=FORMAT):
        self.p.stdin.write(json.dumps({"format": fmt, "team": text}) + "\n"); self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        return None if r["valid"] else "; ".join(r["errors"])
    def close(self):
        self.p.stdin.close(); self.p.wait()

# ---------------------------------------------------------------- database
def connect(path=DEFAULT_DB):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode = WAL"); con.execute("PRAGMA synchronous = NORMAL")
    con.executescript(SCHEMA)
    con.execute("INSERT OR IGNORE INTO meta VALUES ('format_id', ?)", (FORMAT,))
    con.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', '1')")
    con.commit()                  # never leave a write transaction open: it blocks other writers and backup
    return con

def sha256_file(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()[:n]

def get_policy(con, row_policy=None, col_policy=None, simulator=None, note=None):
    """The fixed battle configuration. Default: behaviour-cloning policy on both sides,
    identified by checkpoint hash, on the local Showdown revision."""
    if row_policy is None:
        row_policy = f"bc_100.zip:{sha256_file(POLICY_FILE)}"
    col_policy = col_policy or row_policy
    if simulator is None:
        try:
            rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SHOWDOWN,
                                 capture_output=True, text=True).stdout.strip()
        except OSError:
            rev = "unknown"
        simulator = f"showdown:{rev or 'unknown'}"
    with con:
        con.execute("INSERT OR IGNORE INTO policy (format_id,row_policy,col_policy,simulator,note) VALUES (?,?,?,?,?)",
                    (FORMAT, row_policy, col_policy, simulator, note))
    return con.execute("SELECT policy_id FROM policy WHERE format_id=? AND row_policy=? AND col_policy=? AND simulator=?",
                       (FORMAT, row_policy, col_policy, simulator)).fetchone()[0]

def add_team(con, paste, origin, validator=None):
    """Insert a team by its canonical form; returns (team_id, created)."""
    canon = canonical_paste(paste); h = canon_hash(canon)
    row = con.execute("SELECT team_id FROM team WHERE format_id=? AND canon_hash=?", (FORMAT, h)).fetchone()
    if row: return row[0], False
    full, base = species_keys(canon)
    legal = err = None
    if validator is not None:
        err = validator(canon); legal = int(err is None)
    with con:
        cur = con.execute("INSERT INTO team (format_id,canon_hash,paste,species_key,base_key,legal,legal_error,origin) "
                          "VALUES (?,?,?,?,?,?,?,?)", (FORMAT, h, canon, full, base, legal, err, origin))
    return cur.lastrowid, True

def paste_file(con, team_id):
    """Path to the canonical paste on disk (written once); this is the file that is battled."""
    paste, h = con.execute("SELECT paste, canon_hash FROM team WHERE team_id=?", (team_id,)).fetchone()
    PASTES.mkdir(parents=True, exist_ok=True)
    p = PASTES / f"{h[:20]}.txt"
    if not p.exists(): p.write_text(paste)
    return str(p)

def record(con, policy_id, results, seconds=None, seed=None, note=None, unattributed=0):
    """Add battle results. results: iterable of (row_team, col_team, row_wins, col_wins, ties).
    One transaction: a batch row, its contributions, and the cell totals."""
    agg = defaultdict(lambda: [0, 0, 0])
    for r, c, rw, cw, t in results:
        if r == c: raise ValueError("a team is not battled against itself")
        lo, hi, lw, hw = (r, c, rw, cw) if r < c else (c, r, cw, rw)
        a = agg[(lo, hi)]; a[0] += lw; a[1] += hw; a[2] += t
    total = sum(sum(v) for v in agg.values())
    with con:
        bid = con.execute("INSERT INTO batch (policy_id,started_at,seconds,seed,battles,unattributed,note) "
                          "VALUES (?,datetime('now'),?,?,?,?,?)",
                          (policy_id, seconds, seed, total, unattributed, note)).lastrowid
        for (lo, hi), (lw, hw, t) in agg.items():
            n = lw + hw + t
            if n == 0: continue
            con.execute("INSERT INTO contribution VALUES (?,?,?,?,?,?,?)", (bid, lo, hi, lw, hw, t, n))
            con.execute("INSERT INTO cell VALUES (?,?,?,?,?,?,?,datetime('now')) "
                        "ON CONFLICT (policy_id,lo,hi) DO UPDATE SET lo_wins=lo_wins+excluded.lo_wins, "
                        "hi_wins=hi_wins+excluded.hi_wins, ties=ties+excluded.ties, battles=battles+excluded.battles, "
                        "updated_at=excluded.updated_at", (policy_id, lo, hi, lw, hw, t, n))
    return bid

def battles_done(con, policy_id, a, b):
    lo, hi = (a, b) if a < b else (b, a)
    r = con.execute("SELECT battles FROM cell WHERE policy_id=? AND lo=? AND hi=?", (policy_id, lo, hi)).fetchone()
    return r[0] if r else 0

def set_members(con, set_name):
    return [r[0] for r in con.execute("SELECT team_id FROM team_set WHERE set_name=? ORDER BY team_id", (set_name,))]

# ---------------------------------------------------------------- scheduling: battle only what is missing
def plan(con, policy_id, rows, cols, per_cell):
    """Deficits: battles still needed so every (row, col) cell holds >= per_cell battles.
    A pair asked for in both directions is planned once; a team never plays itself."""
    need, seen = [], set()
    for r in rows:
        for c in cols:
            if r == c: continue
            pair = (min(r, c), max(r, c))
            if pair in seen: continue
            seen.add(pair)
            d = per_cell - battles_done(con, policy_id, r, c)
            if d > 0: need.append((r, c, d))
    return need

def jobs_for(con, need):
    """Group deficits into battle jobs: one row team against a schedule of column teams.
    Columns inside one job must have distinct species sets, because the opponent in each
    finished battle is identified from the six species it shows at team preview."""
    by_row = defaultdict(list)
    for r, c, d in need: by_row[r].append((c, d))
    keys = dict(con.execute("SELECT team_id, species_key FROM team"))
    bases = dict(con.execute("SELECT team_id, base_key FROM team"))
    jobs = []
    for r, cs in by_row.items():
        groups = []                                    # first-fit: put each column in a group without its species set
        for c, d in cs:
            for g in groups:
                if keys[c] not in g["keys"] and bases[c] not in g["bases"]:
                    g["cols"].append((c, d)); g["keys"].add(keys[c]); g["bases"].add(bases[c]); break
            else:
                groups.append({"cols": [(c, d)], "keys": {keys[c]}, "bases": {bases[c]}})
        for g in groups:
            sched, left = [], dict(g["cols"])           # round-robin so a cut-short job still spreads over columns
            while left:
                for c in list(left):
                    sched.append(c); left[c] -= 1
                    if left[c] == 0: del left[c]
            jobs.append({"row": r, "schedule": sched})
    return jobs

def run_jobs(con, jobs, seed=None, conc=50, quiet=False):
    """Run battle jobs across every live Showdown server with src/pair_shard.py.
    Returns ([(row, col, row_wins, col_wins, ties)], unattributed, seconds)."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import pool
    ports = pool.live_ports()
    if not ports: raise RuntimeError("no Showdown server is listening; start one on 8123")
    keys = dict(con.execute("SELECT team_id, species_key FROM team"))
    bases = dict(con.execute("SELECT team_id, base_key FROM team"))
    spec_jobs = []
    for j in jobs:
        cols = sorted(set(j["schedule"]))
        spec_jobs.append({"row": j["row"], "row_file": paste_file(con, j["row"]),
                          "schedule": [paste_file(con, c) for c in j["schedule"]],
                          "key2col": {keys[c]: c for c in cols}, "base2col": {bases[c]: c for c in cols}})
    k = min(len(ports), len(spec_jobs)); shards = [[] for _ in range(k)]
    for i, j in enumerate(sorted(spec_jobs, key=lambda j: -len(j["schedule"]))):
        shards[i % k].append(j)
    tag = f"{os.getpid()}_{int(time.time())}"
    procs, outs, t0 = [], [], time.perf_counter()
    for i, sh in enumerate(shards):
        spec, out = f"/tmp/vgc-pilot/_mshard_{tag}_{i}.json", f"/tmp/vgc-pilot/_mshard_{tag}_{i}.out.json"
        json.dump({"jobs": sh, "conc": conc, "port": ports[i], "seed": seed}, open(spec, "w"))
        env = dict(os.environ, PYTHONPATH=f"{pool.REPO}:/tmp/vgc-pilot/src")
        procs.append(subprocess.Popen([pool.PY, "/tmp/vgc-pilot/src/pair_shard.py", spec, out], cwd=pool.REPO,
                                      env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        outs.append(out)
    results, unattributed, failed = [], 0, []
    for p, out in zip(procs, outs):
        _, err = p.communicate()
        if p.returncode != 0 or not Path(out).exists():
            failed.append((err or b"").decode()[-300:]); continue
        for job in json.load(open(out)):
            unattributed += job["unattributed"]
            for c, (w, l, t) in job["by_col"].items():
                results.append((job["row"], int(c), w, l, t))
    secs = time.perf_counter() - t0
    if not quiet:
        n = sum(w + l + t for _, _, w, l, t in results)
        print(f"  {len(ports)} servers · {n} battles attributed · {unattributed} unattributed · {secs:.1f}s · "
              f"{n / max(secs, 1e-9):.1f}/sec", flush=True)
    if failed: print(f"  WARNING {len(failed)} shard(s) failed: {failed[0][:200]}", flush=True)
    return results, unattributed, secs

def ensure(con, policy_id, rows, cols, per_cell, seed=None, runner=None, note=None):
    """The cache entry point: make every (row, col) cell hold >= per_cell battles, running
    only the deficit. Returns the number of battles actually run (0 when fully cached)."""
    need = plan(con, policy_id, rows, cols, per_cell)
    if not need: return 0
    jobs = jobs_for(con, need)
    results, unattributed, secs = (runner or run_jobs)(con, jobs, seed=seed)
    record(con, policy_id, results, seconds=secs, seed=seed, note=note, unattributed=unattributed)
    return sum(w + l + t for _, _, w, l, t in results)

def score_pastes(con, pastes, per_cell=8, set_name="top50", seed=None, origin="search", policy_id=None):
    """Drop-in objective for search loops: add each paste (validated), battle only the cells
    it is missing against the set, return its score. Repeated or re-ordered teams cost nothing."""
    policy_id = policy_id or get_policy(con)
    val = Validator()
    try:
        with con: ids = [add_team(con, p, origin, val)[0] for p in pastes]
    finally:
        val.close()
    legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
    ensure(con, policy_id, sorted(legal), set_members(con, set_name), per_cell, seed=seed, note=f"score_pastes:{origin}")
    return [score(con, policy_id, t, set_name) if t in legal else {"team_id": t, "score": None, "illegal": True} for t in ids]

# ---------------------------------------------------------------- reading: scores and labels
def row_cells(con, policy_id, team, cols):
    """{col: (wins, battles)} for the tested cells of one row; untested columns are absent."""
    out = {}
    for c in cols:
        if c == team: continue
        lo, hi = (team, c) if team < c else (c, team)
        r = con.execute("SELECT lo_wins, hi_wins, battles FROM cell WHERE policy_id=? AND lo=? AND hi=?",
                        (policy_id, lo, hi)).fetchone()
        if r and r[2] > 0: out[c] = (r[0] if team == lo else r[1], r[2])
    return out

def score(con, policy_id, team, set_name="top50"):
    """A team's objective value: its row averaged over the set's columns (set weights,
    equal by default), with coverage and standard error. The team's own column is skipped."""
    members = con.execute("SELECT team_id, weight FROM team_set WHERE set_name=?", (set_name,)).fetchall()
    cols = {c: w for c, w in members if c != team}
    cells = row_cells(con, policy_id, team, cols)
    if not cells:
        return {"team_id": team, "score": None, "coverage": 0, "columns": len(cols), "battles": 0, "se": None}
    W = sum(cols[c] for c in cells)
    s = sum(cols[c] * w / n for c, (w, n) in cells.items()) / W
    var = sum((cols[c] / W) ** 2 * (w / n) * (1 - w / n) / n for c, (w, n) in cells.items())
    return {"team_id": team, "score": s, "coverage": len(cells), "columns": len(cols),
            "battles": sum(n for _, n in cells.values()), "se": math.sqrt(var)}

def labels(con, policy_id, set_name="top50", min_coverage=None):
    """Team-level surrogate labels: every team with at least one cell against the set."""
    members = set_members(con, set_name)
    min_coverage = len(members) - 1 if min_coverage is None else min_coverage
    rows = [r[0] for r in con.execute(
        "SELECT DISTINCT t FROM (SELECT lo AS t FROM cell WHERE policy_id=? UNION SELECT hi FROM cell WHERE policy_id=?)",
        (policy_id, policy_id))]
    out = []
    for t in rows:
        s = score(con, policy_id, t, set_name)
        if s["score"] is not None and s["coverage"] >= min_coverage: out.append(s)
    return out

def stats(con):
    q = lambda sql, *a: con.execute(sql, a).fetchone()[0]
    return {"teams": q("SELECT COUNT(*) FROM team"), "legal_teams": q("SELECT COUNT(*) FROM team WHERE legal=1"),
            "sources": q("SELECT COUNT(*) FROM team_source"), "sets": dict(con.execute(
                "SELECT set_name, COUNT(*) FROM team_set GROUP BY set_name").fetchall()),
            "policies": q("SELECT COUNT(*) FROM policy"), "cells": q("SELECT COUNT(*) FROM cell"),
            "battles": q("SELECT COALESCE(SUM(battles),0) FROM cell"), "batches": q("SELECT COUNT(*) FROM batch")}

# ---------------------------------------------------------------- seeding from VGCPastes
def seed(con, pool_dir=POOL_DIR, top50=TOP50, validate=True):
    """Import every VGCPastes Reg M-B paste (pool and featured/), keep one canonical team per
    distinct team, record every source file, and build the 'top50' column set."""
    meta = {t["id"]: t for t in json.load(open(top50))} if Path(top50).exists() else {}
    con.execute("DELETE FROM team_set WHERE set_name='top50'")
    val = Validator() if validate else None
    made = dup = 0
    try:
        with con:
            for f in sorted(Path(pool_dir).rglob("*.txt")):
                tid, created = add_team(con, f.read_text(), "vgcpastes", val)
                made += created; dup += not created
                m = meta.get(f.stem, {})
                con.execute("INSERT OR REPLACE INTO team_source VALUES (?,?,?,?,?,?,?)",
                            ("vgcpastes", f.stem, tid, str(f), m.get("event"), m.get("rank"), m.get("date")))
            missing = []
            for i in meta:
                r = con.execute("SELECT team_id FROM team_source WHERE source='vgcpastes' AND source_id=?", (i,)).fetchone()
                if r: con.execute("INSERT INTO team_set VALUES ('top50', ?, 1) ON CONFLICT (set_name, team_id) "
                                  "DO UPDATE SET weight = weight + 1", (r[0],))   # same team listed twice counts twice
                else: missing.append(i)
    finally:
        if val: val.close()
    return {"new_teams": made, "duplicate_files": dup, "top50_missing": missing, **stats(con)}

# ---------------------------------------------------------------- cost of coverage
def bytes_per_cell(n=100_000):
    """Measured on-disk bytes per cell (+ one contribution row), after VACUUM."""
    with tempfile.TemporaryDirectory() as d:
        con = sqlite3.connect(Path(d) / "m.sqlite"); con.executescript(SCHEMA)
        con.execute("PRAGMA foreign_keys = OFF")          # synthetic cells have no team rows; size is unaffected
        con.execute("INSERT INTO policy VALUES (1,'f','p','p','s',NULL)")
        con.execute("INSERT INTO batch (policy_id,started_at,battles) VALUES (1,'t',0)")
        rng = random.Random(0); m = int(math.isqrt(2 * n)) + 2; rows = []
        for lo in range(1, m):
            for hi in range(lo + 1, m):
                w = rng.randint(0, 24); l = rng.randint(0, 24 - w); t = 0
                rows.append((1, lo, hi, w, l, t, w + l + t, "2026-09-30 00:00:00"))
                if len(rows) == n: break
            if len(rows) == n: break
        con.executemany("INSERT INTO cell VALUES (?,?,?,?,?,?,?,?)", rows)
        base = Path(d) / "m.sqlite"; con.commit(); con.execute("VACUUM"); s1 = base.stat().st_size
        con.executemany("INSERT INTO contribution VALUES (1,?,?,?,?,?,?)", [r[1:7] for r in rows])
        con.commit(); con.execute("VACUUM"); s2 = base.stat().st_size; con.close()
    return s1 / n, (s2 - s1) / n

def cost_report(rate, n_seed, n_top=50, per_cell=(8, 24, 96), log10_legal=None):
    cell_b, contrib_b = bytes_per_cell()
    def line(name, pairs):
        st = pairs * (cell_b + contrib_b)
        sims = {k: pairs * k / rate / 3600 for k in per_cell}
        return {"scope": name, "cells": pairs, "storage_MB": st / 1e6,
                **{f"hours_at_{k}": h for k, h in sims.items()}}
    seed_pairs = n_seed * (n_seed - 1) // 2
    out = {"bytes_per_cell": cell_b, "bytes_per_contribution": contrib_b, "battles_per_sec": rate, "rows": [
        line("one new team vs the top-50 columns", n_top),
        line("seed pool vs the top-50 columns", n_seed * n_top - n_top * (n_top + 1) // 2),
        line(f"full seed pool, every pair ({n_seed} teams)", seed_pairs),
        line("100,000 search teams vs the top-50 columns", 100_000 * n_top)]}
    if log10_legal:
        out["full_legal_space"] = {"log10_teams": log10_legal, "log10_cells": 2 * log10_legal - math.log10(2),
                                   "log10_battles_at_8": 2 * log10_legal - math.log10(2) + math.log10(8)}
    return out

# ---------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed", help="import the VGCPastes Reg M-B pool and the top-50 set")
    s.add_argument("--no-validate", action="store_true")
    e = sub.add_parser("ensure", help="battle only the missing battles so cells reach --per-cell")
    e.add_argument("--rows", help="comma-separated team ids, or a set name", required=True)
    e.add_argument("--cols", default="top50"); e.add_argument("--per-cell", type=int, default=8)
    e.add_argument("--seed", type=int, default=101); e.add_argument("--note")
    sc = sub.add_parser("score", help="row averaged over a column set")
    sc.add_argument("--teams", required=True); sc.add_argument("--set", default="top50")
    lb = sub.add_parser("labels", help="write surrogate labels as JSON lines")
    lb.add_argument("--set", default="top50"); lb.add_argument("--out", required=True)
    lb.add_argument("--min-coverage", type=int)
    sub.add_parser("stats")
    c = sub.add_parser("cost", help="storage and simulation cost of coverage")
    c.add_argument("--rate", type=float, default=55.0, help="measured battles/sec across all servers")
    bk = sub.add_parser("backup", help="consistent snapshot (SQLite backup API)"); bk.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    con = connect(a.db)
    ids = lambda spec: set_members(con, spec) if not spec[0].isdigit() else [int(x) for x in spec.split(",")]
    if a.cmd == "seed":
        print(json.dumps(seed(con, validate=not a.no_validate), indent=1))
    elif a.cmd == "ensure":
        pid = get_policy(con)
        n = ensure(con, pid, ids(a.rows), ids(a.cols), a.per_cell, seed=a.seed, note=a.note)
        print(f"battles run: {n}")
    elif a.cmd == "score":
        pid = get_policy(con)
        for t in ids(a.teams): print(json.dumps(score(con, pid, t, a.set)))
    elif a.cmd == "labels":
        pid = get_policy(con)
        with open(a.out, "w") as f:
            for r in labels(con, pid, a.set, a.min_coverage): f.write(json.dumps(r) + "\n")
    elif a.cmd == "stats":
        print(json.dumps(stats(con), indent=1))
    elif a.cmd == "cost":
        n_seed = con.execute("SELECT COUNT(*) FROM team WHERE legal=1").fetchone()[0] or 1
        counts = Path("/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the "
                      "whole search of possible vgc pokemon format to find the right counter to a pokemon team/"
                      "Blog/visuals/team-search-space/counts.json")
        lg = json.load(open(counts))["stages"][-1]["log10"] if counts.exists() else None
        print(json.dumps(cost_report(a.rate, n_seed, log10_legal=lg), indent=1))
    elif a.cmd == "backup":
        # snapshot to a local temp file, then copy: SQLite locking stalls inside iCloud-synced folders
        import shutil
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "snap.sqlite"; dst = sqlite3.connect(tmp); con.backup(dst); dst.close()
            shutil.copyfile(tmp, a.out)
        print("wrote", a.out)

if __name__ == "__main__":
    main()
