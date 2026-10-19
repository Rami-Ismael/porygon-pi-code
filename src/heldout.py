"""The held-out evaluation set: placement ranks 51-100 of the ordering behind results/top50_evs.json.

Same source (the VGCPastes Reg M-B sheet scraped by VGC-Bench), same placement order (ordinal
ascending, then MB id descending within an ordinal), same record format as top50_evs.json
({id, ordinal, rank, event, date, evs, on_disk}). The file is results/heldout_next50_evs.json; the
50 ids are pinned below so a silent edit of that file cannot move the boundary.

The set is never searched against, trained on, or used to pick a candidate. Its one permitted use
is to score the final selected counter once, after selection: matchup_db.evaluate_final (matrix) or
heldout.evaluate (battle pool). Everything else that touches it raises HeldoutLeak.

IDENTICAL = THE SAME 48 FIELDS. A team is its six slots' species, ability, item, four moves and
alignment (what the generators train on); slot order, move order, nicknames, Stat Points and the
spelling of a neutral alignment are not part of it. Two pastes with the same 48 fields are one team
here, so a held-out team with another Stat Point spread is still the held-out team. One identity
rule serves every filter: files, pastes, parsed teams, encoded rows and matrix teams.

Vocabularies and rule tables are NOT filtered: removing the held-out teams shrinks them (species
190 -> 187, moves 317 -> 311) and would shift every token id in every saved checkpoint. encode.Vocab
and encode.Legality read the excluded teams back in (excluded_teams) so ids stay where they were.

    python src/heldout.py verify     # the 50, their order, their separation from the top-50
"""
import functools, hashlib, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
import top50 as T50

N = 50
SET_NAME = "heldout51_100"
HELDOUT = Path(os.environ.get("HELDOUT_EVS", T50.TOP50.parent / "heldout_next50_evs.json"))
PINNED = ("MB50", "MB492", "MB198", "MB599", "MB522", "MB491", "MB369", "MB358", "MB193", "MB680",
          "MB490", "MB266", "MB192", "MB754", "MB597", "MB514", "MB413", "MB403", "MB402", "MB401",
          "MB398", "MB365", "MB521", "MB489", "MB127", "MB595", "MB526", "MB488", "MB356", "MB594",
          "MB574", "MB520", "MB487", "MB367", "MB400", "MB355", "MB265", "MB354", "MB194", "MB264",
          "MB679", "MB519", "MB486", "MB191", "MB732", "MB404", "MB353", "MB263", "MB727", "MB485")
FIELDS = ("id", "ordinal", "rank", "event", "date", "evs", "on_disk")
NEUTRAL = {"bashful", "docile", "hardy", "quirky", "serious"}


class HeldoutLeak(Exception):
    """Ranks 51-100 were about to be used for training, search, fitness or candidate selection."""


# ---------------------------------------------------------------- the set
def entries(path=HELDOUT):
    """The 50 records, in file order. Raises unless they are exactly the pinned ids."""
    with open(path) as f: rows = json.load(f)
    got = tuple(t["id"] for t in rows)
    if got != PINNED:
        raise ValueError(f"{path} does not hold the pinned held-out set: "
                         f"{len(got)} ids, {len(set(got) - set(PINNED))} not pinned, {len(set(PINNED) - set(got))} missing")
    return rows

def ids(path=HELDOUT):
    return [t["id"] for t in entries(path)]

def files(root=T50.TEAMS, path=HELDOUT):
    """The 50 team files (str, file order); raises if one is missing."""
    got, missing = [], []
    for i in ids(path):
        try: got.append(T50.resolve(i, root))
        except FileNotFoundError: missing.append(i)
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {N} held-out teams have no file under {root}: {missing}")
    return got

def _ordinal_key(t):
    return (t["ordinal"], -int(t["id"][2:]))

def verify(heldout=HELDOUT, top50=T50.TOP50, root=T50.TEAMS):
    """Check the set is the 50 teams ranked 51-100. Raises ValueError on the first failure; returns the facts."""
    held = entries(heldout)
    with open(top50) as f: top = json.load(f)
    if len(held) != N: raise ValueError(f"{len(held)} teams, expected {N}")
    if len({t["id"] for t in held}) != N: raise ValueError("duplicate ids")
    for t in held:
        if tuple(t) != FIELDS: raise ValueError(f"{t['id']}: fields {tuple(t)} differ from top50_evs.json's {FIELDS}")
    if set(top[0]) != set(held[0]): raise ValueError("record format differs from top50_evs.json")
    shared = {t["id"] for t in top} & {t["id"] for t in held}
    if shared: raise ValueError(f"ids also in the top-50: {sorted(shared)}")
    if any(t["evs"] != "Yes" or not t["on_disk"] for t in held):
        raise ValueError("a held-out team is not an EVs=Yes team on disk")
    seq = top + held                                  # ranks 1..100: must be one strictly ordered list
    for a, b in zip(seq, seq[1:]):
        if not _ordinal_key(a) < _ordinal_key(b):
            raise ValueError(f"order broken between {a['id']} and {b['id']}: ranks 51-100 do not continue the top-50")
    got = files(root, heldout)
    top_keys = {grid_key_text(Path(T50.resolve(t["id"], root)).read_text()) for t in top}
    held_keys = {grid_key_text(Path(f).read_text()) for f in got}
    if len(held_keys) != N: raise ValueError(f"{len(held_keys)} distinct teams (48 fields) in the 50 held-out files")
    if held_keys & top_keys: raise ValueError("a held-out team is the same 48 fields as a top-50 team")
    return {"n": len(held), "distinct": len(held_keys), "top50_distinct": len(top_keys),
            "first": held[0]["id"], "last": held[-1]["id"], "max_id": max(int(t["id"][2:]) for t in held),
            "pool_twins": twins(root)}

def twins(root=T50.TEAMS):
    """{held-out id: [other pool files with the same 48 fields]}: the 'identical teams elsewhere'."""
    want = {grid_key_text(Path(f).read_text()): i for i, f in zip(ids(), files(root))}
    out = {}
    for f in sorted(Path(root).rglob("*.txt")):
        k = grid_key_text(f.read_text())
        if k in want and f.stem != want[k]: out.setdefault(want[k], []).append(f.stem)
    return out


# ---------------------------------------------------------------- identity: the 48 fields
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def _key(fields):
    return hashlib.sha1("|".join(fields).encode()).hexdigest()

def grid_fields(team):
    """The 48 normalised fields of a parsed team (corpus.parse_team_text dicts): slots sorted by species,
    moves sorted within a slot, neutral alignments one spelling. Stat Points are not in it."""
    out = []
    for s in sorted(team, key=lambda s: (_norm(s["species"]), _norm(s["item"]), _norm(s["ability"]))):
        ms = sorted(_norm(m) for m in s["moves"])
        ms += [""] * (4 - len(ms))
        nat = _norm(s.get("nature"))
        out += [_norm(s["species"]), _norm(s["ability"]), _norm(s["item"]), *ms[:4], "serious" if nat in NEUTRAL else nat]
    return out

def grid_key_team(team):
    return _key(grid_fields(team))

def grid_key_text(text):
    """Key of a Showdown paste. Parsed block by block: corpus.parse_team_text merges an item-less slot
    into the slot before it (MB77, MB502, MB661), which would make a six-slot team look like five."""
    from corpus import parse_team_text
    team = []
    for block in text.strip().split("\n\n"): team += parse_team_text(block)
    return grid_key_team(team)

def row_key(row, V):
    """Key of one encoded row under vocabulary V (a decode, so move order inside the row is free)."""
    slots = []
    for i in range(6):
        f = [_norm(V.decode_field(i * 8 + j, int(row[i * 8 + j]))) for j in range(8)]
        ms = sorted(m for m in f[3:7] if m and m != "mask"); ms += [""] * (4 - len(ms))
        nat = f[7]
        slots.append((f[0], f[1], f[2], *ms[:4], "serious" if nat in NEUTRAL else nat))
    slots.sort(key=lambda s: (s[0], s[2], s[1]))
    return _key([x for s in slots for x in s])


@functools.lru_cache(maxsize=None)
def identities():
    """frozenset of the 50 keys: the held-out teams, by 48 fields."""
    return frozenset(grid_key_text(Path(f).read_text()) for f in files())

def reset():
    identities.cache_clear(); _file_flag.cache_clear(); _excluded.cache_clear()

def is_heldout_text(text):
    return grid_key_text(text) in identities()

def is_heldout_team(team):
    return grid_key_team(team) in identities()

@functools.lru_cache(maxsize=1 << 16)
def _file_flag(path, mtime_ns, size):
    with open(path) as f: return is_heldout_text(f.read())

def is_heldout_file(path):
    """True if the file holds a held-out team under any name. Cached per (path, mtime, size)."""
    st = os.stat(path)
    return _file_flag(str(path), st.st_mtime_ns, st.st_size)


# ---------------------------------------------------------------- filters (training / candidates)
def training_pairs(pairs, V=None):
    """[(path, team)] without held-out teams (48-field identity). V is accepted for older callers."""
    held = identities()
    return [(f, t) for f, t in pairs if grid_key_team(t) not in held]

@functools.lru_cache(maxsize=4)
def _excluded(root):
    import corpus
    held = identities()
    out = []
    for f in sorted(Path(root).rglob("*.txt")):
        t = corpus.parse_team(f)
        if len(t) == 6 and grid_key_team(t) in held: out.append(t)
    return tuple(out)

def excluded_teams(root=None):
    """Parsed teams of every pool file that is held out: what the vocabulary and rule tables still read."""
    import corpus
    return list(_excluded(str(root or corpus.TEAMS)))

def grid_predicate(V=None):
    """team -> True if it is a held-out team."""
    return is_heldout_team

def exclude_pastes(pastes):
    """The pastes that are not a held-out team (candidate selection)."""
    held = identities()
    return [p for p in pastes if grid_key_text(p) not in held]

def row_mask(X, V):
    """Boolean mask over encoded rows X: True = keep (the row is not a held-out team)."""
    import numpy as np
    held = identities()
    cache, out = {}, np.empty(len(X), bool)
    for i, r in enumerate(X):
        b = r.tobytes()
        k = cache.get(b)
        if k is None: k = cache[b] = row_key(r, V) in held
        out[i] = not k
    return out

def team_ids(con):
    """frozenset of team_ids in a matchup database that are a held-out team (any source, any file name,
    any Stat Points). Fills team.grid_hash for teams written by a version that did not."""
    fill_grid_hash(con)
    h = sorted(identities())
    q = ",".join("?" * len(h))
    return frozenset(r[0] for r in con.execute(f"SELECT team_id FROM team WHERE grid_hash IN ({q})", h))

def fill_grid_hash(con):
    """Add team.grid_hash if the database predates it, and fill every NULL. Returns rows filled."""
    cols = [r[1] for r in con.execute("PRAGMA table_info(team)")]
    if "grid_hash" not in cols:
        with con: con.execute("ALTER TABLE team ADD COLUMN grid_hash TEXT")
    rows = con.execute("SELECT team_id, paste FROM team WHERE grid_hash IS NULL").fetchall()
    if rows:
        with con:
            con.executemany("UPDATE team SET grid_hash=? WHERE team_id=?", [(grid_key_text(p), t) for t, p in rows])
    with con: con.execute("CREATE INDEX IF NOT EXISTS team_grid ON team (grid_hash)")
    return len(rows)


# ---------------------------------------------------------------- guards (fitness / selection)
def assert_clear_files(paths, role):
    bad = sorted({Path(p).name for p in set(map(str, paths)) if is_heldout_file(p)})
    if bad:
        raise HeldoutLeak(f"{len(bad)} held-out team(s) used as {role}: {bad[:5]}. Ranks 51-100 are scored only "
                          f"by heldout.evaluate / matchup_db.evaluate_final, on the final selected teams.")

def evaluate(team_files, battles=96, conc=50, **kw):
    """The permitted use, battle-pool route: the final selected teams against the 50 held-out teams.
    Returns pool.score's {file: {wins, battles, win_rate, se}}."""
    import pool
    assert_clear_files(team_files, "a final counter under evaluation (a held-out team is not a result)")
    opp = files()
    return pool.score([(str(f), opp) for f in team_files], battles=battles, conc=conc, allow_heldout=True, **kw)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd != "verify": sys.exit(__doc__)
    print(json.dumps(verify(), indent=1))
