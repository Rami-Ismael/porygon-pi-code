"""The top-50 meta teams of results/top50_evs.json, resolved to their team files.

14 of the 50 live in teams/reg_mb/featured/, not the top folder (MB759 MB686 MB684 MB604
MB582 MB494 MB691 MB683 MB581 MB580 MB682 MB583 MB728 MB681). Building f"{root}/{id}.txt"
and filtering with os.path.exists silently drops them: interaction_order.py, ruggedness.py,
ruggedness2.py and hps_eval.py scored against 36 opponents that way (found 2026-09-30,
docs/top50-36-opponents.md). Resolve through here; it raises instead of shrinking the list.

MB493 and MB494 are the same team, byte for byte (2nd and 1st, Ranked Season M-3), so the
50 files hold 49 distinct teams. Both stay in the list: that team carries weight 2 in a
uniform draw, as in matchup_db's 'top50' column set.
"""
import json
from pathlib import Path

TEAMS = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
TOP50 = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/top50_evs.json")
N = 50
DUPLICATES = {"MB494": "MB493"}          # same paste; 49 distinct teams in the 50

def ids(top50=TOP50):
    """The 50 ids, in top50_evs.json order."""
    out = [t["id"] for t in json.load(open(top50))]
    if len(out) != N:
        raise ValueError(f"{top50} lists {len(out)} teams, expected {N}")
    return out

def resolve(tid, root=TEAMS):
    """Path of one team file, from the top folder or featured/."""
    root = Path(root)
    for c in (root / f"{tid}.txt", root / "featured" / f"{tid}.txt"):
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"{tid}.txt is in neither {root} nor {root / 'featured'}")

def files(root=TEAMS, top50=TOP50):
    """All 50 opponent files (str, top50_evs.json order); raises if any one is missing."""
    got, missing = [], []
    for i in ids(top50):
        try:
            got.append(resolve(i, root))
        except FileNotFoundError:
            missing.append(i)
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {N} top-50 teams have no file under {root}: {missing}")
    return got
