"""The training-free baseline the critics showed is the one to beat: take a corpus
team, replace one candidate (species + its whole set) with a candidate from another
corpus team, and keep it if Showdown validates it."""
import sys
from pathlib import Path
import numpy as np
from corpus import load_corpus
from encode import canon
from propose import Validator, team_to_text

n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 24
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(seed)
teams, _ = load_corpus(); teams = [t for t in teams if len(t) == 6]
out = Path("/tmp/vgc-pilot/proposals"); out.mkdir(exist_ok=True)
for f in out.glob("slotcopy_*.txt"): f.unlink()
val = Validator()
kept = tries = 0
while kept < n_want and tries < 4000:
    tries += 1
    base = canon(teams[rng.integers(0, len(teams))])
    s = int(rng.integers(0, 6))
    donor = canon(teams[rng.integers(0, len(teams))])
    cand = [dict(x) for x in base]; cand[s] = dict(donor[int(rng.integers(0, 6))])
    txt = team_to_text(cand)
    if val(txt) is None:
        (out/f"slotcopy_{kept}.txt").write_text(txt); kept += 1
val.close()
print(f"slotcopy: kept {kept}/{tries} attempts -> Showdown-valid rate {kept/tries:.1%}")
