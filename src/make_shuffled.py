"""Control arm isolating the Stat Points confound.

Generated teams inherit each candidate's spread from a RANDOM corpus slot of the
same species; real corpus teams carry spreads tuned for that specific team. So a
win-rate gap between them could be composition OR spreads.

This arm takes real corpus teams and applies the generated arm's spread treatment:
same six candidates, same sets, but every spread swapped for another corpus spread
of the same species. Comparing it to the untouched corpus arm prices the confound.
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from corpus import load_corpus, norm
from encode import canon
from propose import Validator, team_to_text

n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 16
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(seed)
teams, names = load_corpus()
keep = [(t, n) for t, n in zip(teams, names) if len(t) == 6]
teams = [t for t, _ in keep]
spreads = defaultdict(list)
for t in teams:
    for s in t: spreads[norm(s["species"])].append(dict(s["evs"]))
out = Path("/tmp/vgc-pilot/proposals"); out.mkdir(exist_ok=True)
for f in out.glob("shuffled_*.txt"): f.unlink()
val = Validator()
kept = tries = swapped = 0
while kept < n_want and tries < 2000:
    tries += 1
    base = canon(teams[rng.integers(0, len(teams))])
    cand = []
    for s in base:
        s2 = dict(s); pool = spreads[norm(s["species"])]
        if len(pool) > 1:
            new = pool[rng.integers(0, len(pool))]
            if new != s["evs"]: swapped += 1
            s2["evs"] = new
        cand.append(s2)
    txt = team_to_text(cand)
    if val(txt) is None:
        (out/f"shuffled_{kept}.txt").write_text(txt); kept += 1
val.close()
print(f"shuffled-spread control: {kept} teams from {tries} attempts, "
      f"{swapped/max(kept*6,1):.2f} of 6 slots actually re-spread")
