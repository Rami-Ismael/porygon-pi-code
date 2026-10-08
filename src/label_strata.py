"""Label every stratum against the top-50 meta pool, then report the range covered."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/tmp/vgc-pilot/src")
import pool

root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
opp = []
for i in [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]:
    for c in (root/f"{i}.txt", root/"featured"/f"{i}.txt"):
        if c.exists(): opp.append(str(c)); break
meta = json.load(open("/tmp/vgc-pilot/strata_meta.json"))
teams = sorted(meta)
print(f"labelling {len(teams)} teams x 24 battles vs {len(opp)} meta teams "
      f"= {len(teams)*24:,} battles", flush=True)
t0 = time.perf_counter()
res = pool.score([(t, opp) for t in teams], battles=24, conc=50)
for t, v in res.items(): v["stratum"] = meta[t]["stratum"]
json.dump(res, open("/tmp/vgc-pilot/strata_labels.json", "w"), indent=1)
print(f"wall {time.perf_counter()-t0:.0f}s\n")
order = ["real","shuffled","slotcopy1","slotcopy2","slotcopy3","modeledit","invent","random"]
print(f"{'stratum':11s} {'n':>4s} {'mean':>7s} {'clustered SE':>13s} {'sd':>6s} {'p10':>6s} {'p90':>6s} {'max':>6s}")
for k in order:
    w = np.array([v["win_rate"] for v in res.values() if v["stratum"] == k])
    if not len(w): continue
    print(f"{k:11s} {len(w):4d} {w.mean():7.4f} {w.std(ddof=1)/np.sqrt(len(w)):13.4f} "
          f"{w.std():6.3f} {np.percentile(w,10):6.3f} {np.percentile(w,90):6.3f} {w.max():6.3f}")
allw = np.array([v["win_rate"] for v in res.values()])
print(f"\nwhole dataset: {len(allw)} labelled teams, win rate {allw.min():.3f} - {allw.max():.3f}, "
      f"sd {allw.std():.3f}")
