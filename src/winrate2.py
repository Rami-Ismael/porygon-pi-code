"""Second pass: the shuffled-spread control arm, run against the same opponents
with the same pilot so it is directly comparable to the first pass."""
import sys
sys.argv = ["winrate.py", "--teams-per-arm", "16", "--opponents", "16",
            "--battles", "8", "--conc", "25", "--out", "/tmp/vgc-pilot/winrate2.json"]
import json, asyncio, os, time
from pathlib import Path
import numpy as np, torch
from stable_baselines3 import PPO
import winrate as W

rng = np.random.default_rng(0)
root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
opp_ids = [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]
opp = []
for i in opp_ids:
    for c in (root/f"{i}.txt", root/"featured"/f"{i}.txt"):
        if c.exists(): opp.append(c); break
opp = opp[:16]
arms = {"shuffled": [str(p) for p in sorted(Path("/tmp/vgc-pilot/proposals").glob("shuffled_*.txt"))][:16]}
policy = PPO.load(W.CKPT, device=torch.device("cpu")).policy
res = {}
for arm, files in arms.items():
    wins = fin = 0; per = []
    for ti, tf in enumerate(files):
        tw = tf_ = 0
        for of in opp:
            w, f = asyncio.run(W.play(policy, tf, str(of), 8, 25, os.urandom(3).hex()))
            tw += w; tf_ += f
        wins += tw; fin += tf_
        per.append(dict(team=Path(tf).name, wins=tw, battles=tf_, wr=round(tw/max(tf_,1),4)))
        print(f"  {arm} [{ti+1}/{len(files)}] {Path(tf).name:16s} wr {tw/max(tf_,1):.3f} ({fin} battles)", flush=True)
    wr = wins/max(fin,1); se = (wr*(1-wr)/max(fin,1))**.5
    res[arm] = dict(win_rate=round(wr,4), se=round(se,4), battles=fin, per_team=per)
    print(f"== {arm}: {wr:.4f} +/- {se:.4f} over {fin} battles")
json.dump(res, open("/tmp/vgc-pilot/winrate2.json","w"), indent=2)
print("wrote /tmp/vgc-pilot/winrate2.json")
