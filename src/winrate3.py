"""Final arms: does the learned local move beat a random slot copy on WIN RATE,
and does playstyle guidance help or cost win rate when combined with f?"""
import json, asyncio, os
from pathlib import Path
import numpy as np, torch
from stable_baselines3 import PPO
import winrate as W

root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
opp_ids = [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]
opp = []
for i in opp_ids:
    for c in (root/f"{i}.txt", root/"featured"/f"{i}.txt"):
        if c.exists(): opp.append(c); break
opp = opp[:16]
P = Path("/tmp/vgc-pilot/proposals")
arms = {"modeledit":  [str(p) for p in sorted(P.glob("modeledit_*.txt"))][:16],
        "modelstyle": [str(p) for p in sorted(P.glob("modelstyle_*.txt"))][:16]}
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
        print(f"  {arm} [{ti+1}/{len(files)}] wr {tw/max(tf_,1):.3f} ({fin} battles)", flush=True)
    wr = wins/max(fin,1); se = (wr*(1-wr)/max(fin,1))**.5
    res[arm] = dict(win_rate=round(wr,4), se=round(se,4), battles=fin, per_team=per)
    print(f"== {arm}: {wr:.4f} +/- {se:.4f} over {fin} battles", flush=True)
json.dump(res, open("/tmp/vgc-pilot/winrate3.json","w"), indent=2)
print("wrote /tmp/vgc-pilot/winrate3.json")
