"""Score teams by expected win rate against the top-50 meta opponent set.

Both sides are piloted by the same behaviour-cloning policy from VGC-Bench, so the
comparison isolates the TEAM: any win-rate difference between arms is attributable
to the team, not to the pilot.

Arms:
  diffusion - teams sampled from scratch by the constrained masked-diffusion model
  slotcopy  - a corpus team with one candidate replaced by another corpus team's
  corpus    - unmodified corpus teams (the control: is an edit even an improvement?)

This is the measurement that legality could never substitute for.
"""
import argparse, asyncio, json, os, random, sys, time
from pathlib import Path
import numpy as np, torch
from poke_env.ps_client import ServerConfiguration, AccountConfiguration
from stable_baselines3 import PPO
from vgc_bench.src.policy_player import BatchPolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map

SERVER = ServerConfiguration("ws://localhost:8123/showdown/websocket",
                             "https://play.pokemonshowdown.com/action.php?")
FMT = format_map["mb"]
CKPT = "/tmp/bc_100.zip"

async def play(policy, team_path, opp_path, n, conc, tag):
    # SERVER is read at call time so a shard can point it at its own port
    kw = dict(server_configuration=SERVER, battle_format=FMT, log_level=40,
              max_concurrent_battles=conc, accept_open_team_sheet=True, open_timeout=None)
    a = BatchPolicyPlayer(policy=policy,
                          account_configuration=AccountConfiguration(f"a{tag}", None),
                          team=RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(team_path)]), **kw)
    b = BatchPolicyPlayer(policy=policy,
                          account_configuration=AccountConfiguration(f"b{tag}", None),
                          team=RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(opp_path)]), **kw)
    await a.battle_against(b, n_battles=n)
    w = sum(1 for x in a.battles.values() if x.won)
    f = a.n_finished_battles
    for p in (a, b):
        try: await p.ps_client.stop_listening()
        except Exception: pass
    return w, f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams-per-arm", type=int, default=16)
    ap.add_argument("--opponents", type=int, default=16)
    ap.add_argument("--battles", type=int, default=4)
    ap.add_argument("--conc", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/vgc-pilot/winrate.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    root = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
    opp_ids = [t["id"] for t in json.load(open("/tmp/vgc-pilot/top50_evs.json"))]
    opp_files = []
    for i in opp_ids:
        for c in (root/f"{i}.txt", root/"featured"/f"{i}.txt"):
            if c.exists(): opp_files.append(c); break
    opp_files = opp_files[:a.opponents]
    print(f"opponent set: {len(opp_files)} of the top-50 meta teams")

    arms = {}
    gen = sorted(Path("/tmp/vgc-pilot/valid_gen").glob("*.txt"))
    arms["diffusion"] = [str(p) for p in gen[:a.teams_per_arm]]
    prop = Path("/tmp/vgc-pilot/proposals")
    sc = sorted(prop.glob("slotcopy_*.txt")) if prop.exists() else []
    if sc: arms["slotcopy"] = [str(p) for p in sc[:a.teams_per_arm]]
    allc = sorted(root.glob("*.txt"))
    ctrl = [str(allc[i]) for i in rng.choice(len(allc), a.teams_per_arm, replace=False)]
    arms["corpus"] = ctrl
    for k, v in arms.items(): print(f"  arm {k}: {len(v)} teams")

    policy = PPO.load(CKPT, device=torch.device("cpu")).policy
    res, t0 = {}, time.perf_counter()
    for arm, files in arms.items():
        wins = fin = 0; per = []
        for ti, tf in enumerate(files):
            tw = tf_ = 0
            for oi, of in enumerate(opp_files):
                tag = os.urandom(3).hex()
                w, f = asyncio.run(play(policy, tf, str(of), a.battles, a.conc, tag))
                tw += w; tf_ += f
            wins += tw; fin += tf_
            per.append(dict(team=Path(tf).name, wins=tw, battles=tf_,
                            wr=round(tw/max(tf_,1), 4)))
            print(f"  {arm:9s} [{ti+1}/{len(files)}] {Path(tf).name:18s} "
                  f"wr {tw/max(tf_,1):.3f}  ({fin} battles, {time.perf_counter()-t0:.0f}s)", flush=True)
        wr = wins/max(fin,1); se = (wr*(1-wr)/max(fin,1))**.5
        res[arm] = dict(win_rate=round(wr,4), se=round(se,4), battles=fin, per_team=per)
        print(f"== {arm}: win rate {wr:.4f} +/- {se:.4f} over {fin} battles")
    json.dump(dict(config=vars(a), opponents=[p.name for p in opp_files], arms=res),
              open(a.out,"w"), indent=2)
    print("\nwrote", a.out)
    for k,v in res.items(): print(f"  {k:9s} {v['win_rate']:.4f} +/- {v['se']:.4f}  (n={v['battles']})")

if __name__ == "__main__":
    main()
