"""How far does battle throughput scale with independent (Showdown server, worker) pairs?

The single-threaded Node server is the suspected bottleneck: 1 python process gave
8.0 battles/s and 4 processes only 19.4, which is 2.4x for 4x the workers. If the
server is the cap, giving each worker its OWN server should scale far better.
"""
import asyncio, os, sys, time
from pathlib import Path
import torch
from poke_env.ps_client import ServerConfiguration, AccountConfiguration
from stable_baselines3 import PPO
from vgc_bench.src.policy_player import BatchPolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map

FMT = format_map["mb"]

async def run(port, n, conc, t1, t2):
    srv = ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                              "https://play.pokemonshowdown.com/action.php?")
    pol = PPO.load("/tmp/bc_100.zip", device=torch.device("cpu")).policy
    kw = dict(server_configuration=srv, battle_format=FMT, log_level=40,
              max_concurrent_battles=conc, accept_open_team_sheet=True, open_timeout=None)
    tag = os.urandom(3).hex()
    a = BatchPolicyPlayer(policy=pol, account_configuration=AccountConfiguration(f"p{tag}", None),
                          team=RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(t1)]), **kw)
    b = BatchPolicyPlayer(policy=pol, account_configuration=AccountConfiguration(f"q{tag}", None),
                          team=RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(t2)]), **kw)
    await a.battle_against(b, n_battles=n)
    return a.n_finished_battles

if __name__ == "__main__":
    port, n, conc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    t0 = time.perf_counter()
    done = asyncio.run(run(port, n, conc, "teams/reg_mb/MB1.txt", "teams/reg_mb/MB300.txt"))
    dt = time.perf_counter() - t0
    print(f"PORT {port} battles {done} secs {dt:.1f} rate {done/dt:.2f}", flush=True)
