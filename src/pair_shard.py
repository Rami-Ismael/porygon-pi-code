"""One shard of matchup-matrix battles, on a single Showdown server.

Like shard.py, each job is one row team against a fixed ordered schedule of column
teams (one player pair per job, so setup overhead stays per row, not per cell). The
difference: results are returned PER COLUMN. The column team of each finished battle
is read from the six species it showed at team preview; matchup_db.jobs_for guarantees
every column in a job has a distinct species set. Battles that cannot be matched are
counted as unattributed and never written to a cell.

Spec: {"jobs": [{"row", "row_file", "schedule": [files], "key2col", "base2col"}], "conc", "port", "seed"}
Out:  [{"row", "by_col": {col: [row_wins, col_wins, ties]}, "unattributed", "finished"}]
"""
import asyncio, json, os, random, sys
from collections import defaultdict
from pathlib import Path
import torch
from poke_env.ps_client import ServerConfiguration, AccountConfiguration
from stable_baselines3 import PPO
from vgc_bench.src.policy_player import BatchPolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map
from shard import ScheduledTeamBuilder

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import dex_entry

def _norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())

def _base(s):
    e = dex_entry(s)
    return _norm(e.get("baseSpecies", s)) if e else _norm(s)

async def run_job(pol, job, conc, port, seed):
    if seed is not None:
        import numpy as _np
        random.seed(seed); torch.manual_seed(seed); _np.random.seed(seed)
    srv = ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                              "https://play.pokemonshowdown.com/action.php?")
    kw = dict(server_configuration=srv, battle_format=format_map["mb"], log_level=40,
              max_concurrent_battles=conc, accept_open_team_sheet=True, open_timeout=None)
    tag = os.urandom(4).hex()
    a = BatchPolicyPlayer(policy=pol, account_configuration=AccountConfiguration(f"r{tag}", None),
                          team=RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(job["row_file"])]), **kw)
    cols = sorted(set(job["schedule"]))
    b = BatchPolicyPlayer(policy=pol, account_configuration=AccountConfiguration(f"c{tag}", None),
                          team=ScheduledTeamBuilder(job["schedule"], 1, None, "mb",
                                                    custom_team_paths=[Path(c) for c in cols]), **kw)
    await a.battle_against(b, n_battles=len(job["schedule"]))
    by_col, unattributed, finished = defaultdict(lambda: [0, 0, 0]), 0, 0
    for x in a.battles.values():
        if not x.finished: continue
        finished += 1
        seen = [m.species for m in x.teampreview_opponent_team]
        full = ",".join(sorted(_norm(s) for s in seen))
        col = job["key2col"].get(full)
        if col is None:
            col = job["base2col"].get(",".join(sorted(_base(s) for s in seen)))
        if col is None or len(seen) != 6:
            unattributed += 1; continue
        t = by_col[str(col)]
        if x.won is True: t[0] += 1
        elif x.won is False: t[1] += 1
        else: t[2] += 1
    for p in (a, b):
        try: await p.ps_client.stop_listening()
        except Exception: pass
    return {"row": job["row"], "by_col": dict(by_col), "unattributed": unattributed, "finished": finished,
            "scheduled": len(job["schedule"])}

def main():
    spec = json.load(open(sys.argv[1])); out = sys.argv[2]
    pol = PPO.load("/tmp/bc_100.zip", device=torch.device("cpu")).policy
    res = []
    for job in spec["jobs"]:
        res.append(asyncio.run(run_job(pol, job, spec["conc"], spec["port"], spec.get("seed"))))
        json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w"))

if __name__ == "__main__":
    main()
