"""One shard of a labelling campaign, on a single Showdown server.

Each team plays N battles against an opponent who draws a FRESH team from the top-50
meta pool every battle (RandomTeamBuilder.yield_team samples uniformly). That is the
same quantity as "win rate against the meta" but costs one player-pair setup per
team instead of one per opponent — the pilot showed setup overhead, not simulation,
was the bottleneck (10.5 battles/sec at 2 battles per call vs 63 at 50).
"""
import asyncio, json, os, random, sys
from pathlib import Path
import torch
from poke_env.ps_client import ServerConfiguration, AccountConfiguration
from stable_baselines3 import PPO
from vgc_bench.src.policy_player import BatchPolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map

class ScheduledTeamBuilder(RandomTeamBuilder):
    """Opponent builder that cycles a FIXED ordered schedule instead of drawing
    uniformly — common random numbers across candidates (added 2026-09-02 for the
    TRS port; every candidate in a batch faces the identical opponent multiset,
    which is the dominant reducible variance at n_battles < pool size)."""
    def __init__(self, schedule, *args, **kw):
        super().__init__(*args, **kw)
        self._schedule = [Path(p) for p in schedule]
        self._si = 0
    def yield_team(self):
        p = self._schedule[self._si % len(self._schedule)]
        self._si += 1
        return self._load_team(p)

async def score_one(pol, team, opps, n, conc, port, opp_schedule=None, seed=None):
    if seed is not None:
        # per-candidate reseed: opponent draws and policy action sampling start
        # from the same state for every candidate (battles diverge after that)
        import numpy as _np
        random.seed(seed); torch.manual_seed(seed); _np.random.seed(seed)
    srv = ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                              "https://play.pokemonshowdown.com/action.php?")
    kw = dict(server_configuration=srv, battle_format=format_map["mb"], log_level=40,
              max_concurrent_battles=conc, accept_open_team_sheet=True, open_timeout=None)
    tag = os.urandom(4).hex()
    a = BatchPolicyPlayer(policy=pol, account_configuration=AccountConfiguration(f"a{tag}", None),
                          team=RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(team)]), **kw)
    if opp_schedule:
        tb = ScheduledTeamBuilder(opp_schedule, 1, None, "mb",
                                  custom_team_paths=[Path(o) for o in opps])
    else:
        tb = RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(o) for o in opps])
    b = BatchPolicyPlayer(policy=pol, account_configuration=AccountConfiguration(f"b{tag}", None),
                          team=tb, **kw)
    await a.battle_against(b, n_battles=n)
    w = sum(1 for x in a.battles.values() if x.won)
    f = a.n_finished_battles
    m = margin_of(a.battles.values())
    # mean battle length in turns (added 2026-09-30 for the MAP-Elites measure; extra key, nothing else changes)
    tl = [x.turn for x in a.battles.values() if x.finished]
    t = round(sum(tl) / len(tl), 3) if tl else None
    for p in (a, b):
        try: await p.ps_client.stop_listening()
        except Exception: pass
    return w, f, m, t


def margin_of(battles):
    """Mean faint differential, rescaled to [0, 1]. None if poke-env won't say.

    Win/loss is Bernoulli, so at 24 battles most of a weak team's label variance is
    coin noise (73% on the uniform-legal pool, 2026-08-29). The margin -- how many
    Pokemon were left standing, not merely who was -- is the same battles at finer
    resolution, which is the free variance reduction Fontaine et al. (GECCO 2019)
    use. VGC brings 4, so the differential runs -4..+4 and 0.5 is a dead draw.
    Added 2026-09-01 alongside `wins`, never in place of it: the objective f stays
    expected WIN rate, and the margin is only a surrogate/elite-selection signal.
    """
    try:
        vals = []
        for x in battles:
            ours = sum(1 for p in x.team.values() if p.fainted)
            theirs = sum(1 for p in x.opponent_team.values() if p.fainted)
            vals.append(min(1.0, max(0.0, 0.5 + (theirs - ours) / 8.0)))
        return round(sum(vals) / len(vals), 4) if vals else None
    except Exception:
        return None

def main():
    spec = json.load(open(sys.argv[1])); out = sys.argv[2]
    port, nb, conc = spec["port"], spec["battles"], spec["conc"]
    pol = PPO.load("/tmp/bc_100.zip", device=torch.device("cpu")).policy
    res = {}
    sched, seed = spec.get("opp_schedule"), spec.get("seed")
    for team, opps in spec["jobs"]:
        w, n, m, t = asyncio.run(score_one(pol, team, opps, nb, conc, port, sched, seed))
        r = w / max(n, 1)
        res[team] = {"wins": w, "battles": n, "win_rate": round(r, 4),
                     "se": round((r * (1 - r) / max(n, 1)) ** .5, 4), "margin": m, "turns": t}
        json.dump(res, open(out, "w"))
    json.dump(res, open(out, "w"))

if __name__ == "__main__":
    main()
