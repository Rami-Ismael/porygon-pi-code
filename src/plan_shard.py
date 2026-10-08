"""One shard of plan-vs-plan battles, on a single Showdown server.

Team preview is forced instead of left to the policy: each side brings a given plan (two leads,
two in the back). Everything after team preview is the behaviour-cloning policy on both sides,
as in every other cell of the matchup matrix.

A job is one row plan against an ordered schedule of column plans (one player pair per job, so
setup cost is per row plan, not per cell). The column player takes the next plan from the
schedule at each team preview and records which plan it brought in that battle, so every
finished battle is attributed to exactly one (row plan, column plan) cell.

A plan is written as species ids: "lead1+lead2/back1+back2", each pair sorted. Slots are found by
species at team preview, so the plan does not depend on the order of the team file.

Spec: {"jobs": [{"row_file", "col_file", "row_plan", "schedule": [plans]}], "conc", "port", "policy"}
Out:  [{"row_plan", "by_col": {col_plan: [row_wins, col_wins, ties]}, "finished", "scheduled"}]
"""
import asyncio, json, os, sys
from collections import defaultdict
from pathlib import Path
import torch
from poke_env.ps_client import ServerConfiguration, AccountConfiguration
from stable_baselines3 import PPO
from vgc_bench.src.policy_player import BatchPolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder
from vgc_bench.src.utils import format_map


def _norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def order_for(battle, plan):
    """'/team abcd' for a plan, by species; marks the four as selected (the policy's embedding expects it)."""
    leads, back = [p.split("+") for p in plan.split("/")]
    mons = list(battle.team.values())
    idx = {_norm(m.species): i for i, m in enumerate(mons)}
    picks = [idx[s] for s in leads + back]
    for i in picks:
        mons[i]._selected_in_teampreview = True
    return "/team " + "".join(str(i + 1) for i in picks)


def obeys(seen, plan):
    """Every species seen on the field is one of the plan's four (a Mega counts as its base species)."""
    four = plan.replace("/", "+").split("+")
    return all(any(sp.startswith(f) for f in four) for sp in seen)


class FixedPlanPlayer(BatchPolicyPlayer):
    """Brings the same plan in every battle."""
    def __init__(self, plan, *a, **kw):
        super().__init__(*a, **kw); self._plan = plan
        self.seen = defaultdict(lambda: (set(), set()))     # battle -> (my actives, their actives), for the plan check

    def choose_move(self, battle):
        mine, theirs = self.seen[battle.battle_tag]
        for m in battle.active_pokemon:
            if m: mine.add(_norm(m.species))
        for m in battle.opponent_active_pokemon:
            if m: theirs.add(_norm(m.species))
        return super().choose_move(battle)

    def teampreview(self, battle):
        async def go():
            return order_for(battle, self._plan)
        return go()


class ScheduledPlanPlayer(BatchPolicyPlayer):
    """Brings the next plan of a fixed schedule at each team preview and remembers which."""
    def __init__(self, schedule, *a, **kw):
        super().__init__(*a, **kw); self._schedule = list(schedule); self._i = 0; self.plan_of = {}

    def teampreview(self, battle):
        plan = self._schedule[self._i % len(self._schedule)]; self._i += 1
        self.plan_of[battle.battle_tag] = plan
        async def go():
            return order_for(battle, plan)
        return go()


async def run_job(pol, job, conc, port):
    srv = ServerConfiguration(f"ws://localhost:{port}/showdown/websocket",
                              "https://play.pokemonshowdown.com/action.php?")
    kw = dict(server_configuration=srv, battle_format=format_map["mb"], log_level=40,
              max_concurrent_battles=conc, accept_open_team_sheet=True, open_timeout=None)
    tag = os.urandom(4).hex()
    a = FixedPlanPlayer(job["row_plan"], policy=pol, account_configuration=AccountConfiguration(f"p{tag}", None),
                        team=RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(job["row_file"])]), **kw)
    b = ScheduledPlanPlayer(job["schedule"], policy=pol, account_configuration=AccountConfiguration(f"q{tag}", None),
                            team=RandomTeamBuilder(1, None, "mb", custom_team_paths=[Path(job["col_file"])]), **kw)
    await a.battle_against(b, n_battles=len(job["schedule"]))
    by_col, finished, lost, violations = defaultdict(lambda: [0, 0, 0]), 0, 0, 0
    for t, x in a.battles.items():
        if not x.finished: continue
        col = b.plan_of.get(t)
        if col is None: lost += 1; continue
        finished += 1
        # check the plan was obeyed: every Pokémon that took the field is one of that side's four
        mine, theirs = a.seen[t]
        if not obeys(mine, job["row_plan"]) or not obeys(theirs, col):
            violations += 1
        r = by_col[col]
        if x.won is True: r[0] += 1
        elif x.won is False: r[1] += 1
        else: r[2] += 1
    if os.environ.get("PLAN_DEBUG"):
        t = next(iter(a.battles)); print("DEBUG", a.seen[t], job["row_plan"], b.plan_of.get(t), flush=True)
    for p in (a, b):
        try: await p.ps_client.stop_listening()
        except Exception: pass
    return {"row_plan": job["row_plan"], "by_col": dict(by_col), "finished": finished, "unattributed": lost, "plan_violations": violations,
            "scheduled": len(job["schedule"])}


def main():
    spec = json.load(open(sys.argv[1])); out = sys.argv[2]
    pol = PPO.load(spec["policy"], device=torch.device("cpu")).policy
    res = []
    for job in spec["jobs"]:
        res.append(asyncio.run(run_job(pol, job, spec["conc"], spec["port"])))
        json.dump(res, open(out, "w"))       # written after every job, so a crash keeps finished rows
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
