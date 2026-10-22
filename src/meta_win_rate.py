"""Win rate against the top-50 meta: the one door every search loop asks through.

win_rates(pastes) adds each paste to the matchup cache, battles only the cells it is missing against
the top-50 meta (behaviour-cloning policy on both sides), and returns one WinRate per paste, in order.
Spec: scripts/test_meta_win_rate.py."""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import heldout as H
import matchup_db as M

@dataclass(frozen=True)
class WinRate:
    team_id: int
    status: str                 # ok | illegal | heldout | budget_exhausted | failed; win_rate is None unless ok
    win_rate: float | None
    se: float | None
    battles: int
    coverage: int               # opponents with at least one battle

def win_rates(pastes, *, against="top50", per_cell=8, budget=None, seed=None, con=None, runner=None, validator=None):
    con = con or M.connect()
    policy_id = M.get_policy(con)
    ids = _add(con, pastes, validator)
    held = H.team_ids(con)                                   # ranks 51-100: no search fitness, never an opponent
    legal = _legal(con, ids) - held
    cols = [c for c in M.set_members(con, against) if c not in held]
    funded = _within_budget(con, policy_id, ids, legal, cols, per_cell, budget)
    missing = _battle(con, policy_id, funded, cols, per_cell, seed, runner)
    out = []
    for t in ids:
        if t in held:
            out.append(WinRate(t, "heldout", None, None, 0, 0)); continue
        if t not in legal:
            out.append(WinRate(t, "illegal", None, None, 0, 0)); continue
        if t not in funded:
            out.append(WinRate(t, "budget_exhausted", None, None, 0, 0)); continue
        if t in missing:
            out.append(WinRate(t, "failed", None, None, 0, 0)); continue
        s = M.score(con, policy_id, t, against)
        out.append(WinRate(t, "ok", s["score"], s["se"], s["battles"], s["coverage"]))
    return out

def _add(con, pastes, validator):
    """Team ids, one per paste. Legality is checked once per team: when it enters the cache, or now
    if it entered without a check (seed(validate=False), a test fixture)."""
    own = validator is None
    validator = validator or M.Validator()
    try:
        with con:
            ids = [M.add_team(con, p, "win_rates", validator)[0] for p in pastes]
            q = ",".join("?" * len(ids))
            for t, canon in con.execute(f"SELECT team_id, paste FROM team WHERE legal IS NULL AND team_id IN ({q})", ids).fetchall():
                err = validator(canon)
                con.execute("UPDATE team SET legal=?, legal_error=? WHERE team_id=?", (int(err is None), err, t))
        return ids
    finally:
        if own: validator.close()

def final_evaluation(pastes, *, per_cell=96, seed=None, con=None, runner=None, validator=None):
    """The only door to ranks 51-100: score the FINAL selected teams against the held-out set, after
    selection is over. A held-out team is not a result, so it comes back 'heldout' and unbattled."""
    con = con or M.connect()
    policy_id = M.get_policy(con)
    ids = _add(con, pastes, validator)
    held = H.team_ids(con)
    legal = _legal(con, ids) - held
    finals = sorted(legal)
    scores = dict(zip(finals, M.evaluate_final(con, policy_id, finals, per_cell=per_cell, seed=seed, runner=runner)))
    out = []
    for t in ids:
        if t in held:
            out.append(WinRate(t, "heldout", None, None, 0, 0)); continue
        if t not in legal:
            out.append(WinRate(t, "illegal", None, None, 0, 0)); continue
        s = scores[t]
        out.append(WinRate(t, "ok", s["score"], s["se"], s["battles"], s["coverage"]))
    return out

RETRIES = 2

def _battle(con, policy_id, teams, cols, per_cell, seed, runner):
    """Fill every cell the teams still need, re-asking for the ones a crashed shard left empty.
    Returns the teams still short of per_cell after the retries."""
    missing = set(teams)
    for _ in range(1 + RETRIES):
        M.ensure(con, policy_id, sorted(missing), cols, per_cell, seed=seed, runner=runner)
        missing = {r for r, _, _ in M.plan(con, policy_id, sorted(missing), cols, per_cell)}
        if not missing: break
    return missing

def _within_budget(con, policy_id, ids, legal, cols, per_cell, budget):
    """Teams to battle, in input order. A team is funded only when every battle it still needs fits in
    what is left, so a score is never built from part of the meta. Cached cells cost nothing."""
    if budget is None: return set(legal)
    funded, left = set(), budget
    for t in dict.fromkeys(ids):
        if t not in legal: continue
        cost = sum(d for _, _, d in M.plan(con, policy_id, [t], cols, per_cell))
        if cost <= left: funded.add(t); left -= cost
    return funded

def _legal(con, ids):
    q = ",".join("?" * len(ids))
    return {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({q})", ids)}
