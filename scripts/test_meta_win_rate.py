"""Run: /tmp/vgc-pilot/.venv/bin/python scripts/test_meta_win_rate.py   (no battles, no Showdown).

Spec tests for src/meta_win_rate.py, the one door to "what is each team's win rate against the top-50 meta".
Every test goes through win_rates / final_evaluation only. Two fakes stand in at the seam: a runner
whose outcome is fixed per opponent, and a validator that rejects one marked paste."""
import sys, unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import heldout as H
import matchup_db as M
import meta_win_rate as W

POOL = [f for f in sorted(M.POOL_DIR.glob("MB*.txt")) if f.stem not in set(H.ids())][:8]
TOP, CAND = POOL[:4], POOL[4:]

class FakeRunner:
    """Wins every battle against an opponent in `beats`, loses every other one. Records each battle."""
    def __init__(self, beats=()):
        self.beats, self.battles = set(beats), []
    def __call__(self, con, jobs, seed=None):
        out = []
        for j in jobs:
            for c in j["schedule"]:
                self.battles.append((j["row"], c))
                out.append((j["row"], c, 1, 0, 0) if c in self.beats else (j["row"], c, 0, 1, 0))
        return out, 0, 0.0

ILLEGAL = M.canonical_paste(CAND[1].read_text())

class DroppingRunner(FakeRunner):
    """Returns no result for the teams in `drop` on its first `times` calls, the way a crashed shard does."""
    def __init__(self, drop, times, beats=()):
        super().__init__(beats); self.drop, self.times, self.calls = set(drop), times, 0
    def __call__(self, con, jobs, seed=None):
        self.calls += 1
        if self.calls <= self.times: jobs = [j for j in jobs if j["row"] not in self.drop]
        return super().__call__(con, jobs, seed)

def legal(paste):
    """Every paste is legal except CAND[1], which fails the way Showdown's validator reports it."""
    return "Pokemon 1 is not legal" if paste == ILLEGAL else None

class Fixture(unittest.TestCase):
    """An in-memory matchup cache whose 'top50' set is four pool teams."""
    def setUp(self):
        self.con = M.connect(":memory:")
        self.top = [M.add_team(self.con, f.read_text(), "test")[0] for f in TOP]
        with self.con:
            for t in self.top: self.con.execute("INSERT INTO team_set VALUES ('top50', ?, 1)", (t,))
    def tearDown(self):
        self.con.close()
    def run_win_rates(self, pastes, runner, **kw):
        return W.win_rates(pastes, con=self.con, runner=runner, validator=legal, **kw)

class WinRates(Fixture):
    def test_legal_team_scored_against_top50(self):
        runner = FakeRunner(beats=self.top[:3])
        [r] = self.run_win_rates([CAND[0].read_text()], runner, per_cell=2)
        self.assertEqual(r.status, "ok")
        self.assertEqual(r.win_rate, 0.75)                  # beats 3 of the 4 top-50 teams
        self.assertEqual(r.battles, 8)                      # 4 opponents x 2 battles
        self.assertEqual(r.coverage, 4)

    def test_illegal_team_is_never_battled(self):
        runner = FakeRunner(beats=self.top)
        ok, bad = self.run_win_rates([CAND[0].read_text(), CAND[1].read_text()], runner, per_cell=1)
        self.assertEqual((ok.status, bad.status), ("ok", "illegal"))
        self.assertIsNone(bad.win_rate)
        self.assertEqual(bad.battles, 0)
        self.assertNotIn(bad.team_id, {row for row, _ in runner.battles})

    def test_team_cached_without_a_legality_check_is_checked_now(self):
        for f in CAND[:2]: M.add_team(self.con, f.read_text(), "seeded without validation")
        runner = FakeRunner(beats=self.top)
        ok, bad = self.run_win_rates([CAND[0].read_text(), CAND[1].read_text()], runner, per_cell=1)
        self.assertEqual((ok.status, bad.status), ("ok", "illegal"))

    def test_heldout_team_is_never_battled(self):
        """Ranks 51-100 are never a candidate or an opponent during search, even when one is in the set."""
        stray = M.add_team(self.con, Path(H.files()[1]).read_text(), "test")[0]
        with self.con: self.con.execute("INSERT INTO team_set VALUES ('top50', ?, 1)", (stray,))
        runner = FakeRunner(beats=self.top)
        ok, held = self.run_win_rates([CAND[0].read_text(), Path(H.files()[0]).read_text()], runner, per_cell=1)
        self.assertEqual((ok.status, held.status), ("ok", "heldout"))
        self.assertEqual((ok.win_rate, ok.coverage), (1.0, 4))
        self.assertEqual(held.battles, 0)
        battled = {t for pair in runner.battles for t in pair}
        self.assertFalse(battled & {held.team_id, stray})

    def test_same_team_in_another_slot_order_is_battled_once(self):
        paste = CAND[0].read_text()
        reordered = "\n\n".join(paste.strip().split("\n\n")[::-1]) + "\n"
        runner = FakeRunner(beats=self.top[:1])
        a, b = self.run_win_rates([paste, reordered], runner, per_cell=2)
        self.assertEqual(a, b)
        self.assertEqual(a.win_rate, 0.25)
        self.assertEqual(len(runner.battles), 8)            # one team x 4 opponents x 2 battles, not twice that

    def test_second_call_on_the_same_teams_starts_no_battles(self):
        pastes = [CAND[0].read_text(), CAND[2].read_text()]
        first = self.run_win_rates(pastes, FakeRunner(beats=self.top[:2]), per_cell=2)
        again = FakeRunner(beats=())                         # would score 0.0 if it were asked to battle
        second = self.run_win_rates(pastes, again, per_cell=2)
        self.assertEqual(again.battles, [])
        self.assertEqual(first, second)

    def test_budget_caps_battles_and_marks_the_teams_left_over(self):
        """Each team needs 8 battles (4 opponents x 2); a budget of 20 covers two teams, not a third."""
        runner = FakeRunner(beats=self.top)
        rs = self.run_win_rates([CAND[0].read_text(), CAND[2].read_text(), CAND[3].read_text()], runner,
                                per_cell=2, budget=20)
        self.assertEqual([r.status for r in rs], ["ok", "ok", "budget_exhausted"])
        self.assertEqual(len(runner.battles), 16)
        self.assertIsNone(rs[2].win_rate)

    def test_cached_teams_cost_nothing_against_the_budget(self):
        self.run_win_rates([CAND[0].read_text()], FakeRunner(beats=self.top), per_cell=2)
        runner = FakeRunner(beats=self.top)
        rs = self.run_win_rates([CAND[0].read_text(), CAND[2].read_text()], runner, per_cell=2, budget=8)
        self.assertEqual([r.status for r in rs], ["ok", "ok"])
        self.assertEqual(len(runner.battles), 8)

    def test_team_with_no_result_is_retried(self):
        flaky = M.add_team(self.con, CAND[2].read_text(), "test")[0]
        runner = DroppingRunner(drop=[flaky], times=1, beats=self.top[:2])
        a, b = self.run_win_rates([CAND[0].read_text(), CAND[2].read_text()], runner, per_cell=1)
        self.assertEqual((a.status, b.status), ("ok", "ok"))
        self.assertEqual(b.win_rate, 0.5)

    def test_team_that_never_returns_a_result_is_marked_failed(self):
        dead = M.add_team(self.con, CAND[2].read_text(), "test")[0]
        runner = DroppingRunner(drop=[dead], times=99, beats=self.top)
        a, b = self.run_win_rates([CAND[0].read_text(), CAND[2].read_text()], runner, per_cell=1)
        self.assertEqual((a.status, b.status), ("ok", "failed"))
        self.assertIsNone(b.win_rate)
        self.assertEqual(runner.calls, 3)                   # the first try and two retries

class FinalEvaluation(Fixture):
    """The one door to ranks 51-100: the final selected teams, against the held-out set only."""
    def test_final_teams_are_battled_against_the_heldout_set_only(self):
        runner = FakeRunner(beats=())
        [r] = W.final_evaluation([CAND[0].read_text()], per_cell=1, con=self.con, runner=runner, validator=legal)
        held = H.team_ids(self.con)
        opponents = {c for _, c in runner.battles}
        self.assertEqual(opponents, set(M.set_members(self.con, H.SET_NAME)))
        self.assertTrue(opponents <= held)
        self.assertFalse(opponents & set(self.top))
        self.assertEqual((r.status, r.win_rate), ("ok", 0.0))

    def test_a_heldout_team_cannot_be_a_final_team(self):
        runner = FakeRunner(beats=())
        [r] = W.final_evaluation([Path(H.files()[0]).read_text()], per_cell=1, con=self.con, runner=runner,
                                 validator=legal)
        self.assertEqual(r.status, "heldout")
        self.assertEqual(runner.battles, [])

if __name__ == "__main__":
    unittest.main()
