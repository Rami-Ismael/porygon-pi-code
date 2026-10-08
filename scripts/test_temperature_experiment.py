"""Numerical invariants and campaign isolation checks; no simulator required."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from boltzmann_selection import (ARMS, adaptive_temperature, boltz_weights,
                                 concentration, linear_temperature, select_boltz)
from temperature_experiment import (Config, diversity, phase_seed, opponent_schedule,
                                    require_scores, run)


def paste(i):
    return "\n\n".join(f"Species{x + i} @ Item{x}\nAbility: A\n- Move1\n- Move2" for x in range(6))


class SelectionTest(unittest.TestCase):
    def test_softmax_shift_and_ratios(self):
        y = np.array([0., .2, .7, 1.])
        for temp in (.1, .2, .3):
            w = boltz_weights(y, temp)
            np.testing.assert_allclose(w, boltz_weights(y + 1000, temp), rtol=1e-11)
            self.assertAlmostEqual(w.sum(), 1.)
            self.assertAlmostEqual(w[2] / w[1], np.exp(.5 / temp))
        self.assertLess(concentration(boltz_weights(y, .1))["ess"],
                        concentration(boltz_weights(y, .3))["ess"])

    def test_invalid_and_extreme_inputs(self):
        for y, temp in (([], .1), ([float("nan")], .1), ([.5], 0), ([.5], -1),
                        ([[.2]], .1), ([.5], float("inf"))):
            with self.assertRaises(ValueError):
                boltz_weights(y, temp)
        np.testing.assert_equal(boltz_weights([0, 1], 1e-320), [0, 1])

    def test_linear_endpoints_and_hold(self):
        self.assertEqual(linear_temperature(1), .3)
        self.assertAlmostEqual(linear_temperature(6), .2)
        self.assertAlmostEqual(linear_temperature(11), .1)
        self.assertAlmostEqual(linear_temperature(100), .1)
        self.assertEqual(linear_temperature(1, 1), .3)
        with self.assertRaises(ValueError):
            linear_temperature(0)

    def test_adaptive_target_and_unreachable_ties(self):
        y = np.linspace(0, 1, 100)
        temp, diagnostic = adaptive_temperature(y, target=.4)
        self.assertTrue(.1 < temp < .3)
        self.assertAlmostEqual(diagnostic["ess_fraction"], .4)
        self.assertEqual(diagnostic["status"], "target")
        temp, diagnostic = adaptive_temperature([.5] * 20)
        self.assertEqual(temp, .1)
        self.assertEqual(diagnostic["status"], "lower_bound")
        self.assertAlmostEqual(diagnostic["ess_fraction"], 1.)
        self.assertEqual(adaptive_temperature(y, target=.99)[1]["status"], "upper_bound")

    def test_sampling_is_seeded_and_with_replacement(self):
        a, _ = select_boltz([.1, .5], 30, .2, np.random.default_rng(4))
        b, _ = select_boltz([.1, .5], 30, .2, np.random.default_rng(4))
        np.testing.assert_equal(a, b)
        self.assertEqual(len(a), 30)

    def test_diversity_ignores_slot_and_move_order_but_counts_duplicates(self):
        a = paste(0)
        b = "\n\n".join(reversed(a.split("\n\n")))
        d = diversity([a, b, paste(10)])
        self.assertEqual(d["distinct_teams"], 2)
        self.assertEqual(d["distinct_species_sets"], 2)
        self.assertLess(d["effective_species_sets"], 2.)

    def test_schedules_and_incomplete_scores(self):
        opponents = ["a", "b", "c"]
        self.assertEqual(opponent_schedule(opponents, 7, 1), opponent_schedule(opponents, 7, 1))
        self.assertEqual(set(opponent_schedule(opponents, 3, 1)), set(opponents))
        self.assertNotEqual(phase_seed(1, 1, "search_battles"), phase_seed(1, 0, "holdout_battles"))
        with self.assertRaises(RuntimeError):
            require_scores({"a": {"wins": 1, "battles": 1}}, ["a"], 2)


class FakeEngine:
    """Independent deterministic experiment oracle; records training-label flow."""
    opponents = ["opp1", "opp2", "opp3"]
    initial_teams = [paste(100)]
    initial_y = [.25]

    def __init__(self):
        self.fits, self.scores = [], []

    def fit(self, teams, y, seed, generation, arm):
        self.fits.append((arm, seed, generation, list(y)))
        self.current_generation = generation
        return generation, dict(temperature=.2, refit_steps=40)

    def propose(self, fitted, n, directory, seed):
        p = [paste(i + fitted * 10) for i in range(n)]
        return dict(pastes=p, mu=list(range(n)), diversity=diversity(p))

    def score(self, pastes, directory, n, schedule, seed):
        self.scores.append((str(directory), n, list(schedule), seed))
        # Distinguish search and hold-out scores, so contamination is visible.
        wins = 1 if directory.name == "search" else 0
        return [dict(wins=wins, battles=n, win_rate=wins / n) for _ in pastes]

    def parse(self, pastes):
        return pastes


class CampaignTest(unittest.TestCase):
    def test_matching_holdout_isolation_and_resume(self):
        c = Config(seeds=(1, 2), generations=2, propose=3, battle=2, battles=2,
                   final_sample=3, final_battles=4, finalists=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            engine = FakeEngine()
            result = run(c, output, engine, {"checkpoint": "fixed"})
            self.assertEqual(len(result["runs"]), 10)
            self.assertEqual(len(engine.fits), 20)
            self.assertTrue(all(row[3] == [.25, .5, .5] for row in engine.fits if row[2] == 2))
            self.assertTrue(all(row[3] == [.25] for row in engine.fits if row[2] == 1))
            self.assertTrue(all(r["holdout"]["scores"][0]["win_rate"] == 0 for r in result["runs"].values()))
            for seed in c.seeds:
                for phase in ("holdout", "final"):
                    cells = [result["runs"][f"seed{seed}/{a}"][phase] for a in ARMS]
                    self.assertTrue(all(x["schedule"] == cells[0]["schedule"] for x in cells))
            resumed = FakeEngine()
            self.assertEqual(run(c, output, resumed, {"checkpoint": "fixed"}), result)
            self.assertEqual(resumed.fits, [])
            with self.assertRaises(ValueError):
                run(c, output, resumed, {"checkpoint": "changed"})
            self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
