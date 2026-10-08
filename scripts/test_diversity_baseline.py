"""Baseline phase isolation, resume and verification; no model or battles."""
import copy
from pathlib import Path
import tempfile
import unittest

from capture_diversity_baseline import Config, run_baseline, verify_baseline
from diversity_metrics import Reference, measure
from test_diversity_metrics import team


class FakeBaseline:
    opponents = ["a", "b"]
    def __init__(self):
        self.diversity_reference = Reference([team()], scope="initial_training", format_id="fixture")
        self.loaded = self.samples = self.scored = 0

    def baseline(self):
        self.loaded += 1
        return "original checkpoint"

    def parse(self, pastes):
        return [team() for _ in pastes]

    def propose(self, fitted, n, directory, seed):
        assert fitted == "original checkpoint"
        self.samples += 1
        metrics = measure([team()] * n, self.diversity_reference, expected_n=n, attempts=n)
        metrics["sampling"]["attempt_limit"] = n * 8
        return dict(pastes=["repeat"] * n, diversity=dict(team_diversity=metrics))

    def score(self, pastes, directory, battles, schedule, seed):
        self.scored += 1
        return [dict(wins=1, battles=battles, win_rate=1/battles) for _ in pastes]


class BaselineTest(unittest.TestCase):
    def test_complete_resume_and_corruption(self):
        config = Config(seeds=(1, 2), propose=3, final_sample=2, final_battles=4)
        engine = FakeBaseline()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            result = run_baseline(config, path, engine, {"checkpoint": "fixed"}, "source-hash")
            self.assertEqual(result["status"], "complete")
            self.assertEqual((engine.loaded, engine.samples, engine.scored), (1, 4, 2))
            resumed = FakeBaseline()
            self.assertEqual(run_baseline(config, path, resumed, {"checkpoint": "fixed"}, "source-hash"), result)
            self.assertEqual((resumed.loaded, resumed.samples, resumed.scored), (0, 0, 0))
            for corrupt in (lambda d: d["runs"]["1"]["scores"].pop(),
                            lambda d: d["runs"]["1"]["proposal"]["diversity"]["team_diversity"]["with_spreads"].update(unique_fraction=1),
                            lambda d: d["runs"]["1"].update(battle_seed=0)):
                damaged = copy.deepcopy(result)
                corrupt(damaged)
                with self.assertRaises(ValueError):
                    verify_baseline(damaged, engine.parse)
            with self.assertRaises(ValueError):
                run_baseline(config, path, resumed, {"checkpoint": "changed"}, "source-hash")


if __name__ == "__main__":
    unittest.main()
