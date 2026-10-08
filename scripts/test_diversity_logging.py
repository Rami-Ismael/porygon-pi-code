"""Behavior tests for proposal logging and historical report boundaries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from diversity_metrics import Reference
from temperature_experiment import BattleEngine, Config, run
from test_diversity_metrics import team
from test_temperature_experiment import FakeEngine
from report_team_diversity import analyze


class LoggingTest(unittest.TestCase):
    def test_engine_logs_the_repeated_unranked_stream_and_exact_attempt_count(self):
        teams = [team(), team()]
        paste = "\n\n".join(f"Species-{i} @ Item\nAbility: Ability\nBold Nature\n- Protect" for i in range(6))
        def propose(*args, sampling_stats):
            sampling_stats.update(attempts=3, accepted=2, rejected=1)
            return [], teams, [paste, paste], 2/3, []
        engine = BattleEngine.__new__(BattleEngine)
        engine.E = SimpleNamespace(GG=SimpleNamespace(propose_guided=propose),
            A=SimpleNamespace(memorisation=lambda *args: {}))
        engine.c = Config()
        engine.C = engine.V = engine.look = engine.spreads = engine.validator = engine.grid = None
        engine.diversity_reference = Reference([team()], scope="corpus", format_id="fixture")
        feats = SimpleNamespace(mat=lambda _: np.array([[1.], [1.]]))
        result = engine.propose((None, feats, np.array([.5]), None), 2, Path("unused"), 10)
        metrics = result["diversity"]["team_diversity"]
        self.assertEqual(metrics["sampling"]["attempts"], 3)
        self.assertEqual(metrics["with_spreads"]["unique_fraction"], .5)
        self.assertEqual(metrics["with_spreads"]["novel_draw_fraction"], 0)
        self.assertEqual(len(result["pastes"]), 2)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                engine.propose((None, feats, np.array([.5]), None), 3, Path(directory), 10)
            failed = json.loads((Path(directory) / "sampling_failure.json").read_text())
            self.assertEqual(failed["diversity"]["team_diversity"]["status"], "sample_size_mismatch")

    def test_resume_keeps_the_frozen_reference(self):
        engine = FakeEngine()
        engine.diversity_reference = Reference([team()], scope="corpus", format_id="fixture")
        config = Config(seeds=(1,), generations=1, propose=2, battle=1, battles=2,
                        final_sample=2, final_battles=2, finalists=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            first = run(config, path, engine, {})
            self.assertEqual(run(config, path, engine, {}), first)
            changed = team()
            changed[0]["item"] = "Changed Reference"
            engine.diversity_reference = Reference([changed], scope="corpus", format_id="fixture")
            with self.assertRaises(ValueError):
                run(config, path, engine, {})

    def fixture(self):
        cell = dict(proposal=dict(pastes=["A", "A"], diversity=dict(validity=.5)),
                    selected=["A"], scores=[dict(wins=1, battles=2)])
        holdout = copy.deepcopy(cell)
        holdout["scores"] = [dict(wins=0, battles=2), dict(wins=2, battles=2)]
        return dict(status="complete", manifest=dict(config=dict(seeds=[1], generations=1,
            propose=2, battle=1, battles=2, final_sample=2, final_battles=2), arms=["a"]),
            runs={"seed1/a": dict(seed=1, arm="a", generations={"1": cell}, holdout=holdout)})

    def test_report_preserves_denominators_and_distinguishes_scored_population(self):
        ref = Reference([team()], scope="initial_training", format_id="fixture")
        rows = analyze(self.fixture(), ref, lambda _: team())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["metrics"]["with_spreads"]["repeat_fraction"], .5)
        self.assertIsNone(rows[0]["metrics"]["sampling"]["attempts"])
        self.assertEqual(rows[0]["win_rate"]["population"], "score_selected")
        self.assertEqual(rows[1]["win_rate"]["population"], "unranked_generator_sample")
        self.assertEqual(rows[1]["win_rate"]["mean"], .5)

    def test_report_rejects_incomplete_runs_samples_and_battles(self):
        ref = Reference([team()], scope="corpus", format_id="fixture")
        for mutation in (lambda d: d["runs"].clear(),
                         lambda d: d["runs"]["seed1/a"]["generations"]["1"]["proposal"]["pastes"].pop(),
                         lambda d: d["runs"]["seed1/a"]["holdout"]["scores"].pop()):
            d = self.fixture()
            mutation(d)
            with self.assertRaises(ValueError):
                analyze(d, ref, lambda _: team())


if __name__ == "__main__":
    unittest.main()
