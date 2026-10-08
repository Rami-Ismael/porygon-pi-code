"""Check baseline/history compatibility and reject missing historical arms."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from capture_diversity_baseline import Config, run_baseline
from report_diversity_baseline import compare, REPO, digest
from test_diversity_baseline import FakeBaseline
from test_diversity_metrics import team


class BaselineReportTest(unittest.TestCase):
    def test_comparison_and_missing_arm(self):
        config = Config(seeds=(1,), generations=1, propose=3, final_sample=2, final_battles=4)
        engine = FakeBaseline()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "history.json"
            source.write_text(json.dumps(dict(manifest=dict(arms=["a", "b"]))))
            baseline = run_baseline(config, Path(directory) / "baseline.json", engine, {}, digest(source))
            sample = baseline["runs"]["1"]
            history = dict(source=str(source), source_sha256=digest(source),
                metric_source_sha256=digest(REPO / "src/diversity_metrics.py"),
                reference=engine.diversity_reference.metadata, rows=[])
            for arm in ("a", "b"):
                for phase in ("generation", "holdout"):
                    sample_phase = "proposal" if phase == "generation" else "holdout"
                    history["rows"].append(dict(seed=1, arm=arm, generation=1, phase=phase,
                        metrics=sample[sample_phase]["diversity"]["team_diversity"], win_rate=dict(mean=.5)))
            with patch("corpus.parse_team_text", side_effect=lambda _: team()):
                rows = compare(baseline, history)
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0]["win_rate"], .25)
                self.assertEqual(rows[1]["win_rate"], .5)
                missing = copy.deepcopy(history)
                missing["rows"] = [r for r in missing["rows"] if r["arm"] == "a"]
                with self.assertRaises(ValueError):
                    compare(baseline, missing)
                changed = copy.deepcopy(history)
                changed["source_sha256"] = "different"
                with self.assertRaises(ValueError):
                    compare(baseline, changed)


if __name__ == "__main__":
    unittest.main()
