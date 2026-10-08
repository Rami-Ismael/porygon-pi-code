"""Statistical and protocol regression tests; no battle server required."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoothness_experiment as experiment
from ruggedness import hartmann6, pest_control


class SmoothnessTests(unittest.TestCase):
    def test_noise_correction_matches_cross_replicate_products(self):
        values = np.array([-.3, .1, .4, .2])
        reference = np.mean([a * b for i, a in enumerate(values) for j, b in enumerate(values) if i != j])
        self.assertAlmostEqual(experiment.corrected_mean_square(values), reference)

    def test_recovers_effect_squared_despite_measurement_noise(self):
        rng = np.random.default_rng(44)
        for effect in (0., .1):
            samples = rng.normal(effect, .2, (30000, 4))
            corrected = samples.mean(axis=1)**2 - samples.var(axis=1, ddof=1) / 4
            self.assertAlmostEqual(float(corrected.mean()), effect**2, delta=.0005)
            self.assertGreater(float(np.mean(samples.mean(axis=1)**2)), effect**2 + .009)

    def test_negative_corrections_are_not_clipped(self):
        self.assertLess(experiment.corrected_mean_square([-.1, .1]), 0)

    def test_zero_wins_still_has_uncertainty(self):
        low, high = experiment.difference_interval(0, 0, 400)
        self.assertLess(low, 0)
        self.assertGreater(high, 0)
        self.assertAlmostEqual(low, -high)

    def test_cluster_bootstrap_preserves_duplicate_weight(self):
        rows = [dict(cluster="same", score=1.), dict(cluster="same", score=1.), dict(cluster="other", score=0.)]
        value = experiment.cluster_bootstrap(rows, "score", 1, iterations=100)
        self.assertEqual(value["clusters"], 2)
        self.assertAlmostEqual(value["mean"], 2/3)
        self.assertTrue(set(value["bootstrap"]).issubset({0., 2/3, 1.}))

    def test_single_cluster_has_no_invented_interval(self):
        result = experiment.cluster_bootstrap([dict(cluster="a", score=.3)], "score", 1)
        self.assertIsNone(result["ci95"])

    def test_edit_changes_only_one_move_and_preserves_ivs(self):
        paste = "\n\n".join(f"Nick (Species{i}) (M) @ Leftovers\nAbility: Ability\nLevel: 50\nIVs: 0 Atk\nQuiet Nature\n- oldmove\n- keepmove" for i in range(6)) + "\n"
        tables = {f"species{i}": ["oldmove", "keepmove", "newmove", "anothermove"] for i in range(6)}
        neighbors = experiment.sample_neighbors(paste, tables, 8, np.random.default_rng(0), lambda text: (True, []))
        self.assertEqual(len({row["paste"] for row in neighbors}), 8)
        for row in neighbors:
            experiment.assert_move_edit(paste, row["paste"])
            self.assertEqual(row["paste"].count("IVs: 0 Atk"), 6)
        with self.assertRaises(ValueError):
            experiment.assert_move_edit(paste, paste.replace("IVs: 0 Atk", "IVs: 31 Atk", 1))

    def test_rejection_exhaustion_never_silently_shrinks_neighborhood(self):
        paste = "\n\n".join(f"Species{i}\n- oldmove" for i in range(6))
        with self.assertRaises(RuntimeError):
            experiment.sample_neighbors(paste, {f"species{i}": ["oldmove", "newmove"] for i in range(6)},
                                        2, np.random.default_rng(1), lambda text: (False, ["illegal"]))

    def test_result_requires_correct_job_budget_and_manifest(self):
        job = dict(id="test", battles=100)
        row = dict(job=job, battles=100, wins=9, manifest_sha256="abc")
        experiment.checked_result(row, job, "abc")
        for field, value in [("battles", 99), ("wins", 100.5), ("wins", -1), ("manifest_sha256", "other")]:
            broken = dict(row, **{field: value})
            with self.assertRaises(ValueError):
                experiment.checked_result(broken, job, "abc")

    def test_random_sampling_is_repeatable_and_without_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "teams.jsonl"
            path.write_text("".join(json.dumps(dict(team=str(i))) + "\n" for i in range(100)))
            first = experiment.sample_random_records(path, 50, 10)
            self.assertEqual(first, experiment.sample_random_records(path, 50, 10))
            self.assertEqual(len({i for i, _ in first}), 50)

    def test_job_seeds_are_distinct_and_reproducible(self):
        manifest = dict(seed=1, replicates=4, battles_per_replicate=100,
                        candidates=[dict(id="a", file="a.txt"), dict(id="b", file="b.txt")])
        jobs = experiment.job_list(manifest)
        self.assertEqual(jobs, experiment.job_list(copy.deepcopy(manifest)))
        self.assertEqual(len({row["seed"] for row in jobs}), 8)

    def test_known_benchmark_values_and_scales(self):
        minimum = [0.20169, .150011, .476874, .275332, .311652, .6573]
        self.assertAlmostEqual(hartmann6(np.array(minimum)), -3.322368, places=5)
        value = pest_control(np.zeros(25, dtype=int), np.random.default_rng(1))
        self.assertTrue(0 <= value <= 25)
        point = np.full(6, .4)
        large = abs(hartmann6(point + np.array([.001, 0, 0, 0, 0, 0])) - hartmann6(point))
        small = abs(hartmann6(point + np.array([.0001, 0, 0, 0, 0, 0])) - hartmann6(point))
        self.assertAlmostEqual(large / small, 10, delta=.2)

    def test_complete_analysis_and_missing_panel_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "labels").mkdir()
            team = directory / "team.txt"
            team.write_text("test-only fixture")
            manifest = dict(schema=experiment.SCHEMA, seed=7, anchors_per_group=2,
                            neighbors=1, replicates=2, battles_per_replicate=50,
                            planned_battles=1200, candidates=[], opponents=[], anchors=[])
            for group in ("random", "tournament"):
                for index in range(2):
                    anchor_id = f"{group}-{index}"
                    anchor = dict(id=anchor_id, group=group, cluster=anchor_id, candidates=[])
                    for kind in ("base", "null", "edit-000"):
                        candidate_id = f"{anchor_id}-{kind}"
                        anchor["candidates"].append(candidate_id)
                        manifest["candidates"].append(dict(id=candidate_id, anchor=anchor_id, kind=kind,
                            file=str(team), sha256=experiment.digest(team), edit=None))
                    manifest["anchors"].append(anchor)
            experiment.atomic_json(directory / "manifest.json", manifest)
            manifest_hash = experiment.digest(directory / "manifest.json")
            jobs = experiment.job_list(manifest)
            for job in jobs:
                wins = 15 if "tournament" in job["id"] and "edit" in job["id"] else 10
                experiment.atomic_json(directory / "labels" / f"{job['id']}.json",
                                       dict(job=job, manifest_sha256=manifest_hash, wins=wins, battles=50))
            with patch("builtins.print"):
                experiment.analyse(SimpleNamespace(output=directory))
            report = experiment.read_json(directory / "analysis.json")
            self.assertAlmostEqual(report["summary"]["random"]["mean"], 0)
            self.assertAlmostEqual(report["summary"]["tournament"]["mean"], .01)
            self.assertTrue((directory / "smoothness.png").is_file())
            self.assertTrue((directory / "smoothness.pdf").is_file())
            self.assertEqual(len(report["edges"]), 8)
            (directory / "labels" / f"{jobs[0]['id']}.json").unlink()
            with self.assertRaises(FileNotFoundError):
                experiment.analyse(SimpleNamespace(output=directory))

    def test_changed_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            team = directory / "team.txt"
            team.write_text("original")
            manifest = dict(schema=experiment.SCHEMA, opponents=[],
                            candidates=[dict(file=str(team), sha256=experiment.digest(team))])
            experiment.atomic_json(directory / "manifest.json", manifest)
            experiment.verify_manifest(directory)
            team.write_text("changed")
            with self.assertRaises(ValueError):
                experiment.verify_manifest(directory)


if __name__ == "__main__":
    unittest.main()
