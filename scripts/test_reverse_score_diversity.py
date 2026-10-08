import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reverse_score_diversity import species_metrics, rarefied_distinct, summarize
from signed_score_guidance import SignedGuide, sample_direction


class Guide:
    def tilt(self, *args, **kwargs):
        return 3.5

    def gap(self, *args, **kwargs):
        return .2


class DirectionAndDiversityTests(unittest.TestCase):
    def test_reverse_changes_direction_without_changing_gain(self):
        guide = Guide()
        reverse = SignedGuide(guide, -1)
        self.assertEqual(reverse.tilt(), -guide.tilt())
        self.assertEqual(reverse.gap(), guide.gap())

    def test_negative_direction_reaches_sampler_positive_magnitude_gate(self):
        def sampler(model, constraints, guide, n, mode, lam):
            self.assertEqual((n, mode, lam), (4, "gloss", 108.))
            self.assertLess(guide.tilt(), 0)
            return "sampled"
        self.assertEqual(sample_direction(sampler, None, None, Guide(), 4, -108.), "sampled")

    def test_zero_selects_unguided_sampler_mode(self):
        result = sample_direction(lambda m, c, g, n, mode, lam: (mode, lam), None, None, Guide(), 2, 0.)
        self.assertEqual(result, ("none", 0.))
        with self.assertRaises(ValueError):
            sample_direction(None, None, None, Guide(), 2, math.nan)

    def test_repeated_and_reordered_combinations_are_one(self):
        a = tuple("abcdef")
        row = species_metrics([a, tuple(reversed(a)), a], [a])
        self.assertEqual(row["distinct_species_sets"], 1)
        self.assertAlmostEqual(row["effective_species_sets"], 1.)
        self.assertEqual(row["novel_distinct_species_sets"], 0)
        self.assertEqual(row["mean_pairwise_jaccard"], 0.)

    def test_novelty_is_relative_to_corpus(self):
        a, b = tuple("abcdef"), tuple("abcdeg")
        row = species_metrics([a, b], [a])
        self.assertEqual(row["distinct_species_sets"], 2)
        self.assertEqual(row["novel_distinct_species_sets"], 1)
        self.assertAlmostEqual(row["corpus_combination_rate"], .5)
        self.assertAlmostEqual(row["mean_nearest_corpus_jaccard"], 1 / 7)

    def test_distance_weights_duplicate_teams(self):
        row = species_metrics([("a", "b"), ("a", "b"), ("a", "c")], [("a", "b")])
        self.assertAlmostEqual(row["mean_pairwise_jaccard"], 4 / 9)

    def test_rarefaction_and_empty_valid_subset(self):
        self.assertAlmostEqual(rarefied_distinct([200]), 1.)
        self.assertAlmostEqual(rarefied_distinct([1] * 200), 128.)
        self.assertIsNone(rarefied_distinct([127]))
        row = species_metrics([], [("a", "b")])
        self.assertEqual(row["distinct_species_sets"], 0)
        self.assertEqual(row["effective_species_sets"], 0.)
        self.assertIsNone(row["mean_pairwise_jaccard"])

    def test_incomplete_experiment_is_not_reported_as_complete(self):
        with self.assertRaises(ValueError):
            summarize({"status": "running"})


if __name__ == "__main__":
    unittest.main()
