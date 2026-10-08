from collections import Counter
from copy import deepcopy
from itertools import combinations
import math
import unittest

from pokemon_id_diversity import analyze, pokemon_id_metrics, SCORE


class PokemonIdDiversityTests(unittest.TestCase):
    A = list("abcdef")
    B = list("ghijkl")

    def test_same_roster_and_reordered_slots_have_zero_distance(self):
        r = pokemon_id_metrics([self.A, self.A[::-1]])
        self.assertEqual(r[SCORE], 0.)
        self.assertEqual(r["mean_shared_ids"], 6.)
        self.assertAlmostEqual(r["effective_ids"], 6.)

    def test_one_replacement_is_one_sixth(self):
        r = pokemon_id_metrics([self.A, list("abcdeg")])
        self.assertEqual(r["mean_id_replacements"], 1.)
        self.assertAlmostEqual(r[SCORE], 100 / 6)

    def test_disjoint_rosters_have_maximum_distance(self):
        r = pokemon_id_metrics([self.A, self.B])
        self.assertEqual(r[SCORE], 100.)
        self.assertEqual(r["distinct_ids"], 12)
        self.assertAlmostEqual(r["effective_ids"], 12.)

    def test_duplicate_draws_are_retained(self):
        r = pokemon_id_metrics([self.A, self.A, self.B])
        self.assertEqual(r["n"], 3)
        self.assertAlmostEqual(r[SCORE], 200 / 3)

    def test_repeated_ids_in_raw_proposals_use_multiset_overlap(self):
        r = pokemon_id_metrics([["a"] * 6, ["a"] * 5 + ["b"]])
        self.assertEqual(r["mean_shared_ids"], 5.)
        self.assertAlmostEqual(r[SCORE], 100 / 6)

    def test_empty_single_and_malformed_rosters(self):
        self.assertIsNone(pokemon_id_metrics([])[SCORE])
        self.assertEqual(pokemon_id_metrics([])["effective_ids"], 0.)
        self.assertIsNone(pokemon_id_metrics([self.A])[SCORE])
        for bad in (["a"] * 5, "abcdef", ["a"] * 5 + [None], ["a"] * 5 + ["[MASK]"]):
            with self.assertRaises(ValueError):
                pokemon_id_metrics([bad])

    def test_normalization_preserves_distinct_form_ids(self):
        a = ["Rotom-Wash"] + self.A[1:]
        b = ["rotomwash"] + self.A[1:]
        c = ["Rotom-Heat"] + self.A[1:]
        self.assertEqual(pokemon_id_metrics([a, b])[SCORE], 0.)
        self.assertAlmostEqual(pokemon_id_metrics([a, c])[SCORE], 100 / 6)

    def test_fast_aggregate_matches_explicit_team_pairs(self):
        teams = [self.A, self.B, self.A, list("abcdeg"), ["a"] * 6, ["a"] * 5 + ["b"]]
        brute = sum((6 - sum((Counter(a) & Counter(b)).values())) / 6 * 100
                    for a, b in combinations(teams, 2)) / math.comb(len(teams), 2)
        self.assertAlmostEqual(pokemon_id_metrics(teams)[SCORE], brute)

    def data(self):
        rows = [dict(index=0, species=self.A, valid=True), dict(index=1, species=self.B, valid=False)]
        return dict(status="complete", manifest=dict(arms=["none"], seeds=[1], attempts=2),
                    runs={"seed1/none": dict(seed=1, arm="none", records=rows)})

    def test_non_id_attributes_do_not_change_analysis(self):
        data = self.data()
        expected = analyze(data)
        changed = deepcopy(data)
        for row in changed["runs"]["seed1/none"]["records"]:
            row.update(evs={"hp": 32}, ivs={"atk": 0}, nature="Bold", ability="changed",
                       moves=["changed"], item="changed", paste="unread", surrogate_score=-100)
            row["species"].reverse()
        self.assertEqual(analyze(changed), expected)
        self.assertEqual(expected["cells"]["seed1/none"]["raw"][SCORE], 100.)
        self.assertIsNone(expected["cells"]["seed1/none"]["legal"][SCORE])

    def test_partial_or_inconsistent_experiments_are_rejected(self):
        for mutation in (lambda d: d.update(status="running"),
                         lambda d: d["runs"].clear(),
                         lambda d: d["manifest"].update(attempts=3),
                         lambda d: d["runs"]["seed1/none"]["records"][1].update(index=0)):
            data = self.data()
            mutation(data)
            with self.assertRaises(ValueError):
                analyze(data)


if __name__ == "__main__":
    unittest.main()
