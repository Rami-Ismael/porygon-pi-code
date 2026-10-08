"""Run: python scripts/test_diversity_metrics.py (standard library only)."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diversity_metrics import Reference, STATS, measure, team_keys


def team():
    return [dict(species=f"Species-{i}", ability="Ability", item="Item", nature="Bold",
                 moves=["Protect", "Move One", "Move Two", "Move Three"],
                 evs=dict(zip(STATS, [32, 0, 0, 0, 0, 32]))) for i in range(6)]


class DiversityTest(unittest.TestCase):
    def reference(self, teams):
        return Reference(teams, scope="corpus", format_id="fixture")

    def test_order_names_and_format_do_not_inflate_uniqueness(self):
        a, b = team(), list(reversed(team()))
        for slot in b:
            slot["moves"].reverse()
            slot["species"] = slot["species"].upper().replace("-", " ")
            slot["nickname"] = "irrelevant"
        m = measure([a, b], self.reference([a]))
        self.assertEqual(m["with_spreads"]["unique_count"], 1)
        self.assertEqual(m["with_spreads"]["novel_draw_fraction"], 0)

    def test_spread_change_is_separate_from_categorical_change(self):
        a, b = team(), team()
        b[0]["evs"]["HP"] = 31
        m = measure([a, b], self.reference([a]))
        self.assertEqual(m["categorical_48"]["unique_count"], 1)
        self.assertEqual(m["with_spreads"]["unique_count"], 2)
        self.assertEqual(m["with_spreads"]["novel_draw_fraction"], .5)
        self.assertEqual(m["composition"]["unique_count"], 1)

    def test_omitted_nature_uses_verified_pilot_default(self):
        a, b = team(), team()
        a[0]["nature"], b[0]["nature"] = "", "Serious"
        self.assertEqual(team_keys(a), team_keys(b))

    def test_known_denominators(self):
        a, b, c = team(), team(), team()
        b[0]["item"], c[0]["item"] = "New Item B", "New Item C"
        m = measure([a, a, b, c], self.reference([a, b]), expected_n=4, attempts=5)
        d = m["with_spreads"]
        self.assertEqual((d["unique_fraction"], d["repeat_fraction"]), (.75, .25))
        self.assertEqual((d["novel_draw_fraction"], d["novel_unique_fraction"]), (.25, 1/3))
        self.assertEqual(m["sampling"]["rejected"], 1)
        self.assertEqual(m["sampling"]["acceptance_fraction"], .8)
        self.assertEqual(m["composition"]["frequencies"][0]["count"], 4)

    def test_repeated_novel_draws_do_not_become_new_unique_teams(self):
        a, b = team(), team()
        b[0]["ability"] = "Other Ability"
        m = measure([a, b, b, b], self.reference([a]))["with_spreads"]
        self.assertEqual(m["novel_draw_fraction"], .75)
        self.assertEqual(m["novel_unique_fraction"], .5)

    def test_forms_and_unknown_names_remain_distinct(self):
        a, b = team(), team()
        b[0]["species"] += "-Form"
        m = measure([a, b], self.reference([a]))
        self.assertEqual(m["composition"]["novel_draw_fraction"], .5)
        b[0]["species"] = a[0]["species"]
        b[0]["item"] = "Never In Vocabulary"
        self.assertNotEqual(team_keys(a)[0], team_keys(b)[0])

    def test_duplicate_species_are_not_deleted_or_order_sensitive(self):
        a = team()
        a[1]["species"] = a[0]["species"]
        a[1]["item"] = "Different Item"
        self.assertEqual(team_keys(a), team_keys(list(reversed(a))))
        self.assertEqual(len(team_keys(a)[2]), 6)

    def test_missing_reference_and_empty_samples_are_explicit(self):
        m = measure([], expected_n=512, attempts=20)
        self.assertEqual(m["status"], "empty")
        self.assertIsNone(m["with_spreads"]["unique_fraction"])
        self.assertIsNone(measure([team()])["with_spreads"]["novel_draw_fraction"])
        self.assertEqual(measure([team()], expected_n=512)["status"], "sample_size_mismatch")
        json.dumps(m, allow_nan=False)
        with self.assertRaises(ValueError):
            self.reference([])

    def test_reference_is_a_snapshot_and_order_independent(self):
        a, b = team(), team()
        b[0]["item"] = "Different"
        ref = self.reference([a, b])
        self.assertEqual(ref.metadata, self.reference([b, a]).metadata)
        a[0]["item"] = "Edited After Freezing"
        self.assertEqual(measure([a], ref)["with_spreads"]["novel_draw_fraction"], 1)

    def test_bad_shapes_and_missing_fields_fail_loudly(self):
        for field in ("species", "moves", "evs", "ability", "nature", "item"):
            a = team()
            del a[0][field]
            with self.assertRaises(ValueError):
                measure([a])
        with self.assertRaises(ValueError):
            measure([team()[:5]])
        with self.assertRaises(ValueError):
            measure([team()], attempts=0)
        a = team()
        a[0]["evs"]["HP"] = True
        with self.assertRaises(ValueError):
            measure([a])


if __name__ == "__main__":
    unittest.main()
