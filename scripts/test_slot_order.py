"""Unit tests for src/slot_order.py: every battle gets the same six Pokémon in a new order."""
import random
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import slot_order as SO

PASTE = "\n\n".join(f"{name} @ Leftovers\nAbility: Pressure\nLevel: 50\n- Protect"
                    for name in ["Basculegion", "Chandelure", "Floette-Eternal", "Garchomp", "Kingambit", "Sylveon"]) + "\n"


def names(paste):
    return [b.split(" @ ")[0] for b in paste.strip().split("\n\n")]


class ShufflePaste(unittest.TestCase):
    def test_keeps_every_block_unchanged(self):
        out = SO.shuffle_paste(PASTE, random.Random(1))
        self.assertEqual(sorted(out.strip().split("\n\n")), sorted(PASTE.strip().split("\n\n")))

    def test_every_pokemon_reaches_slot_one(self):
        rng = random.Random(2)
        first = Counter(names(SO.shuffle_paste(PASTE, rng))[0] for _ in range(600))
        self.assertEqual(len(first), 6)
        self.assertTrue(all(60 <= c <= 140 for c in first.values()), first)

    def test_seeded_rng_repeats(self):
        a = [SO.shuffle_paste(PASTE, r) for r in [random.Random(7)] for _ in range(5)]
        b = [SO.shuffle_paste(PASTE, r) for r in [random.Random(7)] for _ in range(5)]
        self.assertEqual(a, b)


@unittest.skipIf(SO.RandomTeamBuilder is object, "vgc_bench is not importable")
class Builder(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "team.txt"
        self.path.write_text(PASTE)

    def tearDown(self):
        self.dir.cleanup()

    def species_order(self, packed):
        return [mon.split("|")[0] or mon.split("|")[1] for mon in packed.split("]")]

    def test_yield_team_reorders_and_keeps_the_six(self):
        tb = SO.ShuffledTeamBuilder(1, None, "mb", custom_team_paths=[self.path], order_seed=3)
        orders = [tuple(self.species_order(tb.yield_team())) for _ in range(30)]
        self.assertTrue(all(sorted(o) == sorted(orders[0]) for o in orders))
        if SO.MODE == "shuffled":
            self.assertGreater(len(set(orders)), 10)
        else:
            self.assertEqual(len(set(orders)), 1)

    def test_shuffle_does_not_touch_global_random(self):
        random.seed(5); expected = random.random()
        random.seed(5)
        tb = SO.ShuffledTeamBuilder(1, None, "mb", custom_team_paths=[self.path], order_seed=None)
        random.seed(5); tb._load_team(self.path)
        self.assertEqual(random.random(), expected)


if __name__ == "__main__":
    unittest.main()
