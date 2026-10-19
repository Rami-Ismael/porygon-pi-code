import copy
import unittest

import repo_paths  # puts src/ first on sys.path
from llm_completion import apply_completion
from team_masks import GRID, STATS, masked, paths_for


class CompletionContractTests(unittest.TestCase):
    def setUp(self):
        self.team = [dict(species="Pikachu", item="Light Ball", ability="Static",
                          nature="Jolly", moves=["Protect", "Thunderbolt", "Thunder", "Volt Switch"],
                          evs={s: 0 for s in STATS}) for _ in range(6)]

    def test_exact_patch_preserves_every_other_field(self):
        before = copy.deepcopy(self.team)
        paths = [[2, "moves", 1], [4, "item"]]
        response = {"fills": [{"path": paths[0], "value": "Nuzzle"},
                              {"path": paths[1], "value": "Focus Sash"}]}
        out = apply_completion(self.team, paths, response)
        self.assertEqual(masked(out, paths), masked(before, paths))
        self.assertEqual(self.team, before)

    def test_rejects_unmasked_duplicate_missing_and_extra_edits(self):
        paths = [[0, "item"], [1, "item"]]
        cases = [
            {"fills": [{"path": [2, "item"], "value": "x"}, {"path": paths[1], "value": "y"}]},
            {"fills": [{"path": paths[0], "value": "x"}] * 2},
            {"fills": []},
            {"fills": [], "team": self.team},
        ]
        for response in cases:
            with self.subTest(response=response), self.assertRaises(ValueError):
                apply_completion(self.team, paths, response)

    def test_whole_slot_must_have_valid_stat_points(self):
        for value in [33, -1, True, 1.5]:
            slot = copy.deepcopy(self.team[0])
            slot["evs"]["HP"] = value
            with self.assertRaises(ValueError):
                apply_completion(self.team, [[0]], {"fills": [{"path": [0], "value": slot}]})
        slot = copy.deepcopy(self.team[0])
        slot["evs"] = {s: 32 for s in STATS}
        with self.assertRaises(ValueError):
            apply_completion(self.team, [[0]], {"fills": [{"path": [0], "value": slot}]})

    def test_grid_covers_all_seven_tasks_with_distinct_paths(self):
        self.assertEqual(sum(map(len, GRID.values())), 21)
        self.assertEqual(len(paths_for("mixed")), 48)
        for task, sizes in GRID.items():
            paths = paths_for(task)
            self.assertEqual(len(paths), len({tuple(p) for p in paths}))
            self.assertLessEqual(max(sizes), len(paths))


if __name__ == "__main__":
    unittest.main()
