import copy
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

from llm_baseline import GRID, STATS, apply_completion, masked, paths_for


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


class FastPanelTests(unittest.TestCase):
    """The fast evaluator sees private identities; results come back under real team ids."""
    def setUp(self):
        self.con = sqlite3.connect(':memory:')
        self.addCleanup(self.con.close)
        self.con.execute('CREATE TABLE team (team_id INTEGER PRIMARY KEY, base_key TEXT)')
        # 7 and 8 share a preview; 9 is the candidate and also a pool team (a mirror).
        self.con.executemany('INSERT INTO team VALUES (?,?)', [(7, 'x'), (8, 'x'), (9, 'y')])
        self.calls = []

    def runner(self, mem, jobs, seed=None):
        self.calls.append((jobs, seed, dict(mem.execute('SELECT team_id, paste FROM team'))))
        for job in jobs:
            bases = [mem.execute('SELECT base_key FROM team WHERE team_id=?', (c,)).fetchone()[0] for c in job['schedule']]
            self.assertTrue(job['row'] not in job['schedule'])
            self.assertEqual(len(set(bases)), len(set(job['schedule'])))
        out = []
        for job in jobs:
            for col in sorted(set(job['schedule'])):
                n = job['schedule'].count(col)
                out.append((job['row'], col, n, 0, 0) if col % 2 else (job['row'], col, 0, n, 0))
        return out, 0, 0.

    def test_mirror_duplicate_rows_and_preview_split(self):
        import llm_baseline as B
        panels = [(9, [7, 8, 9, 7]), (9, [8, 8])]
        by_panel, unattributed = B.fast_panels(self.runner, self.con, panels, lambda t: f'paste {t}', 5)
        self.assertEqual(unattributed, 0)
        jobs, seed, pastes = self.calls[0]
        self.assertEqual(seed, 5)
        self.assertEqual(len(pastes), 5)  # two candidate rows plus three opponents, each once
        self.assertEqual(len(jobs), 3)    # 7 and 8 split; second panel in its own job
        totals = {i: {r[1]: sum(r[2:]) for r in rows} for i, rows in by_panel.items()}
        self.assertEqual(totals, {0: {7: 2, 8: 1, 9: 1}, 1: {8: 2}})
        self.assertTrue(all(r[0] == 9 for rows in by_panel.values() for r in rows))

    def test_unknown_backend_is_refused(self):
        import llm_baseline as B
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                B.battle(SimpleNamespace(output=Path(folder), battle_backend='quick'))

    def test_fast_unless_legacy_is_named(self):
        import llm_baseline as B
        self.assertEqual(B.resolve_backend(), 'fast')
        self.assertEqual(B.resolve_backend('legacy'), 'legacy')

    def test_switching_engine_archives_labels_and_config_never_deletes(self):
        import llm_baseline as B
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertIsNone(B.switch_engine(folder, 'fast'))  # nothing battled yet
            B.save(folder / 'labels' / 'start-0.json', {'wins': 3, 'battles': 50})
            self.assertEqual(B.recorded_backend(folder), 'legacy')  # labels without a config predate fast
            B.save(folder / 'battle-config.json', {'seed': 1})
            self.assertIsNone(B.switch_engine(folder, 'legacy'))
            archive = B.switch_engine(folder, 'fast')
            self.assertTrue(archive.name.startswith('archive-legacy-battles-'))
            self.assertEqual(B.read_config(archive), {'seed': 1})
            self.assertTrue((archive / 'labels' / 'start-0.json').exists())
            self.assertFalse((folder / 'labels').exists())
            self.assertIsNone(B.recorded_backend(folder))
            B.save(folder / 'battle-config.json', {'seed': 1, 'backend': 'fast'})
            self.assertIsNone(B.switch_engine(folder, 'fast'))


if __name__ == "__main__":
    unittest.main()
