from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

import repo_paths  # puts src/ first on sys.path
import battles
import results


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
        panels = [(9, [7, 8, 9, 7]), (9, [8, 8])]
        by_panel, unattributed = battles.fast_panels(self.runner, self.con, panels, lambda t: f'paste {t}', 5)
        self.assertEqual(unattributed, 0)
        jobs, seed, pastes = self.calls[0]
        self.assertEqual(seed, 5)
        self.assertEqual(len(pastes), 5)  # two candidate rows plus three opponents, each once
        self.assertEqual(len(jobs), 3)    # 7 and 8 split; second panel in its own job
        totals = {i: {r[1]: sum(r[2:]) for r in rows} for i, rows in by_panel.items()}
        self.assertEqual(totals, {0: {7: 2, 8: 1, 9: 1}, 1: {8: 2}})
        self.assertTrue(all(r[0] == 9 for rows in by_panel.values() for r in rows))



class EngineTests(unittest.TestCase):
    """Only the fast evaluator battles; folders the removed legacy engine battled still read."""
    def test_the_legacy_engine_and_unknown_engines_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('legacy', 'quick'):
                with self.subTest(engine=name), self.assertRaisesRegex(ValueError, 'removed on 2026-10-19'):
                    battles.battle(SimpleNamespace(output=Path(folder), battle_backend=name))

    def test_legacy_labels_are_archived_never_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertIsNone(battles.recorded_backend(folder))
            self.assertIsNone(battles.archive_legacy_labels(folder))  # nothing battled yet
            results.save(folder / 'labels' / 'start-0.json', {'wins': 3, 'battles': 50})
            self.assertEqual(battles.recorded_backend(folder), 'legacy')  # labels without a config predate fast
            results.save(folder / 'battle-config.json', {'seed': 1})
            self.assertEqual(battles.recorded_backend(folder), 'legacy')  # a config naming no engine predates fast
            archive = battles.archive_legacy_labels(folder)
            self.assertTrue(archive.name.startswith('archive-legacy-battles-'))
            self.assertEqual(battles.read_config(archive), {'seed': 1})
            self.assertTrue((archive / 'labels' / 'start-0.json').exists())
            self.assertFalse((folder / 'labels').exists())
            self.assertIsNone(battles.recorded_backend(folder))
            results.save(folder / 'battle-config.json', {'seed': 1, 'backend': 'fast'})
            self.assertEqual(battles.recorded_backend(folder), 'fast')
            self.assertIsNone(battles.archive_legacy_labels(folder))


class BattleExportTests(unittest.TestCase):
    def test_zero_point_neutral_export_keeps_investment_and_non_neutral_natures(self):
        text = ('Pikachu\nLevel: 50\nSerious Nature\n- Protect\n\n'
                'Raichu\nEVs: 1 HP\nSerious Nature\n- Protect\n\n'
                'Raichu\nEVs: 0 HP / 0 Spe\nBrave Nature\n- Protect\n')
        result = battles.battle_paste(text)
        self.assertEqual(result, text.replace('Level: 50\nSerious Nature', 'Level: 50\nHardy Nature'))
        self.assertEqual(battles.battle_paste(result), result)


if __name__ == "__main__":
    unittest.main()
