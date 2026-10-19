import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import llm_baseline as B
import reasoning_battles as P


class ParallelBattleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m = dict(format=B.FORMAT, seed=42, battles=50, seed_limit='unseeded simulator',
                      opponents=[{'team_id': 1}, {'team_id': 2}],
                      trials=[dict(id='s0-item-1', start=0)])
        self.runs = [({}, self.root / a, copy.deepcopy(self.m)) for a in ['off', 'on']]
        for _, f, _ in self.runs:
            f.mkdir()

    def test_same_team_across_arms_is_one_panel_with_multiple_destinations(self):
        with patch.object(P.E, 'completion', return_value={'valid': True, 'team': 'same'}):
            groups = P.plan(self.runs, lambda t: t)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['destinations']), 2)
        result = dict(wins=12, battles=50, cache_key='exact')
        P.publish(groups[0], result, {'seed': 42})
        times = {p: Path(p).stat().st_mtime_ns for p in groups[0]['destinations']}
        P.publish(groups[0], result, {'seed': 42})
        self.assertEqual(times, {p: Path(p).stat().st_mtime_ns for p in times})
        with patch.object(P.E, 'completion', return_value={'valid': True, 'team': 'same'}):
            self.assertEqual(P.plan(self.runs, lambda t: t), [])

    def test_different_schedules_never_share_a_panel(self):
        for _, _, m in self.runs:
            m['exclude_starter_team_ids'] = [1, 2]
        self.runs[1][2]['trials'] = [dict(id='s1-item-1', start=1)]
        with patch.object(P.E, 'completion', return_value={'valid': True, 'team': 'same'}):
            self.assertEqual(len(P.plan(self.runs, lambda t: t)), 2)

    def test_invalid_missing_and_mismatched_design(self):
        with patch.object(P.E, 'completion', side_effect=[None, {'valid': False}]):
            self.assertEqual(P.plan(self.runs, lambda t: t), [])
        self.runs[1][2]['seed'] = 99
        with patch.object(P.E, 'completion', return_value=None), self.assertRaisesRegex(ValueError, 'design differs'):
            P.plan(self.runs, lambda t: t)

    def test_publish_rejects_overwriting_results_configs_and_partial_panels(self):
        group = dict(destinations=[str(self.root / 'off/labels/llm-s0-item-1.json')])
        result = dict(wins=12, battles=50, cache_key='exact')
        P.publish(group, result, {'seed': 42})
        for changed, config in [(dict(result, wins=13), {'seed': 42}),
                                (result, {'seed': 43}), (dict(result, battles=49), {'seed': 42})]:
            with self.assertRaises(ValueError):
                P.publish(group, changed, config)
        self.assertEqual(P.E.read(Path(group['destinations'][0])), result)

    def test_saved_duplicate_is_copied_to_new_repeat_without_rescoring_plan_duplicates(self):
        with patch.object(P.E, 'completion', return_value={'valid': True, 'team': 'same'}):
            group = P.plan(self.runs, lambda t: t)[0]
            first = dict(group, destinations=group['destinations'][:1])
            P.publish(first, dict(wins=2, battles=50, cache_key='cached'), {'seed': 42})
            pending = P.plan(self.runs, lambda t: t)
            self.assertEqual(len(pending), 1)
            self.assertEqual(len(pending[0]['destinations']), 2)

    def test_fast_by_default_and_a_legacy_study_is_archived_for_a_full_rebattle(self):
        self.assertEqual(P.study_backend(self.root, self.runs), 'fast')
        B.save(self.root / 'off' / 'battle-config.json', {'seed': 42})  # written before the fast engine
        B.save(self.root / 'off' / 'labels' / 'llm-s0-item-1.json', {'wins': 1, 'battles': 50})
        B.save(self.root / 'status.json', {'rows': []})
        self.assertEqual(P.study_backend(self.root, self.runs, 'legacy'), 'legacy')
        self.assertTrue((self.root / 'off' / 'labels').exists())
        self.assertEqual(P.study_backend(self.root, self.runs), 'fast')
        self.assertFalse((self.root / 'off' / 'labels').exists())
        self.assertFalse((self.root / 'status.json').exists())
        self.assertEqual(len(list(self.root.glob('off/archive-legacy-battles-*/labels/*.json'))), 1)
        self.assertEqual(len(list(self.root.glob('archive-legacy-battles-*/status.json'))), 1)
        # The fast panels then publish under a fast config; publish still refuses a mix.
        group = dict(destinations=[str(self.root / 'off' / 'labels' / 'llm-s0-item-1.json')])
        P.publish(group, dict(wins=2, battles=50), {'backend': 'fast'})
        with self.assertRaisesRegex(ValueError, 'battle configuration changed'):
            P.publish(group, dict(wins=2, battles=50), {'seed': 42})

    def test_simulator_workers_have_independent_writable_roots(self):
        source = self.root / 'source'
        for name in ['dist/lib', 'config', 'logs', 'node_modules']:
            (source / name).mkdir(parents=True)
        (source / 'dist/lib/fs.js').write_text('frozen simulator')
        (source / 'config/config.js').write_text('same settings')
        (source / 'logs/lastbattle.txt').write_text('source')
        one = B.isolated_showdown(source, self.root / 'one')
        two = B.isolated_showdown(source, self.root / 'two')
        self.assertEqual((one/'dist/lib/fs.js').read_bytes(), (source/'dist/lib/fs.js').read_bytes())
        self.assertFalse((one/'dist').is_symlink())
        (one/'config/config.js').write_text('worker one')
        (one/'logs/lastbattle.txt').write_text('1')
        self.assertEqual((two/'config/config.js').read_text(), 'same settings')
        self.assertFalse((two/'logs/lastbattle.txt').exists())
        self.assertEqual((source/'logs/lastbattle.txt').read_text(), 'source')

    def test_zero_point_neutral_export_keeps_investment_and_non_neutral_natures(self):
        text = ('Pikachu\nLevel: 50\nSerious Nature\n- Protect\n\n'
                'Raichu\nEVs: 1 HP\nSerious Nature\n- Protect\n\n'
                'Raichu\nEVs: 0 HP / 0 Spe\nBrave Nature\n- Protect\n')
        result = B.battle_paste(text)
        self.assertEqual(result, text.replace('Level: 50\nSerious Nature', 'Level: 50\nHardy Nature'))
        self.assertEqual(B.battle_paste(result), result)


if __name__ == '__main__':
    unittest.main()
