import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import repo_paths  # puts src/ first on sys.path
import results
import showdown_validator
import reasoning_battles as P


class ParallelBattleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m = dict(format=showdown_validator.FORMAT, seed=42, battles=50, seed_limit='unseeded simulator',
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

    def test_a_legacy_study_is_archived_for_a_full_rebattle(self):
        P.archive_legacy_study(self.root, self.runs)  # nothing battled yet: nothing moves
        self.assertEqual(list(self.root.glob('**/archive-*')), [])
        results.save(self.root / 'off' / 'battle-config.json', {'seed': 42})  # written before the fast engine
        results.save(self.root / 'off' / 'labels' / 'llm-s0-item-1.json', {'wins': 1, 'battles': 50})
        results.save(self.root / 'status.json', {'rows': []})
        P.archive_legacy_study(self.root, self.runs)
        self.assertFalse((self.root / 'off' / 'labels').exists())
        self.assertFalse((self.root / 'status.json').exists())
        self.assertEqual(len(list(self.root.glob('off/archive-legacy-battles-*/labels/*.json'))), 1)
        self.assertEqual(len(list(self.root.glob('archive-legacy-battles-*/status.json'))), 1)
        # The fast panels then publish under a fast config; publish still refuses a mix.
        group = dict(destinations=[str(self.root / 'off' / 'labels' / 'llm-s0-item-1.json')])
        P.publish(group, dict(wins=2, battles=50), {'backend': 'fast'})
        with self.assertRaisesRegex(ValueError, 'battle configuration changed'):
            P.publish(group, dict(wins=2, battles=50), {'seed': 42})

if __name__ == '__main__':
    unittest.main()
