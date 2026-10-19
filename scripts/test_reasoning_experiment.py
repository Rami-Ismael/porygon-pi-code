import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock, AsyncMock

import llm_baseline as B
import reasoning_experiment as E


class ReasoningExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.args = SimpleNamespace(parent=root / 'parent', output=root / 'study',
                                    arms=['off', 'on'], replicates=2, max_tokens=16384, request_timeout=600)
        team = [dict(species='Pikachu', item='Light Ball', ability='Static', nature='Jolly',
                     moves=['Protect', 'Thunderbolt', 'Thunder', 'Volt Switch'], evs={s: 0 for s in B.STATS})
                for _ in range(6)]
        self.manifest = dict(model=B.MODEL, format=B.FORMAT, starts=[team], seed=42,
            trials=[dict(id='s0-item-1', start=0, task='item', k=1, paths=[[0, 'item']], random_team=team)],
            opponents=[dict(team_id=1, paste='frozen opponent')], battles=50, grid={'item': [1]},
            max_tokens=4096, temperature=.7, reasoning={'enabled': False}, seed_limit='simulator unseeded')
        B.save(self.args.parent / 'manifest.json', self.manifest)

    def prepare(self):
        E.prepare(self.args)
        return E.load_runs(self.args.output)

    def test_mercury_keeps_tasks_and_pins_provider(self):
        self.args.model = 'inception/mercury-2.5'
        self.args.provider = 'inception'
        self.args.arms = ['off', 'low', 'medium', 'high']
        study, runs = self.prepare()
        self.assertEqual(study['planned_calls'], 8)
        for entry, folder, manifest in runs:
            self.assertEqual(manifest['model'], self.args.model)
            self.assertEqual(manifest['starts'], self.manifest['starts'])
            self.assertEqual(manifest['trials'], self.manifest['trials'])
            self.assertEqual(manifest['opponents'], self.manifest['opponents'])
            self.assertEqual(manifest['provider']['order'], ['inception'])
            self.assertFalse(manifest['provider']['allow_fallbacks'])
            self.assertTrue(manifest['provider']['require_parameters'])
        self.assertEqual(E.read(self.args.parent/'manifest.json'), self.manifest)

    def test_unvalidated_saved_response_cannot_finish_study(self):
        import reasoning_battles as battles
        _, runs = self.prepare()
        for entry, folder, _ in runs:
            self.record(entry, folder, valid=False)
        entry, folder, manifest = runs[-1]
        path = folder/'completions/s0-item-1.json'
        raw = E.read(path)
        del raw['error']  # Durable network reply, validation still outstanding.
        B.save(path, raw)
        before = path.read_bytes()
        self.assertFalse(battles.progress(self.args.output)['complete'])
        E.report(self.args)
        self.assertFalse(E.read(self.args.output/'status.json')['complete'])
        self.assertEqual(path.read_bytes(), before)
        self.record(entry, folder, valid=True)
        self.assertFalse(battles.progress(self.args.output)['complete'])
        B.save(folder/'labels/llm-s0-item-1.json', dict(battles=50, wins=3))
        self.assertTrue(battles.progress(self.args.output)['complete'])

    def record(self, entry, folder, valid=True, tokens=None):
        row = dict(request_hash=entry['request_hashes']['s0-item-1'], valid=valid,
                   response={'choices': [{'finish_reason': 'stop' if valid else 'length'}],
                             'usage': {'cost': .01, 'completion_tokens_details': {'reasoning_tokens': tokens}}})
        if valid:
            row['team'] = self.manifest['starts'][0]
        else:
            row['error'] = 'truncated/non-stop completion'
        B.save(folder / 'completions/s0-item-1.json', row)

    def test_prepare_freezes_matched_design_without_mutating_parent(self):
        study, runs = self.prepare()
        self.assertEqual(study['planned_calls'], 4)
        self.assertEqual(E.read(self.args.parent / 'manifest.json'), self.manifest)
        a = B.completion_payload(runs[0][2], self.manifest['trials'][0])
        b = B.completion_payload(runs[2][2], self.manifest['trials'][0])
        self.assertNotEqual(a.pop('reasoning'), b.pop('reasoning'))
        self.assertEqual(a, b)
        self.assertTrue(a['provider']['require_parameters'])
        self.assertEqual(a['max_tokens'], 16384)
        with self.assertRaises(ValueError):
            E.prepare(self.args)

    def test_request_builder_preserves_legacy_defaults(self):
        payload = B.completion_payload(self.manifest, self.manifest['trials'][0])
        self.assertEqual(payload['reasoning'], {'enabled': False})
        self.assertEqual(payload['provider'], {'order': ['novita'], 'allow_fallbacks': False})
        self.assertEqual(payload['max_tokens'], 4096)

    def test_changed_manifest_prompt_or_completion_is_rejected(self):
        _, runs = self.prepare()
        entry, folder, manifest = runs[0]
        with patch.object(B, 'prompt', return_value=[]), self.assertRaisesRegex(ValueError, 'request changed'):
            E.load_runs(self.args.output)
        self.record(entry, folder)
        row = E.read(folder / 'completions/s0-item-1.json')
        row['request_hash'] = 'wrong'
        B.save(folder / 'completions/s0-item-1.json', row)
        with self.assertRaisesRegex(ValueError, 'completion request mismatch'):
            E.report(self.args)
        manifest['max_tokens'] += 1
        B.save(folder / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            E.load_runs(self.args.output)

    def test_partial_reports_do_not_pair_different_replicates_or_score_failures(self):
        _, runs = self.prepare()
        self.record(*runs[0][:2], valid=False, tokens=0)  # off r0: truncated
        self.record(*runs[3][:2], tokens=100)  # on r1, no paired off response
        E.report(self.args)
        status = E.read(self.args.output / 'status.json')
        self.assertFalse(status['complete'])
        self.assertEqual(status['arms']['off']['legality_rate'], 0)
        self.assertEqual(status['arms']['on']['missing'], 1)
        self.assertEqual(status['versus_off']['on']['legality_delta']['pairs'], 0)
        self.assertIsNone(status['arms']['on']['win_rate_legal_scored'])
        self.record(*runs[2][:2], tokens=None)  # on r0 is now paired
        E.report(self.args)
        status = E.read(self.args.output / 'status.json')
        self.assertEqual(status['versus_off']['on']['legality_delta']['mean'], 1)
        self.assertEqual(status['versus_off']['on']['win_rate_delta_both_legal']['pairs'], 0)
        self.assertEqual(status['arms']['on']['reasoning_tokens_reported'], 1)

    def test_battle_pairs_and_config_mismatch(self):
        _, runs = self.prepare()
        for entry, folder, _ in runs:
            self.record(entry, folder)
            B.save(folder / 'battle-config.json', {'policy': 'same', 'seed': 42})
            B.save(folder / 'labels/llm-s0-item-1.json', {'wins': 30 if entry['arm'] == 'on' else 20, 'battles': 50})
        E.report(self.args)
        status = E.read(self.args.output / 'status.json')
        self.assertTrue(status['complete'])
        d = status['versus_off']['on']['win_rate_delta_both_legal']
        self.assertAlmostEqual(d['mean'], .2)
        self.assertEqual(d['pairs'], 2)
        self.assertIsNone(d['low'])
        B.save(runs[0][1] / 'battle-config.json', {'policy': 'different'})
        with self.assertRaisesRegex(ValueError, 'configurations differ'):
            E.report(self.args)

    def test_equal_starter_weights(self):
        d = E.cluster_interval([(0, 0)] * 10 + [(1, 1)], 42)
        self.assertEqual(d['mean'], .5)
        self.assertEqual(d['starters'], 2)

    def test_run_interleaves_and_battle_only_receives_legal_teams(self):
        _, runs = self.prepare()
        with patch('openrouter_generation.generate', new_callable=AsyncMock) as generate:
            E.run(self.args)
        self.assertEqual(generate.call_count, 1)
        jobs = generate.call_args.args[0]
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({j.folder for j in jobs}), 4)
        self.assertTrue(all(j.trial['id'] == 's0-item-1' for j in jobs))
        self.record(*runs[0][:2], valid=False)
        self.record(*runs[2][:2])
        with patch.object(B, 'battle') as battle:
            E.battle(self.args)
        self.assertEqual(battle.call_count, 1)
        self.assertEqual(battle.call_args.kwargs['candidates_override'][0][0], 'llm-s0-item-1')

    def test_unsupported_effort_fails_before_generation(self):
        self.args.arms = ['off', 'low', 'high']
        self.prepare()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps({'data': [{'id': B.MODEL}]}).encode()
        with patch.object(E.urllib.request, 'urlopen', return_value=response), patch.object(B, 'complete') as generate:
            with self.assertRaisesRegex(ValueError, 'does not advertise'):
                E.run(self.args)
            generate.assert_not_called()

    def test_invalid_arms_and_counts(self):
        for arms in (['on'], ['off', 'off'], ['off', 'bogus']):
            with self.assertRaises(ValueError):
                E.arm_configs(arms)
        self.args.replicates = 0
        with self.assertRaises(ValueError):
            E.prepare(self.args)

    def test_completion_resume_and_pending_guard_do_not_send_requests(self):
        _, runs = self.prepare()
        entry, folder, m = runs[0]
        h = Mock()
        self.record(entry, folder)
        with patch.object(B, 'runtime', return_value=(h, None)), patch.object(B.urllib.request, 'urlopen') as request:
            B.complete(SimpleNamespace(output=folder))
            request.assert_not_called()
            (folder / 'completions/s0-item-1.json').unlink()
            B.save(folder / 'completions/s0-item-1.pending', {})
            with self.assertRaisesRegex(RuntimeError, 'uncertain previous API request'):
                B.complete(SimpleNamespace(output=folder))
            request.assert_not_called()

    def test_generation_sends_reasoning_and_retains_truncation_usage(self):
        _, runs = self.prepare()
        entry, folder, m = runs[2]
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = json.dumps(dict(choices=[dict(finish_reason='length',
            message={'content': ''})], usage={'completion_tokens_details': {'reasoning_tokens': 16384}})).encode()
        h = Mock()
        with patch.dict(B.os.environ, {'OPENROUTER_API_KEY': 'test-only'}), \
                patch.object(B, 'runtime', return_value=(h, None)), \
                patch.object(B.urllib.request, 'urlopen', return_value=response) as request:
            B.complete(SimpleNamespace(output=folder))
        sent = json.loads(request.call_args.args[0].data)
        self.assertEqual(sent['reasoning'], {'enabled': True})
        self.assertEqual(request.call_args.kwargs['timeout'], 600)
        record = E.read(folder / 'completions/s0-item-1.json')
        self.assertFalse(record['valid'])
        self.assertIn('truncated', record['error'])
        self.assertEqual(record['response']['usage']['completion_tokens_details']['reasoning_tokens'], 16384)
        self.assertGreaterEqual(record['elapsed_seconds'], 0)
        self.assertFalse((folder / 'completions/s0-item-1.pending').exists())


if __name__ == '__main__':
    unittest.main()
