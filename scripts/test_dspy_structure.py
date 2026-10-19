import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from repo_paths import ROOT  # also puts src/ first on sys.path
import llm_completion
import results
import showdown_validator
import team_masks
import reasoning_experiment as E

try:
    import dspy  # noqa: F401
    import dspy_structure as S
    import dspy_structure_experiment as X
except ImportError:  # the runtime without DSPy still runs the other suites
    S = None

SPECIES = [('Pikachu', 'Light Ball', 'Static'), ('Charizard', 'Charizardite Y', 'Blaze'),
           ('Garchomp', 'Life Orb', 'Rough Skin'), ('Incineroar', 'Sitrus Berry', 'Intimidate'),
           ('Rotom-Wash', 'Leftovers', 'Levitate'), ('Kingambit', 'Black Glasses', 'Defiant')]
TEAM = [dict(species=s, item=i, ability=a, nature='Jolly', moves=['Protect', 'Thunderbolt', 'Thunder', 'Volt Switch'],
             evs={k: 0 for k in team_masks.STATS}) for s, i, a in SPECIES]
TRIALS = [dict(id='s0-item-2', start=0, task='item', k=2, paths=[[0, 'item'], [2, 'evs']]),
          dict(id='s0-pokemon-1', start=0, task='pokemon', k=1, paths=[[1]]),
          dict(id='s1-move-2', start=1, task='move', k=2, paths=[[3, 'moves', 2], [4, 'nature']])]
MANIFEST = dict(model=llm_completion.MODEL, format=showdown_validator.FORMAT, starts=[TEAM, TEAM[::-1]], seed=7, trials=TRIALS,
                opponents=[dict(team_id=1, paste='Pikachu @ Light Ball\nAbility: Static\n- Protect')],
                battles=50, grid={'item': [2], 'pokemon': [1], 'move': [2]}, temperature=.7, max_tokens=4096,
                reasoning={'enabled': False}, seed_limit='simulator unseeded', preserve_empty_move_slots=True)
FILLS = {'s0-item-2': [{'path': [0, 'item'], 'value': 'Focus Sash'},
                       {'path': [2, 'evs'], 'value': dict(HP=2, Atk=32, Def=0, SpA=0, SpD=0, Spe=32)}],
         's0-pokemon-1': [{'path': [1], 'value': dict(species='Whimsicott', item='Focus Sash', ability='Prankster',
                                                     nature='Timid', moves=['Tailwind', 'Moonblast', 'Protect', 'Encore'],
                                                     evs=dict(HP=32, Atk=0, Def=2, SpA=0, SpD=0, Spe=32))}],
         's1-move-2': [{'path': [3, 'moves', 2], 'value': 'Fake Out'}, {'path': [4, 'nature'], 'value': 'Calm'}]}


def structured(adapter, version=None):
    m = dict(MANIFEST, structure=S.config(adapter))
    if version:
        m['prompt_version'] = version
    return m


def reply(adapter, fills):
    if adapter == 'chat':
        return f'[[ ## fills ## ]]\n{json.dumps(fills)}\n\n[[ ## completed ## ]]'
    return json.dumps({'fills': fills})


def response(content, cost=.001):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': content}}], 'usage': {'cost': cost}}


@unittest.skipIf(S is None, 'DSPy is not installed')
class StructureTests(unittest.TestCase):
    def test_same_information_and_settings_for_every_prompt_version(self):
        for version in (None, 'mb-grounded-v2', 'mb-regulation-v3'):
            for adapter in S.ADAPTERS:
                m = structured(adapter, version)
                plain = {k: v for k, v in m.items() if k != 'structure'}
                for t in TRIALS:
                    self.assertTrue(S.check_information(m, t))
                    a, b = llm_completion.completion_payload(m, t), llm_completion.completion_payload(plain, t)
                    self.assertNotEqual(a.pop('messages'), b.pop('messages'))
                    self.assertEqual(a, b)

    def test_only_reply_format_sentences_leave_the_instructions(self):
        m = structured('chat', 'mb-regulation-v3')
        system = llm_completion.prompt({k: v for k, v in m.items() if k != 'structure'}, TRIALS[0])[0]['content']
        kept = S.strip_envelope(system)
        words = lambda text: ' '.join(text.split())  # DSPy indents multi-line instructions
        self.assertIn(words(kept), words(llm_completion.completion_payload(m, TRIALS[0])['messages'][0]['content']))
        self.assertNotIn('Return only JSON', kept)
        self.assertEqual(len(system) - len(kept), sum(len(s) for s in S.ENVELOPE if s in system))
        with self.assertRaisesRegex(ValueError, 'reply-format'):
            S.strip_envelope('No envelope here.')

    def test_typed_parse_feeds_the_unchanged_completion_checks(self):
        for adapter in S.ADAPTERS:
            m = structured(adapter)
            for t in TRIALS:
                diagnostics = {}
                answer = S.parse(m, t, reply(adapter, FILLS[t['id']]), diagnostics)
                self.assertEqual(answer, {'fills': FILLS[t['id']]})
                self.assertTrue(diagnostics['strict_json'])
                start = m['starts'][t['start']]
                self.assertEqual(llm_completion.apply_completion(start, t['paths'], answer),
                                 llm_completion.apply_completion(start, t['paths'], {'fills': FILLS[t['id']]}))

    def test_strict_types_refuse_what_the_baseline_refuses(self):
        m = structured('chat')
        bad = copy.deepcopy(FILLS['s0-item-2'])
        bad[1]['value']['HP'] = '2'  # a string Stat Point is not silently coerced
        with self.assertRaisesRegex(ValueError, 'DSPy parse'):
            S.parse(m, TRIALS[0], reply('chat', bad), {})
        extra = copy.deepcopy(FILLS['s0-item-2'])
        extra[0]['note'] = 'x'
        with self.assertRaisesRegex(ValueError, 'DSPy parse'):
            S.parse(m, TRIALS[0], reply('chat', extra), {})
        with self.assertRaises(ValueError):
            S.parse(m, TRIALS[0], json.dumps({'fills': FILLS['s0-item-2']}), {})  # no field markers
        with self.assertRaises(TypeError):
            S.parse(m, TRIALS[0], None, {})

    def test_json_repair_is_recorded_not_hidden(self):
        m = structured('chat')
        text = reply('chat', FILLS['s0-item-2']).replace(']\n\n[[ ## completed', ',]\n\n[[ ## completed')
        diagnostics = {}
        self.assertEqual(S.parse(m, TRIALS[0], text, diagnostics), {'fills': FILLS['s0-item-2']})
        self.assertFalse(diagnostics['strict_json'])

    def test_validate_record_uses_structure_only_when_declared(self):
        h = Mock(slot_to_text=lambda slot: slot['species'])
        validator = Mock(return_value=None)
        m = structured('json')
        record = dict(response=response(reply('json', FILLS['s0-item-2'])), valid=False)
        llm_completion.validate_record(record, m, TRIALS[0], h, validator)
        self.assertTrue(record['valid'])
        self.assertEqual(record['structure'], {'strict_json': True})
        plain = dict(response=response(reply('json', FILLS['s0-item-2'])), valid=False)
        llm_completion.validate_record(plain, MANIFEST, TRIALS[0], h, validator)
        self.assertEqual(plain['team'], record['team'])
        self.assertNotIn('structure', plain)
        chat_reply = dict(response=response(reply('chat', FILLS['s0-item-2'])), valid=False)
        llm_completion.validate_record(chat_reply, MANIFEST, TRIALS[0], h, validator)
        self.assertFalse(chat_reply['valid'])  # the baseline parser cannot read field markers

    def test_frozen_dspy_version_is_enforced(self):
        m = structured('chat')
        m['structure']['dspy_version'] = '0.0.0'
        with self.assertRaisesRegex(ValueError, 'differs from frozen'):
            llm_completion.completion_payload(m, TRIALS[0])


# FILLS answers Fake Out for Garchomp, which the legal-choice type rightly refuses.
LEGAL_FILLS = dict(FILLS, **{'s1-move-2': [{'path': [3, 'moves', 2], 'value': 'Earthquake'},
                                          {'path': [4, 'nature'], 'value': 'Calm'}]})


@unittest.skipIf(S is None, 'DSPy is not installed')
class LegalChoiceTests(unittest.TestCase):
    """The 'legal-choices' answer type limits each requested path to what the regulation allows."""

    def constrained(self, adapter='chat'):
        return dict(MANIFEST, structure=S.config(adapter, constraints='legal-choices'))

    def answer_type(self, trial):
        return S.parts(self.constrained(), trial)[0].output_fields['fills'].annotation

    def choices(self, schema, field):
        prop = schema['properties'][field]
        ref = prop.get('$ref') or prop.get('allOf', [{}])[0].get('$ref')
        return set(schema['$defs'][ref.split('/')[-1]]['enum']) if ref else set(prop['enum'])

    def test_phase_one_manifests_keep_their_hashes(self):
        self.assertNotIn('constraints', S.config('chat'))
        with self.assertRaises(ValueError):
            S.config('chat', constraints='anything')

    def test_each_path_is_limited_to_its_legal_choices(self):
        schema = self.answer_type(TRIALS[0]).model_json_schema()
        items = self.choices(schema, 'slot0_item')
        self.assertIn('Light Ball', items)  # Pikachu with Static
        # Item Clause: teammates already hold these.
        self.assertFalse(items & {'Charizardite Y', 'Life Orb', 'Sitrus Berry', 'Leftovers', 'Black Glasses'})
        spread = schema['$defs']['StatPoints']['properties']
        self.assertEqual((spread['Atk']['minimum'], spread['Atk']['maximum']), (0, 32))
        self.assertEqual(spread['total']['maximum'], 66)
        moves = self.choices(self.answer_type(TRIALS[2]).model_json_schema(), 'slot3_move2')
        self.assertIn('Earthquake', moves)  # slot 3 of the reversed team is Garchomp
        self.assertNotIn('Fake Out', moves)  # Garchomp cannot learn it
        self.assertNotIn('Thunderbolt', moves)  # already in the slot
        whole = self.answer_type(TRIALS[1]).model_json_schema()
        species = set(whole['$defs']['Species']['enum'])
        self.assertIn('Whimsicott', species)
        self.assertNotIn('Garchomp', species)  # Species Clause: already on the team

    def test_inputs_and_instructions_are_unchanged(self):
        for trial in TRIALS:
            self.assertEqual(S.check_information(self.constrained(), trial), ['format', 'masked_team', 'opponents', 'requested_paths'])
            system = S.messages(self.constrained(), trial)[0]['content']
            self.assertIn(S.strip_envelope(llm_completion.prompt(MANIFEST, trial)[0]['content']), system)

    def test_answers_return_in_the_baseline_shape_without_the_total(self):
        for adapter in S.ADAPTERS:
            for trial in TRIALS:
                fills = LEGAL_FILLS[trial['id']]
                values = []
                for fill in fills:
                    v = copy.deepcopy(fill['value'])
                    for spread in (v, v.get('evs') if isinstance(v, dict) else None):
                        if isinstance(spread, dict) and 'HP' in spread:
                            spread['total'] = sum(spread.values())
                    values.append(v)
                fields = dict(zip(self.answer_type(trial).model_fields, values))
                content = (f'[[ ## fills ## ]]\n{json.dumps(fields)}\n\n[[ ## completed ## ]]' if adapter == 'chat'
                           else json.dumps({'fills': fields}))
                answer = S.parse(self.constrained(adapter), trial, content, {})
                self.assertEqual(answer, {'fills': fills})
                llm_completion.apply_completion(TEAM, trial['paths'], answer)

    def test_a_value_outside_the_choices_is_an_invalid_answer(self):
        trial = TRIALS[2]
        content = '[[ ## fills ## ]]\n' + json.dumps({'slot3_move2': 'Fake Out', 'slot4_nature': 'Calm'}) + '\n\n[[ ## completed ## ]]'
        with self.assertRaisesRegex(ValueError, 'DSPy parse') as caught:
            S.parse(self.constrained(), trial, content, {})
        # Counted as a rule violation, not as an unreadable answer.
        self.assertEqual(X.failure_class(str(caught.exception)), 'rule violation')

    def test_a_long_answer_keeps_the_reason_it_was_refused(self):
        # DSPy echoes the answer before the reason; a whole-team answer is long enough to push it out.
        trial = TRIALS[1]
        slot = dict(species='Mewtwo', item='Focus Sash', ability='Prankster', nature='Timid',
                    moves=['Tailwind', 'Moonblast', 'Protect', 'Encore'],
                    evs=dict(HP=32, Atk=0, Def=2, SpA=0, SpD=0, Spe=32, total=66), padding='x' * 2000)
        content = '[[ ## fills ## ]]\n' + json.dumps({'slot1': slot}) + '\n\n[[ ## completed ## ]]'
        with self.assertRaises(ValueError) as caught:
            S.parse(self.constrained(), trial, content, {})
        self.assertIn('validation error', str(caught.exception))
        self.assertEqual(X.failure_class(str(caught.exception)), 'wrong answer shape')  # extra key present

    def test_prepare_records_the_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results.save(root / 'parent/manifest.json', MANIFEST)
            E.prepare(SimpleNamespace(parent=root / 'parent', output=root / 'baseline', arms=['off', 'on'],
                                      replicates=1, max_tokens=4096, request_timeout=600))
            _, runs = E.load_runs(root / 'baseline')
            for entry, folder, m in runs:
                for t in m['trials']:
                    results.save(folder / 'completions' / (t['id'] + '.json'),
                           dict(request_hash=entry['request_hashes'][t['id']], valid=False, error='wrong number of fills',
                                response={'choices': [{'finish_reason': 'stop'}], 'usage': {'cost': .001}}))
            E.report(SimpleNamespace(output=root / 'baseline'))
            X.prepare(SimpleNamespace(output=root / 'dspy', baseline=[root / 'baseline'], adapters=['chat'],
                                      resample_baseline=False, program=None, constraints='legal-choices'))
            study, runs = E.load_runs(root / 'dspy')
            self.assertEqual(study['constraints'], 'legal-choices')
            self.assertTrue(all(m['structure']['constraints'] == 'legal-choices' for _, _, m in runs))


@unittest.skipIf(S is None, 'DSPy is not installed')
class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        results.save(self.root / 'parent/manifest.json', MANIFEST)
        self.baseline = self.root / 'baseline'
        E.prepare(SimpleNamespace(parent=self.root / 'parent', output=self.baseline, arms=['off', 'on'],
                                  replicates=1, max_tokens=4096, request_timeout=600))
        _, runs = E.load_runs(self.baseline)
        for entry, folder, m in runs:
            for t in m['trials']:
                legal = t['id'] != 's1-move-2'
                self.save(folder, entry, t, legal, wins=2 if legal else None)
        E.report(SimpleNamespace(output=self.baseline))
        self.args = SimpleNamespace(output=self.root / 'dspy', baseline=[self.baseline], adapters=['chat', 'json'],
                                    resample_baseline=True, program=None, win_margin=.01, legality_margin=.05)

    def save(self, folder, entry, t, legal, wins=None, strict=None):
        row = dict(request_hash=entry['request_hashes'][t['id']], valid=legal,
                   response={'choices': [{'finish_reason': 'stop'}], 'usage': {'cost': .001}})
        if legal:
            row['team'] = MANIFEST['starts'][t['start']]
        else:
            row['error'] = 'wrong number of fills'
        if strict is not None:
            row['structure'] = {'strict_json': strict}
        results.save(folder / 'completions' / (t['id'] + '.json'), row)
        results.save(folder / 'battle-config.json', {'policy': 'bc', 'seed': 7})
        if wins is not None:
            results.save(folder / 'labels' / ('llm-' + t['id'] + '.json'), dict(battles=50, wins=wins))

    def test_prepare_mirrors_every_baseline_run(self):
        X.prepare(self.args)
        study, runs = E.load_runs(self.args.output)
        self.assertEqual(study['planned_calls'], 2 * 3 * len(TRIALS))
        _, baseline = E.load_runs(self.baseline)
        original = {e['path']: e for e, _, _ in baseline}
        for entry, _, m in runs:
            self.assertEqual(m['reasoning'], {'enabled': entry['reasoning'] == 'on'})
            if entry['variant'] == 'resampled':
                self.assertEqual(entry['request_hashes'], original[entry['baseline_path']]['request_hashes'])
                self.assertNotIn('structure', m)
            else:
                self.assertEqual(m['structure']['adapter'], entry['variant'])
        with self.assertRaises(ValueError):
            X.prepare(self.args)

    def test_report_pairs_structured_answers_with_their_baseline(self):
        X.prepare(self.args)
        _, runs = E.load_runs(self.args.output)
        for entry, folder, m in runs:
            for t in m['trials']:
                # Structured answers rescue the move mask and win more battles.
                self.save(folder, entry, t, True, wins=5 if entry['variant'] != 'resampled' else 2,
                          strict=None if entry['variant'] == 'resampled' else t['id'] != 's1-move-2')
        status = X.report(self.args)
        self.assertTrue(status['complete'])
        cell = next(c for c in status['cells'] if c['variant'] == 'chat' and c['reasoning'] == 'off')
        self.assertAlmostEqual(cell['baseline']['legality_rate'], 2 / 3)
        self.assertEqual(cell['structured']['legal'], 3)
        self.assertEqual(cell['structured']['legal_after_json_repair'], 1)
        self.assertAlmostEqual(cell['versus_baseline']['win_rate_both_legal']['mean'], .06)
        self.assertEqual(cell['versus_baseline']['win_rate_both_legal']['pairs'], 2)
        self.assertAlmostEqual(cell['versus_resampled']['win_rate_both_legal']['mean'], .06)
        self.assertEqual(cell['baseline']['failures'], {'wrong answer shape': 1})
        self.assertIn('Baseline JSON prompt, sampled again', (self.args.output / 'report.md').read_text())

    def test_provider_override_is_recorded_and_reported(self):
        model = MANIFEST['model']
        self.args.provider = [f'{model}=deepinfra']
        X.prepare(self.args)
        study, runs = E.load_runs(self.args.output)
        _, baseline = E.load_runs(self.baseline)
        original = {e['path']: (e, m) for e, _, m in baseline}
        self.assertEqual(study['provider_overrides'][model]['used']['order'], ['deepinfra'])
        for entry, folder, m in runs:
            old_entry, old = original[entry['baseline_path']]
            t = m['trials'][0]
            self.assertEqual(llm_completion.completion_payload(m, t)['provider']['order'], ['deepinfra'])
            if entry['variant'] == 'resampled':
                self.assertEqual(llm_completion.completion_payload(m, t)['messages'], llm_completion.completion_payload(old, t)['messages'])
                self.assertNotEqual(entry['request_hashes'], old_entry['request_hashes'])
            for t in m['trials']:
                self.save(folder, entry, t, True, wins=2)
        X.report(self.args)
        self.assertIn('Provider changed for', (self.args.output / 'report.md').read_text())

    def test_report_refuses_a_different_battle_runtime(self):
        X.prepare(self.args)
        _, runs = E.load_runs(self.args.output)
        entry, folder, m = runs[0]
        self.save(folder, entry, m['trials'][0], True, wins=3)
        results.save(folder / 'battle-config.json', {'policy': 'other', 'seed': 7})
        with self.assertRaisesRegex(ValueError, 'not comparable'):
            X.report(self.args)


    def rebattle_baseline_on_fast(self, wins):
        import reasoning_battles as RB
        _, runs = E.load_runs(self.baseline)
        RB.archive_legacy_study(self.baseline, runs)
        for entry, folder, m in runs:
            results.save(folder / 'battle-config.json', {'policy': 'bc', 'seed': 7, 'backend': 'fast'})
            for t in m['trials']:
                if t['id'] != 's1-move-2':
                    results.save(folder / 'labels' / ('llm-' + t['id'] + '.json'), dict(battles=50, wins=wins))
        E.report(SimpleNamespace(output=self.baseline))

    def test_report_accepts_a_baseline_rebattled_on_another_engine(self):
        X.prepare(self.args)
        self.rebattle_baseline_on_fast(wins=4)
        _, runs = E.load_runs(self.args.output)
        for entry, folder, m in runs:
            for t in m['trials']:
                self.save(folder, entry, t, True, wins=5)
            results.save(folder / 'battle-config.json', {'policy': 'bc', 'seed': 7, 'backend': 'fast'})
        status = X.report(self.args)
        self.assertTrue(status['complete'])
        cell = next(c for c in status['cells'] if c['variant'] == 'chat' and c['reasoning'] == 'off')
        self.assertAlmostEqual(cell['versus_baseline']['win_rate_both_legal']['mean'], .02)  # 5/50 - 4/50

    def test_report_refuses_a_baseline_changed_beyond_its_win_rates(self):
        X.prepare(self.args)
        self.rebattle_baseline_on_fast(wins=4)
        status = E.read(self.baseline / 'status.json')
        status['rows'][0]['legal'] = not status['rows'][0]['legal']
        results.save(self.baseline / 'status.json', status)
        with self.assertRaisesRegex(ValueError, 'baseline status changed since prepare'):
            X.report(self.args)


@unittest.skipIf(S is None, 'DSPy is not installed')
class StatisticsTests(unittest.TestCase):
    def test_failure_classes(self):
        self.assertEqual(X.failure_class('Expecting value: line 1 column 1 (char 0)'), 'unreadable answer')
        self.assertEqual(X.failure_class('DSPy parse: missing fills'), 'unreadable answer')
        self.assertEqual(X.failure_class('duplicate or unmasked path'), 'wrong answer shape')
        self.assertEqual(X.failure_class('Stat Points require 0..32 per stat, total <=66'), 'Stat Points out of range')
        self.assertEqual(X.failure_class("Abomasnow can't learn Psybeam."), 'rule violation')
        self.assertEqual(X.failure_class("'choices'"), 'no answer')
        self.assertEqual(X.failure_class('truncated/non-stop completion'), 'cut off')

    def test_sign_flip_and_verdicts(self):
        self.assertEqual(X.sign_flip_p([1.0] * 5, 0), 2 / 32)
        self.assertEqual(X.sign_flip_p([1, -1, 1, -1], 0), 1.0)
        self.assertLess(X.sign_flip_p([.02] * 20, 0), .001)
        self.assertEqual(X.verdict(dict(mean=.01, low=.001, high=.02), .01), 'improved')
        self.assertEqual(X.verdict(dict(mean=-.01, low=-.02, high=-.001), .01), 'regressed')
        self.assertEqual(X.verdict(dict(mean=0, low=-.004, high=.005), .01), 'no change beyond ±1 pp')
        self.assertEqual(X.verdict(dict(mean=0, low=-.03, high=.02), .01), 'unclear: interval includes zero')

    def test_headroom_reads_published_mask_lab_data(self):
        rows = X.headroom(SimpleNamespace(data=ROOT.parent / 'blog/mask-lab/data.json'))
        off = next(r for r in rows if r['reasoning'] == 'off')
        self.assertEqual((off['attempts'], off['legal']), (420, 119))
        self.assertAlmostEqual(off['ceiling'], (119 + 72 + 48) / 420)


if __name__ == '__main__':
    unittest.main()
