import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import Mock

import repo_paths  # puts src/ first on sys.path
import llm_completion
import results

try:
    import dspy
    from dspy.utils import DummyLM
    import dspy_optimize as O
    import dspy_structure as S
    from test_dspy_structure import FILLS, LEGAL_FILLS, MANIFEST, TRIALS
except ImportError:  # the runtime without DSPy still runs the other suites
    O = None


@unittest.skipIf(O is None, 'DSPy is not installed')
class OptimizeTests(unittest.TestCase):
    def setUp(self):
        self.m = dict(MANIFEST, structure=S.config('chat'), optimization=dict(split={'0': 'train', '1': 'dev'}))
        self.h = Mock(slot_to_text=lambda slot: slot['species'])
        self.validator = Mock(return_value=None)

    def run_program(self, fills, example):
        with dspy.context(lm=DummyLM([{'fills': json.dumps(fills)}]), adapter=S.adapter_for(self.m['structure'])):
            return O.Completion(self.m)(**example.inputs())

    def test_splits_keep_starters_apart(self):
        train, dev = O.examples(self.m, 'train'), O.examples(self.m, 'dev')
        self.assertEqual([e.trial_id for e in train], ['s0-item-2', 's0-pokemon-1'])
        self.assertEqual([e.trial_id for e in dev], ['s1-move-2'])
        self.assertEqual(set(train[0].inputs()), {'format', 'opponents', 'masked_team', 'requested_paths', 'trial_id'})

    def test_metric_scores_legality_with_the_validator_message(self):
        metric = O.Metric(self.m, tempfile.mkdtemp(), validator=self.validator, h=self.h)
        example = O.examples(self.m, 'train')[0]
        self.assertEqual(metric(example, self.run_program(FILLS['s0-item-2'], example)).score, 1.0)
        rejected = metric(example, self.run_program(FILLS['s1-move-2'], example))
        self.assertEqual((rejected.score, rejected.feedback), (0.0, 'Rejected: duplicate or unmasked path'))
        self.validator.return_value = 'Garchomp has 76 total Stat Points'
        self.assertIn('76 total', metric(example, self.run_program(FILLS['s0-item-2'], example)).feedback)

    def test_battle_metric_caches_each_team_panel(self):
        battle = Mock(return_value=.1)
        metric = O.Metric(self.m, tempfile.mkdtemp(), battles=True, alpha=.5,
                          validator=self.validator, h=self.h, battle=battle)
        example = O.examples(self.m, 'train')[0]
        pred = self.run_program(FILLS['s0-item-2'], example)
        self.assertAlmostEqual(metric(example, pred).score, .55)
        self.assertAlmostEqual(metric(example, pred).score, .55)
        self.assertEqual(battle.call_count, 1)

    def test_exported_program_reaches_the_phase_one_request(self):
        program = O.Completion(self.m)
        example = O.examples(self.m, 'train')[0]
        program.predict.signature = program.predict.signature.with_instructions('Fill each null with a legal value.')
        program.predict.demos = [dspy.Example(**example.inputs(), fills=FILLS['s0-item-2'])]
        exported = O.export(program, self.m, {'name': 'test'})
        m = dict(MANIFEST, structure=S.config('chat', exported['instructions'], exported['demos']))
        system, *demo, user = llm_completion.completion_payload(m, TRIALS[2])['messages']
        self.assertIn('Fill each null with a legal value.', system['content'])
        self.assertNotIn('Complete only the null fields', system['content'])
        self.assertEqual(len(demo), 2)  # one demonstration: a user turn and an assistant turn
        self.assertIn('Focus Sash', demo[1]['content'])
        S.check_information(m, TRIALS[2])


@unittest.skipIf(O is None, 'DSPy is not installed')
class LegalChoiceOptimizeTests(unittest.TestCase):
    """GEPA optimizes one predictor's instructions while every task keeps its own legal-choice type."""

    def setUp(self):
        self.m = dict(MANIFEST, structure=S.config('chat', constraints='legal-choices'),
                      optimization=dict(split={'0': 'train', '1': 'dev'}))
        self.h = Mock(slot_to_text=lambda slot: slot['species'])
        self.validator = Mock(return_value=None)
        self.metric = O.Metric(self.m, tempfile.mkdtemp(), validator=self.validator, h=self.h)
        self.example = O.examples(self.m, 'dev')[0]  # s1-move-2: Garchomp's third move and a nature

    def run_program(self, answer, program=None, adapter=None):
        lm = DummyLM([{'fills': json.dumps(answer)}])
        with dspy.context(lm=lm, adapter=adapter or O.lenient(S.adapter_for(self.m['structure']))):
            pred = (program or O.Completion(self.m))(**self.example.inputs())
        return pred, lm.history[-1]['messages']

    def test_each_task_is_sent_its_own_legal_choices(self):
        pred, messages = self.run_program({'slot3_move2': 'Earthquake', 'slot4_nature': 'Calm'})
        self.assertIn('slot3_move2', messages[0]['content'])
        self.assertIn('Earthquake', messages[0]['content'])
        self.assertEqual(self.metric(self.example, pred).score, 1.0)
        self.assertEqual(llm_completion.apply_completion(MANIFEST['starts'][1], TRIALS[2]['paths'],
                                            S.as_fills(pred.fills, TRIALS[2]['paths'])),
                         llm_completion.apply_completion(MANIFEST['starts'][1], TRIALS[2]['paths'], {'fills': LEGAL_FILLS['s1-move-2']}))

    def test_a_refused_value_reaches_gepa_as_feedback(self):
        pred, _ = self.run_program({'slot3_move2': 'Fake Out', 'slot4_nature': 'Calm'})
        result = self.metric(self.example, pred)
        self.assertEqual(result.score, 0.0)
        self.assertIn('slot3_move2', result.feedback)
        self.assertIn('type=enum', result.feedback)
        self.validator.assert_not_called()

    def test_without_the_lenient_adapter_a_refusal_is_a_parse_failure(self):
        with self.assertRaises(dspy.AdapterParseError):
            self.run_program({'slot3_move2': 'Fake Out', 'slot4_nature': 'Calm'}, adapter=S.adapter_for(self.m['structure']))

    def test_instructions_gepa_writes_reach_every_typed_call(self):
        program = O.Completion(self.m)
        program.predict.signature = program.predict.signature.with_instructions('Pick only from the listed choices.')
        _, messages = self.run_program({'slot3_move2': 'Earthquake', 'slot4_nature': 'Calm'}, program=program)
        self.assertIn('Pick only from the listed choices.', messages[0]['content'])
        self.assertNotIn('Complete only the null fields', messages[0]['content'])
        self.assertIn('slot3_move2', messages[0]['content'])

    def test_optimized_requests_match_the_test_requests(self):
        # The phase-one test path sends the adapter's messages and no response_format.
        trial = TRIALS[2]
        _, messages = self.run_program({'slot3_move2': 'Earthquake', 'slot4_nature': 'Calm'})
        self.assertEqual(messages, llm_completion.completion_payload(self.m, trial)['messages'])
        kwargs = {}
        O.lenient(S.adapter_for(S.config('json', constraints='legal-choices')))._prepare_response_format(None, kwargs, None)
        self.assertEqual(kwargs, {})

    def test_gepa_learns_from_refusals_on_the_typed_program(self):
        refused = {'fills': json.dumps({'slot3_move2': 'Fake Out', 'slot4_nature': 'Calm'})}
        legal = {'fills': json.dumps({'slot3_move2': 'Earthquake', 'slot4_nature': 'Calm'})}
        task = DummyLM([refused] * 4 + [legal] * 40)  # base program refused; the proposed one legal
        reflection = DummyLM([{'new_instruction': 'Choose only listed values.'}] * 10, adapter=dspy.JSONAdapter())
        with dspy.context(lm=task, adapter=O.lenient(S.adapter_for(self.m['structure']))):
            gepa = dspy.GEPA(metric=self.metric, max_metric_calls=12, reflection_lm=reflection,
                             add_format_failure_as_feedback=True, num_threads=1, seed=1)
            dev = O.examples(self.m, 'dev')
            optimized = gepa.compile(O.Completion(self.m), trainset=dev, valset=dev)
        # The reflection model was shown the refusal reason, not a generic parse note.
        self.assertIn('type=enum', json.dumps(reflection.history[0]['messages']))
        exported = O.export(optimized, self.m, {'name': 'gepa'})
        self.assertEqual(exported['instructions'], 'Choose only listed values.')
        self.assertEqual(exported['demos'], [])
        m = dict(MANIFEST, structure=S.config('chat', exported['instructions'], constraints='legal-choices'))
        S.check_information(m, TRIALS[2])

    def test_mipro_is_refused_with_legal_choice_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            results.save(Path(tmp) / 'manifest.json', self.m)
            args = SimpleNamespace(output=Path(tmp), optimizer='mipro', num_retries=3)
            with unittest.mock.patch.dict(sys.modules, {'hps_generate': self.h}), \
                 unittest.mock.patch.object(O, 'lm_for', return_value=DummyLM([])):
                self.h.Validator = Mock(return_value=self.validator)
                with self.assertRaisesRegex(ValueError, 'MIPROv2'):
                    O.optimize(args)


@unittest.skipIf(O is None, 'DSPy is not installed')
class ReflectionModelTests(unittest.TestCase):
    """GEPA's prompt writer runs on its own route, not the task model's."""

    def setUp(self):
        self.m = dict(MANIFEST, structure=S.config('chat', constraints='legal-choices'),
                      optimization=dict(split={'0': 'train', '1': 'dev'}),
                      provider={'order': ['inception'], 'allow_fallbacks': False, 'require_parameters': True})

    def test_reflection_model_keeps_none_of_the_task_settings(self):
        with unittest.mock.patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test'}):
            lm = O.reflection_lm_for('openai/gpt-6-luna')
            task = O.lm_for(self.m)
        self.assertEqual(lm.model, 'openrouter/openai/gpt-6-luna')
        self.assertNotIn('temperature', {k: v for k, v in lm.kwargs.items() if v is not None})
        self.assertEqual(lm.kwargs['extra_body'], {'reasoning': {'effort': 'high'}})  # no provider pin
        self.assertEqual(lm.kwargs['max_tokens'], 32000)
        with unittest.mock.patch.dict('os.environ', {'OPENROUTER_API_KEY': 'test'}):
            self.assertEqual(O.reflection_lm_for('openai/gpt-6-luna', num_retries=8).num_retries, 8)
            self.assertEqual(O.lm_for(self.m, num_retries=8).num_retries, 8)
        # The task model still carries the baseline's route and sampling settings.
        self.assertEqual(task.kwargs['extra_body']['provider']['order'], ['inception'])
        self.assertEqual(task.kwargs['temperature'], MANIFEST['temperature'])

    def test_reflection_model_needs_a_key(self):
        with unittest.mock.patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(RuntimeError):
                O.reflection_lm_for('openai/gpt-6-luna')

    def test_optimize_hands_gepa_the_separate_reflection_model(self):
        class Stop(Exception):
            pass
        seen = {}

        def fake_gepa(**kwargs):
            seen.update(kwargs)
            raise Stop

        writer = DummyLM([])
        with tempfile.TemporaryDirectory() as tmp:
            results.save(Path(tmp) / 'manifest.json', self.m)
            args = SimpleNamespace(output=Path(tmp), optimizer='gepa', battles=False, alpha=.5, budget='light',
                                   threads=1, seed=1, reflection_model='openai/gpt-6-luna', reflection_effort='high',
                                   num_retries=8)
            h = Mock(Validator=Mock(return_value=Mock()))
            with unittest.mock.patch.dict(sys.modules, {'hps_generate': h}), \
                 unittest.mock.patch.object(O, 'lm_for', return_value=DummyLM([])), \
                 unittest.mock.patch.object(O, 'reflection_lm_for', return_value=writer) as make_writer, \
                 unittest.mock.patch.object(dspy, 'GEPA', side_effect=fake_gepa):
                with self.assertRaises(Stop):
                    O.optimize(args)
        make_writer.assert_called_once_with('openai/gpt-6-luna', 'high', 8)
        self.assertIs(seen['reflection_lm'], writer)


@unittest.skipIf(O is None, 'DSPy is not installed')
class ReflectionShapeTests(unittest.TestCase):
    """The reflection model is shown the answer shape each task requires, including the Stat Points total."""

    def setUp(self):
        self.m = dict(MANIFEST, structure=S.config('chat', constraints='legal-choices'),
                      optimization=dict(split={'0': 'train', '1': 'dev'}))
        self.metric = O.Metric(self.m, tempfile.mkdtemp(), validator=Mock(return_value=None),
                               h=Mock(slot_to_text=lambda slot: slot['species']))
        self.example = O.examples(self.m, 'train')[0]  # s0-item-2: Pikachu's item and Garchomp's Stat Points

    def test_compact_schema_keeps_the_shape_and_drops_long_lists(self):
        schema = O.compact_schema(S.legal_completion_type(MANIFEST['starts'][0], TRIALS[0]['paths']).model_json_schema())
        spread = schema['$defs']['StatPoints']
        self.assertIn('total', spread['required'])
        self.assertEqual(spread['properties']['total']['maximum'], 66)
        item = schema['$defs']['Slot0Item']
        self.assertNotIn('enum', item)
        self.assertRegex(item['allowed_values'], r'^one of \d+ values')

    def test_gepa_reflection_sees_the_required_total(self):
        spread = dict(HP=2, Atk=32, Def=0, SpA=0, SpD=0, Spe=32)
        refused = {'fills': json.dumps({'slot0_item': 'Not An Item', 'slot2_evs': dict(spread, total=66)})}
        legal = {'fills': json.dumps({'slot0_item': 'Light Ball', 'slot2_evs': dict(spread, total=66)})}
        task = DummyLM([refused] * 4 + [legal] * 40)
        reflection = DummyLM([{'new_instruction': 'Keep every Stat Points total.'}] * 10, adapter=dspy.JSONAdapter())
        with dspy.context(lm=task, adapter=O.lenient(S.adapter_for(self.m['structure']))):
            gepa = dspy.GEPA(metric=self.metric, max_metric_calls=12, reflection_lm=reflection,
                             instruction_proposer=O.ShapeAwareProposer(), add_format_failure_as_feedback=True,
                             num_threads=1, seed=1)
            train = [self.example]
            optimized = gepa.compile(O.Completion(self.m), trainset=train, valset=train)
        prompt = json.dumps(reflection.history[0]['messages'])
        self.assertIn('Required answer shape', prompt)
        self.assertIn('slot2_evs', prompt)
        self.assertIn('must keep its `total` field', prompt)
        self.assertIn('\\"total\\"', prompt)  # the StatPoints schema lists total
        self.assertEqual(O.export(optimized, self.m, {})['instructions'], 'Keep every Stat Points total.')

    def test_optimize_uses_the_shape_aware_proposer_with_legal_choices(self):
        class Stop(Exception):
            pass
        seen = {}

        def fake_gepa(**kwargs):
            seen.update(kwargs)
            raise Stop

        with tempfile.TemporaryDirectory() as tmp:
            results.save(Path(tmp) / 'manifest.json', self.m)
            args = SimpleNamespace(output=Path(tmp), optimizer='gepa', battles=False, alpha=.5, budget='light',
                                   threads=1, seed=1, reflection_model=None, reflection_effort='high', num_retries=3)
            h = Mock(Validator=Mock(return_value=Mock()))
            with unittest.mock.patch.dict(sys.modules, {'hps_generate': h}), \
                 unittest.mock.patch.object(O, 'lm_for', return_value=DummyLM([])), \
                 unittest.mock.patch.object(dspy, 'GEPA', side_effect=fake_gepa):
                with self.assertRaises(Stop):
                    O.optimize(args)
        self.assertIsInstance(seen['instruction_proposer'], O.ShapeAwareProposer)


if __name__ == '__main__':
    unittest.main()
