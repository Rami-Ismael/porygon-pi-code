from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cem_direction_runtime import choose_elites, checked_scores, phase_seed, metrics
from cem_score_direction import Run, configuration, budget
from reverse_score_diversity import digest


class FakeEngine:
    def __init__(self, config, fail=False):
        self.c, self.np, self.fail = config, np, fail
        self.opponents = [f'opp-{i}' for i in range(50)]
        self.samples, self.fits, self.battles = [], [], []

    def load(self, path):
        self.loaded = path

    def sample(self, n, seed, label):
        self.samples.append(label)
        records = [dict(index=i, paste=f'{label}/team{i}', species=list('abcde') + [f'id{i}'],
                        valid=not (self.fail and 'reverse/iteration2' in label)) for i in range(n)]
        return dict(seed=seed, records=records, metrics=metrics(records))

    def train(self, pastes, seed, path, parent):
        self.fits.append(dict(pastes=pastes, parent=str(parent), path=str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(Path(parent).read_text() + '\n' + '\n'.join(pastes))
        return dict(checkpoint=str(path), checkpoint_sha256=digest(path), parent_checkpoint=str(parent),
                    parent_sha256=digest(parent), steps=self.c['steps'])

    def battle(self, pastes, directory, battles, schedule, seed):
        self.battles.append(pastes)
        return dict(labels=[dict(wins=i % (battles + 1), battles=battles, score=(i % (battles + 1)) / battles)
                            for i in range(len(pastes))], total_battles=len(pastes) * battles)


class CemDirectionTests(unittest.TestCase):
    def test_sign_reverses_ranking_without_mutating_scores(self):
        scores = [.2, .8, .4, .1]
        original = scores.copy()
        self.assertEqual(set(choose_elites(scores, 2, 'normal', np.random.default_rng(1))), {1, 2})
        self.assertEqual(set(choose_elites(scores, 2, 'reverse', np.random.default_rng(1))), {0, 3})
        self.assertEqual(scores, original)

    def test_ties_and_random_control_use_the_same_uniform_shuffle(self):
        scores = [.25] * 10
        results = [choose_elites(scores, 3, arm, np.random.default_rng(12)) for arm in ('normal', 'random', 'reverse')]
        self.assertEqual(results[0], results[1]); self.assertEqual(results[1], results[2])
        self.assertEqual(results[1], np.random.default_rng(12).permutation(10)[:3].tolist())

    def test_invalid_selection_fails(self):
        for scores, count, arm in (([float('nan')], 1, 'normal'), ([1], 2, 'reverse'), ([1], 1, 'unknown')):
            with self.assertRaises(ValueError):
                choose_elites(scores, count, arm, np.random.default_rng(1))

    def test_score_checks_use_exact_counts_and_reject_partial_battles(self):
        self.assertEqual(checked_scores({'a': dict(wins=1, battles=24, win_rate=.0417)}, ['a'], 24)[0]['score'], 1/24)
        for result in ({}, {'a': dict(wins=1, battles=23)}, {'a': dict(wins=25, battles=24)}):
            with self.assertRaises(ValueError):
                checked_scores(result, ['a'], 24)

    def test_budget_accounts_for_shared_first_generation(self):
        b = budget(configuration())
        self.assertEqual(b['candidate_attempts'] + b['evaluation_attempts'], 19776)
        self.assertEqual(b['ranking_battles'] + b['quality_battles'], 103680)
        self.assertEqual(b['training_steps'], 2880)

    def test_phase_seeds_keep_training_and_holdouts_separate(self):
        values = [phase_seed(2101, i, p) for i in range(6) for p in ('train', 'candidates', 'evaluation', 'elites', 'policy')]
        self.assertEqual(len(set(values)), len(values))
        self.assertEqual(phase_seed(2101, 1, 'train'), phase_seed(2101, 1, 'train'))

    def scenario(self, directory, fail=False):
        c = configuration(True)
        initial = Path(directory) / 'initial.pt'; initial.write_text('p0')
        c['checkpoint'] = str(initial)
        engine = FakeEngine(c, fail)
        data = dict(manifest=dict(inputs={str(initial): digest(initial)}), shared={}, runs={})
        run = Run(engine, Path(directory) / 'run.json', data)
        run.run()
        return engine, data, run

    def test_full_phase_flow_is_cumulative_has_holdouts_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            engine, data, run = self.scenario(directory)
            self.assertEqual(len(engine.fits), 6)
            self.assertEqual(len(engine.samples), 11)
            self.assertEqual(len(engine.battles), 7)  # four selection batches plus three quality batches
            for arm in ('normal', 'random', 'reverse'):
                r = data['runs'][f'seed9903/{arm}']
                self.assertEqual(r['iterations']['2']['training']['parent_checkpoint'], r['iterations']['1']['training']['checkpoint'])
                self.assertEqual(r['quality']['status'], 'complete')
            self.assertTrue(all('/evaluation/' not in p for fit in engine.fits for p in fit['pastes']))
            before = (len(engine.fits), len(engine.samples), len(engine.battles))
            run.run()
            self.assertEqual(before, (len(engine.fits), len(engine.samples), len(engine.battles)))

    def test_infeasible_arm_is_preserved_and_does_not_stop_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, _ = self.scenario(directory, fail=True)
            self.assertEqual(data['runs']['seed9903/reverse']['stopped']['iteration'], 2)
            self.assertNotIn('quality', data['runs']['seed9903/reverse'])
            for arm in ('normal', 'random'):
                self.assertEqual(data['runs'][f'seed9903/{arm}']['iterations']['2']['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
