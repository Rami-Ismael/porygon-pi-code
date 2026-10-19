"""Exact finite-space checks for the discrete DiBO implementation."""
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dibo_core as D


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[0., 1.2, -.4], [0., -.2, .5]]))

    def forward(self, x, t, c):
        values = self.logits[c].clone()
        values[0] = -float("inf")
        return F.log_softmax(values, -1).expand(len(x), -1)


class DiBOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.prior = D.PathSampler(Tiny().eval(), [0, 1])
        self.space = torch.tensor(list(itertools.product([1, 2], repeat=2)))

    def test_density_normalizes_over_terminal_grids(self):
        p = self.prior.log_prob(self.space).detach().exp()
        self.assertAlmostEqual(float(p.sum()), 1., places=6)
        expected = torch.softmax(torch.tensor([1.2, -.4]), 0)[0] * torch.softmax(torch.tensor([-.2, .5]), 0)[0]
        self.assertAlmostEqual(float(p[0]), float(expected), places=6)
        samples = self.prior.sample(10000)
        freq = ((samples == self.space[0]).all(1)).float().mean()
        self.assertAlmostEqual(float(freq), float(expected), delta=.025)

    def test_rtb_learns_prior_times_reward_not_reward_alone(self):
        posterior = D.posterior_copy(self.prior)
        tilt = lambda x: 1.8 * (x[:, 0] == 2).float() + .4 * (x[:, 1] == 1).float()
        expected = torch.softmax(self.prior.log_prob(self.space).detach() + tilt(self.space), 0)
        before = self.prior.model.logits.detach().clone()
        history, _ = D.train_rtb(self.prior, posterior, tilt, steps=350, batch=64, lr=.04, replay=self.space)
        got = posterior.log_prob(self.space).detach().exp()
        self.assertLess(float((got - expected).abs().max()), .015)
        self.assertLess(history[-1]["loss"], .01)
        self.assertTrue(torch.equal(before, self.prior.model.logits))

    def test_zero_tilt_returns_the_prior(self):
        posterior = D.posterior_copy(self.prior)
        history, _ = D.train_rtb(self.prior, posterior, lambda x: torch.zeros(len(x)), 2, 8, .01)
        self.assertEqual(history[-1]["loss"], 0)
        self.assertTrue(torch.allclose(posterior.log_prob(self.space), self.prior.log_prob(self.space)))

    def test_low_scoring_observations_keep_nonzero_weight(self):
        weights = D.score_weights(torch.tensor([.1, .9]))
        self.assertGreater(float(weights[0]), 0)
        self.assertAlmostEqual(float(weights.sum()), 1., places=6)
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_filter_uses_density_and_deduplicates_seen_teams(self):
        records = [dict(key="A", log_target=2, mu=.1), dict(key="B", log_target=1, mu=.9),
                   dict(key="A", log_target=0, mu=.2), dict(key="seen", log_target=3, mu=1)]
        self.assertEqual([r["key"] for r in D.unique_top(records, 2, {"seen"})], ["A", "B"])

    def test_uncertainty_is_ensemble_disagreement(self):
        class Constant(nn.Module):
            def __init__(self, value):
                super().__init__(); self.value = value
            def forward(self, x):
                return torch.full((len(x),), self.value)
        ensemble = D.PredictorEnsemble([Constant(.2), Constant(.8)])
        mu, sigma = ensemble(self.space)
        self.assertTrue(torch.allclose(mu, torch.full((4,), .5)))
        self.assertTrue(torch.allclose(sigma, torch.full((4,), .3)))

    def test_malformed_paths_fail_before_training(self):
        with self.assertRaises(ValueError):
            D.PathSampler(Tiny(), [0, 0])
        with self.assertRaises(ValueError):
            self.prior.log_prob(torch.zeros(1, 2, dtype=torch.long))
        with self.assertRaises(ValueError):
            D.score_weights(torch.tensor([float("nan")]))
        with self.assertRaises(ValueError):
            D.unique_top([], 0)

    def test_local_refinement_cannot_return_an_excluded_high_score(self):
        import dibo_v3 as DB
        runtime = DB.Runtime.__new__(DB.Runtime)
        runtime.cfg = dict(local_steps=1, local_neighbors=2, sample_batch=8)
        def score(terminal, prior, ensemble):
            return [dict(x=row.tolist(), key=''.join(map(str, row.tolist())),
                         log_target=10. if int(row[0]) == 2 else 0.) for row in terminal.cpu()]
        runtime.records = score
        records = [dict(x=[1, 1], key='11', log_target=0.)]
        with patch.object(DB.L, 'COLS', 2), patch.object(DB.L, 'DEV', 'cpu'), \
             patch('numpy.random.randint', return_value=np.zeros(1, dtype=int)):
            result = runtime.refine(records, self.prior, None, seen={'21'})
        self.assertEqual(result[0]['key'], '11')


if __name__ == "__main__":
    unittest.main(verbosity=2)
