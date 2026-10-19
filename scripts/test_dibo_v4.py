"""Regression contracts for the corrected categorical DiBO adaptation."""
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import dibo_core_v4 as D
from test_dibo import Tiny


class Corrections(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)

    def test_weights_are_invariant_to_positive_score_units(self):
        y=torch.tensor([.2,.4,.5,.9])
        self.assertTrue(torch.allclose(D.score_weights(y),D.score_weights(100*y+17),atol=1e-6))

    def test_constant_scores_have_uniform_finite_weights(self):
        self.assertTrue(torch.allclose(D.score_weights(torch.ones(5)),torch.full((5,),.2)))

    def test_predictor_preserves_ownership_and_slot_permutation_invariance(self):
        sizes={k:10 for k in ['species','ability','item','move','nature']}
        fields=['species','ability','item','move','move','move','move','nature']
        net=D.Predictor(sizes,fields,width=8,hidden=32).eval()
        x=torch.randint(1,10,(16,6,8))
        changed=x.clone();changed[:,:,2]=x[:,:,2].roll(1,1)
        self.assertGreater(float((net(x.flatten(1))-net(changed.flatten(1))).abs().max().detach()),1e-4)
        self.assertTrue(torch.allclose(net(x.flatten(1)),net(x.flip(1).flatten(1)),atol=1e-6))

    def test_predictor_ignores_move_order_within_pokemon(self):
        sizes={k:10 for k in ['species','ability','item','move','nature']}
        fields=['species','ability','item','move','move','move','move','nature']
        net=D.Predictor(sizes,fields,width=8,hidden=32).eval()
        x=torch.randint(1,10,(8,6,8));changed=x.clone();changed[:,:,3:7]=x[:,:,3:7].flip(2)
        self.assertTrue(torch.allclose(net(x.flatten(1)),net(changed.flatten(1)),atol=1e-6))

    def test_exact_suffix_rescoring_matches_full_density(self):
        prior=D.PathSampler(Tiny().eval(),[0,1])
        x=torch.tensor([[1,1],[2,2]])
        old=prior.log_prob_terms(x);changed=x.clone();changed[:,1]=3-changed[:,1]
        cached=prior.log_prob_terms(changed,start=1,previous=old)
        self.assertTrue(torch.allclose(cached.sum(1),prior.log_prob(changed)))

    def test_prioritized_replay_and_scaled_rtb_reach_target(self):
        import itertools
        prior=D.PathSampler(Tiny().eval(),[0,1]);posterior=D.posterior_copy(prior)
        space=torch.tensor(list(itertools.product([1,2],repeat=2)))
        tilt=lambda x:1.8*(x[:,0]==2).float()+.4*(x[:,1]==1).float()
        p=D.replay_probabilities(torch.tensor([0.,4.]),temperature=2)
        self.assertTrue(torch.allclose(p,torch.softmax(torch.tensor([0.,2.]),0)))
        target=torch.softmax(prior.log_prob(space).detach()+tilt(space),0)
        before=prior.model.logits.detach().clone()
        history,_=D.train_rtb(prior,posterior,tilt,350,64,.04,replay=space,temperature=2,microbatch=32)
        self.assertLess(float((posterior.log_prob(space).detach().exp()-target).abs().max()),.02)
        self.assertTrue(torch.equal(before,prior.model.logits))
        self.assertEqual(history[-1]['replay_size'],4+350*64)
        self.assertTrue(any(r['off_policy'] for r in history))
        self.assertTrue(any(not r['off_policy'] for r in history))

    def test_ensemble_predictions_return_to_win_rate_units(self):
        class Constant(torch.nn.Module):
            def forward(self,x):return torch.ones(len(x))
        ensemble=D.PredictorEnsemble([Constant(),Constant()])
        ensemble.y_mean.fill_(.4);ensemble.y_scale.fill_(.1)
        mean,sigma=ensemble(torch.ones(3,48,dtype=torch.long))
        self.assertTrue(torch.allclose(mean,torch.full((3,),.5)))
        self.assertTrue(torch.equal(sigma,torch.zeros(3)))


if __name__=='__main__':unittest.main(verbosity=2)
