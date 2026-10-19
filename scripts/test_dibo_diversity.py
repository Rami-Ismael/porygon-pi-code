"""Battle-free behavioral regression tests for the DiBO diversity adaptations."""
import copy
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import dibo_diversity as V
import dibo_diversity_runtime as R
import dibo_diversity_experiment as E
from test_dibo import Tiny


def row(key, species, y=.5):
    return dict(key=key, paste='\n\n'.join(f'{s} @ Leftovers\nAbility: Pressure\n- Protect' for s in species),
                y=y, mu=y, log_target=y, src='initial', x=[1, 1], team_id=key)


class DiversityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.a = row('a', ['a','b','c','d','e','f'], .8)
        self.b = row('b', ['a','b','c','d','e','f'], .79)
        self.c = row('c', ['g','h','i','j','k','l'], .7)

    def test_every_arm_has_finite_normalized_training_weights(self):
        for arm in V.ARMS:
            weights, info = V.prior_weights([self.a,self.b,self.c],arm,0,10)
            self.assertTrue(np.isfinite(weights).all(), arm)
            self.assertAlmostEqual(float(weights.sum()),1, msg=arm)
            self.assertGreater(info['ess'],0)

    def test_niche_assigns_mass_only_to_best_build_per_roster(self):
        weights,_ = V.prior_weights([self.a,self.b,self.c],'niche',0,10)
        self.assertEqual(weights[1],0)
        self.assertGreater(weights[0], weights[2])
        self.assertGreater(weights[2],0)

    def test_mmr_changes_buffer_and_zero_penalty_is_top_score(self):
        rows = [self.a,self.b,self.c]
        self.assertEqual([r['key'] for r in V.mmr(rows,2,0)],['a','b'])
        self.assertEqual([r['key'] for r in V.training_rows(rows,'mmr30',2,0)],['a','c'])

    def test_taboo_blocks_previous_and_third_same_composition(self):
        gate = V.Taboo([V.composition(self.c)])
        self.assertFalse(gate.accept(self.c))
        self.assertTrue(gate.accept(self.a)); self.assertTrue(gate.accept(self.b))
        self.assertFalse(gate.accept(dict(self.a,key='d')))

    def test_peel_uses_only_completed_preceding_epochs(self):
        rounds = {'0':{'evaluated':[self.a]},'1':{'evaluated':[self.c]},'2':{'evaluated':[self.c]}}
        self.assertEqual(V.peel_bans(rounds,0),[])
        self.assertEqual(V.peel_bans(rounds,1),['a','b','c'])
        self.assertEqual(V.peel_bans(rounds,2),['a','b','c'])
        self.assertEqual(V.peel_bans(rounds,3),['a','b','c','g','h','i'])

    def test_islands_private_history_migration_is_repeatable(self):
        seeds = V.island_seeds([self.a,self.b,self.c],2)
        self.assertEqual([r['key'] for r in seeds],['a','c'])
        a = dict(self.a,key='newa',src='round0',island=0)
        c = dict(self.c,key='newc',src='round0',island=1)
        history={'0':{'evaluated':[a,c]}}
        private = V.island_observations([self.a,self.c],history,seeds,1)
        self.assertNotIn('newc',[r['key'] for r in private[0]])
        migrated = V.island_observations([self.a,self.c],history,seeds,3)
        self.assertIn('newc',[r['key'] for r in migrated[0]])
        self.assertEqual(migrated,V.island_observations([self.a,self.c],history,seeds,3))

    def test_dpp_picks_different_rosters_and_fills_rank_deficiency(self):
        rows = [dict(r,mu=.5) for r in [self.a,self.b,self.c]]
        selected, rank = V.select_dpp(rows,3)
        self.assertEqual(rank,2)
        self.assertNotEqual(V.composition(selected[0]),V.composition(selected[1]))
        self.assertEqual(len({r['key'] for r in selected}),3)

    def test_temperature_anneals_without_changing_labels(self):
        rows = [self.a,self.b,self.c]; before = copy.deepcopy(rows)
        early, e = V.prior_weights(rows,'boltz',0,10)
        late, l = V.prior_weights(rows,'boltz',9,10)
        self.assertAlmostEqual(e['temperature'],.3); self.assertAlmostEqual(l['temperature'],.1)
        self.assertGreater(late[0], early[0]); self.assertEqual(rows,before)

    def test_proposal_masks_preserve_support_and_fail_closed(self):
        base=Tiny().eval(); x=torch.zeros((2,2),dtype=torch.long); t=torch.ones(2)
        masked=R.ProposalMask(base,[0],banned=[1])
        lp=masked(x,t,0)
        self.assertTrue(torch.isneginf(lp[:,1]).all())
        self.assertTrue(torch.allclose(lp.exp().sum(1),torch.ones(2)))
        with self.assertRaisesRegex(ValueError,'dead prefix'):
            R.ProposalMask(base,[0],banned=[1,2])(x,t,0)
        self.assertTrue(torch.isfinite(base(x,t,0)[:,1]).all())

    def test_reward_reversal_keeps_predictor_in_win_rate_units(self):
        class Ensemble(torch.nn.Module):
            def forward(self,x):return torch.full((len(x),),.7),torch.full((len(x),),.1)
            def standardized(self,x):return torch.ones(len(x)),torch.full((len(x),),.2)
        x=torch.ones(2,2); reverse=R.RewardView(Ensemble(),'negative')
        self.assertTrue(torch.equal(reverse.standardized(x)[0],-torch.ones(2)))
        self.assertTrue(torch.allclose(reverse(x)[0],torch.full((2,),.7)))
        self.assertTrue(torch.equal(R.RewardView(Ensemble(),'no_guidance').standardized(x)[1],torch.zeros(2)))

    def test_refinement_records_cannot_escape_pins_or_peeling(self):
        rt=R.Runtime.__new__(R.Runtime); rt.banned=['g']; rt.pins={0:1}
        self.assertTrue(rt.allowed(self.a)); self.assertFalse(rt.allowed(self.c))
        self.assertFalse(rt.allowed(dict(self.a,x=[2,1])))

    def test_prior_zero_bonus_matches_v4_update(self):
        class Prior(Tiny):
            def denoising_loss(self,x,mask,t):
                return -self(torch.zeros_like(x),t,0).gather(1,x[:,:1]).squeeze(1)
        a=Prior(); b=copy.deepcopy(a)
        x=torch.tensor([[1,1],[2,2],[1,2]]); y=torch.tensor([.8,.79,.7])
        cfg=dict(prior_epochs=2,batch=2,lr=.001)
        torch.manual_seed(13)
        R.DC.train_prior(a,x,y,2,2,.001)
        weights,_=V.prior_weights([self.a,self.b,self.c],'control',0,10)
        torch.manual_seed(13); R.train_prior(b,x,weights,cfg)
        for left,right in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.allclose(left,right,atol=1e-6))

    def test_proposal_selection_enforces_full_batch_and_excludes_seen(self):
        class Posterior:
            model=Tiny(); order=(0,1); device='cpu'
            def sample(self,n):return torch.ones((n,2),dtype=torch.long)
        for arm in ('control','big','dpp','taboo','random'):
            rt=R.Runtime.__new__(R.Runtime); rt.arm=arm
            rt.cfg=dict(seed=1,propose=3,battle=2,pool_multiplier=2,sample_batch=3,
                        metric_sample=3,attempt_factor=3)
            rt.banned=[]; rt.pins={}; rt.previous=set()
            rt.V=type('Vocabulary',(),{'stoi':{'species':{}}})()
            seq=[0]
            def records(x,*args):
                result=[]
                for _ in x:
                    seq[0]+=1
                    result.append(dict(self.a,key=str(seq[0]),paste=self.a['paste'] if seq[0]%2 else self.c['paste'],_terms=[0,0]))
                return result
            rt.records=records; rt.refine=lambda records,*a:records
            out=rt.propose(None,Posterior(),None,0,{'1'})
            self.assertEqual(len(out['selected']),2,arm)
            self.assertNotIn('1',[r['key'] for r in out['selected']],arm)
            self.assertEqual(out['diversity']['raw']['n'],3)

    def test_plan_contains_every_arm_and_launches_no_runtime(self):
        import contextlib,io,json
        output=io.StringIO()
        with patch.object(R.Runtime,'__init__',side_effect=AssertionError('runtime launched')),contextlib.redirect_stdout(output):
            E.main(['plan'])
        plan=json.loads(output.getvalue())
        self.assertEqual(plan['arms'],list(V.ARMS))
        self.assertEqual(plan['search_battles_per_arm'],62720)
        self.assertEqual(plan['total_battles'],68864*len(V.ARMS))

    def test_driver_resumes_without_battles_and_locks_all_finalists_first(self):
        import contextlib, io, json, hashlib
        from types import SimpleNamespace
        import dibo_comparison_v2 as C
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); initial=[self.a,self.c]
            signature=dict(vocabulary={},meta=[[1,1],[2,1]],policy=['fixed'])
            source=root/'initial.json'
            source.write_text(json.dumps(dict(initial=initial,
                initial_hash=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest(),
                runtime_signature=signature,search_columns=signature['meta'])))
            args=SimpleNamespace(directory=root/'run',initial_protocol=source,smoke=False,arms=list(V.ARMS))
            cfg=dict(rounds=1,battle=2,final_n=1,final_battles=4)
            class Runtime:
                def __init__(self,cfg,path):
                    self.cfg=cfg;self.arm=cfg['arm'];self.signature=signature
                def close(self):pass
            calls=[]
            def propose(runtime,initial,out,g):
                return dict(selected=[dict(self.a,key=runtime.arm+'1',island=0),dict(self.c,key=runtime.arm+'2',island=1)])
            def battle(runtime,selected,directory,arm,g,proto):
                calls.append(arm)
                return [dict(r,src='round0',battles=2) for r in selected]
            def final(directory,arm,proto):
                for other in proto['arms']:
                    self.assertIn('final_selected',E.read(directory/f'{other}.json'))
                path=directory/f'{arm}.json';out=E.read(path)
                out.update(confirmation=[dict(wins=2,battles=4)],heldout=[dict(wins=3,battles=4)])
                R.DB.atomic_json(path,out)
            with patch.object(R,'Runtime',Runtime),patch.object(E,'proposal_round',propose), \
                 patch.object(C,'fresh_evaluate',battle),patch.object(C,'final_arm',final), \
                 patch.object(R.DB.HO,'is_heldout_text',return_value=False),contextlib.redirect_stdout(io.StringIO()):
                E.execute(args,cfg)
                self.assertEqual(len(calls),len(V.ARMS))
                self.assertTrue(E.read(args.directory/'summary.json')['complete'])
                E.execute(args,cfg)
                self.assertEqual(len(calls),len(V.ARMS))
                with self.assertRaisesRegex(ValueError,'frozen diversity'):
                    E.execute(args,dict(cfg,rounds=2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
