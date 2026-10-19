import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from encode import Vocab,team_fields
from regmb_training_policy import sample_moves,encode_training_team,training_regulation,mega_stone_users


class TrainingPolicyTests(unittest.TestCase):
    def test_empty_padding_is_known_and_distinct_from_mask(self):
        slot=dict(species='Kangaskhan',ability='Scrappy',item='',nature='Jolly',moves=['Fake Out','Last Resort'])
        team=[dict(slot) for _ in range(6)];v=Vocab([team])
        explicit=[dict(slot,moves=['Fake Out','','Last Resort','']) for _ in range(6)]
        self.assertNotEqual(v.stoi['move'][''],v.stoi['move']['[MASK]'])
        np.testing.assert_array_equal(encode_training_team(v,team),encode_training_team(v,explicit))
        with self.assertRaises(ValueError):encode_training_team(v,[dict(slot,moves=['','','',''])])

    def test_variable_counts_and_target_preservation(self):
        rng=np.random.default_rng(1);counts=set()
        for _ in range(1000):
            moves=sample_moves(['a','b','c','d','e'],rng,target='a')
            self.assertIn('a',moves);self.assertEqual(len(moves),len(set(moves)));counts.add(len(moves))
        self.assertEqual(counts,{1,2,3,4})

    def test_decoder_allows_repeated_empty_but_not_repeated_real_moves(self):
        import torch
        from diffusion_baseline_v3 import restrict_move_duplicates
        x=torch.zeros((1,48),dtype=torch.long);x[0,3:7]=torch.tensor([2,1,1,0])
        allowed=torch.ones((1,5),dtype=torch.bool)
        restrict_move_duplicates(allowed,x,6,empty=1)
        self.assertTrue(allowed[0,1]);self.assertFalse(allowed[0,2]);self.assertTrue(allowed[0,3])

    def test_floettite_forme_mapping_is_exact(self):
        users=mega_stone_users(Path.home()/'.local/share/vgc-pilot-runtime/validator-913da36')
        self.assertEqual(set(users['floettite']),{'floetteeternal','floettemega'})
        reg={'slots':{sp:dict(abilities=['a'],items=['floettite','leftovers'],abilityItems={'a':['floettite','leftovers']}) for sp in ['kingambit','floette','floetteeternal','floettemega']}}
        out=training_regulation(reg,users)
        for sp in ['kingambit','floette']:self.assertNotIn('floettite',out['slots'][sp]['items'])
        for sp in ['floetteeternal','floettemega']:self.assertIn('floettite',out['slots'][sp]['items'])
        self.assertIn('floettite',reg['slots']['kingambit']['items'])


if __name__=='__main__':unittest.main()
