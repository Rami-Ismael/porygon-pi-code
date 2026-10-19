import copy
import json
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
import frozen as F
import retrain_regmb as R
from encode import Vocab


class RegulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v=Vocab.from_regulation(R.MANIFEST)

    def test_all_natures_and_previously_missing_abilities(self):
        self.assertEqual(len([n for n in self.v.regulation['values']['nature'] if n]),25)
        for a in ['fluffy','corrosion']: self.assertIn(a,self.v.stoi['ability'])
        for n in ['rash','hardy','docile','quirky']:self.assertIn(n,self.v.stoi['nature'])

    def test_mega_ability_requires_compatible_stone(self):
        choices=self.v.regulation['slots']['metagross']['abilityItems']['toughclaws']
        self.assertIn('metagrossite',choices)
        self.assertNotIn('',choices)

    def test_identity_migration_with_shuffled_ids(self):
        old=copy.copy(self.v);new=copy.copy(self.v)
        old.itos={k:['[MASK]','first','second'] for k in R.L.KEYS}
        old.sizes={k:3 for k in R.L.KEYS}
        new.itos={k:['[MASK]','second','new','first'] for k in R.L.KEYS}
        new.stoi={k:{v:i for i,v in enumerate(vs)} for k,vs in new.itos.items()}
        new.sizes={k:4 for k in R.L.KEYS}
        a=R.L.Net(old,d=16,nhead=2,nlayer=1);b=R.L.Net(new,d=16,nhead=2,nlayer=1)
        R.migrate(b,{'sd':a.state_dict()},old,new)
        for k in R.L.KEYS:
            for oi,ni in [(0,0),(1,3),(2,1)]:
                torch.testing.assert_close(a.emb[k].weight[oi],b.emb[k].weight[ni])
                torch.testing.assert_close(a.head[k].weight[oi],b.head[k].weight[ni])
                torch.testing.assert_close(a.head[k].bias[oi],b.head[k].bias[ni])
        torch.testing.assert_close(a.pos,b.pos)

    def test_remap_keeps_padding_last(self):
        old=copy.copy(self.v);old.itos=copy.deepcopy(self.v.itos)
        x=np.ones((1,48),dtype=np.int64)
        for s in range(6):x[0,s*8+3:s*8+7]=[4,3,2,1]
        y,valid=R.remap(x,old,self.v)
        self.assertTrue(valid[0]);self.assertEqual(y[0,3:7].tolist(),[2,3,4,1])

    def test_all_frozen_pilot_contexts_representable(self):
        import diffusion_baseline as B
        m=F.load('llm-baseline-instant-pilot/manifest.json')
        for t in m['trials']:
            self.assertFalse(B.encode_context(m['starts'][t['start']],t['paths'],self.v)[2],t['id'])

    def test_unknown_value_is_not_silently_masked(self):
        import diffusion_baseline as B
        m=F.load('llm-baseline-instant-pilot/manifest.json')
        team=copy.deepcopy(m['starts'][0]);team[0]['nature']='Not a nature'
        with self.assertRaisesRegex(ValueError,'outside Regulation M-B'):
            self.v.encode(team)

    def test_joint_table_enforces_mega_stone(self):
        previous=R.L.DEV
        try:
            R.L.DEV='cpu';table=R.L.Tables(self.v,[])
        finally:R.L.DEV=previous
        x=torch.zeros(1,48,dtype=torch.long)
        x[0,0]=self.v.stoi['species']['metagross']
        ab=self.v.stoi['ability']['toughclaws'];item=self.v.stoi['item']['metagrossite']
        x[0,1]=ab;x[0,2]=self.v.stoi['item']['']
        self.assertFalse(table.allowed(x)['ability'][0,0,ab])
        self.assertTrue(table.allowed(x)['item'][0,0,item])
        x[0,2]=item
        self.assertTrue(table.allowed(x)['ability'][0,0,ab])

    def test_checkpoint_rejects_same_size_different_token_order(self):
        swapped=copy.deepcopy(self.v.itos)
        swapped['ability'][1],swapped['ability'][2]=swapped['ability'][2],swapped['ability'][1]
        with patch.object(R.L.torch,'load',return_value={'vocabulary':swapped}):
            with self.assertRaisesRegex(ValueError,'token identities'):
                R.L.load_net('does-not-need-a-real-file',self.v)


if __name__=='__main__':unittest.main()
