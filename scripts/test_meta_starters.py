import json
import random
import unittest
import llm_baseline as B


class MetaStarterTests(unittest.TestCase):
    def setUp(self):
        self.m=dict(seed=20261010,opponents=[dict(team_id=i,paste=f'team-{i}') for i in range(49)],
                    starts=[[]]*49,exclude_starter_team_ids=list(range(49)))

    def test_starter_never_occurs_in_prompt_or_schedule(self):
        for i in range(49):
            panel=B.schedule_for(self.m,i)
            self.assertEqual(len(panel),50)
            self.assertEqual(len(set(panel)),48)
            self.assertNotIn(i,panel)
            prompt=json.loads(B.prompt(self.m,dict(start=i,paths=[]))[1]['content'])
            self.assertNotIn(f'team-{i}',prompt['opponents'])
            self.assertEqual(len(prompt['opponents']),48)

    def test_original_panel_and_all_method_labels_preserved(self):
        m=dict(self.m);m.pop('exclude_starter_team_ids')
        expected=list(range(49));random.Random(m['seed']).shuffle(expected)
        self.assertEqual(B.schedule_for(m,0),(expected*2)[:50])
        for label in ['start-48','random-s48-mixed-48','llm-s48-mixed-48','diffusion-s48-mixed-48']:
            self.assertEqual(B.candidate_start(label),48)

    def test_fixed_empty_moves_are_preserved_but_masked_moves_need_values(self):
        team=[dict(species='Kangaskhan',item='Kangaskhanite',ability='Scrappy',nature='Jolly',
                   moves=['Fake Out','Last Resort','',''],evs={s:0 for s in B.STATS}) for _ in range(6)]
        output=B.apply_completion(team,[[0,'nature']],{'fills':[{'path':[0,'nature'],'value':'Adamant'}]})
        self.assertEqual(output[0]['moves'],team[0]['moves'])
        with self.assertRaises(ValueError):
            B.apply_completion(team,[[0,'moves',2]],{'fills':[{'path':[0,'moves',2],'value':''}]})


if __name__=='__main__': unittest.main()
