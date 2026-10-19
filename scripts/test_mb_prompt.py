import copy
import json
import unittest
import frozen as F
import llm_baseline as B


class GroundedPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=F.load('ling-mb-prompt-comparison/manifest.json')

    def test_no_hidden_value_leaks_into_prompt(self):
        for task in ['pokemon','move','mixed']:
            trial=next(t for t in self.m['trials'] if t['task']==task)
            altered=copy.deepcopy(self.m)
            for path in trial['paths']:B.set_path(altered['starts'][trial['start']],path,'SECRET-HIDDEN-VALUE')
            self.assertEqual(B.prompt(self.m,trial),B.prompt(altered,trial))

    def test_old_request_hash_is_unchanged(self):
        old=F.load('llm-baseline-meta-starters/manifest.json')
        for trial in self.m['trials'][::21]:
            payload=dict(model=B.MODEL,messages=B.prompt(old,trial),temperature=old['temperature'],max_tokens=old['max_tokens'],
                         reasoning=old['reasoning'],provider={'order':['novita'],'allow_fallbacks':False})
            record=F.load('llm-baseline-meta-starters/completions/'+trial['id']+'.json')
            self.assertEqual(B.digest(payload),record['request_hash'])
            self.assertNotEqual(B.prompt(old,trial),B.prompt(self.m,trial))

    def test_matched_tasks_preserved_and_starter_excluded(self):
        old=F.load('llm-baseline-meta-starters/manifest.json')
        source={t['id']:t for t in old['trials']}
        self.assertEqual(len(self.m['trials']),147)
        self.assertEqual(len({t['start'] for t in self.m['trials']}),7)
        for t in self.m['trials']:
            self.assertEqual(t,source[t['id']])
            request=json.loads(B.prompt(self.m,t)[1]['content'])
            excluded=next(o['paste'] for o in self.m['opponents'] if o['team_id']==self.m['exclude_starter_team_ids'][t['start']])
            self.assertNotIn(excluded,request['opponents'])


if __name__=='__main__':unittest.main()
