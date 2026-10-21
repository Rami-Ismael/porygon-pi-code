import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import repo_paths  # puts src/ first on sys.path
import llm_completion
import results
import showdown_validator
import sys
import team_masks


class RecoveryTests(unittest.TestCase):
    def test_atomic_save_leaves_readable_final_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'completion.json'
            results.save(p,{'response':'original'})
            results.save(p,{'response':'replacement'})
            self.assertEqual(json.loads(p.read_text()),{'response':'replacement'})
            self.assertFalse(p.with_suffix('.json.tmp').exists())

    def test_saved_response_is_validated_without_network_or_key(self):
        team=[dict(species='Pikachu',item='Light Ball',ability='Static',nature='Jolly',
                   moves=['Protect','Thunderbolt','Thunder','Volt Switch'],evs={s:0 for s in team_masks.STATS}) for _ in range(6)]
        trial=dict(id='s0-item-1',start=0,paths=[[0,'item']])
        manifest=dict(model=llm_completion.MODEL,format=showdown_validator.FORMAT,starts=[team],opponents=[],trials=[trial],temperature=.7,max_tokens=4096,reasoning={'enabled':False})
        payload=dict(model=llm_completion.MODEL,messages=llm_completion.prompt(manifest,trial),temperature=.7,max_tokens=4096,reasoning={'enabled':False},provider={'order':['novita'],'allow_fallbacks':False})
        response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'fills':[{'path':[0,'item'],'value':'Focus Sash'}]})}}]}
        class Validator:
            def __call__(self,text): return None
            def close(self): pass
        h=SimpleNamespace(Validator=Validator,slot_to_text=lambda s:s['species'])
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)
            results.save(output/'manifest.json',manifest)
            p=output/'completions/s0-item-1.json'
            results.save(p,dict(request_hash=results.digest(payload),response=response,valid=False))
            with patch.dict(sys.modules, {'hps_generate': h}),patch.dict(llm_completion.os.environ,{},clear=True),patch.object(llm_completion.urllib.request,'urlopen') as network:
                llm_completion.complete(SimpleNamespace(output=output))
                llm_completion.complete(SimpleNamespace(output=output))
                network.assert_not_called()
            record=json.loads(p.read_text())
            self.assertTrue(record['valid'])
            self.assertEqual(record['response'],response)


if __name__=='__main__': unittest.main()
