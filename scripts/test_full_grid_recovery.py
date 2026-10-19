import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import llm_baseline as B


class RecoveryTests(unittest.TestCase):
    def test_atomic_save_leaves_readable_final_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'completion.json'
            B.save(p,{'response':'original'})
            B.save(p,{'response':'replacement'})
            self.assertEqual(json.loads(p.read_text()),{'response':'replacement'})
            self.assertFalse(p.with_suffix('.json.tmp').exists())

    def test_saved_response_is_validated_without_network_or_key(self):
        team=[dict(species='Pikachu',item='Light Ball',ability='Static',nature='Jolly',
                   moves=['Protect','Thunderbolt','Thunder','Volt Switch'],evs={s:0 for s in B.STATS}) for _ in range(6)]
        trial=dict(id='s0-item-1',start=0,paths=[[0,'item']])
        manifest=dict(model=B.MODEL,format=B.FORMAT,starts=[team],opponents=[],trials=[trial],temperature=.7,max_tokens=4096,reasoning={'enabled':False})
        payload=dict(model=B.MODEL,messages=B.prompt(manifest,trial),temperature=.7,max_tokens=4096,reasoning={'enabled':False},provider={'order':['novita'],'allow_fallbacks':False})
        response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'fills':[{'path':[0,'item'],'value':'Focus Sash'}]})}}]}
        class Validator:
            def __call__(self,text): return None
            def close(self): pass
        h=SimpleNamespace(Validator=Validator,slot_to_text=lambda s:s['species'])
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)
            B.save(output/'manifest.json',manifest)
            p=output/'completions/s0-item-1.json'
            B.save(p,dict(request_hash=B.digest(payload),response=response,valid=False))
            with patch.object(B,'runtime',return_value=(h,None)),patch.dict(B.os.environ,{},clear=True),patch.object(B.urllib.request,'urlopen') as network:
                B.complete(SimpleNamespace(output=output))
                B.complete(SimpleNamespace(output=output))
                network.assert_not_called()
            record=json.loads(p.read_text())
            self.assertTrue(record['valid'])
            self.assertEqual(record['response'],response)


if __name__=='__main__': unittest.main()
