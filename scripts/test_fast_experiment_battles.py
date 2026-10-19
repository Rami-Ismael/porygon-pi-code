import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fast_experiment_battles import Backend


class Runner:
    def __init__(self, folder, **kwargs):
        self.folder=Path(folder); self.folder.mkdir()
        self.calls=[]; self.closed=False; self.partial=False
        assert kwargs==dict(workers=6,concurrency=24)
    def close(self): self.closed=True
    def __call__(self, con, jobs, seed=None):
        self.calls.append((jobs,seed))
        rows=[(j['row'],c,1,0,0) for j in jobs for c in j['schedule']]
        return (rows[:-1] if self.partial else rows),0,.01
    def policy_id(self, con, mdb): return 99


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.backend=Backend(self.temp.name,Runner);self.addCleanup(self.backend.close)

    def test_aggregates_repeated_pairs_without_changing_budget(self):
        raw,missing,_=self.backend.call(None,[dict(row=1,schedule=[2,2,3])],6060)
        self.assertEqual(raw,[(1,2,2,0,0),(1,3,1,0,0)])
        self.assertEqual(missing,0)
        self.assertEqual(self.backend.runner.calls[0][1],6060)

    def test_failure_is_recorded_and_blocks_retry(self):
        self.backend.runner.partial=True
        with self.assertRaises(RuntimeError):self.backend.call(None,[dict(row=1,schedule=[2])],6060)
        self.assertTrue((self.backend.runner.folder/'failed.json').exists())
        with self.assertRaises(RuntimeError):Backend(self.temp.name,Runner)

    def test_fast_cache_identity_only_for_its_own_results(self):
        con=sqlite3.connect(':memory:');self.addCleanup(con.close)
        raw=[(1,2,1,0,0)]
        from fast_experiment_battles import fingerprint
        self.backend.pending=(id(con),fingerprint(raw))
        seen=[]
        self.backend.original_record=lambda con,pid,results,**kw:seen.append(pid)
        self.backend.record(con,1,[(3,4,1,0,0)])
        self.backend.record(con,1,raw)
        self.assertEqual(seen,[1,99])

    def test_final_schedule_and_exact_pastes_preserved(self):
        root=Path(self.temp.name)
        paths=[]
        for i in range(3):
            p=root/f'{i}.txt';p.write_text(f'exact paste {i}');paths.append(str(p))
        import matchup_db
        with patch.object(matchup_db,'species_keys',return_value=('same','same')), patch('heldout.assert_clear_files'):
            result=self.backend.score([(paths[0],paths[1:])],battles=3,opp_schedule=[paths[1],paths[2],paths[1]],seed=6060)
        self.assertEqual(result[paths[0]]['battles'],3)
        self.assertEqual(result[paths[0]]['wins'],3)
        jobs,_=self.backend.runner.calls[0]
        self.assertEqual([x for j in jobs for x in j['schedule']],[2,3,2])

    def test_search_heldout_guard(self):
        with patch('heldout.team_ids',return_value={2}):
            with self.assertRaises(Exception):self.backend.run_jobs(None,[dict(row=1,schedule=[2])],seed=6060)
        self.assertEqual(self.backend.runner.calls,[])


if __name__=='__main__':unittest.main()
