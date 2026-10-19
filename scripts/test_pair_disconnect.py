"""Exercise the worker at its actual job boundary with a lost listener."""
import asyncio
import concurrent.futures
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path[:0] = ['/tmp/vgc-pilot/src', '/tmp/vgc-pilot/vgc-bench']
import pair_shard
import shard


class DisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_completion_closes_listeners(self):
        class Player:
            def __init__(self):
                self.listener = concurrent.futures.Future()
                self.ps_client = SimpleNamespace(_listening_coroutine=self.listener, stop_listening=self.stop)
            async def stop(self): self.listener.set_result(None)
            async def battle_against(self, other, n_battles): return
        a, b = Player(), Player()
        result = await asyncio.wait_for(shard.finish_or_disconnect(a, b, 2), 1)
        self.assertIsNone(result)
        self.assertTrue(a.listener.done() and b.listener.done())

    async def test_final_stage_disconnect_returns_partial_count(self):
        class Player:
            def __init__(self, **kw):
                listener = concurrent.futures.Future()
                listener.set_result(None)
                self.ps_client = SimpleNamespace(_listening_coroutine=listener, stop_listening=self.stop)
                self.battles = {'done': SimpleNamespace(finished=True, won=True, turn=5)}
                self.n_finished_battles = 1
            async def stop(self): pass
            async def battle_against(self, other, n_battles): await asyncio.Event().wait()
        with patch.object(shard, 'BatchPolicyPlayer', Player), patch.object(shard, 'RandomTeamBuilder'):
            result = await asyncio.wait_for(shard.score_one(None, 'row', ['col'], 2, 50, 8123), 1)
        self.assertEqual(result[:2], (1, 1))

    async def test_disconnect_returns_completed_results_without_waiting_forever(self):
        players = []
        class Player:
            def __init__(self, **kw):
                listener = concurrent.futures.Future()
                listener.set_result(None)  # poke-env swallows connection errors.
                self.ps_client = SimpleNamespace(_listening_coroutine=listener,
                                                 stop_listening=self.stop)
                self.battles = {'done': SimpleNamespace(finished=True, won=True,
                    teampreview_opponent_team=[SimpleNamespace(species=s) for s in 'abcdef'])}
                players.append(self)
            async def stop(self): pass
            async def battle_against(self, other, n_battles):
                await asyncio.Event().wait()
        job = dict(row=7, row_file='row', schedule=['col', 'col'],
                   key2col={'a,b,c,d,e,f': 8}, base2col={})
        with patch.object(pair_shard, 'BatchPolicyPlayer', Player), \
             patch.object(pair_shard, 'RandomTeamBuilder'), \
             patch.object(pair_shard, 'ScheduledTeamBuilder'):
            result = await asyncio.wait_for(pair_shard.run_job(None, job, 50, 8123, None), 1)
        self.assertEqual(result['finished'], 1)
        self.assertEqual(result['scheduled'], 2)
        self.assertEqual(result['by_col'], {'8': [1, 0, 0]})
        self.assertIn('connection_error', result)


if __name__ == '__main__': unittest.main()
