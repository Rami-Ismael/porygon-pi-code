import asyncio
from collections import Counter
from email.utils import formatdate
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch

import repo_paths  # puts src/ first on sys.path
import llm_completion
import results
import openrouter_generation as G


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.jobs = [G.Job(self.root, {}, {'id': str(i)}) for i in range(8)]
        self.payload = patch.object(llm_completion, 'completion_payload', side_effect=lambda m, t: {'id': t['id']})
        self.payload.start()
        self.addCleanup(self.payload.stop)

    def validate(self, record, job):
        # The response must survive even if validation fails or is interrupted.
        self.assertIn('response', json.loads(job.path.read_text()))
        record['valid'] = True

    async def run_jobs(self, transport, jobs=None, validate=None, **kwargs):
        return await G.generate(self.jobs if jobs is None else jobs, self.root,
            G.Options(min_interval=0, **kwargs), transport=transport,
            validate=validate or self.validate, key='test-only')

    @staticmethod
    def ok():
        return G.Reply(200, {}, b'{"choices": []}')

    async def test_overlapping_bounded_dispatch_and_resume(self):
        lock = threading.Lock()
        active = maximum = 0
        calls = []
        def transport(payload, key, timeout):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                calls.append(payload)
            time.sleep(.025)
            with lock:
                active -= 1
            return self.ok()
        result = await self.run_jobs(transport, healthy_responses=2)
        self.assertEqual(result['completed'], 8)
        self.assertGreaterEqual(maximum, 2)
        self.assertLessEqual(maximum, 4)
        self.assertCountEqual(calls, [{'id': str(i)} for i in range(8)])
        before = [job.path.read_bytes() for job in self.jobs]
        result = await self.run_jobs(lambda *a: self.fail('re-sent saved response'))
        self.assertEqual(result['reused'], 8)
        self.assertEqual(before, [job.path.read_bytes() for job in self.jobs])

    async def test_pending_and_hash_mismatch_preflight_sends_nothing(self):
        last = self.jobs[-1].path
        results.save(last.with_suffix('.pending'), {})
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            await self.run_jobs(lambda *a: self.fail('sent before full preflight'))
        last.with_suffix('.pending').unlink()
        results.save(last, dict(request_hash='wrong', valid=True))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            await self.run_jobs(lambda *a: self.fail('sent before full preflight'))

    async def test_429_shared_cooldown_and_retry_bound(self):
        calls = []
        counts = Counter()
        def transport(payload, *args):
            ident = payload['id']
            calls.append((ident, time.monotonic()))
            counts[ident] += 1
            if ident == '0' and counts[ident] == 1:
                return G.Reply(429, {'Retry-After': '.08'}, b'limited')
            time.sleep(.005)
            return self.ok()
        result = await self.run_jobs(transport, jobs=self.jobs[:4])
        self.assertEqual(result['rate_limits'], 1)
        retry = [stamp for ident, stamp in calls if ident == '0']
        self.assertGreaterEqual(retry[1]-retry[0], .075)
        self.assertEqual(counts, {'0': 2, '1': 1, '2': 1, '3': 1})
        # Only a confirmed rejection is retried; exhaust without a pending marker.
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            await self.run_jobs(lambda *a: G.Reply(429, {'Retry-After': '0'}, b''),
                                jobs=self.jobs[4:5], max_attempts=2)
        self.assertFalse(self.jobs[4].path.with_suffix('.pending').exists())

    async def test_ambiguous_failure_drains_other_response_without_retry(self):
        barrier = threading.Barrier(2)
        calls = []
        def transport(payload, *args):
            calls.append(payload['id'])
            barrier.wait(timeout=2)
            if payload['id'] == '0':
                raise TimeoutError('unknown upstream result')
            time.sleep(.03)
            return self.ok()
        with self.assertRaisesRegex(RuntimeError, 'unknown upstream'):
            await self.run_jobs(transport)
        self.assertCountEqual(calls, ['0', '1'])
        self.assertTrue(self.jobs[0].path.with_suffix('.pending').exists())
        self.assertTrue(json.loads(self.jobs[1].path.read_text())['valid'])

    async def test_validation_failure_recovers_raw_without_network(self):
        def broken(record, job):
            self.assertTrue(job.path.exists())
            raise RuntimeError('validator unavailable')
        with self.assertRaisesRegex(RuntimeError, 'validator unavailable'):
            await self.run_jobs(lambda *a: self.ok(), jobs=self.jobs[:2], validate=broken)
        await self.run_jobs(lambda *a: self.fail('raw response re-sent'), jobs=self.jobs[:2])
        self.assertTrue(all(json.loads(j.path.read_text())['valid'] for j in self.jobs[:2]))

    async def test_cancel_drains_inflight(self):
        started = threading.Event()
        release = threading.Event()
        calls = []
        def transport(payload, *args):
            calls.append(payload['id'])
            if len(calls) == 2:
                started.set()
            release.wait(timeout=2)
            return self.ok()
        task = asyncio.create_task(self.run_jobs(transport))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        task.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(j.path.exists() for j in self.jobs[:2]))
        self.assertEqual(json.loads((self.root/'generation-status.json').read_text())['stage'], 'stopped')

    async def test_pending_write_failure_drains_existing_request(self):
        save = results.save
        def failing_save(path, record):
            if path == self.jobs[1].path.with_suffix('.pending'):
                raise OSError('write failed')
            save(path, record)
        with patch.object(results, 'save', side_effect=failing_save):
            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                await self.run_jobs(lambda *a: self.ok())
        self.assertTrue(self.jobs[0].path.exists())
        self.assertFalse(self.jobs[2].path.exists())

    async def test_http_unknown_outcome_and_definite_rejection(self):
        for index, status in enumerate([500, 408, 401]):
            with self.assertRaisesRegex(RuntimeError, str(status)):
                await self.run_jobs(lambda *a: G.Reply(status, {}, b'error'), jobs=self.jobs[index:index+1])
            self.assertEqual(self.jobs[index].path.with_suffix('.pending').exists(), status != 401)

    async def test_malformed_response_saved_once(self):
        await self.run_jobs(lambda *a: G.Reply(200, {}, b'not json'), jobs=self.jobs[:1])
        record = json.loads(self.jobs[0].path.read_text())
        self.assertEqual(record['raw_response'], 'not json')
        await self.run_jobs(lambda *a: self.fail('re-sent malformed response'), jobs=self.jobs[:1])

    async def test_global_pacing_and_stop_file(self):
        starts = []
        def transport(*args):
            starts.append(time.monotonic())
            return self.ok()
        await G.generate(self.jobs[:3], self.root, G.Options(min_interval=.03),
                         transport=transport, validate=self.validate, key='test-only')
        self.assertTrue(all(b-a >= .025 for a, b in zip(starts, starts[1:])))
        (self.root/'STOP_GENERATION').touch()
        result = await self.run_jobs(lambda *a: self.fail('sent after stop'))
        self.assertEqual(result['stage'], 'stopped')

    async def test_status_write_failure_drains_existing_request(self):
        save = results.save
        def failing_save(path, record):
            if path.name == 'generation-status.json' and record['attempts']:
                raise OSError('status disk failed')
            save(path, record)
        with patch.object(results, 'save', side_effect=failing_save):
            with self.assertRaisesRegex(RuntimeError, 'status disk failed'):
                await self.run_jobs(lambda *a: self.ok())
        self.assertTrue(self.jobs[0].path.exists())
        self.assertFalse(self.jobs[1].path.exists())

    def test_controller_growth_and_retry_after(self):
        gate = G.Controller(G.Options(healthy_responses=2))
        gate.success(0, 0)
        gate.success(0, 0)
        self.assertEqual(gate.limit, 3)
        gate.rate_limited({'retry-after': '2'}, 10)
        self.assertEqual(gate.limit, 1)
        for _ in range(20):
            gate.success(0, 20)
        self.assertEqual(gate.limit, 1)
        self.assertEqual(G.retry_after({'Retry-After': formatdate(100, usegmt=True)}, now=90), 10)
        self.assertIsNone(G.retry_after({'Retry-After': 'nan'}))
        self.assertIsNone(G.retry_after({'Retry-After': 'bad'}))
        with self.assertRaises(ValueError):
            G.Options(initial=5, maximum=4)

    async def test_real_http_adapter_through_scheduler(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append((body, self.headers.get('Authorization')))
                limited = len(captured) == 1
                self.send_response(429 if limited else 200)
                self.send_header('Retry-After', '0')
                self.end_headers()
                self.wfile.write(b'{}' if limited else b'{"choices": []}')

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        request = urllib.request.Request
        def local_request(url, **kwargs):
            self.assertEqual(url, 'https://openrouter.ai/api/v1/chat/completions')
            return request(f'http://127.0.0.1:{server.server_port}', **kwargs)
        try:
            with patch.object(G.urllib.request, 'Request', side_effect=local_request):
                result = await self.run_jobs(G.send, jobs=self.jobs[:1])
            self.assertEqual(result['attempts'], 2)
            self.assertEqual(captured, [({'id': '0'}, 'Bearer test-only')]*2)
            self.assertTrue(json.loads(self.jobs[0].path.read_text())['valid'])
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
