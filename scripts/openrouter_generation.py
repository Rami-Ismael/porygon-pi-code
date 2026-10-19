"""Bounded async dispatch with one rate controller and durable per-call records.

The caller owns the study lock. Payloads and filenames are supplied unchanged;
this module controls execution only. Network I/O runs in bounded worker threads
so the existing urllib transport needs no new dependency.
"""
import asyncio
from collections import deque
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import urllib.error
import urllib.request

import llm_baseline as B


@dataclass(frozen=True)
class Options:
    initial: int = 2
    maximum: int = 4
    min_interval: float = 1.0
    healthy_responses: int = 20
    max_attempts: int = 6
    backoff_base: float = 20.0
    backoff_cap: float = 300.0

    def __post_init__(self):
        if not 1 <= self.initial <= self.maximum <= 32:
            raise ValueError('require 1 <= initial concurrency <= maximum <= 32')
        if self.healthy_responses < 1 or self.max_attempts < 1:
            raise ValueError('healthy_responses and max_attempts must be positive')
        if any(not math.isfinite(x) or x < 0 for x in
               (self.min_interval, self.backoff_base, self.backoff_cap)):
            raise ValueError('timings must be finite and nonnegative')


@dataclass(frozen=True)
class Job:
    folder: Path
    manifest: dict
    trial: dict

    @property
    def path(self):
        return self.folder / 'completions' / (self.trial['id'] + '.json')


@dataclass(frozen=True)
class Reply:
    status: int
    headers: dict
    body: bytes


def send(payload, key, timeout):
    """Transport adapter: definite HTTP replies vs ambiguous I/O exceptions."""
    request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Authorization': f'Bearer {key}',
                                                   'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Reply(response.status, dict(response.headers), response.read())
    except urllib.error.HTTPError as exc:
        with exc:
            return Reply(exc.code, dict(exc.headers), exc.read())


def retry_after(headers, now=None):
    value = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
    if value is None:
        return None
    try:
        delay = float(value)
    except (ValueError, TypeError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now)
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


class Controller:
    def __init__(self, options):
        self.options = options
        self.limit = options.initial
        self.streak = self.throttles = 0
        self.cooldown = self.next_dispatch = 0.0
        self.epoch = 0

    def rate_limited(self, headers, now):
        self.limit = max(1, self.limit // 2)
        self.streak = 0
        self.throttles += 1
        self.epoch += 1
        delay = retry_after(headers)
        if delay is None:
            ceiling = min(self.options.backoff_cap,
                          self.options.backoff_base * 2 ** min(self.throttles - 1, 20))
            delay = random.uniform(ceiling / 2, ceiling)
        self.cooldown = max(self.cooldown, now + delay)

    def success(self, dispatch_epoch, now):
        # Replies already in flight before a 429 cannot immediately undo it.
        if dispatch_epoch != self.epoch or now < self.cooldown:
            return
        self.streak += 1
        if self.streak >= self.options.healthy_responses:
            self.limit = min(self.options.maximum, self.limit + 1)
            self.streak = self.throttles = 0


async def generate(jobs, output, options=None, *, transport=send, validate=None,
                   key=None, install_signals=False):
    """Resume jobs; return counters, or drain in-flight calls then raise.

    Preflight scans every job before dispatch. Unknown outcomes retain pending
    markers. SIGINT/SIGTERM (when enabled), cancellation, and STOP_GENERATION
    stop new dispatches while responses already in flight are persisted.
    """
    options = options or Options()
    output = Path(output)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = []
    if install_signals:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
            installed.append(sig)
    task = asyncio.create_task(_generate(list(jobs), output, options, transport, validate,
                                         key if key is not None else os.environ.get('OPENROUTER_API_KEY'), stop))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stop.set()
        # asyncio.to_thread cannot cancel an HTTP request already sent.
        # Keep its owner alive to save the eventual reply before propagating.
        try:
            await asyncio.shield(task)
        finally:
            raise
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)


async def _generate(jobs, output, options, transport, validate, key, stop):
    queue, recover, seen = deque(), [], set()
    for job in jobs:
        path = job.path
        if path in seen:
            raise ValueError(f'duplicate output path: {path}')
        seen.add(path)
        payload = B.completion_payload(job.manifest, job.trial)
        request_hash = B.digest(payload)
        if path.exists():
            record = json.loads(path.read_text())
            if record['request_hash'] != request_hash:
                raise ValueError(f'saved completion request mismatch: {path}')
            if not record['valid'] and 'error' not in record:
                recover.append((job, record))
            continue
        if path.with_suffix('.pending').exists():
            raise RuntimeError(f'uncertain previous API request: {path.with_suffix(".pending")}')
        queue.append(dict(job=job, payload=payload, request_hash=request_hash, attempts=0, started=None))
    if queue and not key:
        raise RuntimeError('OPENROUTER_API_KEY is not configured; no requests sent')
    validator = None
    if validate is None and (queue or recover):
        h, _ = B.runtime()
        validator = h.Validator()
        validate = lambda record, job: B.validate_record(record, job.manifest, job.trial, h, validator)
    gate = Controller(options)
    active = {}
    errors = []
    counters = dict(planned=len(jobs), reused=len(jobs)-len(queue), completed=0, attempts=0, rate_limits=0)
    started = time.monotonic()

    def status(stage):
        try:
            B.save(output / 'generation-status.json', dict(counters, stage=stage,
                concurrency=gate.limit, in_flight=len(active), queued=len(queue),
                cooldown_seconds=max(0, gate.cooldown-time.monotonic()),
                elapsed_seconds=time.monotonic()-started, updated_at=time.time(), errors=errors))
        except Exception as exc:
            # Even a status-disk failure must not abandon paid in-flight calls.
            error = f'status record: {type(exc).__name__}: {exc}'
            if error not in errors:
                errors.append(error)
            stop.set()

    async def accept(item, reply):
        job = item['job']
        path = job.path
        pending = path.with_suffix('.pending')
        if reply.status == 429:
            gate.rate_limited(reply.headers, time.monotonic())
            counters['rate_limits'] += 1
            B.save(path.with_suffix('.http-error.json'), dict(status=429,
                detail=reply.body.decode(errors='replace')[:2000], attempts=item['attempts']))
            pending.unlink()
            if item['attempts'] >= options.max_attempts:
                raise RuntimeError(f'rate-limit attempts exhausted: {path}')
            queue.appendleft(item)
            print(f"429: shared cooldown, concurrency={gate.limit}", flush=True)
            return
        if not 200 <= reply.status < 300:
            B.save(path.with_suffix('.http-error.json'), dict(status=reply.status,
                detail=reply.body.decode(errors='replace')[:2000]))
            # 5xx and request timeout can have an unknown upstream outcome.
            if 400 <= reply.status < 500 and reply.status != 408:
                pending.unlink()
            raise RuntimeError(f'OpenRouter HTTP {reply.status}: {path}')
        record = dict(request_hash=item['request_hash'], valid=False,
                      elapsed_seconds=time.monotonic()-item['started'], attempts=item['attempts'])
        try:
            response = json.loads(reply.body)
            if not isinstance(response, dict):
                raise ValueError('provider envelope must be an object')
            record['response'] = response
        except (ValueError, UnicodeError):
            record.update(response={}, raw_response=reply.body.decode(errors='replace'),
                          error='invalid provider JSON envelope')
        B.save(path, record)  # Paid response is durable before answer validation.
        pending.unlink()
        if 'error' not in record:
            await asyncio.to_thread(validate, record, job)
            B.save(path, record)
        counters['completed'] += 1
        gate.success(item['epoch'], time.monotonic())
        print(str(path.relative_to(output)), 'legal' if record['valid'] else 'invalid', flush=True)

    try:
        for job, record in recover:
            await asyncio.to_thread(validate, record, job)
            B.save(job.path, record)
            job.path.with_suffix('.pending').unlink(missing_ok=True)
        status('running')
        while queue or active:
            if (output / 'STOP_GENERATION').exists():
                stop.set()
            now = time.monotonic()
            while queue and not stop.is_set() and len(active) < gate.limit and now >= max(gate.cooldown, gate.next_dispatch):
                item = queue.popleft()
                item['attempts'] += 1
                item['epoch'] = gate.epoch
                item['started'] = item['started'] or now
                job = item['job']
                try:
                    B.save(job.path.with_suffix('.pending'), dict(request_hash=item['request_hash'],
                        payload=item['payload'], started_at=time.time(), attempt=item['attempts']))
                except Exception as exc:
                    queue.appendleft(item)
                    errors.append(f'pending record: {type(exc).__name__}: {exc}')
                    stop.set()
                    break
                task = asyncio.create_task(asyncio.to_thread(transport, item['payload'], key,
                    job.manifest.get('request_timeout', 600)))
                active[task] = item
                counters['attempts'] += 1
                gate.next_dispatch = now + options.min_interval
                status('running')
                now = time.monotonic()
            if stop.is_set() and not active:
                break
            delay = 1.0
            if queue and not stop.is_set() and len(active) < gate.limit:
                delay = min(delay, max(.001, max(gate.cooldown, gate.next_dispatch)-time.monotonic()))
            if not active:
                await asyncio.sleep(delay)
                continue
            done, _ = await asyncio.wait(active, timeout=delay, return_when=asyncio.FIRST_COMPLETED)
            # Process known rate limits first, before successes from the same wave.
            def priority(task):
                return 0 if not task.cancelled() and task.exception() is None and task.result().status == 429 else 1
            for task in sorted(done, key=priority):
                item = active.pop(task)
                try:
                    await accept(item, task.result())
                except Exception as exc:
                    errors.append(f'{type(exc).__name__}: {exc}')
                    stop.set()
            status('draining' if stop.is_set() else 'running')
        stage = 'failed' if errors else 'stopped' if queue else 'complete'
        status(stage)
        if errors:
            raise RuntimeError('; '.join(errors))
        return dict(counters, stage=stage)
    finally:
        if validator is not None:
            validator.close()
