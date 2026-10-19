"""Paired reasoning experiment over a frozen LLM baseline manifest.

See docs/reasoning-experiment.md. Preparation and reports make no API calls.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
import copy
import fcntl
import json
from pathlib import Path
import random
from types import SimpleNamespace
import urllib.request

import repo_paths  # puts src/ first on sys.path
import battles
import llm_completion
import results


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def arm_configs(levels):
    if len(levels) < 2 or len(set(levels)) != len(levels) or levels[0] != 'off':
        raise ValueError('choose distinct arms beginning with off, plus at least one reasoning arm')
    result = {}
    for level in levels:
        if level in ('off', 'on'):
            result[level] = {'enabled': level == 'on'}
        elif level in ('minimal', 'low', 'medium', 'high', 'xhigh', 'max'):
            result[level] = {'effort': level}
        else:
            raise ValueError(f'unsupported reasoning arm: {level}')
    return result


def prepare(args):
    parent = llm_completion.load_manifest(SimpleNamespace(output=args.parent))
    model = getattr(args, 'model', None) or parent['model']
    provider = getattr(args, 'provider', None) or 'novita'
    if model not in (llm_completion.MODEL, 'inception/mercury-2.5'):
        raise ValueError('model must be explicitly supported by the experiment runner')
    arms = arm_configs(args.arms)
    if args.replicates < 1 or args.max_tokens < 1 or args.request_timeout <= 0:
        raise ValueError('replicates, max-tokens and request-timeout must be positive')
    if not parent['trials'] or not parent['opponents']:
        raise ValueError('the parent must contain trials and an opponent pool')
    ids = [t['id'] for t in parent['trials']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate trial IDs')
    if any(Path(tid).name != tid or tid in ('.', '..') for tid in ids):
        raise ValueError('trial IDs must be safe file names')
    if parent.get('battles') != 50:
        raise ValueError('the existing battle runner requires 50-battle panels')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be empty; use run/battle/report to resume')
    study = dict(schema=1, parent_manifest_sha256=results.digest(parent),
                 model=model, format=parent['format'], arms=arms,
                 replicates=args.replicates, max_tokens=args.max_tokens,
                 opponents_sha256=results.digest(parent['opponents']), seed=parent['seed'], runs=[])
    for arm, reasoning in arms.items():
        for replicate in range(args.replicates):
            m = copy.deepcopy(parent)
            m.update(model=model, reasoning=reasoning, max_tokens=args.max_tokens,
                     request_timeout=args.request_timeout, replicates=1,
                     provider={'order': [provider], 'allow_fallbacks': False, 'require_parameters': True},
                     reasoning_experiment=dict(arm=arm, replicate=replicate,
                                               parent_manifest_sha256=results.digest(parent)))
            relative = f'{arm}/r{replicate}'
            # Freeze prompt bytes as well as manifests (prompt helpers may read local data).
            request_hashes = {t['id']: results.digest(llm_completion.completion_payload(m, t)) for t in m['trials']}
            study['runs'].append(dict(path=relative, arm=arm, replicate=replicate,
                                      manifest_sha256=results.digest(m), request_hashes=request_hashes))
            results.save(args.output / relative / 'manifest.json', m)
            results.save(args.output / relative / 'example-request.json', llm_completion.completion_payload(m, m['trials'][0]))
    study['planned_calls'] = len(ids) * len(study['runs'])
    results.save(args.output / 'experiment.json', study)
    print(json.dumps(dict(output=str(args.output), planned_calls=study['planned_calls'], arms=list(arms))))


def load_runs(output):
    study = read(output / 'experiment.json')
    if study is None:
        raise ValueError('no experiment.json; run prepare first')
    runs = []
    for entry in study['runs']:
        folder = output / entry['path']
        m = llm_completion.load_manifest(SimpleNamespace(output=folder))
        if results.digest(m) != entry['manifest_sha256']:
            raise ValueError(f'manifest changed: {folder}')
        for t in m['trials']:
            if results.digest(llm_completion.completion_payload(m, t)) != entry['request_hashes'][t['id']]:
                raise ValueError(f'prompt or request changed: {folder}/{t["id"]}')
        runs.append((entry, folder, m))
    return study, runs


def run(args):
    study, runs = load_runs(args.output)
    efforts = [name for name in study['arms'] if name not in ('off', 'on')]
    if efforts:
        # A reasoning toggle does not imply support for graded effort levels.
        with urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=30) as response:
            models = json.load(response)['data']
        model = next((m for m in models if m['id'] == study['model']), {})
        capabilities = model.get('reasoning') or {}
        supported = capabilities.get('supported_efforts', [])
        if 'supported_efforts' not in capabilities or (supported is not None and not set(efforts) <= set(supported)):
            raise ValueError('model does not advertise the requested effort levels; use --arms off on')
        if capabilities.get('mandatory'):
            raise ValueError('model requires reasoning and cannot supply an off control')
        results.save(args.output / 'reasoning-capabilities.json', capabilities)
    from openrouter_generation import Job, Options, generate
    # Freeze dispatch order across arms/repeats; completion order may differ.
    jobs = []
    rng = random.Random(study['seed'])
    for index in range(len(runs[0][2]['trials'])):
        order = list(runs)
        rng.shuffle(order)
        for _, folder, m in order:
            jobs.append(Job(folder, m, m['trials'][index]))
    options = Options(initial=getattr(args, 'concurrency', 2), maximum=getattr(args, 'max_concurrency', 4),
                      min_interval=getattr(args, 'request_interval', 1.0),
                      max_attempts=getattr(args, 'max_attempts', 6))
    try:
        return asyncio.run(generate(jobs, args.output, options, install_signals=True))
    finally:
        report(args)


def completion(folder, entry, trial):
    row = read(folder / 'completions' / (trial['id'] + '.json'))
    if row is not None and row['request_hash'] != entry['request_hashes'][trial['id']]:
        raise ValueError(f'completion request mismatch: {folder}/{trial["id"]}')
    # A raw reply is durable before validation; it is not yet an illegal result.
    # Keep the watcher alive until validation and any required scoring finish.
    if row is not None and not row['valid'] and 'error' not in row:
        return None
    return row


def battle(args):
    _, runs = load_runs(args.output)
    for entry, folder, m in runs:
        candidates = []
        for t in m['trials']:
            row = completion(folder, entry, t)
            if row and row['valid']:
                candidates.append(('llm-' + t['id'], row['team']))
        if candidates:
            # Reuses the existing exact-panel cache, validator and BC policy.
            battles.battle(SimpleNamespace(output=folder), candidates_override=candidates)
    report(args)


def average(values):
    return sum(values) / len(values) if values else None


def cluster_interval(rows, seed):
    """Equal starter weights; replicates/masks are not independent samples."""
    by_start = defaultdict(list)
    for start, delta in rows:
        by_start[start].append(delta)
    values = [average(v) for _, v in sorted(by_start.items())]
    result = dict(pairs=len(rows), starters=len(values), mean=average(values), low=None, high=None)
    if len(values) >= 2:
        rng = random.Random(seed)
        draws = sorted(average(rng.choices(values, k=len(values))) for _ in range(4000))
        result.update(low=draws[100], high=draws[3899])
    return result


def summarize(rows, arms, seed):
    summaries, paired = {}, {}
    for arm in arms:
        items = [row for row in rows if row['arm'] == arm]
        returned = [r for r in items if r['returned']]
        legal = [r for r in returned if r['legal']]
        scored = [r for r in legal if r['win_rate'] is not None]
        token_rows = [r for r in returned if r['reasoning_tokens'] is not None]
        summaries[arm] = dict(planned=len(items), returned=len(returned), missing=len(items)-len(returned),
            legal=len(legal), legality_rate=average([int(r['legal']) for r in returned]),
            scored=len(scored), win_rate_legal_scored=average([r['win_rate'] for r in scored]),
            reasoning_tokens_reported=len(token_rows),
            mean_reasoning_tokens=average([r['reasoning_tokens'] for r in token_rows]),
            positive_reasoning_token_responses=sum(r['reasoning_tokens'] > 0 for r in token_rows),
            cost_usd=sum(r['cost_usd'] or 0 for r in returned),
            cost_reported=sum(r['cost_usd'] is not None for r in returned),
            mean_elapsed_seconds=average([r['elapsed_seconds'] for r in returned if r['elapsed_seconds'] is not None]),
            finish_reasons=dict(Counter(r['finish_reason'] for r in returned)),
            failures=dict(Counter(r['error'] for r in returned if not r['legal'])))
    baseline = {(r['trial'], r['replicate']): r for r in rows if r['arm'] == 'off'}
    for arm in arms:
        if arm == 'off':
            continue
        legality, wins = [], []
        for row in rows:
            if row['arm'] != arm:
                continue
            control = baseline[(row['trial'], row['replicate'])]
            if row['returned'] and control['returned']:
                legality.append((row['start'], int(row['legal']) - int(control['legal'])))
                if all(r['legal'] and r['win_rate'] is not None for r in (row, control)):
                    wins.append((row['start'], row['win_rate'] - control['win_rate']))
        paired[arm] = dict(legality_delta=cluster_interval(legality, seed),
                           win_rate_delta_both_legal=cluster_interval(wins, seed))
    return dict(arms=summaries, versus_off=paired)


def report(args):
    study, runs = load_runs(args.output)
    rows, configs = [], []
    for entry, folder, m in runs:
        config = read(folder / 'battle-config.json')
        if config is not None:
            configs.append(config)
        for t in m['trials']:
            record = completion(folder, entry, t)
            label = read(folder / 'labels' / ('llm-' + t['id'] + '.json'))
            win = None
            if label is not None:
                if not record or not record['valid'] or config is None:
                    raise ValueError('battle label lacks a legal completion or battle config')
                if label['battles'] != 50 or not 0 <= label['wins'] <= 50:
                    raise ValueError('incomplete or invalid battle panel')
                win = label['wins'] / label['battles']
            response = record.get('response', {}) if record else {}
            usage = response.get('usage') or {}
            choices = response.get('choices') or [{}]
            rows.append(dict(arm=entry['arm'], replicate=entry['replicate'], trial=t['id'],
                start=t['start'], task=t['task'], k=t['k'], returned=record is not None,
                legal=bool(record and record['valid']), win_rate=win,
                finish_reason=choices[0].get('finish_reason', 'unknown'),
                error=record.get('error', 'unknown') if record else None,
                cost_usd=usage.get('cost'), elapsed_seconds=record.get('elapsed_seconds') if record else None,
                reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')))
    if configs and any(c != configs[0] for c in configs[1:]):
        raise ValueError('battle configurations differ across arms; results are not comparable')
    status = summarize(rows, study['arms'], study['seed'])
    status.update(model=study['model'], format=study['format'], opponents_sha256=study['opponents_sha256'],
                  planned_calls=study['planned_calls'], rows=rows,
                  complete=all(r['returned'] and (not r['legal'] or r['win_rate'] is not None) for r in rows),
                  by_task_k={f'{task}:{k}': summarize([r for r in rows if r['task'] == task and r['k'] == k],
                                                     study['arms'], study['seed'])
                             for task, k in sorted({(r['task'], r['k']) for r in rows})})
    results.save(args.output / 'status.json', status)
    lines = ['# Reasoning experiment', '', f"Model: `{study['model']}`; format: `{study['format']}`.",
             f"Frozen opponent pool: `{study['opponents_sha256']}`. Complete: {status['complete']}.", '',
             '| Arm | Returned / planned | Legal | Legality | Scored | Mean win rate | Reported cost USD |',
             '|---|---:|---:|---:|---:|---:|---:|']
    fmt = lambda n: '—' if n is None else f'{n:.1%}'
    for arm, s in status['arms'].items():
        lines.append(f"| {arm} | {s['returned']} / {s['planned']} | {s['legal']} | {fmt(s['legality_rate'])} | "
                     f"{s['scored']} | {fmt(s['win_rate_legal_scored'])} | {s['cost_usd']:.6f} |")
    for arm, metrics in status['versus_off'].items():
        for metric, d in metrics.items():
            lines += ['', f"{arm} − off, {metric}: {fmt(d['mean'])}; exploratory 95% starter-bootstrap "
                      f"interval [{fmt(d['low'])}, {fmt(d['high'])}], {d['pairs']} pairs / {d['starters']} starters."]
    lines += ['', 'Legality uses returned attempts, including truncations and invalid JSON. Missing calls are shown separately. '
              'Battle differences pair the same trial and replicate only when both outputs are legal and fully scored. '
              'Illegal outputs are not counted as battle losses. Repeated identical teams reuse exact panels. '
              'Intervals weight starters equally; a single starter has no interval.', '',
              'Requested settings do not prove realized reasoning effort. Inspect per-arm reasoning-token coverage, '
              'finish reasons, failures, latency, cost coverage, and task/mask breakdowns in status.json. '
              'Missing usage is unknown, not zero. The total output cap is shared across arms and may include reasoning tokens.', '',
              'The supplied pool is frozen and visible in prompts. This is an in-pool experiment, not a claim about '
              'the live meta or held-out performance. Policy seeds match; simulator RNG is not fixed. '
              'No causal or statistical conclusion should be drawn from an incomplete or tiny pilot.']
    (args.output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(args.output / 'report.md'), complete=status['complete'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'battle', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path, help='baseline folder containing manifest.json (prepare only)')
    parser.add_argument('--arms', nargs='+', default=['off', 'on'])
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--model', help='Explicit model override for a new study only')
    parser.add_argument('--provider', help='Pinned OpenRouter provider for a new study (default novita)')
    parser.add_argument('--max-tokens', type=int, default=16384, help='same total output cap for every arm')
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--max-concurrency', type=int, default=4)
    parser.add_argument('--request-interval', type=float, default=1.0, help='minimum seconds between global dispatches')
    parser.add_argument('--max-attempts', type=int, default=6, help='maximum attempts per request after definite 429 rejections')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == 'prepare':
        if args.parent is None:
            parser.error('prepare requires --parent')
        args.parent = args.parent.resolve()
    # Lock outside the output so prepare can require an empty destination.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / ('.' + args.output.name + '.reasoning.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.command](args)


if __name__ == '__main__':
    main()
