"""Prepare a paired study and run generation followed by battle scoring."""
import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import llm_baseline as B
import reasoning_experiment as E


def fresh_starters(parent, count, seed):
    """Sample legal starters without generating unused random-fill controls."""
    import numpy as np
    h, db = B.runtime()
    tables = h.build_tables()
    rng = np.random.default_rng(seed)
    m = copy.deepcopy(parent)
    m.update(starts=[], trials=[], seed=seed)
    for key in ('exclude_starter_team_ids', 'starter_provenance', 'pilot_selection'):
        m.pop(key, None)
    m['starter_kind'] = 'fresh random legal teams outside the frozen opponent pool'
    seen = {db.canon_hash(db.canonical_paste(o['paste'])) for o in m['opponents']}
    # Exclude earlier pilot starters too: the follow-up is a fresh sample.
    seen.update(db.canon_hash(db.canonical_paste(B.paste(t, h))) for t in parent['starts'])
    validator = h.Validator()
    try:
        for o in m['opponents']:
            error = validator(o['paste'])
            if error:
                raise ValueError(f'invalid frozen opponent: {error}')
        for _ in range(10000):
            team = h.one_team(tables, rng)
            if team is None:
                continue
            text = B.paste(team, h)
            key = db.canon_hash(db.canonical_paste(text))
            if key in seen or validator(text) is not None:
                continue
            B.validate_shape(team)
            seen.add(key)
            m['starts'].append(team)
            if len(m['starts']) == count:
                break
        if len(m['starts']) != count:
            raise ValueError('could not sample enough legal starters')
    finally:
        validator.close()
    for start in range(count):
        for task, sizes in m['grid'].items():
            paths = B.paths_for(task)
            random.Random(f'{seed}:{start}:{task}').shuffle(paths)
            k = max(sizes)
            m['trials'].append(dict(id=f's{start}-{task}-{k}', start=start,
                                    task=task, k=k, paths=paths[:k]))
    m['control_design'] = 'paired reasoning off/on; no random-fill arm'
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run'])
    p.add_argument('--parent', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--starters', type=int, default=3)
    p.add_argument('--replicates', type=int, default=1)
    p.add_argument('--sample-new', action='store_true', help='sample fresh legal starters using the parent pool and prompt')
    p.add_argument('--seed', type=int, default=20261012)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--max-concurrency', type=int, default=4)
    p.add_argument('--request-interval', type=float, default=1.0)
    p.add_argument('--max-attempts', type=int, default=6)
    args = p.parse_args()
    args.output = args.output.resolve()
    if not 1 <= args.workers <= 32:
        p.error('workers must be between 1 and 32')
    if args.command == 'prepare':
        if not args.parent:
            p.error('prepare requires --parent')
        if args.starters < 1 or args.replicates < 1:
            p.error('starters and replicates must be positive')
        if args.output.exists() and any(args.output.iterdir()):
            p.error('output must be empty')
        parent = B.load_manifest(SimpleNamespace(output=args.parent))
        m = fresh_starters(parent, args.starters, args.seed) if args.sample_new else copy.deepcopy(parent)
        starters = sorted({t['start'] for t in m['trials']})[:args.starters]
        if len(starters) != args.starters:
            p.error('parent does not contain enough starting teams')
        # Largest mask available in each family tests actual completion ability.
        selected = []
        for start in starters:
            for task in sorted({t['task'] for t in m['trials']}):
                options = [t for t in m['trials'] if t['start'] == start and t['task'] == task]
                if options:
                    selected.append(max(options, key=lambda t: t['k']))
        m['trials'] = selected
        m['grid'] = {task: sorted({t['k'] for t in selected if t['task'] == task})
                     for task in sorted({t['task'] for t in selected})}
        m['pilot_selection'] = dict(parent_sha256=B.digest(parent), starters=starters,
                                    rule=f'first {args.starters} starter indices; largest mask per family')
        subset = args.output.parent / (args.output.name + '-parent')
        if subset.exists():
            raise ValueError('pilot parent already exists')
        B.save(subset / 'manifest.json', m)
        E.prepare(SimpleNamespace(parent=subset, output=args.output, arms=['off', 'on'],
                                  replicates=args.replicates, max_tokens=16384, request_timeout=600))
        E.report(args)
        return
    import fcntl
    with (args.output.parent / ('.' + args.output.name + '.reasoning.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            outcome = E.run(args)
        finally:
            E.report(args)
    if outcome['stage'] != 'complete':
        return
    subprocess.run([sys.executable, str(B.ROOT / 'scripts/reasoning_battles.py'),
                    '--output', str(args.output), '--workers', str(args.workers)], check=True)


if __name__ == '__main__':
    main()
