"""Opt-in, resumable diversity experiments on DiBO v4, with fresh battle receipts.

Use `plan` first. No existing DiBO entry point or frozen experiment is modified.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import numpy as np
import dibo_diversity as DV

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def all_rows(initial, rounds, generation):
    return initial + [r for g, value in rounds.items() if int(g) < generation for r in value.get('evaluated', [])]


def proposal_round(runtime, initial, out, generation):
    from dibo_diversity_runtime import Runtime, DB
    cfg, arm = runtime.cfg, runtime.arm
    rows = all_rows(initial, out['rounds'], generation)
    seen = {r['key'] for r in rows}
    if arm == 'peel':
        runtime.banned = DV.peel_bans(out['rounds'], generation)
        rows = [r for r in rows if not set(DV.composition(r)) & set(runtime.banned)]
    if arm == 'taboo':
        runtime.previous = {tuple(c) for g, result in out['rounds'].items() if int(g) < generation
                            for c in result.get('proposal_compositions', [])}
    if arm != 'island':
        rows = DV.training_rows(rows, arm, cfg['buffer'], cfg['seed'] + generation)
        prior, posterior, ensemble, training = runtime.models(rows, generation)
        return dict(runtime.propose(prior, posterior, ensemble, generation, seen), training=training)
    anchor_ids = {tid for tid, _ in runtime.signature['meta']}
    seeds = DV.island_seeds([r for r in initial if r['team_id'] in anchor_ids], cfg['islands'])
    homes = DV.island_observations(initial, out['rounds'], seeds, generation)
    pieces, selected = [], []
    columns = [c for c in DB.D.ORDER if c in DB.FD.SPECIES_COLS][:3]
    for i, home in enumerate(homes):
        child_cfg = dict(cfg, seed=cfg['seed'] + i * 100000,
                         propose=cfg['propose'] // cfg['islands'], battle=cfg['battle'] // cfg['islands'])
        child = Runtime(child_cfg, runtime.outpath.with_name(f'island_{i}.json'))
        try:
            child.pins = {c: seeds[i]['x'][c] for c in columns}
            own = DV.training_rows(home, 'control', cfg['buffer'], child_cfg['seed'] + generation)
            prior, posterior, ensemble, training = child.models(own, generation)
            part = child.propose(prior, posterior, ensemble, generation, seen)
            for row in part['selected']:
                row['island'] = i
            selected.extend(part['selected']); seen.update(r['key'] for r in part['selected'])
            pieces.append(dict(part, training=training, island=i, seed_key=seeds[i]['key']))
        finally:
            child.close()
    # Interleave raw streams so equal-size metrics represent every island.
    raw = [p['raw_accepted_sample'][j] for j in range(max(len(p['raw_accepted_sample']) for p in pieces))
           for p in pieces if j < len(p['raw_accepted_sample'])][:cfg['metric_sample']]
    proposals = [r for p in pieces for r in p['proposals']]
    return dict(proposals=proposals, selected=selected, raw_accepted_sample=raw, islands=pieces,
                tried=sum(p['tried'] for p in pieces),
                proposal_compositions=sorted({DV.composition(r) for r in proposals}),
                diversity=dict(raw=DV.metrics(raw), output=DV.metrics(proposals), selected=DV.metrics(selected)))


def report(directory):
    from dibo_v4 import atomic_json
    proto = read(directory / 'protocol.json')
    summary = dict(complete=True, expected=proto['physical_battles_per_arm'], arms={})
    lines = ['# DiBO v4 diversity experiments', '', 'Variants are adaptations; these are not reruns of CEM.', '',
             '| Arm | Rounds | Search battles | Confirmation | Heldout | Heldout first | Heldout mean |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in proto['arms']:
        path = directory / f'{arm}.json'
        out = read(path) if path.exists() else {'rounds': {}}
        rounds = [r for r in out['rounds'].values() if 'evaluated' in r]
        value = dict(rounds=len(rounds), search=sum(r['physical_battles'] for r in rounds))
        for stage in ('confirmation', 'heldout'):
            results = out.get(stage, [])
            value[stage] = sum(r['battles'] for r in results)
            value[stage + '_first'] = results[0]['wins'] / results[0]['battles'] if results else None
            value[stage + '_mean'] = float(np.mean([r['wins'] / r['battles'] for r in results])) if results else None
        complete = value['rounds'] == proto['config']['rounds'] and all(
            value[k] == v for k, v in proto['physical_battles_per_arm'].items())
        summary['complete'] &= complete
        summary['arms'][arm] = value
        fmt = lambda v: f'{v:.2%}' if v is not None else 'pending'
        lines.append(f"| {arm} | {value['rounds']} | {value['search']} | {value['confirmation']} | {value['heldout']} | {fmt(value['heldout_first'])} | {fmt(value['heldout_mean'])} |")
    lines += ['', 'One search seed per invocation. No claim of repeatable superiority follows from one run.',
              'Finalists are locked using search labels before fresh evaluation. Raw receipts are saved alongside this report.']
    atomic_json(directory / 'summary.json', summary)
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


def execute(args, cfg):
    from dibo_diversity_runtime import Runtime, DB
    import dibo_comparison_v2 as C
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import os
        if os.environ.get('VGC_FAST_BATTLES') == '1':
            from fast_experiment_battles import install
            install(directory)
        runtime = Runtime(dict(cfg, arm='control'), directory / 'control.json')
        try:
            frozen = read(args.initial_protocol)
            initial = frozen['initial']
            actual = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
            if actual != frozen['initial_hash']:
                raise ValueError('initial observation hash mismatch')
            if frozen['runtime_signature']['vocabulary'] != json.loads(json.dumps(runtime.signature['vocabulary'])):
                raise ValueError('initial vocabulary differs from runtime')
            if frozen['search_columns'] != json.loads(json.dumps(runtime.signature['meta'])):
                raise ValueError('initial meta opponents differ from runtime')
            if frozen['runtime_signature']['policy'] != json.loads(json.dumps(runtime.signature['policy'])):
                raise ValueError('initial labels used a different battle policy')
            if any(DB.HO.is_heldout_text(r['paste']) for r in initial):
                raise ValueError('heldout team in initial observations')
            if args.smoke:
                # Keep all anchors so structural arms have legitimate island seeds.
                ids = {tid for tid, _ in runtime.signature['meta']}
                initial = [r for r in initial if r['team_id'] in ids] + [r for r in initial if r['team_id'] not in ids][:16]
            proto = dict(config=cfg, arms=args.arms, initial=initial,
                         initial_hash=hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest(),
                         source_initial_hash=actual, runtime_signature=runtime.signature,
                         search_columns=runtime.signature['meta'],
                         physical_battles_per_arm=dict(search=cfg['rounds'] * cfg['battle'] * len(runtime.signature['meta']),
                                                      confirmation=cfg['final_n'] * cfg['final_battles'],
                                                      heldout=cfg['final_n'] * cfg['final_battles']))
            proto = json.loads(json.dumps(proto))
            path = directory / 'protocol.json'
            if path.exists() and read(path) != proto:
                raise ValueError('frozen diversity configuration or source changed; use a new directory')
            if not path.exists():
                DB.atomic_json(path, proto)
        finally:
            runtime.close()
        for g in range(cfg['rounds']):
            for arm in args.arms:
                path = directory / f'{arm}.json'
                out = read(path) if path.exists() else dict(initial_hash=proto['initial_hash'], rounds={})
                if out['initial_hash'] != proto['initial_hash']:
                    raise ValueError('arm initial observations mismatch')
                result = out['rounds'].setdefault(str(g), {})
                if 'evaluated' in result:
                    continue
                runtime = Runtime(dict(cfg, arm=arm), path)
                try:
                    if 'selected' not in result:
                        result.update(proposal_round(runtime, initial, out, g))
                        DB.atomic_json(path, out)
                    evaluated = C.fresh_evaluate(runtime, result['selected'], directory, arm, g, proto)
                    if arm == 'island':
                        owners = {r['key']: r['island'] for r in result['selected']}
                        evaluated = [dict(r, island=owners[r['key']]) for r in evaluated]
                    result.update(evaluated=evaluated, mean=float(np.mean([r['y'] for r in evaluated])),
                                  best=max(r['y'] for r in evaluated),
                                  physical_battles=sum(r['battles'] for r in evaluated))
                    DB.atomic_json(path, out)
                finally:
                    runtime.close()
            report(directory)
        # Lock every arm before any evaluation, never choose by final results.
        for arm in args.arms:
            path = directory / f'{arm}.json'; out = read(path)
            if 'final_selected' not in out:
                rows = [dict(r, log_target=r['y']) for result in out['rounds'].values() for r in result['evaluated']]
                out['final_selected'] = DB.DC.unique_top(rows, cfg['final_n'])
                DB.atomic_json(path, out)
        for arm in args.arms:
            C.final_arm(directory, arm, proto)
        report(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'run', 'report'))
    parser.add_argument('--arms', nargs='+', choices=DV.ARMS, default=list(DV.ARMS))
    parser.add_argument('--directory', type=Path, default=ROOT / 'results/dibo_diversity_v4')
    parser.add_argument('--initial-protocol', type=Path, default=ROOT / 'results/dibo_vs_cem_full_20261012/protocol.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, default=6060)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--pool-multiplier', type=int, default=8)
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms):
        parser.error('duplicate arms')
    if args.stage == 'report':
        with (args.directory / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return report(args.directory)
    from dibo_comparison_v2 import config
    cfg = config(args.smoke)
    cfg.update(seed=args.seed, pool_multiplier=args.pool_multiplier, islands=4, metric_sample=512)
    if args.smoke:
        cfg.update(rounds=2, propose=8, battle=4, islands=2, buffer=64, metric_sample=8)
        if args.directory == ROOT / 'results/dibo_diversity_v4':
            args.directory = ROOT / 'results/dibo_diversity_v4_smoke'
    if args.rounds is not None:
        cfg['rounds'] = args.rounds
    if cfg['rounds'] < 1 or cfg['seed'] < 0 or cfg['pool_multiplier'] < 1:
        parser.error('positive rounds and pool multiplier, nonnegative seed required')
    if cfg['final_n'] > cfg['rounds'] * cfg['battle']:
        parser.error('not enough searched teams for the final budget')
    if cfg['propose'] % cfg['islands'] or cfg['battle'] % cfg['islands']:
        parser.error('island budgets must divide exactly')
    if args.stage == 'plan':
        initial = read(args.initial_protocol)
        n = len(initial['search_columns'])
        search = cfg['rounds'] * cfg['battle'] * n
        final = 2 * cfg['final_n'] * cfg['final_battles']
        print(json.dumps(dict(arms=args.arms, config=cfg, search_battles_per_arm=search,
                              final_battles_per_arm=final, total_battles=(search + final) * len(args.arms),
                              note='No training or battles launched. Large-pool arms multiply proposal compute, not battles.'), indent=2))
        return
    execute(args, cfg)


if __name__ == '__main__':
    main()
