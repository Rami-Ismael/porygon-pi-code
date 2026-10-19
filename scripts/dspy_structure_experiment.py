"""Does DSPy request structure change masked-completion legality or win rate?

See docs/dspy-structure-experiment.md. Phase one mirrors finished baseline
studies (reasoning_experiment.py) run for run: same model, provider, reasoning
level, starters, masks, repeats, instructions, input values, output cap,
validator and 50-battle meta panel. Only the request structure changes
(dspy_structure.py); no DSPy optimization runs. Phase two passes an optimized
program from dspy_optimize.py with --program.

prepare, report and headroom make no API calls and launch no battles.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
import copy
import fcntl
import json
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace
import urllib.request

from repo_paths import ROOT  # also puts src/ first on sys.path
import battles
import llm_completion
import results
import reasoning_experiment as E

MODEL_NAMES = {llm_completion.MODEL: 'Ling 3.0 Flash', 'inception/mercury-2.5': 'Mercury 2.5'}
VARIANT_NAMES = {'chat': 'DSPy field-marker format', 'json': 'DSPy JSON-object format',
                 'resampled': 'Baseline JSON prompt, sampled again'}
RESAMPLED = 'resampled'
FAILURE_CLASSES = ('no answer', 'unreadable answer', 'cut off', 'wrong answer shape',
                   'Stat Points out of range', 'rule violation')
SHAPE_ERRORS = ('expected exactly', 'wrong number of fills', 'each fill must contain',
                'duplicate or unmasked path', 'team must have six slots', 'invalid slot fields',
                'slot names must be strings', 'four move strings required', 'unmasked fields changed')
JSON_ERRORS = re.compile(r'^(Expecting|Extra data|Unterminated|Invalid control|Invalid \\escape|'
                         r'DSPy parse|reply has no answer text|invalid provider JSON envelope|the JSON object must be)')


def slug(model):
    return model.split('/')[-1].split('-')[0]


def model_name(model):
    return MODEL_NAMES.get(model, model)


def variant_name(variant, program=None, constraints=None):
    name = VARIANT_NAMES[variant]
    if constraints and variant != RESAMPLED:
        name += ' with legal-choice types'
    if not program or variant == RESAMPLED:
        return name
    return f'Optimized {name}' if name[:2].isupper() else f'Optimized {name[0].lower()}{name[1:]}'  # keep "DSPy"


def payload_without_messages(manifest, trial):
    payload = llm_completion.completion_payload(manifest, trial)
    payload.pop('messages')
    return payload


def battle_design(m):
    import reasoning_battles
    return reasoning_battles.battle_design(m)


def prepare(args):
    import dspy_structure as S
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be empty; use run/battle/report to resume')
    program = json.loads(args.program.read_text()) if args.program else {}
    unknown = set(program) - {'instructions', 'demos', 'optimizer'}
    if unknown:
        raise ValueError(f'unknown program keys {sorted(unknown)}')
    variants = list(args.adapters) + ([RESAMPLED] if args.resample_baseline else [])
    if not variants or len(set(variants)) != len(variants):
        raise ValueError('choose at least one distinct adapter')
    # A model whose baseline route no longer exists can be sent to another provider.
    # The change is recorded, and the report states it next to every comparison.
    overrides = dict(item.split('=', 1) for item in getattr(args, 'provider', None) or ())
    study = dict(schema=1, kind='dspy-structure', dspy_version=S.dspy.__version__, variants=variants,
                 program=program or None, baselines=[], runs=[], arms={})
    if overrides:
        study['provider_overrides'] = {}
    constraints = getattr(args, 'constraints', None)
    if constraints:
        study['constraints'] = constraints
    design = None
    for folder in args.baseline:
        baseline, runs = E.load_runs(folder)  # verifies frozen manifests and request hashes
        status = E.read(folder / 'status.json')
        if status is None or not status.get('complete'):
            raise ValueError(f'baseline study is not complete: {folder}')
        study['baselines'].append(dict(path=str(folder), name=folder.name, model=baseline['model'],
                                       arms=list(baseline['arms']), experiment_sha256=results.digest(baseline),
                                       status_sha256=results.digest(status)))
        for entry, run_folder, m in runs:
            if design is None:
                design = battle_design(m)
                study['seed'] = m['seed']
                study['opponents_sha256'] = results.digest(m['opponents'])
            elif battle_design(m) != design:
                raise ValueError('baseline studies use different starters, opponents or battle settings')
            for variant in variants:
                new = copy.deepcopy(m)
                if variant != RESAMPLED:
                    new['structure'] = S.config(variant, program.get('instructions'), program.get('demos', ()),
                                                constraints=constraints)
                new['structure_experiment'] = dict(baseline_study=folder.name, baseline_run=entry['path'],
                                                   baseline_manifest_sha256=entry['manifest_sha256'])
                rerouted = m['model'] in overrides
                if rerouted:
                    original = llm_completion.completion_payload(m, m['trials'][0])['provider']
                    new['provider'] = dict(original, order=[overrides[m['model']]])
                    new['structure_experiment']['provider_override'] = dict(baseline=original, used=new['provider'])
                    study['provider_overrides'][m['model']] = dict(baseline=original, used=new['provider'])
                for t in new['trials']:
                    # Everything but the message packaging matches the baseline request,
                    # apart from a recorded provider override.
                    mine, theirs = payload_without_messages(new, t), payload_without_messages(m, t)
                    if rerouted:
                        mine.pop('provider'), theirs.pop('provider')
                    if mine != theirs:
                        raise ValueError('structured request changed a sampling or routing setting')
                    if variant == RESAMPLED:
                        if rerouted:
                            if llm_completion.completion_payload(new, t)['messages'] != llm_completion.completion_payload(m, t)['messages']:
                                raise ValueError('resampled baseline messages differ from the original')
                        elif results.digest(llm_completion.completion_payload(new, t)) != entry['request_hashes'][t['id']]:
                            raise ValueError('resampled baseline request differs from the original')
                    else:
                        S.check_information(new, t)
                arm = f"{slug(m['model'])}-{entry['arm']}-{variant}"
                study['arms'][arm] = dict(model=m['model'], reasoning=entry['arm'], variant=variant)
                relative = f"{slug(m['model'])}/{entry['arm']}/{variant}/r{entry['replicate']}"
                hashes = {t['id']: results.digest(llm_completion.completion_payload(new, t)) for t in new['trials']}
                study['runs'].append(dict(path=relative, arm=arm, model=m['model'], reasoning=entry['arm'],
                                          variant=variant, replicate=entry['replicate'],
                                          baseline=folder.name, baseline_path=entry['path'],
                                          manifest_sha256=results.digest(new), request_hashes=hashes))
                results.save(args.output / relative / 'manifest.json', new)
                results.save(args.output / relative / 'example-request.json', llm_completion.completion_payload(new, new['trials'][0]))
    study['planned_calls'] = sum(len(r['request_hashes']) for r in study['runs'])
    results.save(args.output / 'experiment.json', study)
    print(json.dumps(dict(output=str(args.output), planned_calls=study['planned_calls'],
                          runs=len(study['runs']), variants=variants)))


def run(args):
    study, runs = E.load_runs(args.output)
    graded = defaultdict(set)
    for entry, _, _ in runs:
        if entry['reasoning'] not in ('off', 'on'):
            graded[entry['model']].add(entry['reasoning'])
    if graded:
        # Same preflight as reasoning_experiment.run: a toggle is not a graded scale.
        with urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=30) as response:
            catalog = {m['id']: m for m in json.load(response)['data']}
        for model, efforts in graded.items():
            capabilities = catalog.get(model, {}).get('reasoning') or {}
            supported = capabilities.get('supported_efforts', [])
            if 'supported_efforts' not in capabilities or (supported is not None and not efforts <= set(supported)):
                raise ValueError(f'{model} does not advertise the requested effort levels')
            if capabilities.get('mandatory'):
                raise ValueError(f'{model} requires reasoning and cannot supply an off arm')
        results.save(args.output / 'reasoning-capabilities.json', {m: catalog[m].get('reasoning') for m in graded})
    from openrouter_generation import Job, Options, generate
    # Interleave every run within each task to spread provider drift evenly.
    jobs = []
    rng = random.Random(study['seed'])
    for index in range(len(runs[0][2]['trials'])):
        order = list(runs)
        rng.shuffle(order)
        jobs += [Job(folder, m, m['trials'][index]) for _, folder, m in order]
    options = Options(initial=args.concurrency, maximum=args.max_concurrency,
                      min_interval=args.request_interval, max_attempts=args.max_attempts)
    return asyncio.run(generate(jobs, args.output, options, install_signals=True))


def rebattle_baselines(output):
    """A comparison needs its baselines on the same engine: re-battle (in full) any
    baseline study the removed legacy engine battled, under that study's own locks."""
    import reasoning_battles as RB
    study, _ = E.load_runs(output)
    for b in study['baselines']:
        path = Path(b['path'])
        _, runs = E.load_runs(path)
        if all(battles.recorded_backend(folder) in (None, 'fast') for _, folder, _ in runs):
            continue
        print(f"re-battling baseline {b['name']} on the fast evaluator", flush=True)
        with (path / '.parallel-battles.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            RB.score(path)
            with (path.parent / ('.' + path.name + '.reasoning.lock')).open('a') as generation_lock:
                fcntl.flock(generation_lock, fcntl.LOCK_EX)
                E.report(SimpleNamespace(output=path))


def battle(args):
    """Score legal teams with the exact-panel scorer on the fast evaluator, after
    bringing the baselines onto the same engine."""
    import reasoning_battles as RB
    with (args.output / '.parallel-battles.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rebattle_baselines(args.output)
        while True:
            panels = RB.score(args.output)
            state = dict(RB.progress(args.output), workers=args.workers, updated_at=time.time())
            state['stage'] = 'complete' if state['complete'] else 'waiting_for_generation'
            results.save(args.output / 'parallel-battles.json', state)
            print(json.dumps(dict(state, new_panels=panels)), flush=True)
            if state['complete'] or not args.watch:
                break
            time.sleep(30)


def failure_class(error):
    if error in (None, 'unknown') or error.startswith("'"):
        return 'no answer'  # e.g. a provider error body without choices
    if error == 'truncated/non-stop completion':
        return 'cut off'
    if error.startswith('DSPy parse') and 'validation error' in error:
        # Readable JSON refused by a typed field (legal-choice types): classify the cause.
        if re.search(r'type=(missing|extra_forbidden|too_short|too_long|list_type|dict_type|model_type)', error) \
                or 'Field required' in error or 'Extra inputs are not permitted' in error or 'List should have' in error:
            return 'wrong answer shape'
        if re.search(r'type=(less_than_equal|greater_than_equal|int_type)', error) or 'than or equal to' in error:
            return 'Stat Points out of range'
        if 'type=enum' in error or 'Input should be' in error:
            return 'rule violation'
    if JSON_ERRORS.match(error):
        return 'unreadable answer'
    if error.startswith(SHAPE_ERRORS):
        return 'wrong answer shape'
    if error.startswith('Stat Points require'):
        return 'Stat Points out of range'
    return 'rule violation'


def structured_rows(output):
    """One row per planned request, in the reasoning_experiment row format."""
    study, runs = E.load_runs(output)
    rows, configs = [], {}
    for entry, folder, m in runs:
        config = E.read(folder / 'battle-config.json')
        if config is not None:
            configs[entry['path']] = config
        for t in m['trials']:
            record = E.completion(folder, entry, t)
            label = E.read(folder / 'labels' / ('llm-' + t['id'] + '.json'))
            win = None
            if label is not None:
                if not record or not record['valid'] or config is None:
                    raise ValueError('battle label lacks a legal completion or battle config')
                if label['battles'] != 50 or not 0 <= label['wins'] <= 50:
                    raise ValueError('incomplete or invalid battle panel')
                win = label['wins'] / label['battles']
            response = record.get('response', {}) if record else {}
            usage = response.get('usage') or {}
            error = record.get('error', 'unknown') if record else None
            if (record and not record['valid'] and (m.get('structure') or {}).get('constraints')
                    and str(error).startswith('DSPy parse') and 'validation error' not in error):
                # Older records cut the typed-parse reason off; re-derive it from the saved reply (no API call).
                import dspy_structure as S
                try:
                    S.parse(m, t, response['choices'][0]['message']['content'], {})
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    error = str(exc)
            rows.append(dict(model=entry['model'], reasoning=entry['reasoning'], variant=entry['variant'],
                             replicate=entry['replicate'], trial=t['id'], start=t['start'], task=t['task'],
                             returned=record is not None, legal=bool(record and record['valid']), win_rate=win,
                             error=error,
                             strict_json=(record or {}).get('structure', {}).get('strict_json'),
                             cost_usd=usage.get('cost'),
                             reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')))
    return study, rows, configs


def without_outcomes(status):
    return [{k: v for k, v in r.items() if k != 'win_rate'} for r in status['rows']]


def baseline_status(b):
    """The baseline status frozen at prepare, or its re-battle on another engine: the
    prepared status must then sit in an archive-*-battles-* folder of that baseline
    and differ from the current one only in win rates."""
    path = Path(b['path'])
    status = E.read(path / 'status.json')
    if status is not None and results.digest(status) == b['status_sha256']:
        return status
    for archived in sorted(path.glob('archive-*-battles-*/status.json')):
        old = E.read(archived)
        if status is not None and results.digest(old) == b['status_sha256'] and without_outcomes(old) == without_outcomes(status):
            return status
    raise ValueError(f"baseline status changed since prepare: {b['path']}")


def baseline_rows(study):
    rows = []
    for b in study['baselines']:
        status = baseline_status(b)
        for r in status['rows']:
            rows.append(dict(r, model=b['model'], reasoning=r['arm'], variant='baseline'))
    return rows


def check_battle_runtime(study, configs):
    """Win rates are comparable only under the identical policy and simulator."""
    reference = None
    for entry in study['runs']:
        config = configs.get(entry['path'])
        baseline = E.read(Path(next(b['path'] for b in study['baselines'] if b['name'] == entry['baseline']))
                          / entry['baseline_path'] / 'battle-config.json')
        for c in (config, baseline):
            if c is None:
                continue
            reference = reference or c
            if c != reference:
                raise ValueError('battle policy, simulator or panel differs from the baseline; win rates are not comparable')


def sign_flip_p(values, seed, draws=20000):
    """Two-sided starter-level paired permutation p-value (exact up to 16 starters)."""
    n = len(values)
    if n < 2:
        return None
    observed = abs(sum(values)) / n
    if n <= 16:
        signs = ([1 if mask >> i & 1 else -1 for i in range(n)] for mask in range(2 ** n))
        total = 2 ** n
    else:
        rng = random.Random(seed)
        signs = ([rng.choice((-1, 1)) for _ in range(n)] for _ in range(draws))
        total = draws
    hits = sum(abs(sum(s * v for s, v in zip(sign, values))) / n >= observed - 1e-12 for sign in signs)
    return (hits + (n > 16)) / (total + (n > 16))


def interval(pairs, seed):
    """Equal starter weights: bootstrap interval plus sign-flip p-value."""
    result = E.cluster_interval(pairs, seed)
    by_start = defaultdict(list)
    for start, delta in pairs:
        by_start[start].append(delta)
    result['p'] = sign_flip_p([E.average(v) for _, v in sorted(by_start.items())], seed)
    return result


def compare(rows, control, seed):
    """Paired treatment minus control by model, reasoning level, trial and repeat."""
    lookup = {(r['model'], r['reasoning'], r['trial'], r['replicate']): r for r in control}
    legality, both_legal, delivered = [], [], []
    for r in rows:
        c = lookup.get((r['model'], r['reasoning'], r['trial'], r['replicate']))
        if c is None or not (r['returned'] and c['returned']):
            continue
        if any(x['legal'] and x['win_rate'] is None for x in (r, c)):
            continue  # a legal team still waiting for its battle panel
        legality.append((r['start'], int(r['legal']) - int(c['legal'])))
        if r['legal'] and c['legal']:
            both_legal.append((r['start'], r['win_rate'] - c['win_rate']))
        delivered.append((r['start'], (r['win_rate'] or 0) - (c['win_rate'] or 0)))
    return dict(legality=interval(legality, seed), win_rate_both_legal=interval(both_legal, seed),
                wins_delivered_per_request=interval(delivered, seed))


def describe(rows):
    returned = [r for r in rows if r['returned']]
    legal = [r for r in returned if r['legal']]
    scored = [r for r in legal if r['win_rate'] is not None]
    strict = [r for r in returned if r.get('strict_json') is not None]
    return dict(planned=len(rows), returned=len(returned), legal=len(legal),
                legality_rate=E.average([int(r['legal']) for r in returned]), scored=len(scored),
                legal_team_win_rate=E.average([r['win_rate'] for r in scored]),
                failures={k: v for k, v in Counter(failure_class(r['error']) for r in returned
                                                   if not r['legal']).items()},
                strict_json_rate=E.average([int(r['strict_json']) for r in strict]),
                legal_after_json_repair=sum(r['legal'] and r.get('strict_json') is False for r in returned),
                cost_usd=sum(r.get('cost_usd') or 0 for r in returned),
                cost_reported=sum(r.get('cost_usd') is not None for r in returned))


def holm(results, metric):
    """Holm step-down adjustment across every listed comparison of one metric."""
    tests = sorted(((c[metric]['p'], i) for i, c in enumerate(results) if c[metric]['p'] is not None))
    running = 0
    for rank, (p, i) in enumerate(tests):
        running = max(running, min(1.0, (len(tests) - rank) * p))
        results[i][metric]['p_holm'] = running


def verdict(d, margin):
    if d['low'] is None:
        return 'too few starters'
    if d['low'] > 0:
        return 'improved'
    if d['high'] < 0:
        return 'regressed'
    if -margin <= d['low'] and d['high'] <= margin:
        return f'no change beyond ±{100*margin:g} pp'
    return 'unclear: interval includes zero'


def report(args):
    study, rows, configs = structured_rows(args.output)
    base = baseline_rows(study)
    check_battle_runtime(study, configs)
    seed = study['seed']
    # Phase two: also pair the optimized program with the phase-one structured answers.
    reference = structured_rows(args.reference)[1] if getattr(args, 'reference', None) else None
    cells = []
    for model, reasoning, variant in sorted({(r['model'], r['reasoning'], r['variant']) for r in rows}):
        treated = [r for r in rows if (r['model'], r['reasoning'], r['variant']) == (model, reasoning, variant)]
        control = [r for r in base if (r['model'], r['reasoning']) == (model, reasoning)]
        cell = dict(model=model, reasoning=reasoning, variant=variant, structured=describe(treated),
                    baseline=describe(control), versus_baseline=compare(treated, control, seed))
        if variant != RESAMPLED and RESAMPLED in study['variants']:
            same_time = [r for r in rows if (r['model'], r['reasoning'], r['variant']) == (model, reasoning, RESAMPLED)]
            cell['versus_resampled'] = compare(treated, same_time, seed)
        if reference is not None:
            unoptimized = [r for r in reference if (r['model'], r['reasoning'], r['variant']) == (model, reasoning, variant)]
            cell['versus_unoptimized'] = compare(treated, unoptimized, seed)
        cells.append(cell)
    pooled = []
    for model in sorted({r['model'] for r in rows}) + [None]:
        for variant in study['variants']:
            treated = [r for r in rows if r['variant'] == variant and model in (None, r['model'])]
            control = [r for r in base if model in (None, r['model'])]
            pooled.append(dict(model=model, variant=variant, structured=describe(treated),
                               baseline=describe(control), versus_baseline=compare(treated, control, seed)))
    family = [c['versus_baseline'] for c in cells]
    for metric in ('legality', 'win_rate_both_legal', 'wins_delivered_per_request'):
        holm(family, metric)
    for c in cells + pooled:
        v = c['versus_baseline']
        c['verdict'] = dict(legality=verdict(v['legality'], args.legality_margin),
                            win_rate_both_legal=verdict(v['win_rate_both_legal'], args.win_margin),
                            wins_delivered_per_request=verdict(v['wins_delivered_per_request'], args.win_margin))
    complete = all(r['returned'] and (not r['legal'] or r['win_rate'] is not None) for r in rows)
    status = dict(kind=study['kind'], complete=complete, dspy_version=study['dspy_version'],
                  program=study['program'], baselines=study['baselines'], planned_calls=study['planned_calls'],
                  provider_overrides=study.get('provider_overrides') or {},
                  constraints=study.get('constraints'),
                  margins=dict(legality=args.legality_margin, win_rate=args.win_margin),
                  cells=cells, pooled=pooled, rows=rows)
    results.save(args.output / 'status.json', status)
    (args.output / 'report.md').write_text(render(status))
    print(json.dumps(dict(report=str(args.output / 'report.md'), complete=complete)))
    return status


def pp(x):
    return '—' if x is None else f'{100*x:+.1f}'


def pct(x):
    return '—' if x is None else f'{100*x:.1f}%'


def ci(d):
    if d['low'] is None:
        return f"{pp(d['mean'])} (no interval)"
    p = '' if d.get('p_holm') is None else f"; Holm p {d['p_holm']:.2f}"
    return f"{pp(d['mean'])} [{pp(d['low'])}, {pp(d['high'])}]{p}"


def label(model, reasoning=None):
    name = 'All models' if model is None else model_name(model)
    return name if reasoning is None else f'{name}, reasoning {reasoning}'


def render(status):
    title = 'DSPy-optimized structure versus the JSON-prompt baseline' if status['program'] else \
        'DSPy request structure versus the JSON-prompt baseline'
    lines = [f'# {title}', '',
             f"Complete: {status['complete']}. DSPy {status['dspy_version']}. "
             f"Baselines: {', '.join(b['name'] for b in status['baselines'])}.", '',
             'Every structured request keeps the baseline model, provider, reasoning level, starters, masks, '
             'repeats, instructions, input values, output cap, validator and 50-battle meta panel. '
             'Differences are structured minus baseline for the same model, reasoning level, mask and repeat. '
             'Intervals are 95% starter-cluster bootstraps (each starter weighted equally); p-values are '
             'starter-level sign-flip tests, Holm-adjusted across every model × reasoning × format cell.', '']
    if status.get('constraints') == 'legal-choices':
        lines += ['**Legal-choice answer types.** The structured answer has one typed field per requested path, limited to '
                  "the regulation's legal values there (learnable moves, the species' ability/item pairs, items no fixed "
                  'teammate holds, species not already on the team, Stat Points 0 to 32 with a stated total of at most 66). '
                  'Inputs and instructions are unchanged.', '']
    for model, o in status.get('provider_overrides', {}).items():
        lines += [f"**Provider changed for {model_name(model)}.** The baseline route "
                  f"({', '.join(o['baseline']['order'])}) no longer serves this model, so these requests went to "
                  f"{', '.join(o['used']['order'])}. Its rows against the historical baseline mix structure with a "
                  'provider change; the rows against the baseline prompt sampled at the same time use one provider.', '']
    lines += ['## Pooled', '',
             '| Models | Format | Legal: baseline → structured | Legality change, pp | Win-rate change when both legal, pp | Wins delivered per request change, pp | Win-rate verdict |',
             '|---|---|---:|---:|---:|---:|---|']
    for c in status['pooled']:
        v = c['versus_baseline']
        lines.append(f"| {label(c['model'])} | {variant_name(c['variant'], status['program'], status.get('constraints'))} | {pct(c['baseline']['legality_rate'])} → "
                     f"{pct(c['structured']['legality_rate'])} | {ci(v['legality'])} | {ci(v['win_rate_both_legal'])} "
                     f"({v['win_rate_both_legal']['pairs']} pairs) | {ci(v['wins_delivered_per_request'])} | "
                     f"{c['verdict']['win_rate_both_legal']} |")
    lines += ['', '## By model and reasoning level', '',
              '| Setting | Format | Legal: baseline → structured | Legality change, pp | Win-rate change when both legal, pp | Legal-team win rate: baseline → structured | Verdicts (legality; win rate) |',
              '|---|---|---:|---:|---:|---:|---|']
    for c in status['cells']:
        v = c['versus_baseline']
        lines.append(f"| {label(c['model'], c['reasoning'])} | {variant_name(c['variant'], status['program'], status.get('constraints'))} | "
                     f"{pct(c['baseline']['legality_rate'])} → {pct(c['structured']['legality_rate'])} | "
                     f"{ci(v['legality'])} | {ci(v['win_rate_both_legal'])} ({v['win_rate_both_legal']['pairs']} pairs) | "
                     f"{pct(c['baseline']['legal_team_win_rate'])} → {pct(c['structured']['legal_team_win_rate'])} | "
                     f"{c['verdict']['legality']}; {c['verdict']['win_rate_both_legal']} |")
    if any('versus_resampled' in c for c in status['cells']):
        lines += ['', '## Against the baseline prompt sampled at the same time', '',
                  'The historical baseline was sampled on 12–13 October 2026. These rows compare with the '
                  'baseline JSON prompt sampled again alongside the structured requests, which removes provider drift.', '',
                  '| Setting | Format | Legality change, pp | Win-rate change when both legal, pp |', '|---|---|---:|---:|']
        for c in status['cells']:
            if 'versus_resampled' in c:
                v = c['versus_resampled']
                lines.append(f"| {label(c['model'], c['reasoning'])} | {variant_name(c['variant'], status['program'], status.get('constraints'))} | "
                             f"{ci(v['legality'])} | {ci(v['win_rate_both_legal'])} |")
    if any('versus_unoptimized' in c for c in status['cells']):
        lines += ['', '## Against the unoptimized DSPy structure (phase one)', '',
                  '| Setting | Format | Legality change, pp | Win-rate change when both legal, pp |', '|---|---|---:|---:|']
        for c in status['cells']:
            if 'versus_unoptimized' in c:
                v = c['versus_unoptimized']
                lines.append(f"| {label(c['model'], c['reasoning'])} | {variant_name(c['variant'], status['program'], status.get('constraints'))} | "
                             f"{ci(v['legality'])} | {ci(v['win_rate_both_legal'])} |")
    lines += ['', '## Why answers were rejected', '',
              '| Setting | Format | ' + ' | '.join(FAILURE_CLASSES) + ' | Legal only after DSPy JSON repair |',
              '|---|---|' + '---:|' * (len(FAILURE_CLASSES) + 1)]
    seen = set()
    for c in status['cells']:
        for name, d in (('baseline', c['baseline']), (c['variant'], c['structured'])):
            key = (c['model'], c['reasoning'], name)
            if key in seen:
                continue
            seen.add(key)
            variant = 'Baseline JSON prompt' if name == 'baseline' else variant_name(name, status['program'], status.get('constraints'))
            lines.append(f"| {label(c['model'], c['reasoning'])} | {variant} | " +
                         ' | '.join(str(d['failures'].get(k, 0)) for k in FAILURE_CLASSES) +
                         f" | {d['legal_after_json_repair'] if name not in ('baseline', RESAMPLED) else '—'} |")
    lines += ['', '## Reading these numbers', '',
              '- Win-rate change when both legal compares only pairs where both answers passed the validator, '
              'so a format that rescues harder masks can change which pairs are compared.',
              '- Wins delivered per request counts an illegal answer as delivering no team (0 wins). It is not '
              'a battle loss; it combines legality and team strength in one unconditional number.',
              '- Verdicts: improved or regressed when the 95% interval excludes zero; "no change beyond ±m pp" when the '
              'interval lies inside the stated margin; otherwise unclear.',
              '- The pool is frozen and visible in every prompt: this is in-pool optimization, not held-out play. '
              'Policy seeds match; Showdown RNG is not seeded. Repeats are not independent starters.']
    return '\n'.join(lines) + '\n'


def headroom(args):
    """How many baseline rejections a reply format could address (published data only)."""
    data = json.loads(args.data.read_text())
    rows = []
    for arm, a in data['mercury']['final']['arms'].items():
        classes = Counter()
        for error, n in a['failures'].items():
            classes[failure_class(error)] += n
        format_only = classes['unreadable answer'] + classes['wrong answer shape']
        rows.append(dict(reasoning=arm, attempts=a['returned'], legal=a['legal'], **classes,
                         ceiling=(a['legal'] + format_only) / a['returned']))
    lines = ['| Mercury 2.5 reasoning | Attempts | Legal | ' + ' | '.join(FAILURE_CLASSES) +
             ' | Legality if every format failure were fixed |', '|---|---:|---:|' + '---:|' * (len(FAILURE_CLASSES) + 1)]
    for r in rows:
        lines.append(f"| {r['reasoning']} | {r['attempts']} | {r['legal']} ({pct(r['legal']/r['attempts'])}) | " +
                     ' | '.join(str(r.get(k, 0)) for k in FAILURE_CLASSES) + f" | {pct(r['ceiling'])} |")
    print('\n'.join(lines))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'battle', 'report', 'headroom'])
    p.add_argument('--output', type=Path)
    p.add_argument('--baseline', type=Path, nargs='+', help='finished reasoning_experiment study folders (prepare)')
    p.add_argument('--adapters', nargs='+', default=['chat', 'json'], choices=['chat', 'json'])
    p.add_argument('--resample-baseline', action='store_true',
                   help='also sample the unchanged baseline request again, to separate structure from provider drift')
    p.add_argument('--provider', nargs='+', default=[], metavar='MODEL=PROVIDER',
                   help='prepare: route a model to another provider when its baseline route is gone (recorded)')
    p.add_argument('--constraints', choices=['legal-choices'],
                   help='prepare: type each requested path with its legal choices (see dspy_structure.legal_completion_type)')
    p.add_argument('--program', type=Path, help='phase two: optimized program from dspy_optimize.py')
    p.add_argument('--reference', type=Path, help='phase two report: the phase-one structured study to compare with')
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--max-concurrency', type=int, default=4)
    p.add_argument('--request-interval', type=float, default=1.0)
    p.add_argument('--max-attempts', type=int, default=6)
    p.add_argument('--workers', type=int, default=6,
                   help='battle: recorded in parallel-battles.json only; the fast evaluator runs its own six workers')
    p.add_argument('--watch', action='store_true', help='battle: keep scoring while generation runs')
    p.add_argument('--win-margin', type=float, default=0.01)
    p.add_argument('--legality-margin', type=float, default=0.05)
    p.add_argument('--data', type=Path, default=ROOT.parent / 'blog/mask-lab/data.json')
    args = p.parse_args()
    if args.command == 'headroom':
        return headroom(args)
    if args.output is None:
        p.error('--output is required')
    args.output = args.output.resolve()
    if args.command == 'prepare':
        if not args.baseline:
            p.error('prepare requires --baseline')
        args.baseline = [b.resolve() for b in args.baseline]
    if args.reference:
        args.reference = args.reference.resolve()
    if args.command == 'battle':
        return battle(args)
    # Same lock name as reasoning_experiment, so its tools see one generation lock.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent / ('.' + args.output.name + '.reasoning.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.command](args)


if __name__ == '__main__':
    main()
