"""Parallelize independent exact battle panels, optionally as generation arrives."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import repo_paths  # puts src/ first on sys.path
import battles
import results
import showdown_validator
import reasoning_experiment as E


def battle_design(m):
    return {k: m.get(k) for k in ('format', 'seed', 'battles', 'opponents',
                                 'seed_limit', 'exclude_starter_team_ids')}


def plan(runs, canonical):
    """One group per canonical team AND schedule across every arm/repeat."""
    groups = {}
    design = battle_design(runs[0][2])
    for entry, folder, m in runs:
        if battle_design(m) != design:
            raise ValueError('battle design differs across runs')
        for trial in m['trials']:
            row = E.completion(folder, entry, trial)
            if row is None or not row['valid']:
                continue
            key = results.digest(dict(team=canonical(row['team']),
                                schedule=battles.schedule_for(m, trial['start'])))
            group = groups.setdefault(key, dict(key=key, team=row['team'], start=trial['start'], destinations=[]))
            group['destinations'].append(str(folder / 'labels' / ('llm-' + trial['id'] + '.json')))
    return [g for g in groups.values() if any(not Path(p).exists() for p in g['destinations'])]


def publish(group, result, config):
    """Serialize config publication and never replace an existing result."""
    if result['battles'] != 50 or not 0 <= result['wins'] <= 50:
        raise ValueError('incomplete or invalid panel')
    for destination in group['destinations']:
        destination = Path(destination)
        folder = destination.parent.parent
        with (folder / '.battle-publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            config_path = folder / 'battle-config.json'
            old_config = E.read(config_path)
            if old_config is not None and old_config != config:
                raise ValueError(f'battle configuration changed: {folder}')
            old = E.read(destination)
            if old is not None and old != result:
                raise ValueError(f'refusing to replace saved battle result: {destination}')
            if old_config is None:
                results.save(config_path, config)
            if old is None:
                results.save(destination, result)


def archive_legacy_study(output, runs):
    """Run folders battled on the removed legacy engine are archived before planning,
    so the whole study is re-battled on the fast evaluator; the study summaries
    computed from the old labels move with them."""
    legacy = any(battles.recorded_backend(folder) == 'legacy' for _, folder, _ in runs)
    for _, folder, _ in runs:
        battles.archive_legacy_labels(folder)
    if legacy:
        archive = Path(output) / f"archive-legacy-battles-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        archive.mkdir()
        for name in ('status.json', 'report.md', 'parallel-battles.json'):
            if (Path(output) / name).exists():
                (Path(output) / name).rename(archive / name)


def fast_worker(output, manifest, groups):
    """All groups through one persistent six-worker evaluator, in this process."""
    folder = Path(output) / 'battle-fast'
    results.save(folder / 'manifest.json', manifest)
    by_label = {f"p{g['key']}-s{g['start']}": g for g in groups}
    def finished(label, result):
        publish(by_label[label], result, E.read(folder / 'battle-config.json'))
    battles.battle(SimpleNamespace(output=folder),
                   candidates_override=[(label, g['team']) for label, g in by_label.items()],
                   on_result=finished)
    return len(groups)


def score(output):
    _, runs = E.load_runs(output)
    archive_legacy_study(output, runs)
    import hps_generate as h, matchup_db as db
    groups = plan(runs, lambda team: db.canonical_paste(showdown_validator.paste(team, h)))
    # A team's first insert sets its origin: record this study, not the evaluator's battle-fast folder.
    con = db.connect()
    validator = h.Validator()
    try:
        for group in groups:
            db.add_team(con, showdown_validator.paste(group['team'], h), 'reasoning-parallel:' + output.name, validator)
    finally:
        validator.close()
        con.close()
    return fast_worker(output, runs[0][2], groups) if groups else 0


def progress(output):
    study, runs = E.load_runs(output)
    returned = legal = scored = 0
    for entry, folder, m in runs:
        for trial in m['trials']:
            row = E.completion(folder, entry, trial)
            if row is not None:
                returned += 1
                if row['valid']:
                    legal += 1
                    scored += (folder / 'labels' / ('llm-' + trial['id'] + '.json')).exists()
    return dict(returned=returned, planned=study['planned_calls'], legal=legal, scored=scored,
                complete=returned == study['planned_calls'] and scored == legal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4,
                        help='recorded in parallel-battles.json only: the fast evaluator runs its own six workers')
    parser.add_argument('--watch', action='store_true', help='score new legal responses as generation continues')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be between 1 and 32')
    args.output = args.output.resolve()
    # Separate from the generation lock; only one scorer can publish at a time.
    with (args.output / '.parallel-battles.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                state = dict(progress(args.output), workers=args.workers, stage='scoring', updated_at=time.time())
                results.save(args.output / 'parallel-battles.json', state)
                panels = score(args.output)
                state = dict(progress(args.output), workers=args.workers, updated_at=time.time())
                state['stage'] = 'complete' if state['complete'] else 'waiting_for_generation'
                results.save(args.output / 'parallel-battles.json', state)
                print(json.dumps(dict(state, new_panels=panels)), flush=True)
                if state['complete'] or not args.watch:
                    # An older generation process may still write its end-of-run
                    # report. Wait for it before publishing the final report.
                    with (args.output.parent / ('.' + args.output.name + '.reasoning.lock')).open('a') as generation_lock:
                        fcntl.flock(generation_lock, fcntl.LOCK_EX)
                        E.report(SimpleNamespace(output=args.output))
                    break
                time.sleep(30)
        except Exception as exc:
            results.save(args.output / 'parallel-battles.json', dict(stage='failed', error=f'{type(exc).__name__}: {exc}', updated_at=time.time()))
            raise


if __name__ == '__main__':
    main()
