"""Independently verify completed CEM trajectories, labels, and ID diversity."""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_reverse_score_diversity import from_paste
from cem_score_direction import ARMS, ROOT, configuration


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(seed, iteration, phase):
    return int.from_bytes(hashlib.sha256(f'{seed}/{iteration}/{phase}'.encode()).digest()[:4], 'little') % (2**31 - 1)


def close(a, b):
    assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)), (a, b)


class Audit:
    def __init__(self, path, review_source_drift=False):
        self.path = path
        self.raw = path.read_bytes()
        self.d = json.loads(self.raw)
        self.c = self.d['manifest']['config']
        self.source_review = None
        if review_source_drift:
            from cem_source_review import review_sources, require_finished
            self.source_review = review_sources(self.d)
            require_finished(path, self.d, self.source_review)
            assert json.loads(path.with_suffix('.source-review.json').read_text()) == self.source_review
        else:
            assert self.d['status'] == 'complete'
        assert self.c == configuration(self.c['smoke'])
        assert self.d['manifest']['schema'] == 'cem-score-direction-v1'
        if self.source_review is None:
            for p, h in self.d['manifest']['inputs'].items():
                assert sha(p) == h, ('changed input', p)
        versions = self.d['manifest']['versions']
        assert versions['python'] == sys.version
        for name in ('torch', 'numpy'):
            assert importlib.metadata.version(name) == versions[name]
        battle_versions = subprocess.check_output([self.c['battle_python'], '-c',
            "import sys,importlib.metadata as m,json;print(json.dumps({'python':sys.version,**{n:m.version(n) for n in ['torch','numpy','poke-env','stable-baselines3']}}))"], text=True).strip()
        assert json.loads(battle_versions) == json.loads(versions['battle_runtime'])
        assert subprocess.check_output([self.c['node'], '--version'], text=True).strip() == self.d['manifest']['versions']['node']
        validator = Path('/tmp/vgc-pilot/vgc-bench/pokemon-showdown/validate-teams-batch.js')
        assert str(validator) in self.d['manifest']['inputs']
        self.validator = subprocess.Popen([self.c['node'], validator.name], cwd=validator.parent,
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.attempts = self.battles = self.steps = self.pairs = self.updates = 0
        self.battle_phases = self.retry_phases = 0
        tree = ast.parse((ROOT / 'src/filterrate.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'elite_indices')
        ns = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'frozen_cem_selector', 'exec'), ns)
        self.original_selector = ns['elite_indices']

    def batch(self, batch, n, expected_seed):
        assert batch['seed'] == expected_seed
        rows = batch['records']
        assert len(rows) == n and [r['index'] for r in rows] == list(range(n))
        for row in rows:
            assert tuple(row['species']) == from_paste(row['paste'])
            assert type(row['valid']) is bool and row['valid'] == (row['validation_error'] is None)
            self.validator.stdin.write(json.dumps({'format': 'gen9championsvgc2026regmb', 'team': row['paste']}) + '\n')
            self.validator.stdin.flush()
            assert json.loads(self.validator.stdout.readline())['valid'] == row['valid']
        self.attempts += n
        for subset in ('raw', 'legal'):
            selected = [r['species'] for r in rows if subset == 'raw' or r['valid']]
            counters = [Counter(ids) for ids in selected]
            count = len(selected)
            pairs = math.comb(count, 2)
            replacements = sum(6 - sum((a & b).values()) for a, b in combinations(counters, 2)) / pairs if pairs else None
            ids = Counter(s for team in selected for s in team)
            total = sum(ids.values())
            effective = math.exp(math.log(total) - sum(f * math.log(f) for f in ids.values()) / total) if total else 0.
            saved = batch['metrics'][subset]
            assert saved['n'] == count and saved['distinct_ids'] == len(ids)
            assert saved['id_slot_counts'] == dict(ids)
            assert saved['distinct_rosters'] == len({tuple(sorted(t)) for t in selected})
            close(saved['effective_ids'], effective)
            close(saved['mean_id_replacements'], replacements)
            close(saved['mean_shared_ids'], 6 - replacements if pairs else None)
            close(saved['pokemon_id_diversity_percent'], 100 * replacements / 6 if pairs else None)
            self.pairs += pairs
        close(batch['metrics']['legality_rate'], sum(r['valid'] for r in rows) / n)

    def battle(self, scores, pastes, n, expected_schedule, expected_seed):
        manifest = Path(scores['manifest'])
        assert sha(manifest) == scores['manifest_sha256']
        spec = json.loads(manifest.read_text())
        files = scores['files']
        assert len(files) == len(pastes) and set(spec['files']) == set(files)
        assert spec['battles'] == n and spec['schedule'] == expected_schedule and spec['seed'] == expected_seed
        assert spec['ports'] == self.c['ports'] and spec['opponents'] == self.d['manifest']['opponents']
        assert scores['schedule'] == expected_schedule and scores['seed'] == expected_seed
        collected = {}
        attempts = sorted(manifest.parent.glob('attempt-*'))
        self.battle_phases += 1
        self.retry_phases += len(attempts) > 1
        for attempt in attempts:
            for inp in attempt.glob('shard-*.json'):
                if inp.name.endswith('.out.json'):
                    continue
                job = json.loads(inp.read_text())
                assert job['battles'] == n and job['seed'] == expected_seed and job['opp_schedule'] == expected_schedule
                assert job['port'] in self.c['ports'] and job['conc'] == min(24, n)
                assigned = {file for file, opps in job['jobs']}
                assert assigned <= set(files) and all(opps == spec['opponents'] for _, opps in job['jobs'])
                out = inp.with_suffix('.out.json')
                if not out.exists():
                    continue
                values = json.loads(out.read_text())
                assert set(values) <= assigned
                for file, row in values.items():
                    if row['battles'] != n:
                        continue
                    assert file not in collected, 'a completed candidate was battled again'
                    collected[file] = row
        assert set(collected) == set(files)
        for file, paste, label in zip(files, pastes, scores['labels']):
            assert Path(file).read_text() == paste and sha(file) == spec['files'][file]
            original = collected[file]
            assert type(original['wins']) is int and 0 <= original['wins'] <= n
            assert label == dict(wins=original['wins'], battles=n, score=original['wins'] / n)
        assert len(scores['labels']) == len(files)
        saved = json.loads((manifest.parent / 'scores.json').read_text())
        assert saved == dict(labels=scores['labels'], manifest_sha256=scores['manifest_sha256'])
        assert scores['total_battles'] == n * len(files)
        self.battles += n * len(files)

    def schedule(self, seed, iteration, phase, n):
        return np.random.default_rng(seed_for(seed, iteration, phase)).choice(self.d['manifest']['opponents'], n, replace=False).tolist()

    def candidates(self, row, seed, iteration):
        c = self.c
        self.batch(row['candidate_batch'], c['candidate_attempts'], seed_for(seed, iteration, 'candidates'))
        rows = row['candidate_batch']['records']
        legal = [r['index'] for r in rows if r['valid']]
        if len(legal) < c['evaluated_candidates']:
            assert row['infeasible'] == dict(reason='insufficient_legal_candidates', legal=len(legal), required=c['evaluated_candidates'], iteration=iteration)
            assert 'scores' not in row
            return False
        expected = np.random.default_rng(seed_for(seed, iteration, 'candidate_pick')).choice(legal, c['evaluated_candidates'], replace=False).tolist()
        assert row['candidate_indices'] == expected
        self.battle(row['scores'], [rows[i]['paste'] for i in expected], c['battles'],
                    self.schedule(seed, iteration, 'opponents', c['battles']), seed_for(seed, iteration, 'policy'))
        return True

    def run(self):
        d, c = self.d, self.c
        assert set(d['shared']) == {str(s) for s in c['seeds']}
        assert set(d['runs']) == {f'seed{s}/{a}' for s in c['seeds'] for a in ARMS}
        for seed in c['seeds']:
            shared = d['shared'][str(seed)]
            self.batch(shared['baseline'], c['evaluation_attempts'], seed_for(seed, 0, 'evaluation'))
            initial_ok = self.candidates(shared['initial'], seed, 1)
            for arm in ARMS:
                run = d['runs'][f'seed{seed}/{arm}']
                assert run['seed'] == seed and run['arm'] == arm
                stop = run.get('stopped', {}).get('iteration')
                assert set(run['iterations']) == {str(i) for i in range(1, (stop or c['iterations']) + 1)}
                parent, parent_hash = c['checkpoint'], sha(c['checkpoint'])
                for iteration in range(1, (stop or c['iterations']) + 1):
                    row = run['iterations'][str(iteration)]
                    if iteration == 1:
                        assert row['shared_candidates'] == str(seed) and 'candidate_batch' not in row
                        source, feasible = shared['initial'], initial_ok
                    else:
                        source, feasible = row, self.candidates(row, seed, iteration)
                    if not feasible:
                        assert stop == iteration and row['status'] == 'infeasible'
                        assert all(run['stopped'][k] == v for k, v in source['infeasible'].items())
                        assert 'training' not in row and 'evaluation' not in row
                        continue
                    assert row['status'] == 'complete'
                    values = np.array([r['score'] for r in source['scores']['labels']])
                    sign = {'normal': 1, 'random': 0, 'reverse': -1}[arm]
                    indices = self.original_selector(sign * values, .2, np.random.default_rng(seed_for(seed, iteration, 'elites')), floor=c['elites']).tolist()
                    selected = row['selection']
                    assert selected['score_indices'] == indices
                    assert selected['raw_indices'] == [source['candidate_indices'][i] for i in indices]
                    assert selected['original_scores'] == values[indices].tolist()
                    assert selected['ranking_scores'] == (sign * values[indices]).tolist()
                    assert selected['seed'] == seed_for(seed, iteration, 'elites')
                    train = row['training']
                    assert train['parent_checkpoint'] == parent and train['parent_sha256'] == parent_hash
                    assert train['steps'] == c['steps'] and len(train['losses']) == c['steps']
                    assert train['training_examples'] == c['elites'] and train['seed'] == seed_for(seed, iteration, 'train')
                    assert train['parameter_l2_delta'] > 0 and all(math.isfinite(v) for v in train['losses'])
                    assert sha(train['checkpoint']) == train['checkpoint_sha256']
                    parent, parent_hash = train['checkpoint'], train['checkpoint_sha256']
                    self.steps += c['steps']; self.updates += 1
                    self.batch(row['evaluation'], c['evaluation_attempts'], seed_for(seed, iteration, 'evaluation'))
                if stop:
                    assert 'quality' not in run
                    continue
                quality = run['quality']
                records = run['iterations'][str(c['iterations'])]['evaluation']['records']
                legal = [r['index'] for r in records if r['valid']]
                if len(legal) < c['quality_teams']:
                    assert quality['status'] == 'unavailable'
                else:
                    assert quality['status'] == 'complete'
                    idx = np.random.default_rng(seed_for(seed, c['iterations'], 'quality_pick')).choice(legal, c['quality_teams'], replace=False).tolist()
                    assert quality['indices'] == idx
                    self.battle(quality['scores'], [records[i]['paste'] for i in idx], c['quality_battles'],
                                self.schedule(seed, c['iterations'], 'quality_opponents', c['quality_battles']),
                                seed_for(seed, c['iterations'], 'quality_policy'))
                print(f'Verified seed{seed}/{arm}', flush=True)
        summary_path = self.path.with_suffix('.analysis.json')
        summary = json.loads(summary_path.read_text())
        assert summary.get('source_review') == self.source_review
        assert summary['raw_result_sha256'] == hashlib.sha256(self.raw).hexdigest()
        assert summary['analysis_source_sha256'] == sha(ROOT / 'scripts/report_cem_score_direction.py')
        assert (summary['total_raw_attempts'], summary['total_recorded_battles'], summary['total_training_steps'], summary['completed_updates']) == (self.attempts, self.battles, self.steps, self.updates)
        for arm in ARMS:
            for iteration, series in enumerate(summary['series'][arm]):
                expected_seeds = []
                for seed in c['seeds']:
                    source = d['shared'][str(seed)]['baseline'] if iteration == 0 else d['runs'][f'seed{seed}/{arm}']['iterations'].get(str(iteration), {}).get('evaluation')
                    m = source['metrics'] if source else None
                    expected = dict(legal_id_diversity=m['legal']['pokemon_id_diversity_percent'] if m else None,
                                    raw_id_diversity=m['raw']['pokemon_id_diversity_percent'] if m else None,
                                    legality_percent=100 * m['legality_rate'] if m else None,
                                    legal_effective_ids=m['legal']['effective_ids'] if m else None,
                                    legal_distinct_ids=m['legal']['distinct_ids'] if m else None,
                                    legal_distinct_rosters=m['legal']['distinct_rosters'] if m else None)
                    for metric, value in expected.items():
                        close(series['seeds'][str(seed)][metric], value)
                    expected_seeds.append(expected)
                for metric in expected_seeds[0]:
                    values = [s[metric] for s in expected_seeds]
                    close(series['means'][metric], sum(values) / len(values) if all(v is not None for v in values) else None)
            changes, quality = [], []
            for seed in c['seeds']:
                first = summary['series'][arm][0]['seeds'][str(seed)]['legal_id_diversity']
                last = summary['series'][arm][-1]['seeds'][str(seed)]['legal_id_diversity']
                changes.append(last - first if first is not None and last is not None else None)
                result = d['runs'][f'seed{seed}/{arm}'].get('quality', {})
                labels = result.get('scores', {}).get('labels', [])
                quality.append(sum(v['score'] for v in labels) / len(labels) if result.get('status') == 'complete' else None)
            for actual, expected in zip(summary['change_from_baseline_pp'][arm]['seed_changes_percentage_points'], changes):
                close(actual, expected)
            close(summary['change_from_baseline_pp'][arm]['mean_percentage_points'], sum(changes) / len(changes) if all(v is not None for v in changes) else None)
            for actual, expected in zip(summary['fresh_final_quality'][arm]['seed_win_rates'], quality):
                close(actual, expected)
            close(summary['fresh_final_quality'][arm]['mean_win_rate'], sum(quality) / len(quality) if all(v is not None for v in quality) else None)
        for arm in ('normal', 'random'):
            diffs = []
            for seed in c['seeds']:
                a = summary['series']['reverse'][-1]['seeds'][str(seed)]['legal_id_diversity']
                b = summary['series'][arm][-1]['seeds'][str(seed)]['legal_id_diversity']
                diffs.append(a - b if a is not None and b is not None else None)
            comparison = summary['final_reverse_minus_comparator_pp'][arm]
            assert comparison['differences'] == diffs
            mean = sum(diffs) / len(diffs) if all(v is not None for v in diffs) else None
            close(comparison['mean'], mean)
            if len(diffs) == 3 and mean is not None:
                lo, hi = 0., 100.
                for _ in range(80):
                    mid = (lo + hi) / 2
                    if .5 + mid / (2 * math.sqrt(mid * mid + 2)) < .9875: lo = mid
                    else: hi = mid
                half = (lo + hi) / 2 * math.sqrt(sum((v - mean) ** 2 for v in diffs) / 6)
                for actual, expected in zip(comparison['ci95_two_comparisons'], (mean - half, mean + half)):
                    close(actual, expected)
        if not summary['stopped'] and all(r['quality']['status'] == 'complete' for r in d['runs'].values()):
            assert (self.attempts, self.battles, self.steps) == ((132, 60, 12) if c['smoke'] else (19776, 103680, 2880))
        if not c['smoke']:
            archive = ROOT / 'results/cem_score_direction_source'
            archived = json.loads((archive / 'manifest.json').read_text())
            assert archived['experiment_manifest'] == d['manifest']
            for relative, expected in archived['files'].items():
                assert sha(archive / relative) == expected
        return dict(status='passed_with_reviewed_source_deviation' if self.source_review else 'passed',
                    source_review=self.source_review, smoke=c['smoke'], raw_result_sha256=hashlib.sha256(self.raw).hexdigest(),
                    analysis_sha256=sha(summary_path), audit_source_sha256=sha(__file__),
                    inputs_checked=len(d['manifest']['inputs']), raw_attempts=self.attempts,
                    recorded_battles=self.battles, optimizer_steps=self.steps, completed_updates=self.updates,
                    explicit_team_pairs_checked=self.pairs, all_proposals_revalidated=True,
                    battle_phases=self.battle_phases, battle_phases_with_retries=self.retry_phases,
                    verified_at=datetime.now(timezone.utc).isoformat())

    def close(self):
        self.validator.stdin.close(); self.validator.wait()
        assert self.validator.returncode == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--review-source-drift', action='store_true')
    args = parser.parse_args()
    path = args.results.resolve()
    audit = Audit(path, args.review_source_drift)
    try:
        receipt = audit.run()
    finally:
        audit.close()
    path.with_suffix('.verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
