"""Run cumulative CEM with normal, random, or negative-score elite ranking."""

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from cem_direction_runtime import Engine, ROOT, RUNTIME, phase_seed, choose_elites
from reverse_score_diversity import atomic_json, digest

ARMS = ('normal', 'random', 'reverse')


def now():
    return datetime.now(timezone.utc).isoformat()


def configuration(smoke=False):
    c = dict(seeds=[2101, 2202, 2303], iterations=5, candidate_attempts=192,
             evaluated_candidates=96, elites=20, evaluation_attempts=256,
             steps=64, batch=64, lr=1e-4, chunk=48, battles=24,
             quality_teams=32, quality_battles=48, ports=list(range(8134, 8142)),
             checkpoint=str(ROOT / 'results/temperature_p0.pt'),
             checkpoint_metadata=str(ROOT / 'results/temperature_p0.training.json'),
             node='/Users/ramiismael/.nvm/versions/node/v22.22.0/bin/node',
             battle_python='/tmp/vgc-pilot/.venv/bin/python', smoke=smoke,
             decoding_temperature=1., sampling_guidance=0., entropy_bonus=0.,
             optimizer='AdamW', weight_decay=.01, gradient_clip=1.,
             schedule='cosine per optimizer step; reset each iteration',
             conditioning='WNULL', training_start='previous iteration checkpoint',
             selection_pool='current iteration only', candidate_selection='uniform legal subset')
    if smoke:
        c.update(seeds=[9903], iterations=2, candidate_attempts=12, evaluated_candidates=6,
                 elites=2, evaluation_attempts=12, steps=2, batch=4, chunk=12,
                 battles=2, quality_teams=2, quality_battles=2, ports=[8134, 8135])
    return c


def budget(c):
    seeds, arms, rounds = len(c['seeds']), len(ARMS), c['iterations']
    candidate_batches = seeds * (1 + arms * (rounds - 1))
    return dict(candidate_batches=candidate_batches,
                candidate_attempts=candidate_batches * c['candidate_attempts'],
                evaluation_attempts=seeds * (1 + arms * rounds) * c['evaluation_attempts'],
                ranking_battles=candidate_batches * c['evaluated_candidates'] * c['battles'],
                quality_battles=seeds * arms * c['quality_teams'] * c['quality_battles'],
                training_steps=seeds * arms * rounds * c['steps'])


class Run:
    def __init__(self, engine, path, data):
        self.e, self.c, self.path, self.d = engine, engine.c, path, data
        self.work = path.parent / (path.stem + '_work')

    def save(self):
        atomic_json(self.path, self.d)

    def active(self, stage, **context):
        self.d['active'] = dict(stage=stage, **context, at=now())
        self.save()
        print(json.dumps(self.d['active']), flush=True)

    def schedule(self, seed, iteration, phase, n):
        rng = self.e.np.random.default_rng(phase_seed(seed, iteration, phase))
        return rng.choice(self.e.opponents, n, replace=False).tolist()

    def candidates(self, row, parent, seed, iteration, label, directory):
        c = self.c
        if 'candidate_batch' not in row:
            self.active('generate_candidates', label=label, seed=seed, iteration=iteration)
            self.e.load(parent)
            row['candidate_batch'] = self.e.sample(c['candidate_attempts'], phase_seed(seed, iteration, 'candidates'), label)
            self.save()
        records = row['candidate_batch']['records']
        legal = [r['index'] for r in records if r['valid']]
        if len(legal) < c['evaluated_candidates']:
            row['infeasible'] = dict(reason='insufficient_legal_candidates', legal=len(legal),
                                      required=c['evaluated_candidates'], iteration=iteration)
            self.save()
            return False
        if 'candidate_indices' not in row:
            rng = self.e.np.random.default_rng(phase_seed(seed, iteration, 'candidate_pick'))
            row['candidate_indices'] = rng.choice(legal, c['evaluated_candidates'], replace=False).tolist()
            self.save()
        if 'scores' not in row:
            self.active('battle_candidates', label=label, seed=seed, iteration=iteration)
            pastes = [records[i]['paste'] for i in row['candidate_indices']]
            row['scores'] = self.e.battle(pastes, directory, c['battles'],
                                           self.schedule(seed, iteration, 'opponents', c['battles']),
                                           phase_seed(seed, iteration, 'policy'))
            self.save()
        return True

    def run(self):
        c = self.c
        for seed_index, seed in enumerate(c['seeds']):
            shared = self.d['shared'].setdefault(str(seed), {})
            if 'baseline' not in shared:
                self.active('baseline', seed=seed)
                self.e.load(c['checkpoint'])
                shared['baseline'] = self.e.sample(c['evaluation_attempts'], phase_seed(seed, 0, 'evaluation'), f'seed{seed}/baseline')
                self.save()
            initial = shared.setdefault('initial', {})
            self.candidates(initial, c['checkpoint'], seed, 1, f'seed{seed}/shared', self.work / f'seed{seed}' / 'shared-battles')
            for arm in ARMS:
                self.d['runs'].setdefault(f'seed{seed}/{arm}', dict(seed=seed, arm=arm, iterations={}))
            self.save()
            for iteration in range(1, c['iterations'] + 1):
                start = (seed_index + iteration - 1) % len(ARMS)
                order = ARMS[start:] + ARMS[:start]
                for arm in order:
                    key = f'seed{seed}/{arm}'
                    run = self.d['runs'][key]
                    if 'stopped' in run:
                        continue
                    row = run['iterations'].setdefault(str(iteration), {})
                    if row.get('status') == 'complete':
                        continue
                    label = f'{key}/iteration{iteration}'
                    directory = self.work / f'seed{seed}' / arm / f'iteration{iteration}'
                    parent = c['checkpoint'] if iteration == 1 else run['iterations'][str(iteration - 1)]['training']['checkpoint']
                    expected_parent = self.d['manifest']['inputs'][str(Path(c['checkpoint']))] if iteration == 1 else run['iterations'][str(iteration - 1)]['training']['checkpoint_sha256']
                    if digest(parent) != expected_parent:
                        raise ValueError("parent checkpoint changed")
                    source = initial if iteration == 1 else row
                    if iteration == 1:
                        row['shared_candidates'] = str(seed)
                        feasible = 'infeasible' not in initial
                    else:
                        feasible = self.candidates(row, parent, seed, iteration, label, directory / 'battles')
                    if not feasible:
                        run['stopped'] = dict(source['infeasible'], at=now())
                        row['status'] = 'infeasible'
                        self.save()
                        continue
                    scores = [r['score'] for r in source['scores']['labels']]
                    if 'selection' not in row:
                        rng = self.e.np.random.default_rng(phase_seed(seed, iteration, 'elites'))
                        idx = choose_elites(scores, c['elites'], arm, rng)
                        raw_indices = [source['candidate_indices'][i] for i in idx]
                        sign = {'normal': 1., 'random': 0., 'reverse': -1.}[arm]
                        row['selection'] = dict(score_indices=idx, raw_indices=raw_indices,
                                                 original_scores=[scores[i] for i in idx],
                                                 ranking_scores=[sign * scores[i] for i in idx],
                                                 seed=phase_seed(seed, iteration, 'elites'))
                        self.save()
                    if 'training' not in row:
                        self.active('train', label=label, seed=seed, arm=arm, iteration=iteration)
                        pastes = [source['candidate_batch']['records'][i]['paste'] for i in row['selection']['raw_indices']]
                        row['training'] = self.e.train(pastes, phase_seed(seed, iteration, 'train'), directory / 'model.pt', parent)
                        self.save()
                    training = row['training']
                    if digest(training['checkpoint']) != training['checkpoint_sha256']:
                        raise ValueError('trained checkpoint changed')
                    if 'evaluation' not in row:
                        self.active('evaluate_diversity', label=label, seed=seed, arm=arm, iteration=iteration)
                        self.e.load(training['checkpoint'])
                        row['evaluation'] = self.e.sample(c['evaluation_attempts'], phase_seed(seed, iteration, 'evaluation'), label + '/evaluation')
                        self.save()
                    row['status'], row['completed_at'] = 'complete', now()
                    self.save()
            for arm in ARMS:
                run = self.d['runs'][f'seed{seed}/{arm}']
                if 'stopped' in run or 'quality' in run:
                    continue
                records = run['iterations'][str(c['iterations'])]['evaluation']['records']
                legal = [r['index'] for r in records if r['valid']]
                if len(legal) < c['quality_teams']:
                    run['quality'] = dict(status='unavailable', reason='insufficient_legal_evaluation_outputs')
                    self.save()
                    continue
                rng = self.e.np.random.default_rng(phase_seed(seed, c['iterations'], 'quality_pick'))
                idx = rng.choice(legal, c['quality_teams'], replace=False).tolist()
                self.active('final_quality', seed=seed, arm=arm)
                scores = self.e.battle([records[i]['paste'] for i in idx], self.work / f'seed{seed}' / arm / 'quality',
                                      c['quality_battles'], self.schedule(seed, c['iterations'], 'quality_opponents', c['quality_battles']),
                                      phase_seed(seed, c['iterations'], 'quality_policy'))
                run['quality'] = dict(status='complete', indices=idx, scores=scores)
                self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    c = configuration(args.smoke)
    path = (args.output or ROOT / 'results' / ('cem_score_direction_smoke.json' if args.smoke else 'cem_score_direction.json')).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    for port in c['ports']:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            pass
    with path.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(c)
        data = None
        try:
            inputs = engine.fingerprints()
            versions = {name: importlib.metadata.version(name) for name in ('torch', 'numpy')}
            versions['python'] = sys.version
            versions['node'] = subprocess.check_output([c['node'], '--version'], text=True).strip()
            versions['battle_runtime'] = subprocess.check_output([c['battle_python'], '-c',
                "import sys,importlib.metadata as m,json;print(json.dumps({'python':sys.version,**{n:m.version(n) for n in ['torch','numpy','poke-env','stable-baselines3']}}))"], text=True).strip()
            manifest = dict(schema='cem-score-direction-v1', config=c, arms=list(ARMS), budget=budget(c),
                            inputs=inputs, opponents=engine.opponents, versions=versions,
                            device=str(engine.A.D.DEV))
            if path.exists():
                data = json.loads(path.read_text())
                if data['manifest'] != manifest:
                    raise ValueError('configuration or input fingerprints changed; cannot resume')
            else:
                data = dict(manifest=manifest, status='running', created_at=now(), shared={}, runs={})
                atomic_json(path, data)
            if data['status'] == 'complete':
                print('Experiment already complete; no generation, training, or battles repeated.', flush=True)
                return
            run = Run(engine, path, data)
            run.run()
            for p, expected in inputs.items():
                if digest(p) != expected:
                    raise ValueError(f'input changed during experiment: {p}')
            data.update(status='complete', completed_at=now(), active=None)
            data.pop('last_error', None)
            atomic_json(path, data)
            print(f'CEM_DIRECTION_COMPLETE {path}', flush=True)
        except Exception as error:
            if data is not None and data.get('status') != 'complete':
                data['last_error'] = dict(type=type(error).__name__, message=str(error), at=now())
                atomic_json(path, data)
            raise
        finally:
            engine.close()


if __name__ == '__main__':
    main()
