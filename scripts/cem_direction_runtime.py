"""Model and durable battle operations for the CEM direction experiment."""

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from reverse_score_diversity import atomic_json, digest
from pokemon_id_diversity import pokemon_id_metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUNTIME = Path('/Users/ramiismael/.local/share/vgc-pilot-runtime')


def phase_seed(seed, iteration, phase):
    return int.from_bytes(hashlib.sha256(f"{seed}/{iteration}/{phase}".encode()).digest()[:4], "little") % (2**31 - 1)


def choose_elites(scores, count, arm, rng):
    import numpy as np
    values = np.asarray(scores, dtype=float)
    if arm not in ("normal", "random", "reverse") or not 1 <= count <= len(values) or not np.isfinite(values).all():
        raise ValueError("invalid elite selection")
    perm = rng.permutation(len(values))
    sign = {"normal": 1., "random": 0., "reverse": -1.}[arm]
    return perm[np.argsort(-(sign * values)[perm], kind="stable")][:count].tolist()


def metrics(records):
    out = {}
    for subset in ("raw", "legal"):
        rows = [r for r in records if subset == "raw" or r["valid"]]
        ids = [r["species"] for r in rows]
        out[subset] = pokemon_id_metrics(ids)
        out[subset]["distinct_rosters"] = len({tuple(sorted(v)) for v in ids})
    out["legality_rate"] = out["legal"]["n"] / len(records) if records else None
    return out


def checked_scores(results, files, battles):
    out = []
    for path in files:
        row = results.get(str(path))
        if not row or row.get("battles") != battles or type(row.get("wins")) is not int or not 0 <= row["wins"] <= battles:
            raise ValueError(f"missing or invalid battle label: {path}")
        out.append(dict(wins=row["wins"], battles=battles, score=row["wins"] / battles))
    return out


class Engine:
    def __init__(self, config):
        import numpy as np
        import torch
        import activesearch as A
        import gradguide as GG
        self.np, self.torch, self.A, self.GG, self.c = np, torch, A, GG, config
        self.corpus_files = A.load_corpus_files()
        self.corpus = [t for _, t in self.corpus_files]
        meta = json.loads(Path(config["checkpoint_metadata"]).read_text())
        if digest(config["checkpoint"]) != meta["checkpoint_sha256"] or len(self.corpus) != meta["corpus_size"]:
            raise ValueError("checkpoint provenance mismatch")
        for p, h in meta["corpus"].items():
            if digest(p) != h:
                raise ValueError(f"training corpus changed: {p}")
        if {p for p, _ in self.corpus_files} != {p for p in meta['corpus'] if len(A.parse_team(p)) == 6}:
            raise ValueError("corpus membership changed")
        for name in ("corpus.py", "encode.py", "diffusion.py", "hpsdiffusion.py"):
            if digest(ROOT / 'src' / name) != meta['source'][str(ROOT / 'src' / name)]:
                raise ValueError("checkpoint architecture or vocabulary changed")
        self.V = A.Vocab(self.corpus)
        self.C = A.D.Constraints(self.V, A.Legality(self.corpus))
        self.look, self.spreads = A.decode_tables(self.corpus)
        self.opponents = A.opponents()
        if len(self.opponents) != 50:
            raise ValueError("expected fifty frozen opponents")
        self.model = A.H.TeamDiffusionHPS(self.V).to(A.D.DEV)
        self.load(config["checkpoint"])
        os.environ['PATH'] = str(Path(config['node']).parent) + os.pathsep + os.environ.get('PATH', '')
        self.validator = A.Validator()

    def load(self, checkpoint):
        self.model.load_state_dict(self.torch.load(checkpoint, map_location=self.A.D.DEV)["sd"])
        self.model.eval()

    def close(self):
        self.validator.close()

    def sample(self, n, seed, label):
        self.torch.manual_seed(seed)
        rng = self.np.random.default_rng(seed)
        self.model.eval()
        rows = []
        for offset in range(0, n, self.c['chunk']):
            size = min(self.c['chunk'], n - offset)
            with self.torch.no_grad():
                x, _ = self.GG.sample_guided(self.model, self.C, None, size, 'none', 0.)
            for tokens in x.cpu().numpy():
                paste = self.A.row_to_paste(self.V, tokens, self.look, self.spreads, rng)
                team = self.A.parse_team_text(paste)
                if len(team) != 6:
                    raise ValueError("decoder did not produce six slots")
                error = self.validator(paste)
                rows.append(dict(index=len(rows), paste=paste, species=list(self.A.species_set(team)),
                                 valid=error is None, validation_error=error))
            print(f"{label}: {len(rows)}/{n} raw attempts", flush=True)
        return dict(seed=seed, records=rows, metrics=metrics(rows))

    def train(self, pastes, seed, path, parent):
        torch, np, A = self.torch, self.np, self.A
        self.load(parent)
        before = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        teams = [A.parse_team_text(p) for p in pastes]
        x = torch.tensor(np.stack([self.V.encode(t) for t in teams]), device=A.D.DEV)
        if (x == 0).any():
            raise ValueError("elite contains an out-of-vocabulary field")
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.c['lr'], weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.c['steps'])
        losses = []
        self.model.train()
        for step in range(self.c['steps']):
            ix = rng.integers(0, len(teams), size=self.c['batch'])
            xb = x[ix].view(-1, A.NSLOT, A.NF)
            order = torch.rand(xb.shape[0], A.NSLOT, device=xb.device).argsort(1)
            xb = torch.gather(xb, 1, order.unsqueeze(-1).expand(-1, -1, A.NF)).reshape(-1, A.D.COLS)
            w = torch.full((len(xb),), A.H.WNULL, device=xb.device, dtype=torch.long)
            loss = self.model.loss(xb, w)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite denoising loss")
            opt.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite training gradient")
            opt.step(); scheduler.step()
            losses.append(float(loss.detach()))
        self.model.eval()
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        delta = math.sqrt(sum(float((state[k] - before[k]).square().sum()) for k in state))
        if not delta > 0:
            raise ValueError("training made no parameter update")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save(dict(sd=state), temporary)
        temporary.replace(path)
        return dict(seed=seed, steps=len(losses), losses=losses, parameter_l2_delta=delta,
                    checkpoint=str(path), checkpoint_sha256=digest(path), parent_checkpoint=str(parent),
                    parent_sha256=digest(parent), training_examples=len(teams))

    def battle(self, pastes, directory, battles, schedule, seed):
        directory.mkdir(parents=True, exist_ok=True)
        files = []
        for i, paste in enumerate(pastes):
            path = directory / f'team-{i:03d}.txt'
            if path.exists() and path.read_text() != paste:
                raise ValueError("battle candidate changed during resume")
            path.write_text(paste)
            files.append(path)
        spec = dict(files={str(p): digest(p) for p in files}, battles=battles,
                    opponents=self.opponents, schedule=schedule, seed=seed, ports=self.c['ports'])
        spec_path = directory / 'manifest.json'
        if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
            raise ValueError("battle manifest changed")
        atomic_json(spec_path, spec)
        for recorded in directory.glob('attempt-*/processes.json'):
            for pid in json.loads(recorded.read_text())['pids']:
                command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
                if 'src/shard.py' in command and str(recorded.parent) in command:
                    raise RuntimeError(f"previous battle worker {pid} is still running; wait before resuming")
        collected = {}
        for output in sorted(directory.glob('attempt-*/shard-*.out.json')):
            try:
                existing = json.loads(output.read_text())
            except json.JSONDecodeError:
                continue
            for file, value in existing.items():
                if file in spec['files']:
                    try:
                        checked_scores({file: value}, [file], battles)
                    except ValueError:
                        continue
                    if file in collected and collected[file] != value:
                        raise ValueError("duplicate conflicting completed battle label")
                    collected[file] = value
        missing = [str(p) for p in files if str(p) not in collected]
        if missing:
            attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))):03d}"
            attempt.mkdir()
            processes = []
            streams = []
            try:
                for i, port in enumerate(self.c['ports'][:len(missing)]):
                    jobs = [(p, self.opponents) for p in missing[i::min(len(missing), len(self.c['ports']))]]
                    inp, out = attempt / f'shard-{i}.json', attempt / f'shard-{i}.out.json'
                    atomic_json(inp, dict(jobs=jobs, battles=battles, conc=min(24, battles), port=port,
                                          opp_schedule=schedule, seed=seed))
                    stream = (attempt / f'shard-{i}.log').open('w'); streams.append(stream)
                    env = dict(os.environ, PYTHONPATH='/tmp/vgc-pilot/vgc-bench:/tmp/vgc-pilot/src', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                    proc = subprocess.Popen([self.c['battle_python'], '/tmp/vgc-pilot/src/shard.py', str(inp), str(out)],
                                            cwd='/tmp/vgc-pilot/vgc-bench', env=env, stdout=stream, stderr=stream,
                                            start_new_session=True)
                    processes.append((proc, out))
                atomic_json(attempt / 'processes.json', dict(pids=[p.pid for p, _ in processes], started=time.time()))
                deadline = time.monotonic() + 600
                while any(p.poll() is None for p, _ in processes):
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"battle phase exceeded ten minutes: {attempt}")
                    time.sleep(1)
                for proc, out in processes:
                    if proc.returncode != 0 or not out.exists():
                        raise RuntimeError(f"battle shard failed; inspect {attempt}")
                    collected.update(json.loads(out.read_text()))
            finally:
                for proc, _ in processes:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                for stream in streams:
                    stream.close()
        labels = checked_scores(collected, files, battles)
        atomic_json(directory / 'scores.json', dict(labels=labels, manifest_sha256=digest(spec_path)))
        return dict(labels=labels, manifest=str(spec_path), manifest_sha256=digest(spec_path),
                    files=[str(p) for p in files], schedule=schedule, seed=seed,
                    total_battles=sum(r['battles'] for r in labels))

    def fingerprints(self):
        files = set(ROOT.joinpath('src').glob('*.py'))
        files.update(Path(p) for p, _ in self.corpus_files)
        files.update(map(Path, self.opponents))
        files.update(ROOT.joinpath('scripts', name) for name in ('cem_score_direction.py', 'cem_direction_runtime.py',
                                                               'pokemon_id_diversity.py', 'reverse_score_diversity.py'))
        files.add(ROOT / 'docs/cem-score-direction-protocol.md')
        files.update(map(Path, (self.c['checkpoint'], self.c['checkpoint_metadata'], '/tmp/bc_100.zip',
                                '/tmp/vgc-pilot/learnset_true.json', '/tmp/vgc-pilot/top50_evs.json')))
        bench = Path('/tmp/vgc-pilot/vgc-bench')
        files.update((bench / 'vgc_bench').rglob('*.py'))
        files.update((bench / 'data').glob('*.json'))
        files.update(Path('/tmp/vgc-pilot/data').glob('*.json'))
        files.update((bench / 'pokemon-showdown/dist').rglob('*.js'))
        files.add(bench / 'pokemon-showdown/validate-teams-batch.js')
        base = {str(p.relative_to(bench / 'pokemon-showdown/dist')): digest(p)
                for p in (bench / 'pokemon-showdown/dist').rglob('*.js')}
        for port in self.c['ports']:
            server = RUNTIME / 'servers' / str(port)
            current = {str(p.relative_to(server / 'dist')): digest(p) for p in (server / 'dist').rglob('*.js')}
            if current != base:
                raise ValueError(f"simulator {port} differs from the validator")
            files.update((server / 'dist').rglob('*.js'))
            files.add(server / 'config/config.js')
        return {str(p): digest(p) for p in sorted(files)}
