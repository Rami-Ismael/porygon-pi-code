"""One authorized repair of shard 0; never relaunch or overwrite results."""
import asyncio
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / 'results/dibo_v4_vs_cem_20261013/stall-20261014'
sys.path.insert(0, str(ROOT / 'src'))
from showdown_validator import VGC_BENCH
BENCH = str(VGC_BENCH)
os.chdir(BENCH)
sys.path[:0] = [str(ROOT / 'src'), BENCH]
import pair_shard as P


def prepare():
    spec = json.loads((DIR / '_mshard_64782_1791940735_0.json').read_text())
    snap = next(x for x in json.loads((DIR / 'worker-snapshot.json').read_text()) if x['function'] == 'run_job')
    job = spec['jobs'][0]
    assert snap['job'] == job
    a, b = snap['a']['battles'], snap['b']['battles']
    assert len(a) == len(b) == 48
    opposite = {x['tag']: x for x in b}
    con = sqlite3.connect('file:/Users/ramiismael/vgc-data/matchup_regmb.sqlite?mode=ro', uri=True)
    paths = {}
    for col in set(job['key2col'].values()):
        paste, h = con.execute('SELECT paste, canon_hash FROM team WHERE team_id=?', (col,)).fetchone()
        path = Path('/Users/ramiismael/vgc-data/pastes') / (h[:20] + '.txt')
        assert path.read_text() == paste
        paths[col] = str(path)
    con.close()
    complete, lost, counts = [], [], defaultdict(lambda: [0, 0, 0])
    for battle in a:
        other = opposite[battle['tag']]
        assert battle['finished'] == other['finished']
        key = ','.join(sorted(P._norm(s) for s in battle['opponent']))
        col = job['key2col'].get(key)
        if col is None:
            col = job['base2col'][','.join(sorted(P._base(s) for s in battle['opponent']))]
        if battle['finished']:
            assert (battle['won'] is None and other['won'] is None) or battle['won'] is not other['won']
            counts[str(col)][0 if battle['won'] is True else 1 if battle['won'] is False else 2] += 1
            complete.append(paths[col])
        else:
            lost.append(paths[col])
    assert len(complete) == 32 and len(lost) == 16
    assert Counter(complete + lost) == Counter(job['schedule'])
    # Preserve original opponent ordering among the remaining occurrences.
    need = Counter(lost)
    schedule = []
    for path in job['schedule']:
        if need[path]: schedule.append(path); need[path] -= 1
    return spec, dict(job, schedule=schedule), dict(counts)


def main():
    spec, replacement, preserved = prepare()
    if '--plan' in sys.argv:
        print(json.dumps(dict(preserved=32, replacements=16, unstarted=sum(len(j['schedule']) for j in spec['jobs'][1:]), extra_attempts=16)))
        return
    with (DIR / 'repair.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Exclusive marker deliberately prevents automatic retries after any failure.
        with (DIR / 'repair-started.json').open('x') as f:
            json.dump(dict(pid=os.getpid(), authorized_extra_attempts=16,
                           reason='User approved replacing only the 16 ping-timeout interrupted battles on 2026-10-14',
                           worker_sha256=hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest()), f)
        pol = P.PPO.load('/tmp/bc_100.zip', device=P.torch.device('cpu')).policy
        results = []
        for index, job in enumerate([replacement, *spec['jobs'][1:]]):
            result = asyncio.run(P.run_job(pol, job, spec['conc'], spec['port'], spec['seed']))
            with (DIR / f'repair-job-{index}.json').open('x') as f: json.dump(result, f, indent=2)
            assert result['finished'] == len(job['schedule']) and result['unattributed'] == 0
            assert not result.get('connection_error')
            if index == 0:
                for col, values in preserved.items():
                    target = result['by_col'].setdefault(col, [0, 0, 0])
                    for k in range(3): target[k] += values[k]
                result['finished'] += 32
                result['scheduled'] += 32
            results.append(result)
            print(f'job {index}: {result["finished"]}/{result["scheduled"]} saved', flush=True)
        assert sum(j['finished'] for j in results) == 146
        with (DIR / 'recovered-shard-output.json').open('x') as f: json.dump(results, f, indent=2)
        print('Repair complete; awaiting independent validation before releasing original driver.', flush=True)


if __name__ == '__main__': main()
