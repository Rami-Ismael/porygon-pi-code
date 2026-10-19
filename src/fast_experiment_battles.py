"""Opt-in persistent fast battles, retaining the comparison's fresh-result contract."""
import atexit
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

PIPELINE = Path('/Users/ramiismael/Documents/yakumsi-vault/Personal Project/Using Advance method search in the whole search of possible vgc pokemon format to find the right counter to a pokemon team/team_completion_pipeline')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Backend:
    def __init__(self, directory, runner_factory=None):
        if runner_factory is None:
            sys.path.insert(0, str(PIPELINE))
            from fast_battle_runner import FastBattleRunner
            runner_factory = FastBattleRunner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Each process gets a new evidence directory, but failures block resumption.
        if list(self.directory.glob('session-*/failed.json')) or list(self.directory.glob('session-*/inflight.json')):
            raise RuntimeError('A fast battle batch failed; inspect its receipts before resuming')
        self.runner = runner_factory(self.directory / f'session-{os.getpid()}-{time.time_ns()}', workers=6, concurrency=24)
        if hasattr(self.runner, 'identity'):
            manifest = self.directory / 'backend.json'
            value = {'identity':self.runner.identity, 'provenance':self.runner.provenance}
            if manifest.exists() and json.loads(manifest.read_text()) != value:
                self.runner.close()
                raise RuntimeError('Fast backend differs from the frozen execution manifest')
            if not manifest.exists():
                with manifest.open('x') as f: json.dump(value,f,indent=2)
        self.pending = None
        self.original_record = None
        atexit.register(self.close)

    def close(self):
        self.runner.close()

    def call(self, con, jobs, seed):
        expected = Counter((j['row'], c) for j in jobs for c in j['schedule'])
        request = {'jobs': jobs, 'seed': seed}
        marker = self.runner.folder / 'inflight.json'
        marker.write_text(json.dumps(request))
        try:
            raw, unattributed, seconds = self.runner(con, jobs, seed=seed)
            actual = Counter()
            merged = defaultdict(lambda: [0, 0, 0])
            for row, col, w, l, t in raw:
                if min(w, l, t) < 0: raise ValueError('Negative battle count')
                actual[row, col] += w+l+t
                for i, value in enumerate((w,l,t)): merged[row,col][i] += value
            if unattributed or actual != expected:
                raise RuntimeError('Fast results do not exactly match requested battles')
            result = [(r,c,*v) for (r,c),v in sorted(merged.items())]
            marker.unlink()
            print(f'fast battles: {sum(actual.values())} completed in {seconds:.2f}s; six persistent workers', flush=True)
            return result, 0, seconds
        except BaseException as exc:
            (self.runner.folder / 'failed.json').write_text(json.dumps({'request':request,'error':str(exc)}))
            self.close()
            raise

    def run_jobs(self, con, jobs, seed=None, conc=50, quiet=False):
        import heldout
        bad = heldout.team_ids(con) & {tid for j in jobs for tid in [j['row'], *j['schedule']]}
        if bad: raise heldout.HeldoutLeak('Heldout teams in fast search evaluation')
        result = self.call(con, jobs, seed)
        self.pending = (id(con), fingerprint(result[0]))
        return result

    def record(self, con, policy_id, results, **kwargs):
        import matchup_db as M
        results = list(results)
        if self.pending == (id(con), fingerprint(results)):
            policy_id = self.runner.policy_id(con, M)
            self.pending = None
        return self.original_record(con, policy_id, results, **kwargs)

    def score(self, jobs, battles=24, conc=50, ports=None, quiet=False, opp_schedule=None,
              seed=None, allow_heldout=False):
        import heldout as H
        import matchup_db as M
        if opp_schedule is None or len(opp_schedule) != battles:
            raise ValueError('Experiment fast finals require the exact predeclared opponent schedule')
        if not allow_heldout:
            H.assert_clear_files([f for f,_ in jobs], 'a scored candidate')
            H.assert_clear_files([o for _,opps in jobs for o in opps], 'an opponent')
            H.assert_clear_files(opp_schedule, 'an opponent')
        con = sqlite3.connect(':memory:')
        con.execute('CREATE TABLE team(team_id INTEGER PRIMARY KEY,paste TEXT,base_key TEXT,format_id TEXT,legal INTEGER)')
        identities = {}
        # Keep the exact file contents, including slot and move order.
        for path in dict.fromkeys([f for f,_ in jobs] + list(opp_schedule)):
            paste = Path(path).read_text()
            key = len(identities)+1
            identities[path] = key
            con.execute('INSERT INTO team VALUES (?,?,?,?,1)', (key,paste,M.species_keys(paste)[1],M.FORMAT))
        requests = []
        bases = dict(con.execute('SELECT team_id,base_key FROM team'))
        for path,_ in jobs:
            schedule, seen = [], {}
            for opponent in opp_schedule:
                col = identities[opponent]
                base = bases[col]
                if base in seen and seen[base] != col:
                    requests.append({'row':identities[path], 'schedule':schedule})
                    schedule, seen = [], {}
                schedule.append(col); seen[base] = col
            if schedule: requests.append({'row':identities[path], 'schedule':schedule})
        try:
            raw, _, _ = self.call(con, requests, seed)
        finally:
            con.close()
        totals = defaultdict(lambda: [0,0])
        for row,_,w,l,t in raw:
            totals[row][0] += w
            totals[row][1] += w+l+t
        results = {}
        for path,_ in jobs:
            w,n = totals[identities[path]]
            if n != battles: raise RuntimeError('Incomplete final evaluation')
            rate=w/n
            results[path] = dict(wins=w,battles=n,win_rate=round(rate,4),
                                 se=round((rate*(1-rate)/n)**.5,4),margin=None,turns=None)
        return results


_backend = None


def install(directory):
    """Install only in the requesting experiment process, between battle calls."""
    global _backend
    if _backend is not None: return _backend
    import matchup_db as M
    import pool
    backend = Backend(Path(directory)/'fast-battles')
    backend.original_record = M.record
    M.run_jobs, M.record, pool.score = backend.run_jobs, backend.record, backend.score
    _backend = backend
    print('Fast battle backend enabled: six workers, concurrency 24; policy and opponent counts preserved.', flush=True)
    return backend
