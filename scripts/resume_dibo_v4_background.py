"""One-shot, launchd-owned recovery of the frozen October 13 DiBO v4 run.

No retry loop and no protocol changes. An incomplete battle receipt is a hard
stop. The launchd job owns its simulator children and writes durable logs.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/dibo_v4_vs_cem_20261013'
PYTHON = '/tmp/vgc-pilot/.venv/bin/python'
NODE = '/Users/ramiismael/.nvm/versions/node/v22.22.0/bin/node'
SERVERS = Path.home() / '.local/share/vgc-pilot-runtime/servers'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(stage, **fields):
    path = RUN / 'recovery-status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(stage=stage, updated_at=now(), supervisor_pid=os.getpid(), **fields), indent=2)+'\n')
    temp.replace(path)


def ready(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.3):
            return True
    except OSError:
        return False


def validate_saved_state(root):
    incomplete = list(root.glob('*incomplete*.json'))
    if incomplete:
        raise RuntimeError('Incomplete battle receipts require manual reconciliation: '+str(incomplete))
    snapshot = {}
    for path in sorted(root.glob('*_round*_battles_*.json')):
        snapshot[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not snapshot:
        raise RuntimeError('Expected saved battle receipts for this recovery')
    for arm in ('cem','dibo'):
        data=json.loads((root/f'{arm}.json').read_text())
        for generation,result in data['rounds'].items():
            if 'selected' in result and 'evaluated' not in result:
                raise RuntimeError(f'{arm} round {generation} has pending battles; reconcile interruption before recovery')
    return snapshot


def main():
    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
             PATH='/Users/ramiismael/.nvm/versions/node/v22.22.0/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin')
    children=[]; driver=None; wake=None
    with (RUN/'recovery-supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            before=validate_saved_state(RUN)
            # Do not race a manually resumed driver or another simulator owner.
            with (RUN/'run.lock').open('a') as driver_lock:
                fcntl.flock(driver_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            occupied=[port for port in range(8123,8130) if ready(port)]
            if occupied:
                raise RuntimeError(f'Simulator ports already owned; inspect before recovery: {occupied}')
            write_status('preflight')
            with (RUN/'recovery-audit.log').open('a') as log:
                subprocess.run([PYTHON,'scripts/audit_dibo_comparison.py',str(RUN)],cwd=ROOT,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            baseline=RUN/'recovery-receipt-baseline.json'
            if baseline.exists():
                saved=json.loads(baseline.read_text())
                if any(before.get(k)!=v for k,v in saved.items()):
                    raise RuntimeError('Previously saved battle receipts changed')
            else:
                baseline.write_text(json.dumps(before,indent=2)+'\n')
            for port in range(8123,8130):
                with (RUN/f'recovery-server-{port}.log').open('a') as log:
                    child=subprocess.Popen([NODE,'pokemon-showdown','start',str(port),'--no-security'],
                        cwd=SERVERS/str(port+11),env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child)
            deadline=time.monotonic()+45
            while not all(ready(port) for port in range(8123,8130)):
                if any(p.poll() is not None for p in children) or time.monotonic()>deadline:
                    raise RuntimeError('Simulator startup failed; no experiment started')
                time.sleep(1)
            with (RUN/'recovery-driver.log').open('a') as log:
                log.write('\nRecovery started '+now()+'\n');log.flush()
                driver=subprocess.Popen([PYTHON,'-u','src/dibo_comparison_v2.py','run','--directory',str(RUN)],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True)
            wake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            sys.path.insert(0,str(ROOT/'scripts'))
            from finalize_dibo_v4_comparison import progress
            while driver.poll() is None:
                if any(p.poll() is not None for p in children):
                    raise RuntimeError('An owned simulator stopped; refusing to continue with a partial server pool')
                write_status('running',driver_pid=driver.pid,server_pids=[p.pid for p in children])
                progress(RUN)
                time.sleep(30)
            if driver.returncode:
                raise RuntimeError(f'Driver exited {driver.returncode}; inspect recovery-driver.log, no retry attempted')
            if not json.loads((RUN/'summary.json').read_text()).get('complete'):
                raise RuntimeError('Driver exited without a complete experiment')
            for name,digest in before.items():
                if hashlib.sha256((RUN/name).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('An existing receipt changed: '+name)
            write_status('finalizing',driver_pid=driver.pid)
            with (RUN/'recovery-finalizer.log').open('a') as log:
                subprocess.run([PYTHON,'scripts/finalize_dibo_v4_comparison.py',str(RUN)],cwd=ROOT,env=env,
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            write_status('complete',existing_receipts_preserved=len(before))
        except Exception as exc:
            write_status('failed',error=str(exc),driver_pid=driver.pid if driver else None)
            raise
        finally:
            owned=([driver] if driver else [])+children
            for process in owned:
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
            for process in owned:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            if wake and wake.poll() is None:
                wake.terminate()


if __name__=='__main__':
    main()
