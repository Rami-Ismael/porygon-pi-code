"""One-time activation inside the existing v4 driver; never starts another driver."""
import os
import sys
import threading

assert os.getpid() == 64782, 'This activation is only for the inspected live v4 driver'
assert 'dibo_comparison_v2.py' in sys.argv[0]


def _activate_fast_v4():
    import hashlib
    import json
    from pathlib import Path
    import time
    root = Path('/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/dibo_v4_vs_cem_20261013')
    try:
        from fast_experiment_battles import install
        before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*_round*_battles_*.json')}
        backend = install(root)
        with (root/'fast-backend-activation.json').open('x') as f:
            json.dump(dict(pid=os.getpid(),enabled_at=time.time(),preserved_receipts=before,
                           identity=backend.runner.identity,
                           note='User requested the fastest battle pipeline. Subsequent batches use six persistent workers at concurrency 24, fast encoder and actor inference, with the same checkpoint, simulator and opponent counts. This changes execution and random-draw ordering partway through v4; do not claim a single unchanged execution configuration.'),f,indent=2)
    except BaseException as exc:
        (root/'fast-backend-activation-failed.json').write_text(json.dumps({'error':repr(exc)}))
        raise


_fast_activation_thread = threading.Thread(target=_activate_fast_v4, name='activate-fast-v4')
_fast_activation_thread.start()
