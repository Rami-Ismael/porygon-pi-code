"""Independently check the authorized repair before the original driver resumes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from corpus import dex_entry
norm = lambda s: ''.join(c for c in s.lower() if c.isalnum())
D = ROOT / 'results/dibo_v4_vs_cem_20261013/stall-20261014'
read = lambda name: json.loads((D / name).read_text())
spec = read('_mshard_64782_1791940735_0.json')
merged = read('recovered-shard-output.json')
snap = next(x for x in read('worker-snapshot.json') if x['function'] == 'run_job')
con = sqlite3.connect('file:/Users/ramiismael/vgc-data/matchup_regmb.sqlite?mode=ro', uri=True)
path_to_col = {str(Path('/Users/ramiismael/vgc-data/pastes') / (h[:20]+'.txt')): col
               for col, h in con.execute('SELECT team_id, canon_hash FROM team')}
con.close()
assert len(merged) == len(spec['jobs']) == 5
saved = Counter()
for battle in snap['a']['battles']:
    if battle['finished']:
        key = ','.join(sorted(''.join(c for c in s.lower() if c.isalnum()) for s in battle['opponent']))
        col = spec['jobs'][0]['key2col'].get(key)
        if col is None:
            base = ','.join(sorted(norm((dex_entry(s) or {}).get('baseSpecies',s)) for s in battle['opponent']))
            col = spec['jobs'][0]['base2col'][base]
        saved[col, 0 if battle['won'] is True else 1 if battle['won'] is False else 2] += 1
assert sum(saved.values()) == 32
for i, (job, result) in enumerate(zip(spec['jobs'], merged)):
    assert result['row'] == job['row']
    assert result['finished'] == result['scheduled'] == len(job['schedule'])
    assert result['unattributed'] == 0 and not result.get('connection_error')
    expected = Counter(path_to_col[p] for p in job['schedule'])
    assert {int(c): sum(v) for c,v in result['by_col'].items()} == dict(expected)
    fresh = read(f'repair-job-{i}.json')
    assert fresh['finished'] == fresh['scheduled'] == (16 if i == 0 else len(job['schedule']))
    for col in result['by_col']:
        for outcome in range(3):
            assert result['by_col'][col][outcome] == fresh['by_col'].get(col,[0,0,0])[outcome] + (saved[int(col),outcome] if i == 0 else 0)
for i in range(1,7):
    name=f'_mshard_64782_1791940735_{i}.out.json'
    assert (D/name).read_bytes() == (Path('/tmp/vgc-pilot')/name).read_bytes()
print('PASS: all 146 results match original opponent counts; 32 preserved, 16 replacements, 98 previously unstarted; six other shard outputs unchanged.')
