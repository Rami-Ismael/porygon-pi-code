"""Narrow, explicit review of the three unrelated edits observed during CEM.

The original raw result and frozen manifest are never rewritten. This is a
post-hoc provenance deviation, not a successful strict source-freeze check.
"""

import ast
import fcntl
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/cem_score_direction_source'
KNOWN = {
    'src/entropyloop.py': '82dfda7bca0e41978a2c3402dc463c807085c12d8fdd93b80bc1f8fe8380cfad',
    'src/gradguide.py': '3d2c18602aae55cc1ba67a218de8c19302f254c98c8ba511c6b292f7ec9f570e',
    'src/temperature_experiment.py': '38da5466775985b402ed86de3b9c4136d08d4153aec6cbc540c5a24c2c4d6dab',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_sources(data):
    archive_manifest = json.loads((ARCHIVE / 'manifest.json').read_text())
    for relative, expected in archive_manifest['files'].items():
        assert sha(ARCHIVE / relative) == expected, ('changed archive', relative)
    if not data['manifest']['config']['smoke']:
        assert archive_manifest['experiment_manifest'] == data['manifest']
    changed = []
    for path, expected in data['manifest']['inputs'].items():
        actual = sha(path)
        if actual == expected:
            continue
        relative = str(Path(path).relative_to(ROOT))
        assert relative in KNOWN and actual == KNOWN[relative], ('unreviewed source drift', path)
        assert sha(ARCHIVE / relative) == expected
        changed.append(dict(path=path, frozen_sha256=expected, current_sha256=actual))
    assert changed, 'no source deviation to review'
    files = {p.stem: p for folder in ('src', 'scripts') for p in (ARCHIVE / folder).glob('*.py')}
    seen, pending = set(), ['cem_score_direction', 'cem_direction_runtime', 'shard']
    while pending:
        name = pending.pop()
        if name in seen or name not in files:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse(files[name].read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                pending.append(node.module.split('.')[0])
    assert not {'entropyloop', 'temperature_experiment'} & seen
    original = ast.parse((ARCHIVE / 'src/gradguide.py').read_text())
    current = ast.parse((ROOT / 'src/gradguide.py').read_text())
    def without_helper(tree):
        return ast.dump(ast.Module(body=[n for n in tree.body if not
            (isinstance(n, ast.FunctionDef) and n.name == 'propose_guided')], type_ignores=[]))
    assert without_helper(original) == without_helper(current)
    for filename in ('scripts/cem_direction_runtime.py', 'scripts/cem_score_direction.py', 'src/shard.py'):
        assert 'propose_guided' not in (ARCHIVE / filename).read_text()
    sampler = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_guided')
    assert 'propose_guided' not in ast.unparse(sampler)
    return dict(status='reviewed_source_deviation', strict_source_freeze_passed=False,
                changed_inputs=changed, conservative_local_import_closure=sorted(seen),
                archive_manifest_sha256=sha(ARCHIVE / 'manifest.json'), review_source_sha256=sha(__file__),
                explanation='Two changed modules are outside the CEM import closure. The sole AST change in gradguide is propose_guided, which this CEM driver does not call. The directly called sample_guided and all ranking, training, battle, and metric code are unchanged. The uninterrupted Python controller imported gradguide before the edits. This is a post-hoc review; the strict source-freeze check still failed.')


def require_finished(path, data, review):
    """Accept only complete experimental phases after the controller has exited."""
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if data['status'] == 'complete':
        return
    assert review and data['status'] == 'running'
    error = data.get('last_error', {})
    assert error.get('type') == 'ValueError'
    assert error.get('message') in {'input changed during experiment: ' + row['path'] for row in review['changed_inputs']}
    config = data['manifest']['config']
    assert set(data['runs']) == {f'seed{s}/{a}' for s in config['seeds'] for a in ('normal', 'random', 'reverse')}
    for run in data['runs'].values():
        assert not run.get('stopped'), 'deviation completion override requires every planned update'
        assert set(run['iterations']) == {str(i) for i in range(1, config['iterations'] + 1)}
        assert all(row.get('status') == 'complete' for row in run['iterations'].values())
        assert run.get('quality', {}).get('status') == 'complete'
