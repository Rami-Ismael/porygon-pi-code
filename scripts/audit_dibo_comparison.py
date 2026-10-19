"""Independent audit of budgets, scores, separation and frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import heldout as H
import matchup_db as M
import top50


def load(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    args = ap.parse_args()
    root = args.directory.resolve()
    proto = load(root / 'protocol.json')
    cfg = proto['config']
    signature = proto['runtime_signature']
    initial_hash = hashlib.sha256(json.dumps(proto['initial'], sort_keys=True).encode()).hexdigest()
    assert initial_hash == proto['initial_hash']
    project = Path(__file__).resolve().parents[1]
    def verified_source(path, expected):
        path=Path(path)
        if path.exists() and sha(path)==expected:
            return
        archived=root/'source'/path.name
        assert archived.exists() and sha(archived)==expected, f'source changed without matching snapshot: {path}'
    verified_source(proto.get('comparison_source_file',project / 'src/dibo_comparison.py'), proto['comparison_source'])
    if 'initial_reference' in proto:
        assert sha(proto['initial_reference']) == proto['initial_reference_sha256']
        original=load(Path(proto['initial_reference']))
        assert proto['initial']==original['initial'][:cfg['initial_limit']]
    assert sha(proto['loaded_source_paths']['cem']) == proto['cem_source']
    for name, value in signature['implementation'].items():
        verified_source(project / 'src' / name, value)
    checkpoints = Path.home() / '.local/share/vgc-pilot-runtime/lossdown'
    for name, value in signature['foundation'].items():
        assert sha(checkpoints / (name + '.pt')) == value
    policy_digest_prefix = signature['policy'][1].rsplit(':', 1)[-1]
    assert len(policy_digest_prefix) >= 16 and sha(M.POLICY_FILE).startswith(policy_digest_prefix)
    if 'prior_comparison_source' in proto:
        assert sha(root / 'comparison-before-canonical-final.py') == proto['prior_comparison_source']
        old = load(root / 'protocol.before-canonical-final.json')
        assert old['initial_hash'] == proto['initial_hash'] and old['config'] == cfg
    weights = dict(proto['search_columns'])
    initial_keys = {r['key'] for r in proto['initial']}
    assert all(not H.is_heldout_text(r['paste']) for r in proto['initial'])
    totals, schedules = {}, {}
    complete = True
    for arm in proto['arms']:
        out = load(root / (arm + '.json'))
        assert out['initial_hash'] == initial_hash
        seen = set(initial_keys)
        count, completed = 0, 0
        for generation, result in sorted(out['rounds'].items(), key=lambda kv: int(kv[0])):
            if 'evaluated' not in result:
                continue
            completed += 1
            assert len(result['evaluated']) == cfg['battle']
            receipt_rows = []
            for start in range(0, cfg['battle'], 16):
                receipt = load(root / f'{arm}_round{generation}_battles_{start}.json')
                raw = receipt['raw']
                physical = sum(w + loss + tie for _, _, w, loss, tie in raw)
                assert physical == receipt['physical_battles']
                assert receipt['unattributed'] == 0
                assert receipt['keys'] == [r['key'] for r in result['selected'][start:start + 16]]
                cells = {(row, col): (win, loss, tie) for row, col, win, loss, tie in raw}
                for row in receipt['evaluated']:
                    assert row['key'] not in seen
                    seen.add(row['key'])
                    assert row['key'] == H.grid_key_text(row['paste'])
                    assert not H.is_heldout_text(row['paste'])
                    values = [cells[row['team_id'], col] for col in weights]
                    assert all(sum(value) == 1 for value in values)
                    score = sum(weights[col] * cells[row['team_id'], col][0] for col in weights) / sum(weights.values())
                    assert abs(score - row['y']) < 1e-12
                    assert row['battles'] == len(weights)
                receipt_rows.extend(receipt['evaluated'])
                count += physical
            assert receipt_rows == result['evaluated']
            assert result['physical_battles'] == cfg['battle'] * len(weights)
            assert abs(result['mean'] - np.mean([r['y'] for r in receipt_rows])) < 1e-12
        assert count == completed * cfg['battle'] * len(weights)
        totals[arm] = dict(rounds=completed, search=count)
        if completed != cfg['rounds']:
            complete = False
        if 'final_selected' in out:
            candidates = [r for _, result in sorted(out['rounds'].items(), key=lambda kv: int(kv[0])) for r in result.get('evaluated', [])]
            expected, used = [], set()
            for row in sorted(candidates, key=lambda r: -r['y']):
                if row['key'] not in used:
                    used.add(row['key']); expected.append(row['key'])
                    if len(expected) == cfg['final_n']:
                        break
            assert expected == [r['key'] for r in out['final_selected']]
            for i, row in enumerate(out['final_selected']):
                assert (root / f'{arm}_final_{i:02d}.txt').read_text() == M.canonical_paste(row['paste'])
        for stage in ('confirmation', 'heldout'):
            if stage not in out:
                complete = False
                continue
            assert len(out[stage]) == cfg['final_n']
            assert all(r['battles'] == cfg['final_battles'] and 0 <= r['wins'] <= r['battles'] and
                       abs(r['win_rate'] - r['wins']/r['battles']) < .0001 for r in out[stage])
            opponents = top50.files() if stage == 'confirmation' else H.files()
            rng = np.random.default_rng(cfg['seed'] + (100000 if stage == 'confirmation' else 200000))
            expected = [opponents[i] for i in rng.integers(len(opponents), size=cfg['final_battles'])]
            assert out[stage + '_opponent_schedule'] == expected
            schedules.setdefault(stage, expected)
            assert schedules[stage] == expected
            totals[arm][stage] = sum(r['battles'] for r in out[stage])
    if complete:
        assert totals['cem'] == totals['dibo']
        assert totals['cem']['search'] == proto['physical_battles_per_arm']['search']
    print(json.dumps(dict(audit_passed=True, full_experiment_complete=complete, verified_budgets=totals), indent=2))


if __name__ == '__main__':
    main()
