"""Real-foundation training/proposal smoke checks; never starts battles.

Each selected arm trains its actual prior, predictors and RTB posterior with tiny
settings. This verifies plumbing, not diversity or competitive performance.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import dibo_diversity as DV
import dibo_diversity_runtime as R
from dibo_diversity_experiment import proposal_round, ROOT
from dibo_comparison_v2 import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', choices=DV.ARMS, default=list(DV.ARMS))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / 'results/dibo_vs_cem_full_20261012/protocol.json').read_text())
    ids = {tid for tid, _ in frozen['search_columns']}
    initial = [r for r in frozen['initial'] if r['team_id'] in ids]
    cfg = config(True)
    cfg.update(rounds=2,propose=4,battle=2,islands=2,pool_multiplier=2,metric_sample=4,
               buffer=64,rtb_steps=1,local_steps=1,rtb_batch=2,rtb_microbatch=2,seed=7171)
    summary = dict(kind='real-model proposal-only smoke; zero battles', arms={})
    for arm in args.arms:
        rt = R.Runtime(dict(cfg,arm=arm),args.directory / f'{arm}.json')
        try:
            result = proposal_round(rt,initial,{'rounds':{}},0)
            assert len(result['selected']) == cfg['battle']
            assert len({r['key'] for r in result['selected']}) == cfg['battle']
            assert not ({r['key'] for r in initial} & {r['key'] for r in result['selected']})
            assert all(rt.validator(r['paste']) is None and not R.DB.HO.is_heldout_text(r['paste'])
                       for r in result['selected'])
            assert all(abs(r['log_target'] - r['log_prior'] - cfg['beta'] * r['standardized_ucb']) < 1e-5
                       for r in result['selected'])
            R.DB.atomic_json(args.directory / f'{arm}.json',result)
            summary['arms'][arm] = dict(passed=True,selected=len(result['selected']),diversity=result['diversity'])
            R.DB.atomic_json(args.directory / 'verification.json',summary)
            print(f'{arm}: PASS; legal unseen proposals, density identity, zero battles',flush=True)
        finally:
            rt.close()


if __name__ == '__main__':
    main()
