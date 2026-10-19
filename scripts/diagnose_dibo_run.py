"""Read-only diagnosis of a completed comparison; creates separate diagnostic receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import dibo_core as DC
from checkpoints import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-item-ownership', action='store_true',
                        help='Exit nonzero if trained predictors ignore changed item ownership.')
    args = parser.parse_args()
    directory = args.directory.resolve()
    proto = json.loads((directory / 'protocol.json').read_text())
    arms = {a: json.loads((directory / (a + '.json')).read_text()) for a in ['dibo', 'cem']}
    cfg = proto['config']
    torch.set_num_threads(1)
    rows = []
    def arr(records, key):
        return np.asarray([r[key] for r in records], float)
    def mean(v):
        return float(np.mean(v))
    for g in range(cfg['rounds']):
        r = arms['dibo']['rounds'][str(g)]
        p, s, e = r['proposals'], r['selected'], r['evaluated']
        assert [a['key'] for a in s] == [a['key'] for a in e]
        target_keys = {a['key'] for a in s}
        prior_only = sorted(p, key=lambda a: -a['log_prior'])[:cfg['battle']]
        reward_only = sorted(p, key=lambda a: -a['ucb'])[:cfg['battle']]
        mu, y = arr(s, 'mu'), arr(e, 'y')
        training_rows = list(proto['initial'])
        training_rows += [a for j in range(g) for a in arms['dibo']['rounds'][str(j)]['evaluated']]
        training_rows = sorted(training_rows, key=lambda a: -a['y'])[:cfg['buffer']]
        train_y = arr(training_rows, 'y')
        weights = np.exp(train_y - max(train_y)); weights /= weights.sum()
        z = (train_y - train_y.mean()) / (train_y.std(ddof=1) + 1e-7)
        zw = np.exp(z - max(z)); zw /= zw.sum()
        rt = r['training']['rtb']
        rows.append(dict(round=g + 1, actual_mean=mean(y), cem_actual_mean=arms['cem']['rounds'][str(g)]['mean'],
                         prior_sd=float(arr(p, 'log_prior').std()), tilt_sd=float((cfg['beta'] * arr(p, 'ucb')).std()),
                         prior_only_overlap=len(target_keys & {a['key'] for a in prior_only}),
                         selected_predicted_mean=mean(mu), selected_sigma=mean(arr(s, 'sigma')),
                         reward_only_predicted_mean=mean(arr(reward_only, 'mu')),
                         prediction_correlation=float(np.corrcoef(mu, y)[0, 1]), rmse=float(np.mean((mu-y)**2)**.5),
                         rtb_first10=mean([a['loss'] for a in rt[:10]]), rtb_last10=mean([a['loss'] for a in rt[-10:]]),
                         weight_ess=float(1 / (weights**2).sum()), standardized_weight_ess=float(1 / (zw**2).sum()),
                         weight_max_min_ratio=float(weights.max()/weights.min()),
                         buffer_new_rows=sum(a['src'] != 'initial' for a in training_rows), draws=r['tried']))
    # Check the real trained predictors, without loading the generator or battle service.
    ckdir = Path.home() / '.local/share/vgc-pilot-runtime/lossdown' / ('dibo_v2_' + hashlib.sha256(str(directory / 'dibo.json').encode()).hexdigest()[:12])
    sizes = {k: len(v) for k, v in proto['runtime_signature']['vocabulary'].items()}
    keys = ['species', 'ability', 'item', 'move', 'move', 'move', 'move', 'nature']
    probes = []
    for g in [0, cfg['rounds']-1]:
        ck = load_checkpoint(ckdir / f'round{g}.pt', map_location='cpu')
        ensemble = DC.PredictorEnsemble([DC.Predictor(sizes, keys, width=cfg['width'], hidden=cfg['hidden']) for _ in range(cfg['ensemble'])])
        ensemble.load_state_dict(ck['ensemble']); ensemble.eval()
        x = torch.tensor([a['x'] for a in arms['dibo']['rounds'][str(g)]['selected']])
        changed = x.clone()
        # Rotate item ownership only, keeping species and the remaining fields fixed.
        changed[:, 2::8] = x[:, 2::8].roll(1, dims=1)
        with torch.no_grad():
            before = torch.stack([m(x) for m in ensemble.members])
            after = torch.stack([m(changed) for m in ensemble.members])
        delta = float((before-after).abs().max())
        changed_count = int((x != changed).any(1).sum())
        assert changed_count > 0
        probes.append(dict(round=g+1, changed_teams=changed_count, max_member_prediction_change=delta,
                           ownership_blindness_reproduced=delta < 1e-6))
    result = dict(rounds=rows, predictor_ownership_probes=probes,
                  note='Post-hoc diagnostics on saved data. No fresh battles, hyperparameter tuning, or changes to frozen experiment sources. Reward-only scores are predictions, not measured counterfactual wins.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(predictor_probes=probes,
                         mean_prior_overlap=mean([r['prior_only_overlap'] for r in rows]),
                         mean_weight_ess=mean([r['weight_ess'] for r in rows]),
                         mean_standardized_weight_ess=mean([r['standardized_weight_ess'] for r in rows]),
                         mean_selected_prediction=mean([r['selected_predicted_mean'] for r in rows]),
                         mean_reward_only_prediction=mean([r['reward_only_predicted_mean'] for r in rows]),
                         output=str(args.output)), indent=2))
    if args.require_item_ownership and any(p['ownership_blindness_reproduced'] for p in probes):
        raise SystemExit('FAIL: trained predictors cannot distinguish changed item ownership')


if __name__ == '__main__':
    main()
