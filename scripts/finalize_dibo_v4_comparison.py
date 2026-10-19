"""Watch a full run, then audit and produce win-rate/diversity artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def progress(root):
    lines=['# Revised DiBO comparison — running', '',
           'Updated '+datetime.now(timezone.utc).isoformat(), '',
           'The full experiment is still running. No final winner has been determined.', '',
           '| Method | Completed search rounds /10 | Completed fresh search battles /62,720 | Confirmation battles /3,072 | Heldout battles /3,072 |',
           '|---|---:|---:|---:|---:|']
    for arm in ['cem','dibo']:
        p=root/f'{arm}.json'; data=json.loads(p.read_text()) if p.exists() else {}
        rounds=list(data.get('rounds',{}).values())
        receipts=[json.loads(p.read_text()) for p in root.glob(f'{arm}_round*_battles_*.json') if '.incomplete.' not in p.name]
        counts=[sum(r.get('battles',0) for r in data.get(stage,[])) for stage in ['confirmation','heldout']]
        lines.append(f"| {arm.upper() if arm=='cem' else 'DiBO'} | {sum('evaluated' in r for r in rounds)} | {sum(r['physical_battles'] for r in receipts)} | {counts[0]} | {counts[1]} |")
    lines += ['', 'Automatic finalization will write comparison.md only after the full run, budget/source audit, diversity analysis, and trained-predictor checks succeed.', '',
              '[What was fixed and what remains different from the paper](../../docs/dibo-v4.md) · [Execution and resume details](execution.md)']
    (root/'progress.md').write_text('\n'.join(lines)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    ap.add_argument('--wait-pid', type=int)
    args = ap.parse_args()
    root = args.directory.resolve()
    if args.wait_pid:
        while True:
            progress(root)
            try:
                os.kill(args.wait_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(30)
    summary = json.loads((root/'summary.json').read_text())
    if not summary['complete']:
        raise RuntimeError('Experiment stopped before completion; do not issue a final comparison.')
    audit = subprocess.run([sys.executable, str(ROOT/'scripts/audit_dibo_comparison.py'), str(root)], check=True, cwd=ROOT, capture_output=True, text=True)
    evidence = json.loads(audit.stdout)
    assert evidence['audit_passed'] and evidence['full_experiment_complete']
    recovery_path = root/'authorized-recovery.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    if recovery:
        evidence['authorized_recovery'] = recovery
        evidence['physical_attempt_budgets_equal'] = False
    backend_path = root/'fast-backend-activation.json'
    backend = json.loads(backend_path.read_text()) if backend_path.exists() else None
    if backend:
        evidence['execution_backend_change'] = backend
    (root/'audit.json').write_text(json.dumps(evidence, indent=2)+'\n')
    for script in ['plot_dibo_comparison.py', 'analyze_dibo_diversity_v2.py', 'verify_dibo_v4_trained_predictor.py']:
        subprocess.run([sys.executable, str(ROOT/'scripts'/script), str(root)], check=True, cwd=ROOT)
    diversity = json.loads((root/'diversity/metrics.json').read_text())['arms']
    lines = ['# Revised DiBO versus CEM', '',
             'Completed experiment: **68,864 completed fresh battles per method**. The revised DiBO preserves Pokémon/build ownership, standardizes rewards, expands prioritized posterior replay, and uses the larger proposal pool and bounded buffer described in the paper. See [implementation details](../../docs/dibo-v4.md).', '',
             '| Metric | CEM | Revised DiBO |', '|---|---:|---:|']
    held = summary['heldout']
    lines += [f"| Highest-search-score finalist: heldout win rate | {held['cem_first']:.1%} | {held['dibo_first']:.1%} |",
              f"| All 16 locked finalists: heldout mean | {held['cem_mean']:.1%} | {held['dibo_mean']:.1%} |"]
    for scope, label in [('all_battled','All 1,280 battled teams'), ('finalists','16 locked finalists')]:
        c, d = diversity['cem'][scope], diversity['dibo'][scope]
        for key, name, fmt in [('distinct_compositions','distinct rosters','d'),('effective_compositions','effective rosters','.2f'),('largest_composition_share','largest roster share','.1%')]:
            lines.append(f"| {label}: {name} | {format(c[key],fmt)} | {format(d[key],fmt)} |")
    lo, hi = held['team_cluster_bootstrap_95']
    lines += ['', f"The finalist-set heldout mean difference (DiBO − CEM) is **{held['mean_difference']:+.1%}**, with a team-cluster bootstrap 95% interval of **[{lo:+.1%}, {hi:+.1%}]**. This interval describes the selected teams, not variation over independent training seeds.", '',
              'This is one categorical adaptation and one training seed. Completed battle counts match; computation differs: DiBO filters 12,800 proposals per round versus CEM’s 512. Diversity comparisons use equal-size raw, battled, and final populations. Heldout opponents were already examined in the earlier experiment, although battle seeds are fresh. The changes were made together, so this run cannot identify which individual repair caused a performance change.', '',
              '![Win rate and diversity](diversity/quality-and-diversity.png)', '',
              '[Detailed win-rate analysis](analysis.md) · [Diversity definitions and per-round results](diversity/report.md) · [Budget/source audit](audit.json) · [Frozen protocol](protocol.json)']
    if recovery:
        lines[3:3] = [recovery['note'], '', '[Authorized recovery details](stall-20261014/repair.md)', '']
    if backend:
        lines[3:3] = [backend['note'], '', '[Fast battle execution details](fast-battles.md)', '']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    (root/'progress.md').write_text('# Revised DiBO comparison — complete\n\nThe full experiment, completed-result budget/source audit, and diversity analysis passed.\n\n' + (recovery['note']+'\n\n' if recovery else '') + '[Read the win-rate and diversity comparison](comparison.md).\n')
    with (root/'execution.md').open('a') as f:
        f.write('\nFull experiment and automatic budget/source audit, win-rate plots, and diversity analysis completed. See comparison.md.\n')
    print(root/'comparison.md', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if len(sys.argv)>1:
            root=Path(sys.argv[1]).resolve()
            if root.is_dir():
                (root/'progress.md').write_text('# Revised DiBO comparison — needs attention\n\nThe experiment or final checks stopped before a verified final report.\n\n'+str(exc)+'\n\nSee execution.md for resume details. No final winner is established.\n')
        raise
