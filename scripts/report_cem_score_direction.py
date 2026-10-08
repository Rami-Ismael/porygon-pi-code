"""Report the completed CEM score-direction experiment, including all seed paths."""

import argparse
import json
from pathlib import Path
import statistics

from cem_score_direction import ARMS, ROOT
from reverse_score_diversity import atomic_json, digest, paired

ID_SCORE = 'pokemon_id_diversity_percent'
NAMES = dict(normal='Normal CEM', random='Random elites', reverse='Reversed CEM')


def complete_mean(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def summarize(data, source_review=None):
    if data['status'] != 'complete' and source_review is None:
        raise ValueError('cannot report an unfinished experiment as complete')
    c = data['manifest']['config']
    if set(data['runs']) != {f'seed{s}/{a}' for s in c['seeds'] for a in ARMS}:
        raise ValueError('missing trajectories')
    series, retention, final_quality = {}, {}, {}
    for arm in ARMS:
        series[arm] = []
        for iteration in range(c['iterations'] + 1):
            seeds = {}
            for seed in c['seeds']:
                if iteration == 0:
                    batch = data['shared'][str(seed)]['baseline']
                else:
                    row = data['runs'][f'seed{seed}/{arm}']['iterations'].get(str(iteration), {})
                    batch = row.get('evaluation') if row.get('status') == 'complete' else None
                m = batch['metrics'] if batch else None
                seeds[str(seed)] = dict(
                    legal_id_diversity=m['legal'][ID_SCORE] if m else None,
                    raw_id_diversity=m['raw'][ID_SCORE] if m else None,
                    legality_percent=100 * m['legality_rate'] if m else None,
                    legal_effective_ids=m['legal']['effective_ids'] if m else None,
                    legal_distinct_ids=m['legal']['distinct_ids'] if m else None,
                    legal_distinct_rosters=m['legal']['distinct_rosters'] if m else None)
            means = {key: complete_mean([r[key] for r in seeds.values()]) for key in next(iter(seeds.values()))}
            series[arm].append(dict(iteration=iteration, means=means, seeds=seeds))
        changes = []
        quality = []
        for seed in c['seeds']:
            first = series[arm][0]['seeds'][str(seed)]['legal_id_diversity']
            last = series[arm][-1]['seeds'][str(seed)]['legal_id_diversity']
            changes.append(last - first if last is not None and first is not None else None)
            q = data['runs'][f'seed{seed}/{arm}'].get('quality', {})
            labels = q.get('scores', {}).get('labels', [])
            quality.append(statistics.mean(r['score'] for r in labels) if q.get('status') == 'complete' else None)
        retention[arm] = dict(seed_changes_percentage_points=changes, mean_percentage_points=complete_mean(changes))
        final_quality[arm] = dict(seed_win_rates=quality, mean_win_rate=complete_mean(quality))
    contrasts = {}
    for arm in ('normal', 'random'):
        differences = []
        for seed in c['seeds']:
            a = series['reverse'][-1]['seeds'][str(seed)]['legal_id_diversity']
            b = series[arm][-1]['seeds'][str(seed)]['legal_id_diversity']
            differences.append(a - b if a is not None and b is not None else None)
        contrasts[arm] = paired(differences) if all(v is not None for v in differences) else dict(
            differences=differences, mean=None, note='complete paired final comparison unavailable')
    attempts = battles = steps = 0
    for shared in data['shared'].values():
        attempts += len(shared['baseline']['records']) + len(shared['initial']['candidate_batch']['records'])
        battles += shared['initial'].get('scores', {}).get('total_battles', 0)
    for run in data['runs'].values():
        for row in run['iterations'].values():
            attempts += len(row.get('candidate_batch', {}).get('records', [])) + len(row.get('evaluation', {}).get('records', []))
            battles += row.get('scores', {}).get('total_battles', 0)
            steps += row.get('training', {}).get('steps', 0)
        battles += run.get('quality', {}).get('scores', {}).get('total_battles', 0)
    return dict(series=series, final_reverse_minus_comparator_pp=contrasts, source_review=source_review,
                change_from_baseline_pp=retention, fresh_final_quality=final_quality,
                stopped={key: r['stopped'] for key, r in data['runs'].items() if 'stopped' in r},
                completed_updates=sum(row.get('status') == 'complete' for run in data['runs'].values() for row in run['iterations'].values()),
                total_raw_attempts=attempts, total_recorded_battles=battles, total_training_steps=steps,
                inference_unit='trajectory seed, conditional on a fixed initial checkpoint and battle stack')


def show(value, digits=2):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def report(path, data, summary):
    c = data['manifest']['config']
    lines = ['# CEM score direction across iterations', '',
             '**Reduced plumbing run; not evidence for the full experiment.**' if c['smoke'] else
             f"Completed {len(c['seeds'])} seeds, three arms, and up to {c['iterations']} cumulative CEM updates.", '',
             'Normal CEM ranks by score, reversed CEM ranks by negative score, and the random control selects uniformly. Recorded battle scores and the denoising training objective remain unchanged. Sampling has no score guidance.', '',
             'Pokemon-ID diversity is the average fraction of the six roster IDs that must be replaced between two teams. Identical rosters score 0%, one replacement scores 16.67%, and disjoint rosters score 100%. Slot order and non-ID attributes are ignored; different form IDs remain distinct.', '',
             f"Each checkpoint is evaluated with {c['evaluation_attempts']} independent raw attempts. Repeats and invalid outputs are retained; legal-output diversity is reported alongside the raw view and legality.", '',
             '## Final checkpoint', '',
             '| Arm | Legal ID diversity (%) | Raw ID diversity (%) | Legal attempts (%) | Effective IDs among legal outputs | Change from initial legal ID diversity (pp) |',
             '|---|---:|---:|---:|---:|---:|']
    if summary.get('source_review'):
        timing = ('The completed run predates three shared-source edits, so a later audit against current source files fails the strict hash check.'
                  if data['status'] == 'complete' else
                  'All experimental phases completed, but the strict final source-freeze check failed after three shared source files changed during the run.')
        lines[4:4] = [f'**Provenance deviation:** {timing} A separate source review found that the edits affect unused modules/helper code. The original raw status and any error remain preserved. Results below are reported with this deviation, not as a clean current-source hash pass.', '',
                      f"[Source review]({path.with_suffix('.source-review.json')})", '']
    for arm in ARMS:
        m = summary['series'][arm][-1]['means']
        lines.append(f"| {NAMES[arm]} | {show(m['legal_id_diversity'])} | {show(m['raw_id_diversity'])} | {show(m['legality_percent'])} | {show(m['legal_effective_ids'])} | {show(summary['change_from_baseline_pp'][arm]['mean_percentage_points'])} |")
    lines += ['', 'Means give equal weight to every trajectory seed. An incomplete or infeasible seed makes the corresponding full mean unavailable; it is not silently dropped.', '',
              '## Distinct legal rosters and IDs', '',
              f"These are mean counts in the legal portion of each {c['evaluation_attempts']}-attempt evaluation batch. A roster is the sorted multiset of its six Pokemon IDs. The diversity percentage above measures average roster distance; it is not the percentage of unique teams.", '',
              '| Checkpoint | Distinct legal ID rosters | Distinct Pokemon IDs in legal outputs |', '|---|---:|---:|']
    initial = summary['series']['normal'][0]['means']
    lines.append(f"| Initial generator | {show(initial['legal_distinct_rosters'], 1)} | {show(initial['legal_distinct_ids'], 1)} |")
    for arm in ARMS:
        m = summary['series'][arm][-1]['means']
        lines.append(f"| {NAMES[arm]} | {show(m['legal_distinct_rosters'], 1)} | {show(m['legal_distinct_ids'], 1)} |")
    lines += ['',
              '## Legal Pokemon-ID diversity by iteration', '',
              '| Iteration | Normal CEM (%) | Random elites (%) | Reversed CEM (%) |', '|---|---:|---:|---:|']
    for iteration in range(c['iterations'] + 1):
        values = [show(summary['series'][a][iteration]['means']['legal_id_diversity']) for a in ARMS]
        lines.append(f"| {iteration} | {' | '.join(values)} |")
    lines += ['', 'Iteration zero is the shared starting model. Later checkpoints continue from the preceding model; they do not restart from p0.', '',
              '## Final paired differences', '',
              '| Comparator | Reversed minus comparator (pp) | Seed differences (pp) | Adjusted 95% paired t interval (pp) |',
              '|---|---:|---|---|']
    for arm, comparison in summary['final_reverse_minus_comparator_pp'].items():
        ci = comparison.get('ci95_two_comparisons')
        lines.append(f"| {NAMES[arm]} | {show(comparison['mean'])} | {', '.join(show(x) for x in comparison['differences'])} | {f'[{ci[0]:.2f}, {ci[1]:.2f}]' if ci else 'unavailable'} |")
    lines += ['', 'The three-seed intervals are exploratory, assume normally distributed seed effects, and use Bonferroni adjustment for the two final comparisons. Other measurements and trajectory changes are descriptive. A higher value than another arm can still coexist with a loss relative to the initial generator.', '',
              '## Fresh final battle evaluation', '',
              f"Uniformly sampled {c['quality_teams']} legal final evaluation outputs per arm/seed received {c['quality_battles']} fresh battles each. These labels were never used for selection or retraining.", '',
              '| Arm | Mean fresh win rate (%) | Seed win rates (%) |', '|---|---:|---|']
    for arm in ARMS:
        q = summary['fresh_final_quality'][arm]
        value = q['mean_win_rate']
        lines.append(f"| {NAMES[arm]} | {show(100 * value if value is not None else None)} | {', '.join(show(100 * x if x is not None else None) for x in q['seed_win_rates'])} |")
    lines += ['', '## Completion and limitations', '',
              f"Recorded {summary['total_raw_attempts']:,} raw generation attempts, {summary['total_recorded_battles']:,} completed battles, and {summary['total_training_steps']:,} optimizer steps across {summary['completed_updates']} completed updates. Shared initial selection batches and battles are counted once.", '',
              f"Stopped trajectories: {json.dumps(summary['stopped'])}.", '',
              'The experiment is conditional on one pretrained checkpoint, its fixed species vocabulary, this battle policy, and these update budgets. Policy seeds and opponent schedules are matched; they do not control Showdown\'s internal battle RNG. ID overlap does not establish strategic diversity. Candidate scores use finite battle samples, so ranking can amplify noise.', '',
              f"![Diversity trajectories]({path.with_suffix('.png')})", '',
              f"[Protocol]({ROOT / 'docs/cem-score-direction-protocol.md'}) · [Raw results]({path}) · [Analysis JSON]({path.with_suffix('.analysis.json')}) · [Verification receipt]({path.with_suffix('.verification.json')})", '']
    return '\n'.join(lines)


def plot(path, data, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(normal='#21618c', random='#777777', reverse='#b94735')
    metrics = [('legal_id_diversity', 'Pokemon-ID diversity: legal outputs (%)'),
               ('raw_id_diversity', 'Pokemon-ID diversity: all raw outputs (%)'),
               ('legality_percent', 'Legal generation attempts (%)'),
               ('legal_effective_ids', 'Effective Pokemon IDs: legal outputs')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, metrics):
        for arm in ARMS:
            series = summary['series'][arm]
            xs = [v['iteration'] for v in series]
            for seed in data['manifest']['config']['seeds']:
                values = [v['seeds'][str(seed)][key] for v in series]
                ax.plot(xs, values, color=colors[arm], alpha=.23, linewidth=1)
            ax.plot(xs, [v['means'][key] for v in series], '-o', color=colors[arm], label=NAMES[arm], linewidth=2, markersize=4)
        ax.set_title(title); ax.set_xlabel('Cumulative CEM update')
        ax.set_xticks(range(data['manifest']['config']['iterations'] + 1))
        ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.15)
        if key == 'legality_percent':
            ax.set_ylim(0, 100)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('CEM ranking score and generated diversity\nThin lines: individual seeds; thick lines: means')
    fig.savefig(path.with_suffix('.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--review-source-drift', action='store_true', help='Explicitly review the three known unused-code edits; preserve the failed strict check')
    args = parser.parse_args()
    path = args.results.resolve()
    data = json.loads(path.read_text())
    source_review = None
    if args.review_source_drift:
        from cem_source_review import review_sources, require_finished
        source_review = review_sources(data)
        require_finished(path, data, source_review)
        atomic_json(path.with_suffix('.source-review.json'), source_review)
    summary = summarize(data, source_review)
    summary.update(raw_result_sha256=digest(path), analysis_source_sha256=digest(__file__))
    atomic_json(path.with_suffix('.analysis.json'), summary)
    plot(path, data, summary)
    path.with_suffix('.md').write_text(report(path, data, summary))
    print(json.dumps({key: summary[key] for key in ('completed_updates', 'total_raw_attempts', 'total_recorded_battles', 'total_training_steps')}, indent=2))
