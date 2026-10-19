"""Post-hoc structural diversity of the frozen DiBO/CEM comparison. No battles."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from corpus import parse_team_text
from diversity_metrics import team_keys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def keys(paste):
    # Parsing blocks separately preserves itemless slots in the legacy parser.
    slots = [s for block in paste.strip().split('\n\n') for s in parse_team_text(block)]
    return team_keys(slots)


def composition_stats(compositions):
    n = len(compositions)
    if not n:
        raise ValueError('empty population')
    if any(len(c) != 6 or len(set(c)) != 6 for c in compositions):
        raise ValueError('species replacement distance assumes six distinct species/forms')
    counts = Counter(tuple(sorted(c)) for c in compositions)
    probs = np.array(list(counts.values()), float) / n
    # Sum shared species over unordered distinct team pairs without a large matrix.
    species = Counter(s for c in compositions for s in c)
    shared_pairs = sum(k * (k-1) / 2 for k in species.values())
    pairs = n * (n-1) / 2
    return dict(n=n, distinct_compositions=len(counts), effective_compositions=float(np.exp(-(probs*np.log(probs)).sum())),
                largest_composition_share=float(max(probs)), species_richness=len(species),
                mean_species_replacements=6-shared_pairs/pairs if pairs else None,
                composition_frequencies=[dict(species=list(c), count=k) for c,k in sorted(counts.items(), key=lambda p: (-p[1],p[0]))],
                species_frequencies=[dict(species=s,count=k) for s,k in species.most_common()])


def metrics(records, initial):
    representations = [keys(r['paste']) for r in records]
    out = composition_stats([k[2] for k in representations])
    out['unique_categorical_teams'] = len({k[0] for k in representations})
    out['unique_teams_with_spreads'] = len({k[1] for k in representations})
    out['composition_novel_draw_fraction'] = sum(k[2] not in initial for k in representations)/len(records)
    # Within a fixed roster, count how many complete Pokemon builds differ.
    # A build includes ability, item, nature and unordered moves, excluding spreads.
    groups = {}
    for categorical, _, composition in representations:
        groups.setdefault(composition, []).append(categorical)
    pairs = 0; identical_build_pairs = 0
    for teams in groups.values():
        pairs += len(teams)*(len(teams)-1)//2
        counts = Counter(slot for team in teams for slot in team)
        identical_build_pairs += sum(k*(k-1)//2 for k in counts.values())
    out['same_roster_pairs'] = pairs
    out['mean_build_changes_within_same_roster'] = 6-identical_build_pairs/pairs if pairs else None
    return out


def self_check():
    a = tuple('abcdef'); b = tuple('abcdeg'); c = tuple('hijklm')
    same = composition_stats([a, tuple(reversed(a))])
    assert same['distinct_compositions'] == 1 and same['mean_species_replacements'] == 0
    assert same['effective_compositions'] == 1
    assert composition_stats([a,b])['mean_species_replacements'] == 1
    assert composition_stats([a,c])['mean_species_replacements'] == 6
    three = composition_stats([a,a,b])
    assert math.isclose(three['mean_species_replacements'], 2/3)
    assert math.isclose(three['effective_compositions'], math.exp(-2/3*math.log(2/3)-1/3*math.log(1/3)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', type=Path)
    args = ap.parse_args(); root = args.directory.resolve()
    self_check()
    proto = json.loads((root/'protocol.json').read_text())
    data = {a:json.loads((root/f'{a}.json').read_text()) for a in ['cem','dibo']}
    initial = {keys(r['paste'])[2] for r in proto['initial']}
    result = dict(inputs={n:sha(root/n) for n in ['protocol.json','cem.json','dibo.json','summary.json']},
                  sources={str(p):sha(p) for p in [Path(__file__),ROOT/'src/diversity_metrics.py',ROOT/'src/corpus.py']},
                  parser_data={n:sha(Path('/tmp/vgc-pilot/data')/(n+'.json')) for n in ['pokedex','learnsets','items','abilities','moves']},
                  initial_reference='The same frozen 5000 starting observations; not the full foundation-training corpus.',
                  populations={'proposals':'Unequal saved pools: CEM 512; DiBO 12800. Do not compare raw roster counts across these pools.',
                               'raw':'First 512 legal accepted samples before deduplication and local refinement; equal-sized generator populations',
                               'selected':'128 distinct teams actually battled per round',
                               'finalists':'16 teams locked by search score before independent evaluations'},
                  arms={})
    for arm,out in data.items():
        rounds=[]; accumulated=[]
        for g,r in sorted(out['rounds'].items(),key=lambda p:int(p[0])):
            assert len(r['selected'])==len(r['evaluated'])==128 and len(r['proposals'])==proto['proposal_counts'][arm]
            assert [a['key'] for a in r['selected']]==[a['key'] for a in r['evaluated']]
            pm=metrics(r['proposals'],initial); sm=metrics(r['selected'],initial)
            assert len(r['raw_accepted_sample']) == 512
            raw=metrics(r['raw_accepted_sample'],initial)
            # Independently recompute the experiment's stored proposal composition count.
            assert pm['distinct_compositions']==r['species_sets']
            accumulated.extend(r['evaluated'])
            qualified=[x for x in accumulated if x['y']>=.531]
            rounds.append(dict(round=int(g)+1,proposals=pm,raw=raw,selected=sm,search_mean=float(np.mean([x['y'] for x in r['evaluated']])),
                               cumulative_compositions=len({keys(x['paste'])[2] for x in accumulated}),
                               cumulative_qualifying_compositions=len({keys(x['paste'])[2] for x in qualified})))
        final=metrics(out['final_selected'],initial)
        assert len(out['final_selected'])==len(out['heldout'])==len(out['confirmation'])==16
        final['confirmation_mean']=sum(r['wins'] for r in out['confirmation'])/sum(r['battles'] for r in out['confirmation'])
        final['heldout_mean']=sum(r['wins'] for r in out['heldout'])/sum(r['battles'] for r in out['heldout'])
        final['teams']=[dict(index=i,species=list(keys(r['paste'])[2]),search_score=r['y'],
                             heldout_wins=out['heldout'][i]['wins'],heldout_battles=out['heldout'][i]['battles']) for i,r in enumerate(out['final_selected'])]
        cumulative=metrics(accumulated,initial)
        cumulative['search_mean']=float(np.mean([r['y'] for r in accumulated]))
        cumulative['qualifying_compositions']=rounds[-1]['cumulative_qualifying_compositions']
        result['arms'][arm]=dict(rounds=rounds,all_battled=cumulative,finalists=final)
    dest=root/'diversity';dest.mkdir(exist_ok=True)
    (dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result,dest)
    report(result,dest)
    print(json.dumps({a:{k:{x:v for x,v in d[k].items() if x not in ['composition_frequencies','species_frequencies','teams']} for k in ['all_battled','finalists']} for a,d in result['arms'].items()},indent=2))


def plot(result,dest):
    colors={'cem':'#7c4d94','dibo':'#087e8b'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for arm,d in result['arms'].items():
        label='CEM' if arm=='cem' else 'DiBO'; c=colors[arm];rs=d['rounds'];x=[r['round'] for r in rs]
        axs[0,0].plot(x,[r['selected']['distinct_compositions'] for r in rs],'-o',color=c,label=label)
        axs[0,1].plot(x,[r['selected']['largest_composition_share'] for r in rs],'-o',color=c,label=label)
        axs[1,0].plot(x,[r['selected']['mean_species_replacements'] for r in rs],'-o',color=c,label=label)
        axs[1,1].scatter([r['selected']['mean_species_replacements'] for r in rs],[r['search_mean'] for r in rs],color=c,s=50,label=label)
    axs[0,0].set(title='Distinct rosters among 128 battled teams',ylim=(0,128))
    axs[0,1].set(title='Most common roster’s share — lower is broader',ylim=(0,1));axs[0,1].yaxis.set_major_formatter(PercentFormatter(1))
    axs[1,0].set(title='Average species replacements between teams',ylim=(0,6))
    axs[1,1].set(title='Win rate and diversity of the same battle batch',xlabel='Average species replacements',ylabel='Search win rate',xlim=(0,6),ylim=(0,1));axs[1,1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set_xlabel('Round');ax.set_xticks(range(1,11))
    for ax in axs.flat:ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.suptitle('Structural diversity alongside performance · one matched-budget run')
    fig.savefig(dest/'quality-and-diversity.png',dpi=160);fig.savefig(dest/'quality-and-diversity.svg');plt.close(fig)


def report(result,dest):
    arms=result['arms'];lines=['# Diversity and win rate: DiBO versus CEM','',
        'Diversity endpoints were specified before this revised equal-battle-budget experiment. No additional battles or changes to finalist selection. Each method used 68,864 fresh battles. These are structural measurements of this run, not proof of different strategies or superiority across seeds.','',
        f"Across 1,280 battled teams, CEM covered **{arms['cem']['all_battled']['distinct_compositions']} rosters versus DiBO’s {arms['dibo']['all_battled']['distinct_compositions']}**. The 16 finalists contained {arms['cem']['finalists']['distinct_compositions']} CEM rosters versus {arms['dibo']['finalists']['distinct_compositions']} DiBO roster. Moves, items, abilities and natures can still differ within a roster. The two kinds of variety should be read separately: broad roster coverage and variation of builds within a roster.",'',
        '**Final outputs: the same 16 locked finalists used for independent evaluation.**','',
        '| Method | Heldout mean | Confirmation mean | Distinct rosters /16 | Effective rosters | Largest roster share | Mean species replacements | Species represented |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,d in arms.items():
        f=d['finalists'];lines.append(f"| {a.upper() if a=='cem' else 'DiBO'} | {f['heldout_mean']:.1%} | {f['confirmation_mean']:.1%} | {f['distinct_compositions']} | {f['effective_compositions']:.2f} | {f['largest_composition_share']:.1%} | {f['mean_species_replacements']:.2f} /6 | {f['species_richness']} |")
    lines += ['', '**Variation within the same roster.** All 16 finalists in each method are distinct in their categorical builds. Among pairs sharing the same six species, the following measures how many Pokémon differ in at least one ability, item, nature or move. Stat Points are excluded from this build measure.','',
              '| Method | Same-roster finalist pairs | Mean Pokémon builds changed /6 |','|---|---:|---:|']
    for a,d in arms.items():
        f=d['finalists'];lines.append(f"| {a.upper() if a=='cem' else 'DiBO'} | {f['same_roster_pairs']} | {format(f['mean_build_changes_within_same_roster'], '.2f') if f['same_roster_pairs'] else 'N/A'} |")
    lines += ['', 'A changed build can be a small nature/item change or a larger move change; this metric does not establish a new strategy or better quality. The averages condition on same-roster pairs, whose counts differ between methods. Neither metric alone is an overall diversity score.']
    lines += ['', '**Search outputs: all 1,280 battled teams per method.**','',
              '| Method | Search mean | Distinct rosters | Effective rosters | Largest roster share | Mean species replacements | Rosters with at least one ≥53.1% search label |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for a,d in arms.items():
        f=d['all_battled'];lines.append(f"| {a.upper() if a=='cem' else 'DiBO'} | {f['search_mean']:.1%} | {f['distinct_compositions']} | {f['effective_compositions']:.2f} | {f['largest_composition_share']:.1%} | {f['mean_species_replacements']:.2f} /6 | {f['qualifying_compositions']} |")
    lines += ['', 'The 53.1% cutoff is the existing CEM elite threshold, reused as an explicitly post-hoc quality-and-coverage summary for both methods. These are noisy search labels; one qualifying variant does not establish that its roster reliably exceeds the cutoff. Search scores are weighted across the top50 list and are not independent heldout performance.','',
              '![Win rate and structural diversity](quality-and-diversity.png)','',
              '**How diversity changed during search.** Win rate and selected-roster count below refer to the same 128 teams each round. Raw counts refer to the first 512 legal samples before deduplication and refinement in each method.','',
              '| Round | CEM raw rosters /512 | DiBO raw rosters /512 | CEM battled rosters /128 | DiBO battled rosters /128 | CEM search mean | DiBO search mean |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for c,d in zip(arms['cem']['rounds'],arms['dibo']['rounds']):
        lines.append(f"| {c['round']} | {c['raw']['distinct_compositions']} | {d['raw']['distinct_compositions']} | {c['selected']['distinct_compositions']} | {d['selected']['distinct_compositions']} | {c['search_mean']:.1%} | {d['search_mean']:.1%} |")
    lines += ['', '**Definitions and limits.** A roster is the unordered six species/form identifiers; changing moves, items or Stat Points does not create a new roster. Effective rosters is exp(Shannon entropy): 16 equally represented rosters score 16, while one repeated roster scores 1. Largest share exposes concentration hidden by a count of rare alternatives. Mean species replacements is 6 minus the average shared-species count across every unordered pair of different output entries; identical rosters score 0, disjoint rosters 6. Repeated rosters retain their frequencies.','',
              'Exact team uniqueness is also recorded with and without Stat Points, ignoring slot and move ordering. Because the experiment deliberately selected previously unseen distinct teams, exact uniqueness is partly enforced and is not sufficient evidence of diversity. Counts should only be compared at the same sample size and stage.','',
              'Raw populations retain duplicate draws. The separately saved proposal pools have unequal sizes (CEM 512; DiBO 12,800), so their unique counts are not directly comparable. DiBO proposal pools are after local refinement; CEM retains its nature repair. Legal-acceptance filtering remains part of both pipelines.','',
              'The metrics file includes composition frequencies, species frequencies, novelty relative to the frozen 5,000 initial observations, per-finalist results, and input/source hashes. Novelty to that initial pool is not novelty to the full pretrained corpus. Heldout scores have not been used to choose a different final team set.','',
              '[Machine-readable metrics](metrics.json) · [Original win-rate report](../analysis.md)']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
