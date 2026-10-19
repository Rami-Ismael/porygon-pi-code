"""Paired prompt-version test on the frozen 147-task pilot. See docs/prompt-changelog.md."""
import argparse
import collections
import copy
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import llm_baseline as B
import prompt_versions as V

OUT=B.ROOT/'results/prompt-regulation-test'
ARMS=['mb-original-v1','mb-regulation-v3']
MIN_STARTERS=6
FAILURES=[('truncated','truncated'),('Expecting|Extra data|Unterminated|delimiter|Invalid','unparseable JSON'),
          ('wrong number of fills|each fill must|expected exactly|duplicate or unmasked|invalid slot fields|four move|unmasked fields','wrong answer shape'),
          ('Stat Points require','Stat Points out of range'),('multiple copies|nonempty','repeated or empty move'),
          ("can't learn",'move not learnable'),('does not exist','name does not exist'),('Item Clause','Item Clause'),
          ('Species Clause|limited to one','Species Clause'),("can't have",'ability not available'),('item','item not allowed')]


def read(path):return json.loads(Path(path).read_text()) if Path(path).exists() else None


def folder(arm,replicate):return OUT/arm/f'r{replicate}'


def prepare(args):
    parent=read(args.parent/'manifest.json')
    if parent is None:raise ValueError(f'no frozen manifest in {args.parent}')
    starters={t['start'] for t in parent['trials']}
    if len(starters)<MIN_STARTERS:raise ValueError(f'{len(starters)} starters; the test needs at least {MIN_STARTERS}')
    for arm in ARMS:
        for replicate in range(args.replicates):
            m=copy.deepcopy(parent);m.pop('prompt_comparison',None)
            m.update(prompt_version=arm,prompt_test=dict(parent_manifest_sha256=B.digest(parent),replicate=replicate,
                     system_sha256=V.system_sha256(arm,m),arms=ARMS,starters=sorted(starters)))
            if arm==V.ORIGINAL:del m['prompt_version']  # identical request bytes to the first run
            path=folder(arm,replicate)/'manifest.json'
            if path.exists():assert read(path)==m,f'{path} differs; use another --output'
            else:B.save(path,m)
    print(json.dumps(dict(prepared=str(OUT),arms=ARMS,replicates=args.replicates,tasks=len(parent['trials']),
                          starters=len(starters),calls=len(ARMS)*args.replicates*len(parent['trials']))))


def folders():return sorted(p.parent for p in OUT.glob('*/r*/manifest.json'))


def run(args):
    todo=[f for f in folders() if any(not (f/'completions'/(t['id']+'.json')).exists() for t in read(f/'manifest.json')['trials'])]
    if todo and not os.environ.get('OPENROUTER_API_KEY'):
        os.environ['OPENROUTER_API_KEY']=getpass.getpass('OpenRouter key (not saved): ')
    # All arms run at the same time, so provider drift cannot favour one prompt.
    jobs=[(f,subprocess.Popen([sys.executable,str(B.ROOT/'scripts/llm_baseline.py'),'complete','--output',str(f)],
                               stdout=(f/'generation.log').open('a'),stderr=subprocess.STDOUT)) for f in todo]
    failed=[str(f) for f,job in jobs if job.wait()]
    report(args)
    if failed:raise RuntimeError(f'generation stopped in {failed}; see generation.log, then rerun')


def battle_worker(index,count,backend='fast'):
    """Score the legal completions only; the comparison is between the two prompts."""
    h,db=B.runtime();jobs=[]
    for f in folders():
        m=read(f/'manifest.json');tag=f.parent.name.split('-')[-1]+f.name  # v3r0: no hyphen, so the starter index still parses
        for t in m['trials']:
            r=read(f/'completions'/(t['id']+'.json'))
            if r and r['valid'] and not (f/'labels'/('llm-'+t['id']+'.json')).exists():jobs.append((tag+'-'+t['id'],r['team'],f/'labels'/('llm-'+t['id']+'.json')))
    # Identical teams always go to the same process, preventing cache races.
    jobs=[j for j in jobs if int(B.digest(db.canonical_paste(B.paste(j[1],h))),16)%count==index]
    out=OUT/'workers'/str(index);B.save(out/'manifest.json',m)
    try:B.battle(SimpleNamespace(output=out,battle_backend=backend),candidates_override=[(label,team) for label,team,_ in jobs])
    finally:
        for label,_,dest in jobs:
            done=read(out/'labels'/(label+'.json'))
            if done:
                B.save(dest,done)
                # Records the destination's engine, so a later run does not archive it again.
                if not (dest.parent.parent/'battle-config.json').exists():B.save(dest.parent.parent/'battle-config.json',read(out/'battle-config.json'))
    return len(jobs)


def battle(args):
    if args.worker is not None:return print(battle_worker(args.worker,args.workers,args.battle_backend),'panels')
    # The fast evaluator already runs six workers; one process feeds it every panel.
    if args.battle_backend=='fast':args.workers=1
    # Labels from the other engine are archived, so every arm is re-battled on one engine.
    for f in folders():B.switch_engine(f,args.battle_backend)
    jobs=[subprocess.Popen([sys.executable,__file__,'battle','--output',str(OUT),'--workers',str(args.workers),'--worker',str(i),
                            '--battle-backend',args.battle_backend],
                           stdout=(OUT/f'battles-{i}.log').open('a'),stderr=subprocess.STDOUT) for i in range(args.workers)]
    failed=[i for i,job in enumerate(jobs) if job.wait()]
    report(args)
    if failed:raise RuntimeError(f'battle workers {failed} stopped; see battles-<n>.log, then rerun')


def failure(record):
    if record['valid']:return None
    error=record.get('error') or 'unknown'
    return next((name for pattern,name in FAILURES if re.search(pattern,error)),'other')


def interval(by_start,seed):
    import numpy as np
    values=np.array([sum(v)/len(v) for _,v in sorted(by_start.items())])
    draws=np.random.default_rng(seed).choice(values,(4000,len(values))).mean(axis=1)
    return float(values.mean()),*map(float,np.quantile(draws,[.025,.975]))


def report(args):
    rows=collections.defaultdict(dict);cost=collections.Counter();why={arm:collections.Counter() for arm in ARMS};manifest=None
    for f in folders():
        m=read(f/'manifest.json');manifest=manifest or m;arm=m.get('prompt_version',V.ORIGINAL)
        for t in m['trials']:
            r=read(f/'completions'/(t['id']+'.json'))
            if r is None:continue
            score=read(f/'labels'/('llm-'+t['id']+'.json'))
            rows[t['id']].setdefault(arm,[]).append(dict(valid=r['valid'],win=score['wins']/score['battles'] if score else None))
            cost[arm]+=r['response'].get('usage',{}).get('cost',0) or 0
            if not r['valid']:why[arm][failure(r)]+=1
    if manifest is None:raise ValueError('nothing prepared')
    trials={t['id']:t for t in manifest['trials']};old,new=ARMS
    paired=[tid for tid in trials if old in rows[tid] and new in rows[tid]]
    rate=lambda tid,arm:sum(x['valid'] for x in rows[tid][arm])/len(rows[tid][arm])
    families=[dict(task=task,tasks=sum(trials[t]['task']==task for t in paired),
                   **{arm:sum(rate(t,arm) for t in paired if trials[t]['task']==task) for arm in ARMS}) for task in manifest['grid']]
    status=dict(arms=ARMS,system_sha256={arm:V.system_sha256(arm,manifest) for arm in ARMS},tasks=len(trials),paired_tasks=len(paired),
                starters=len({t['start'] for t in trials.values()}),responses={arm:sum(len(rows[t].get(arm,[])) for t in trials) for arm in ARMS},
                legal={arm:sum(x['valid'] for t in trials for x in rows[t].get(arm,[])) for arm in ARMS},families=families,
                failures={arm:dict(why[arm].most_common()) for arm in ARMS},cost_usd=dict(cost))
    lines=['# Regulation prompt test','',f'`{new}` against `{old}` on the same {len(trials)} masked-completion tasks '
           f'({status["starters"]} meta starting teams, seven mask families, three mask sizes).','',
           '| Prompt | Responses | Legal | Share | Cost (USD) |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        n=status['responses'][arm]
        lines.append(f'| `{arm}` | {n} | {status["legal"][arm]} | {status["legal"][arm]/n:.1%} | {cost[arm]:.4f} |' if n else f'| `{arm}` | 0 | — | — | — |')
    if paired:
        by_start=collections.defaultdict(list)
        for t in paired:by_start[trials[t]['start']].append(rate(t,new)-rate(t,old))
        status['legal_difference']=dict(zip(('mean','low','high'),interval(by_start,manifest['seed'])))
        d=status['legal_difference']
        lines+=['',f'Legal share, new minus old, each starting team weighted equally: **{100*d["mean"]:+.1f} points** '
                f'[{100*d["low"]:+.1f}, {100*d["high"]:+.1f}] (95% bootstrap over {len(by_start)} starting teams, {len(paired)} paired tasks).',
                '','| Mask family | Tasks | '+' | '.join(f'`{a}` legal per repeat' for a in ARMS)+' |','|---|---:|---:|---:|']
        lines+=[f'| {f["task"]} | {f["tasks"]} | {f[old]:.1f} | {f[new]:.1f} |' for f in families]
        both=[t for t in paired if all(x['win'] is not None for a in ARMS for x in rows[t][a]) and all(rate(t,a)==1 for a in ARMS)]
        if both:
            win=lambda t,a:sum(x['win'] for x in rows[t][a])/len(rows[t][a])
            by_start=collections.defaultdict(list)
            for t in both:by_start[trials[t]['start']].append(win(t,new)-win(t,old))
            flat=[win(t,new)-win(t,old) for t in both];mean=sum(flat)/len(flat)
            error=(sum((x-mean)**2 for x in flat)/(len(flat)-1)/len(flat))**.5 if len(flat)>1 else float('nan')
            status['win_difference']=dict(zip(('mean','low','high'),interval(by_start,manifest['seed'])),tasks=len(both),
                                          task_standard_error=error,tasks_with_any_difference=sum(x!=0 for x in flat))
            d=status['win_difference']
            lines+=['',f'Win rate against the supplied meta pool (50 battles per team, behaviour-cloning policy on both sides), '
                    f'new minus old on the {len(both)} tasks where every response was legal: **{100*d["mean"]:+.1f} points**, '
                    f'standard error {100*error:.1f} over tasks; starting-team bootstrap [{100*d["low"]:+.1f}, {100*d["high"]:+.1f}]. '
                    f'Only {d["tasks_with_any_difference"]} of those tasks differ at all; identical teams share one battle panel. '
                    'Conditional on legality, so it is not a randomised comparison.']
        else:lines+=['','Battle win rates: not scored yet.']
        scored={arm:[x['win'] for t in trials for x in rows[t].get(arm,[]) if x['win'] is not None] for arm in ARMS}
        if all(scored.values()):
            status['win_rate_legal']={arm:dict(teams=len(v),mean=sum(v)/len(v)) for arm,v in scored.items()}
            lines+=['','Mean win rate of every legal completion that was battled: '+'; '.join(f'`{arm}` {sum(v)/len(v):.3f} over {len(v)} teams' for arm,v in scored.items())+'.']
    lines+=['','Rejected responses by reason:','','| Reason | '+' | '.join(f'`{a}`' for a in ARMS)+' |','|---|---:|---:|']
    lines+=[f'| {name} | {why[old][name]} | {why[new][name]} |' for name in sorted(set(why[old])|set(why[new]),key=lambda n:-why[old][n]-why[new][n])]
    B.save(OUT/'status.json',status);(OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','battle','report'])
    p.add_argument('--parent',type=Path,default=B.ROOT/'results/ling-mb-prompt-comparison',help='folder holding the frozen 147-task manifest')
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--replicates',type=int,default=2)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--worker',type=int)
    p.add_argument('--battle-backend',choices=['legacy','fast'],default='fast',
                   help='battle: fast = one process on the six-worker evaluator; legacy = --workers pair_shard processes')
    args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
    globals()[args.command](args)
