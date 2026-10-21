"""Full integer-mask experiment starting from the frozen meta pool."""
import argparse
import copy
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
from repo_paths import ROOT  # also puts src/ first on sys.path
import completion_study
import results
import showdown_validator
import team_masks
from resume_full_grid import restore_runtime

BASE=ROOT/'results/llm-baseline-meta-starters'
SELECTION=BASE/'diffusion-selection.json'
selection=json.loads(SELECTION.read_text()) if SELECTION.exists() else {}
DIFF=ROOT/'results'/selection.get('results','diffusion-baseline-meta-starters')
MEMBERS=selection.get('members',['F1_final_medium_noheldout_regmb_v2','G2_family_matrix_noprtrain_noheldout_regmb_v2'])


def random_control(team,paths,tables,rng,h,validator,regulation):
    from corpus import norm,NATURES
    if any(len(p)==1 for p in paths): return completion_study.random_fill(team,paths,tables,rng,h,validator)
    allowed={tuple(p) for p in paths}
    names=regulation['names']
    pool={norm(x) for x in tables['items']}
    for _ in range(1000):
        result=copy.deepcopy(team)
        for i,slot in enumerate(team):
            spec=regulation['slots'][norm(slot['species'])]
            ma=(i,'ability') in allowed;mi=(i,'item') in allowed
            if ma or mi:
                pairs=[(a,it) for a,items in spec['abilityItems'].items() for it in items
                       if (ma or a==norm(slot['ability'])) and (mi or it==norm(slot['item']))
                       and (not mi or it in pool)]
                if not pairs: raise ValueError(f'No compatible random-fill pair for {slot["species"]}')
                ability,item=pairs[int(rng.integers(len(pairs)))]
                if ma: result[i]['ability']=names.get(ability,ability)
                if mi: result[i]['item']=names.get(item,item)
            missing=[j for j in range(4) if (i,'moves',j) in allowed]
            fixed={norm(v) for j,v in enumerate(slot['moves']) if j not in missing}
            moves=[v for v in h.TRUE[norm(slot['species'])] if norm(v) not in fixed]
            if missing:
                for j,value in zip(missing,rng.choice(moves,len(missing),replace=False)): result[i]['moves'][j]=str(value)
            if (i,'nature') in allowed: result[i]['nature']=str(rng.choice(NATURES))
            if (i,'evs') in allowed: result[i]['evs']=h.sample_spread(rng)
        team_masks.validate_shape(result)
        if validator(showdown_validator.paste(result,h)) is None:
            assert team_masks.masked(result,paths)==team_masks.masked(team,paths)
            return result
    raise RuntimeError('Compatible random-fill control failed final validation')


def prepare():
    import numpy as np
    if (BASE/'manifest.json').exists(): return
    old=json.loads((ROOT/'results/llm-baseline-full-grid/manifest.json').read_text())
    import hps_generate as h, matchup_db as db
    tables=h.build_tables()
    regulation=json.loads((ROOT/'data/regmb-vocabulary.json').read_text())
    m={k:copy.deepcopy(v) for k,v in old.items() if k not in ('starts','trials','extension','source_sha256')}
    m.update(seed=20261010,starts=[],trials=[],starter_kind='frozen meta pool',exclude_starter_team_ids=[o['team_id'] for o in old['opponents']],
             starter_provenance=[],parent_manifest_sha256=results.digest(old),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    m['sampler']='random control: compatible ability/item pairs (corpus item pool), distinct legal moves, random spreads/natures; whole slots use hierarchical product; final simulator validation'
    m['preserve_empty_move_slots']=True
    validator=h.Validator()
    con=sqlite3.connect(f'file:{db.DEFAULT_DB}?mode=ro',uri=True)
    try:
        for i,opponent in enumerate(m['opponents']):
            team=db.canonical_slots(opponent['paste'])
            for slot in team: slot['moves'] += ['']*(4-len(slot['moves']))
            team_masks.validate_shape(team)
            assert db.canonical_paste(showdown_validator.paste(team,h))==db.canonical_paste(opponent['paste'])
            error=validator(showdown_validator.paste(team,h))
            if error: raise ValueError(error)
            m['starts'].append(team)
            sources=con.execute('select source,source_id,event,placement,date from team_source where team_id=?',(opponent['team_id'],)).fetchall()
            m['starter_provenance'].append(dict(team_id=opponent['team_id'],sources=sources))
            for task,sizes in m['grid'].items():
                order=team_masks.paths_for(task)
                random.Random(f"{m['seed']}:{i}:{task}").shuffle(order)
                for k in sizes:
                    tid=f's{i}-{task}-{k}'
                    checkpoint=BASE/'prepared-trials-v2'/(tid+'.json')
                    if checkpoint.exists():
                        trial=json.loads(checkpoint.read_text())
                        assert trial['paths']==order[:k]
                    else:
                        seed=int(hashlib.sha256(f"meta:{m['seed']}:{tid}".encode()).hexdigest()[:16],16)
                        control=random_control(team,order[:k],tables,np.random.default_rng(seed),h,validator,regulation)
                        trial=dict(id=tid,start=i,task=task,k=k,paths=order[:k],random_team=control)
                        results.save(checkpoint,trial)
                    m['trials'].append(trial)
            print('Prepared starter',i+1,'of',len(m['opponents']),flush=True)
    finally:
        con.close();validator.close()
    assert len(m['starts'])==49 and len(m['trials'])==4998
    for dest in (BASE,DIFF): results.save(dest/'manifest.json',m)


def status():
    mpath=BASE/'manifest.json'
    if not mpath.exists(): return dict(stage='preparing',prepared_trials=len(list((BASE/'prepared-trials-v2').glob('*.json'))))
    m=json.loads(mpath.read_text())
    result=dict(starters=len(m['starts']),tasks=len(m['trials']))
    for name,folder in [('ling',BASE),('diffusion',DIFF)]:
        rows=[json.loads(p.read_text()) for p in (folder/'completions').glob('*.json') if not p.name.endswith('.http-error.json')]
        result[name]=dict(completed=len(rows),legal=sum(r['valid'] for r in rows),saved_labels=len(list((folder/'labels').glob('*.json'))))
        if name=='ling': result[name]['cost_usd']=sum(r['response'].get('usage',{}).get('cost',0) or 0 for r in rows)
    result['worker_panels']=len(list((BASE/'workers').glob('*/labels/*.json')))
    return result


def run():
    if selection.get('version')=='v3':
        raise RuntimeError('This experiment is upgrading to v3; resume with upgrade_meta_v3.py')
    prepare()
    m=json.loads((BASE/'manifest.json').read_text())
    missing=[t for t in m['trials'] if not (BASE/'completions'/(t['id']+'.json')).exists()]
    uncertain=[t['id'] for t in missing if (BASE/'completions'/(t['id']+'.pending')).exists()]
    if uncertain: raise RuntimeError('Reconcile uncertain requests before retrying: '+str(uncertain))
    if missing and not os.environ.get('OPENROUTER_API_KEY'):
        os.environ['OPENROUTER_API_KEY']=getpass.getpass('OpenRouter key (not saved): ')
    def launch(name,argv):
        log=(BASE/(name+'.log')).open('a')
        proc=subprocess.Popen([sys.executable,str(ROOT/'scripts'/argv[0]),*argv[1:]],stdout=log,stderr=subprocess.STDOUT)
        log.close()
        return proc
    jobs=[('ling-'+str(i),['llm_baseline.py','complete','--output',str(BASE),'--shards','4','--shard',str(i)]) for i in range(4)]
    jobs.append(('diffusion',['diffusion_baseline.py','generate','--baseline',str(BASE),'--output',str(DIFF),'--members',*MEMBERS]))
    results.save(BASE/'pipeline-progress.json',dict(stage='generation',status='running',time=time.time()))
    launch('progress',['watch_meta_grid.py'])
    active=[(name,launch(name,argv)) for name,argv in jobs]
    codes=[(name,p.wait()) for name,p in active]
    if any(code for _,code in codes): raise RuntimeError('Generation interrupted; saved outputs retained: '+str(codes))
    results.save(BASE/'pipeline-progress.json',dict(stage='battles',status='running',time=time.time()))
    proc=launch('battles',['dense_grid_battles.py','--baseline',str(BASE),'--diffusion',str(DIFF),'--workers','6'])
    if proc.wait(): raise RuntimeError('Scoring interrupted; rerun to resume cached panels')
    report()
    results.save(BASE/'pipeline-progress.json',dict(stage='complete',status='complete',time=time.time()))


def report():
    from types import SimpleNamespace
    completion_study.report(SimpleNamespace(output=BASE))
    m=json.loads((BASE/'manifest.json').read_text())
    s=status()
    assert json.loads((BASE/'status.json').read_text())['complete']
    assert s['diffusion']['completed']==len(m['trials'])
    assert s['diffusion']['saved_labels']==s['diffusion']['legal']
    assert json.loads((BASE/'battle-config.json').read_text())==json.loads((DIFF/'battle-config.json').read_text())
    pairs=[]
    for t in m['trials']:
        a=BASE/'labels'/('llm-'+t['id']+'.json');b=DIFF/'labels'/('diffusion-'+t['id']+'.json')
        if a.exists() and b.exists():
            x,y=json.loads(a.read_text()),json.loads(b.read_text())
            pairs.append(dict(trial=t['id'],start=t['start'],delta_pp=2*(y['wins']-x['wins'])))
    results.save(BASE/'experiment-summary.json',dict(s,paired=pairs))
    supported=sum(json.loads(p.read_text())['supported'] for p in (DIFF/'completions').glob('*.json'))
    results.save(DIFF/'summary.json',dict(supported=supported,legal=s['diffusion']['legal'],scored=s['diffusion']['saved_labels'],pairs=pairs))
    version=selection.get('version','v2')
    lines=['# Meta-starter full grid','',f"49 frozen meta teams; 4,998 tasks. Ling: {s['ling']['legal']} legal. Diffusion: {s['diffusion']['legal']} legal. API cost: ${s['ling']['cost_usd']:.6f}.",'',f'{len(pairs)} jointly scored tasks. Each starter is excluded from its own prompt and opponent panel. Each panel contains 50 battles over the other 48 opponents. Diffusion: F1/G2 {version}, ask 0.5, guidance 2, temperature 1. BC policy on both sides. This is an in-pool improvement experiment, not held-out generalization.']
    (ROOT/'docs/meta-starter-grid-results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','status','report'],default='run',nargs='?');args=p.parse_args()
    if args.command=='status': print(json.dumps(status()));sys.exit()
    restore_runtime();BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'.pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try: globals()[args.command]()
        except Exception as exc:
            results.save(BASE/'pipeline-progress.json',dict(status='interrupted',error=str(exc),time=time.time()))
            raise
