"""Paired prompt pilot and live charts; retain the full experiment unchanged."""
import argparse
import copy
import fcntl
import getpass
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import llm_baseline as B

PARENT=B.ROOT/'results/llm-baseline-meta-starters'
DIFF=B.ROOT/'results/diffusion-baseline-meta-starters-v3'
OUT=B.ROOT/'results/ling-mb-prompt-comparison'
HERE=B.ROOT/'dashboard'


def prepare():
    old=json.loads((PARENT/'manifest.json').read_text());m=copy.deepcopy(old)
    starts=sorted(random.Random(20261011).sample(range(len(m['starts'])),7))
    grid={task:sorted({1,(max(ks)+1)//2,max(ks)}) for task,ks in m['grid'].items()}
    m['trials']=[t for t in m['trials'] if t['start'] in starts and t['k'] in grid[t['task']]]
    m.update(prompt_version='mb-grounded-v2',grid=grid,prompt_comparison=dict(selected_starts=starts,selection_seed=20261011,
         selection='seven starters sampled before observing revised results; min/mid/max mask per family',parent_manifest_sha256=B.digest(old)))
    assert len(m['trials'])==147
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==m
    else:B.save(OUT/'manifest.json',m)
    return m


def read(path):return json.loads(path.read_text()) if path.exists() else None


def label(folder,name,worker_cache):
    return read(folder/'labels'/(name+'.json')) or worker_cache.get(name)


def publish():
    m=read(OUT/'manifest.json');records=[]
    worker_cache={p.stem:read(p) for p in (PARENT/'workers').glob('*/labels/*.json')}
    worker_cache.update({p.stem:read(p) for p in (OUT/'workers').glob('*/labels/*.json')})
    for t in m['trials']:
        tid=t['id'];old=read(PARENT/'completions'/(tid+'.json'));new=read(OUT/'completions'/(tid+'.json'));diff=read(DIFF/'completions'/(tid+'.json'))
        # Old and revised Ling labels have identical names; don't conflate caches.
        oldscore=read(PARENT/'labels'/('llm-'+tid+'.json'))
        if not oldscore:
            for p in (PARENT/'workers').glob('*/labels/llm-'+tid+'.json'):oldscore=read(p);break
        newscore=read(OUT/'labels'/('llm-'+tid+'.json'))
        if not newscore:
            for p in (OUT/'workers').glob('*/labels/llm-'+tid+'.json'):newscore=read(p);break
        records.append(dict(id=tid,start=t['start'],task=t['task'],k=t['k'],old=old['valid'],
             new=new['valid'] if new and (new['valid'] or 'error' in new) else None,diffusion=diff['valid'],
             oldCost=old['response']['usage'].get('cost',0),newCost=new['response']['usage'].get('cost',0) if new else 0,
             oldScore=oldscore,newScore=newscore,diffScore=label(DIFF,'diffusion-'+tid,worker_cache),error=new.get('error') if new else None))
    done=[r for r in records if r['new'] is not None]
    battle=[r for r in done if r['oldScore'] and r['newScore'] and r['diffScore']]
    names=['old','new','diffusion'];scores={}
    for name,key in zip(names,['oldScore','newScore','diffScore']):
        groups={r['start'] for r in battle}
        scores[name]=sum(sum(r[key]['wins']/r[key]['battles'] for r in battle if r['start']==s)/sum(r['start']==s for r in battle) for s in groups)/len(groups) if groups else None
    families=[dict(task=task,n=sum(r['task']==task for r in done),**{name:sum(r[name] for r in done if r['task']==task) for name in names}) for task in m['grid']]
    state=read(OUT/'progress.json') or {'stage':'prepared'}
    result=dict(updated=time.time(),stage=state['stage'],total=147,completed=len(done),starters=[i+1 for i in m['prompt_comparison']['selected_starts']],
         legal={name:sum(r[name] for r in done) for name in names},families=families,battlePairs=len(battle),scores=scores,
         cost={'old':sum(r['oldCost'] for r in done),'new':sum(r['newCost'] for r in done)},
         full={'oldLegal':1600,'diffusionLegal':4998,'total':4998},rows=records)
    B.save(HERE/'prompt-comparison.json',result)


def watch():
    with (OUT/'.watch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            publish()
            if (read(OUT/'progress.json') or {}).get('stage') in ('complete','interrupted'):return
            time.sleep(15)


def run():
    m=prepare()
    B.save(OUT/'progress.json',dict(stage='generating'))
    with (OUT/'watch.log').open('a') as log:subprocess.Popen([sys.executable,__file__,'watch'],stdout=log,stderr=subprocess.STDOUT)
    if not all((OUT/'completions'/(t['id']+'.json')).exists() for t in m['trials']) and not os.environ.get('OPENROUTER_API_KEY'):
        os.environ['OPENROUTER_API_KEY']=getpass.getpass('OpenRouter key (not saved): ')
    def command(logname,argv):
        with (OUT/logname).open('a') as log:
            subprocess.run([sys.executable,*argv],stdout=log,stderr=subprocess.STDOUT,check=True)
    command('generation.log',[str(B.ROOT/'scripts/llm_baseline.py'),'complete','--output',str(OUT)])
    B.save(OUT/'progress.json',dict(stage='waiting_for_main_battles'));publish()
    # Avoid simultaneous writers calculating the same exact-panel cache entry.
    while (read(PARENT/'pipeline-progress.json') or {}).get('status')!='complete':
        if (read(PARENT/'pipeline-progress.json') or {}).get('status')=='interrupted':
            raise RuntimeError('Main battle run interrupted; resume it before this queued comparison')
        time.sleep(30)
    B.save(OUT/'progress.json',dict(stage='scoring'))
    frozen=OUT/'diffusion-reference';B.save(frozen/'manifest.json',m)
    for t in m['trials']:B.save(frozen/'completions'/(t['id']+'.json'),read(DIFF/'completions'/(t['id']+'.json')))
    command('battles.log',[str(B.ROOT/'scripts/dense_grid_battles.py'),'--baseline',str(OUT),'--diffusion',str(frozen),'--workers','6'])
    B.save(OUT/'progress.json',dict(stage='complete'));publish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','watch','publish']);args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.command=='watch':watch()
    else:
        with (OUT/'.run.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:globals()[args.command]()
            except Exception:
                B.save(OUT/'progress.json',dict(stage='interrupted'));raise
