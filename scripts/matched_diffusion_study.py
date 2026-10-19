"""Frozen 20-starter F1/G2 v3 system comparison; no API calls or reranking."""
import collections
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace
import llm_baseline as B
import diffusion_baseline_v3 as D
import reasoning_battles as RB
import reasoning_experiment as E

OUT=B.ROOT/'results/diffusion-matched-20261013'
SOURCE=B.ROOT/'results/mercury-reasoning-20261013'

def snapshot(stage):
    m=json.loads((OUT/'r0/manifest.json').read_text())
    rows=[]
    for rep in range(3):
        for t in m['trials']:
            f=OUT/f'r{rep}/completions'/(t['id']+'.json')
            r=json.loads(f.read_text()) if f.exists() else None
            label=OUT/f'r{rep}/labels'/('diffusion-'+t['id']+'.json')
            lab=json.loads(label.read_text()) if label.exists() else None
            if lab: assert r['valid'] and lab['battles']==50
            rows.append(dict(arm='diffusion',replicate=rep,trial=t['id'],start=t['start'],task=t['task'],returned=r is not None,legal=bool(r and r['valid']),supported=bool(r and r['supported']),win_rate=lab['wins']/50 if lab else None,cache_key=lab['cache_key'] if lab else None))
    pairs={}
    for study in ['mercury-reasoning-20261013','reasoning-expanded-20261012']:
        old=json.loads((B.ROOT/'results'/study/'status.json').read_text())
        assert old['complete']
        for arm in old['arms']:
            controls={(r['trial'],r['replicate']):r for r in old['rows'] if r['arm']==arm}
            legal=[];wins=[]
            for row in rows:
                c=controls[(row['trial'],row['replicate'])]
                if row['returned']:
                    legal.append((row['start'],int(row['legal'])-int(c['legal'])))
                    if row['legal'] and c['legal'] and row['win_rate'] is not None and c['win_rate'] is not None:
                        wins.append((row['start'],row['win_rate']-c['win_rate']))
            pairs[study+'/'+arm]=dict(legality=E.cluster_interval(legal,m['seed']),battles=E.cluster_interval(wins,m['seed']))
    scored=[r for r in rows if r['win_rate'] is not None]
    result=dict(stage=stage,updated_at=time.time(),planned=420,returned=sum(r['returned'] for r in rows),supported=sum(r['supported'] for r in rows),legal=sum(r['legal'] for r in rows),scored=len(scored),unique_panels=len({r['cache_key'] for r in scored}),all_legal_win_rate=E.average([r['win_rate'] for r in scored]),comparisons=pairs,rows=rows,api_cost_usd=0)
    result['complete']=result['returned']==420 and result['legal']==result['scored']
    B.save(OUT/'status.json',result)
    return result

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=json.loads((SOURCE/'off/r0/manifest.json').read_text())
        config=json.loads((SOURCE/'off/r0/battle-config.json').read_text())
        assert len(m['starts'])==20 and len(m['trials'])==140
        for f,h in config['fingerprints'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
        for rep in range(3):
            folder=OUT/f'r{rep}';folder.mkdir(exist_ok=True)
            for name,data in [('manifest.json',m),('battle-config.json',config)]:
                path=folder/name
                if path.exists():assert json.loads(path.read_text())==data
                else:B.save(path,data)
        audit=OUT/'preservation-audit.json'
        if not audit.exists():
            hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for study in ['mercury-reasoning-20261013','reasoning-expanded-20261012'] for f in (B.ROOT/'results'/study).glob('*/r*/**/*.json')}
            B.save(audit,dict(before=hashes))
        B.save(OUT/'design.json',dict(starters=20,masks=140,repeats=3,members=D.MEMBERS,generation_seeds=[m['seed']+100000*rep for rep in range(3)],battle_seed=m['seed'],attempts=1,stat_points='Corpus spread or random legal fallback, not neural prediction',input='Categorical context only; no opponent-paste input',comparison='Matched complete systems, not equal-input architecture or reasoning effect',started_at=time.time()))
        snapshot('generating')
        for rep in range(3):
            gen=copy.deepcopy(m);gen['seed']=m['seed']+100000*rep
            D.generate(SimpleNamespace(output=OUT/f'r{rep}',manifest=gen,members=D.MEMBERS))
            snapshot('generating')
        h,db=B.runtime();groups={}
        for rep in range(3):
            for t in m['trials']:
                row=json.loads((OUT/f'r{rep}/completions'/(t['id']+'.json')).read_text())
                if not row['valid']:continue
                assert B.masked(row['team'],t['paths'])==B.masked(m['starts'][t['start']],t['paths'])
                key=B.digest(dict(team=db.canonical_paste(B.paste(row['team'],h)),schedule=B.schedule_for(m,t['start'])))
                g=groups.setdefault(key,dict(key=key,team=row['team'],start=t['start'],destinations=[]))
                g['destinations'].append(str(OUT/f'r{rep}/labels'/('diffusion-'+t['id']+'.json')))
        pending=[g for g in groups.values() if any(not Path(x).exists() for x in g['destinations'])]
        con=db.connect();validator=h.Validator()
        try:
            for g in pending:
                tid=db.add_team(con,B.paste(g['team'],h),'matched-diffusion',validator)[0];db.paste_file(con,tid)
            for opponent in m['opponents']:db.paste_file(con,opponent['team_id'])
        finally:validator.close();con.close()
        snapshot('scoring')
        buckets=[[] for _ in range(4)]
        for g in pending:buckets[int(g['key'],16)%4].append(g)
        for i in range(4):B.save(OUT/f'battle-workers/{i}/battle-config.json',config)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs=[pool.submit(RB.worker,str(OUT),i,m,b) for i,b in enumerate(buckets) if b]
            for job in concurrent.futures.as_completed(jobs):job.result();snapshot('scoring')
        assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==v for f,v in json.loads(audit.read_text())['before'].items())
        assert snapshot('complete')['complete']

if __name__=='__main__':
    try:run()
    except Exception as exc:
        if (OUT/'r0/manifest.json').exists():snapshot('failed')
        B.save(OUT/'failure.json',dict(error=str(exc),time=time.time()))
        raise
