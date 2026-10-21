"""Score independent panels in isolated processes; preserve per-panel settings."""
import argparse
import fcntl
import concurrent.futures
import json
from types import SimpleNamespace
from repo_paths import ROOT  # also puts src/ first on sys.path
import battles
import results
import showdown_validator

BASE=ROOT/'results/llm-baseline-full-grid'
DIFF=ROOT/'results/diffusion-baseline-full-grid'


def worker(index,count,baseline=None,diffusion=None):
    global BASE,DIFF
    if baseline is not None:
        from pathlib import Path
        BASE,DIFF=Path(baseline),Path(diffusion)
    m=json.loads((BASE/'manifest.json').read_text())
    out=BASE/'workers'/str(index)
    results.save(out/'manifest.json',m)
    candidates=[(f'start-{i}',team,BASE) for i,team in enumerate(m['starts'])]
    for t in m['trials']:
        candidates.append(('random-'+t['id'],t['random_team'],BASE))
        for source,prefix in ((BASE,'llm-'),(DIFF,'diffusion-')):
            p=source/'completions'/(t['id']+'.json')
            if p.exists():
                r=json.loads(p.read_text())
                if r['valid']: candidates.append((prefix+t['id'],r['team'],source))
    import hps_generate as h, matchup_db as db
    # Identical teams always go to the same process, preventing cache races.
    selected=[(label,team,dest) for label,team,dest in candidates
              if int(results.digest(db.canonical_paste(showdown_validator.paste(team,h))),16)%count==index]
    battles.battle(SimpleNamespace(output=out),candidates_override=[(label,team) for label,team,_ in selected])
    for label,_,dest in selected:
        p=out/'labels'/(label+'.json')
        results.save(dest/'labels'/p.name,json.loads(p.read_text()))
    return index,len(selected)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6,help='ignored: the fast evaluator runs its own six workers, fed by one process')
    parser.add_argument('--baseline',default=str(BASE))
    parser.add_argument('--diffusion',default=str(DIFF))
    args=parser.parse_args()
    from pathlib import Path
    BASE,DIFF=Path(args.baseline).resolve(),Path(args.diffusion).resolve()
    lock=(BASE/'.battle-workers.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # The fast evaluator already runs six workers; one process feeds it every panel.
    args.workers=1
    # Labels from the removed legacy engine are archived, so every destination is re-battled on one engine.
    for dest in (BASE,DIFF):battles.archive_legacy_labels(dest)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,i,args.workers,str(BASE),str(DIFF)) for i in range(args.workers)]
        for future in concurrent.futures.as_completed(futures): print('Finished worker',future.result(),flush=True)
    config=json.loads((BASE/'workers/0/battle-config.json').read_text())
    for i in range(args.workers): assert json.loads((BASE/f'workers/{i}/battle-config.json').read_text())==config
    for dest in (BASE,DIFF): results.save(dest/'battle-config.json',config)
