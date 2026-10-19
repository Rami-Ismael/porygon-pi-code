"""Publish local run progress and the final dashboard; makes no model calls."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import llm_baseline as B
from meta_starter_grid import BASE,DIFF,selection


def main():
    with (BASE/'.progress-watcher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        receipts={};diffusion={}
        while True:
            for folder,cache in ((BASE,receipts),(DIFF,diffusion)):
                for p in (folder/'completions').glob('*.json'):
                    if p.name.endswith('.http-error.json'): continue
                    stamp=p.stat().st_mtime_ns
                    if p.name in cache and cache[p.name][0]==stamp: continue
                    r=json.loads(p.read_text())
                    cache[p.name]=(stamp,r['valid'],r.get('supported',True),r.get('response',{}).get('usage',{}).get('cost',0) or 0)
            progress=json.loads((BASE/'pipeline-progress.json').read_text())
            payload=dict(updated=time.time(),starters=49,tasks=4998,stage=progress.get('stage',progress['status']),status=progress['status'],
                         ling=len(receipts),lingLegal=sum(r[1] for r in receipts.values()),cost=sum(r[3] for r in receipts.values()),
                         diffusion=len(diffusion),diffusionLegal=sum(r[1] for r in diffusion.values()),
                         panels=len(list((BASE/'workers').glob('*/labels/*.json'))),error=progress.get('error'))
            payload['diffusionVersion']=selection.get('version','v2')
            payload['previousDiffusion']=len(list((B.ROOT/'results/diffusion-baseline-meta-starters/completions').glob('*.json')))
            training=list((Path.home()/'.local/share/vgc-pilot-runtime/regmb-v3').glob('*-progress.json'))
            if training:
                latest=max(training,key=lambda p:p.stat().st_mtime_ns)
                payload['training']=dict(json.loads(latest.read_text()),member=latest.name.replace('-progress.json',''))
            B.save(B.ROOT/'dashboard/meta-progress.json',payload)
            if progress['status']=='complete':
                # Report through the current code so a long-running coordinator
                # can still publish the latest dashboard implementation.
                result=subprocess.run([sys.executable,str(B.ROOT/'scripts/meta_starter_grid.py'),'report'])
                if result.returncode: time.sleep(15);continue
                subprocess.run([sys.executable,str(B.ROOT/'dashboard/build.py')],check=True)
                return
            if progress['status']=='interrupted': return
            time.sleep(15)


if __name__=='__main__': main()
