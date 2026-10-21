"""Resume the full grid from disk, without repeating saved model calls."""
import fcntl
import getpass
import json
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
from repo_paths import ROOT  # also puts src/ first on sys.path
from showdown_validator import SHOWDOWN, VGC_BENCH
import results

# sha256 of the behaviour-cloning policy checkpoint (bc_100.zip) both sides play.
BC_SHA256 = "57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b"


def restore_runtime():
    """Recreate disposable aliases after OS cleanup; keep durable data untouched."""
    runtime=Path.home()/'.local/share/vgc-pilot-runtime'
    scratch=Path('/tmp/vgc-pilot')
    scratch.mkdir(parents=True,exist_ok=True)
    bench=VGC_BENCH
    aliases={scratch/'.venv':runtime/'venv',scratch/'data':runtime/'data',
             scratch/'src':ROOT/'src',scratch/'learnset_true.json':runtime/'files/learnset_true.json',
             scratch/'top50_evs.json':runtime/'files/top50_evs.json',
             scratch/'vgc-bench/teams':bench/'teams',scratch/'vgc-bench/vgc_bench':bench/'vgc_bench',
             scratch/'vgc-bench/pokemon-showdown':SHOWDOWN}
    for alias,target in aliases.items():
        if not alias.exists():
            if not target.exists(): raise RuntimeError('Missing durable dependency: '+str(target))
            if alias.is_symlink(): alias.unlink()
            alias.parent.mkdir(parents=True,exist_ok=True)
            alias.symlink_to(target,target_is_directory=target.is_dir())
    policy=Path('/tmp/bc_100.zip')
    durable=runtime/'files/bc_100.zip'
    if hashlib.sha256(durable.read_bytes()).hexdigest()!=BC_SHA256:
        raise RuntimeError('Durable BC checkpoint fingerprint mismatch')
    if not policy.exists(): shutil.copy2(durable,policy)
    if hashlib.sha256(policy.read_bytes()).hexdigest()!=BC_SHA256:
        raise RuntimeError('Runtime BC checkpoint fingerprint mismatch')


def main():
    restore_runtime()
    output=ROOT/'results/llm-baseline-full-grid'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=[('prepare',['dense_mask_grid.py','prepare']),
              ('ling',['llm_baseline.py','complete','--output',str(output)]),
              ('diffusion',['diffusion_baseline.py','generate','--baseline',str(output),'--output',str(ROOT/'results/diffusion-baseline-full-grid'),'--members','F1_final_medium_noheldout_regmb_v2','G2_family_matrix_noprtrain_noheldout_regmb_v2']),
              ('battles',['dense_grid_battles.py','--workers','6']),
              ('report',['dense_mask_grid.py','report']),
              ('dashboard',[str(ROOT/'dashboard/build.py')])]
        for stage,command in jobs:
            if stage=='prepare' and (output/'manifest.json').exists(): continue
            if stage=='ling':
                manifest=json.loads((output/'manifest.json').read_text())
                missing=[t['id'] for t in manifest['trials'] if not (output/'completions'/(t['id']+'.json')).exists()]
                uncertain=[t for t in missing if (output/'completions'/(t+'.pending')).exists()]
                if uncertain: raise RuntimeError('Uncertain sent requests require reconciliation before retrying: '+', '.join(uncertain))
                if missing and not os.environ.get('OPENROUTER_API_KEY'):
                    os.environ['OPENROUTER_API_KEY']=getpass.getpass('OpenRouter API key (not saved): ')
            command[0]=str(ROOT/'scripts'/command[0]) if not command[0].startswith('/') else command[0]
            results.save(output/'pipeline-progress.json',dict(stage=stage,status='running',updated_at=time.time()))
            print('Stage:',stage,flush=True)
            with (output/'pipeline.log').open('a') as log:
                result=subprocess.run([sys.executable,*command],stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                results.save(output/'pipeline-progress.json',dict(stage=stage,status='interrupted',returncode=result.returncode,updated_at=time.time()))
                raise RuntimeError('Stage failed: '+stage+'; inspect '+str(output/'pipeline.log'))
        results.save(output/'pipeline-progress.json',dict(stage='complete',status='complete',updated_at=time.time()))
        print('Full grid complete. Dashboard rebuilt.',flush=True)


if __name__=='__main__': main()
