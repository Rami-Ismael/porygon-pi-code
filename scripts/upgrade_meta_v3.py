"""Finish existing diffusion generation, train v3, regenerate, then score."""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from repo_paths import ROOT  # also puts src/ first on sys.path
import results
from resume_full_grid import restore_runtime

BASE=ROOT/'results/llm-baseline-meta-starters'
OLD=ROOT/'results/diffusion-baseline-meta-starters'
NEW=ROOT/'results/diffusion-baseline-meta-starters-v3'
MEMBERS=['F1_final_medium_noheldout_regmb_v3','G2_family_matrix_noprtrain_noheldout_regmb_v3']


def locked(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(f,fcntl.LOCK_UN);return False


def stage(name):
    results.save(BASE/'pipeline-progress.json',dict(stage=name,status='running',time=time.time()))
    print(name,flush=True)


def execute(name,arguments):
    with (BASE/(name+'.log')).open('a') as log:
        result=subprocess.run([sys.executable,str(ROOT/'scripts'/arguments[0]),*arguments[1:]],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(name+' failed; inspect '+str(BASE/(name+'.log')))


def finished(folder,trials):
    return all((folder/'completions'/(t['id']+'.json')).exists() for t in trials)


def main(args):
    restore_runtime()
    m=json.loads((BASE/'manifest.json').read_text());trials=m['trials']
    with (BASE/'.upgrade.lock').open('w') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results.save(BASE/'diffusion-selection.json',dict(version='v3',results=NEW.name,members=MEMBERS,
                 previous_results=OLD.name,target_win_rate=.5,guidance=2,temperature=1))
        stage('waiting_for_v2_generation')
        if not locked(BASE/'.progress-watcher.lock'):
            with (BASE/'progress-v3.log').open('a') as log:
                subprocess.Popen([sys.executable,str(ROOT/'scripts/watch_meta_grid.py')],stdout=log,stderr=subprocess.STDOUT)
        while locked(OLD/'.lock'): time.sleep(10)
        if not finished(OLD,trials):
            execute('finish-v2',['diffusion_baseline.py','generate','--baseline',str(BASE),'--output',str(OLD),'--members','F1_final_medium_noheldout_regmb_v2','G2_family_matrix_noprtrain_noheldout_regmb_v2'])
        assert finished(OLD,trials)
        stage('training_v3')
        execute('train-v3',['retrain_regmb.py','train','--version','v3'])
        stage('generating_v3')
        execute('generate-v3',['diffusion_baseline_v3.py','generate','--baseline',str(BASE),'--output',str(NEW),'--members',*MEMBERS])
        assert finished(NEW,trials)
        stage('waiting_for_ling')
        while any(locked(BASE/f'.complete-{i}-of-4.lock') for i in range(4)): time.sleep(10)
        if not finished(BASE,trials):
            uncertain=[t['id'] for t in trials if not (BASE/'completions'/(t['id']+'.json')).exists() and (BASE/'completions'/(t['id']+'.pending')).exists()]
            if uncertain:raise RuntimeError('Reconcile uncertain Ling requests: '+str(uncertain))
            if not os.environ.get('OPENROUTER_API_KEY'):os.environ['OPENROUTER_API_KEY']=getpass.getpass('OpenRouter key for missing calls (not saved): ')
            execute('resume-ling',['llm_baseline.py','complete','--output',str(BASE)])
        assert finished(BASE,trials)
        # The original coordinator is paused, not its children. Only retire that
        # explicitly supplied live coordinator after all its generation is saved.
        if args.old_coordinator:
            command=subprocess.run(['ps','-p',str(args.old_coordinator),'-o','args='],capture_output=True,text=True).stdout
            if 'meta_starter_grid.py run' in command:os.kill(args.old_coordinator,signal.SIGKILL)
        stage('battles_v3')
        execute('battles-v3',['dense_grid_battles.py','--baseline',str(BASE),'--diffusion',str(NEW),'--workers','6'])
        execute('report-v3',['meta_starter_grid.py','report'])
        subprocess.run([sys.executable,str(ROOT/'dashboard/build.py')],check=True)
        results.save(BASE/'pipeline-progress.json',dict(stage='complete',status='complete',time=time.time()))
        print('V3 comparison complete.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old-coordinator',type=int);args=p.parse_args()
    try:main(args)
    except BlockingIOError:
        print('The v3 upgrade is already running; no work was restarted.',flush=True)
    except Exception as exc:
        results.save(BASE/'pipeline-progress.json',dict(status='interrupted',error=str(exc),time=time.time()))
        raise
