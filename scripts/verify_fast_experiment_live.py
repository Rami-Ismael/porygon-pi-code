"""Isolated 52-battle adapter check; never writes campaign scores or matrix cells."""
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fast_experiment_battles import Backend

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/fast-experiment-validation-20261014'
out.mkdir(exist_ok=False)
con=sqlite3.connect('file:/Users/ramiismael/vgc-data/matchup_regmb.sqlite?mode=ro',uri=True)
old=ROOT/'results/dibo_v4_vs_cem_20261013/stall-20261014'
jobs=[]
for i in range(6):
    original=json.loads((old/f'_mshard_64782_1791940735_{i}.json').read_text())['jobs'][0]
    cols=list(original['key2col'].values())[:4]
    jobs.append(dict(row=original['row'],schedule=cols))
b=Backend(out/'fast-battles')
try:
    for seed in (830001,830002):
        raw,u,t=b.run_jobs(con,jobs,seed=seed)
        assert not u and sum(sum(r[2:]) for r in raw)==24
        print(f'PASS seed {seed}: 24/24 fresh results, {t:.2f}s',flush=True)
    before=[w.proc.pid for w in b.runner.workers]
    def paste(tid):
        text=con.execute('SELECT paste FROM team WHERE team_id=?',(tid,)).fetchone()[0]
        p=out/f'{tid}.txt';p.write_text(text);return str(p)
    candidate=paste(jobs[0]['row']);opponents=[paste(t) for t in jobs[0]['schedule']]
    result=b.score([(candidate,opponents)],battles=4,opp_schedule=opponents,seed=830003)
    assert result[candidate]['battles']==4
    assert before==[w.proc.pid for w in b.runner.workers]
    (out/'verified.json').write_text(json.dumps(dict(completed_battles=52,worker_reuse=True,workers=before,provenance=b.runner.identity),indent=2))
    print('PASS: finals schedule, exact counts, and persistent worker reuse',flush=True)
finally:
    b.close();con.close()
