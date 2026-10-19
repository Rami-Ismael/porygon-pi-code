"""Extend the frozen pilot without regenerating or reordering existing trials."""
import argparse
import copy
import hashlib
import json
import random
import shutil

import llm_baseline as B


def prepare():
    import numpy as np
    source = B.ROOT / 'results/llm-baseline-instant-pilot'
    old_diff = B.ROOT / 'results/diffusion-baseline-f1g2-regmb-v2'
    target = B.ROOT / 'results/llm-baseline-full-grid'
    diff = B.ROOT / 'results/diffusion-baseline-full-grid'
    original = json.loads((source/'manifest.json').read_text())
    m = copy.deepcopy(original)
    m['grid'] = {task:list(range(1,len(B.paths_for(task))+1)) for task in B.GRID}
    h, _ = B.runtime()
    tables = h.build_tables()
    validator = h.Validator()
    old_ids = {t['id'] for t in m['trials']}
    try:
        for i, team in enumerate(m['starts']):
            for task, sizes in m['grid'].items():
                order = B.paths_for(task)
                random.Random(f"{m['seed']}:{i}:{task}").shuffle(order)
                for k in sizes:
                    tid = f's{i}-{task}-{k}'
                    if tid in old_ids:
                        old = next(t for t in original['trials'] if t['id']==tid)
                        assert old['paths'] == order[:k]
                        continue
                    seed = int(hashlib.sha256(f"dense:{m['seed']}:{tid}".encode()).hexdigest()[:16],16)
                    control = B.random_fill(team,order[:k],tables,np.random.default_rng(seed),h,validator)
                    m['trials'].append(dict(id=tid,start=i,task=task,k=k,paths=order[:k],random_team=control))
    finally:
        validator.close()
    assert len(m['trials']) == 306 and m['trials'][:63] == original['trials']
    m['extension'] = dict(parent_manifest_sha256=B.digest(original),reused=63,added=243,
                          control_seed='sha256(dense:seed:trial) first 16 hex digits',
                          diffusion_seed='manifest seed + trial index; original indices preserved')
    for dest in (target,diff):
        if (dest/'manifest.json').exists():
            assert json.loads((dest/'manifest.json').read_text()) == m
        B.save(dest/'manifest.json',m)
    for src,dest in ((source,target),(old_diff,diff)):
        for directory in ('completions','labels'):
            for path in (src/directory).glob('*.json'):
                out=dest/directory/path.name
                out.parent.mkdir(parents=True,exist_ok=True)
                if out.exists(): assert out.read_bytes()==path.read_bytes()
                else: shutil.copy2(path,out)
        for name in ('battle-config.json','generation-config.json'):
            if (src/name).exists():
                out=dest/name
                if out.exists(): assert out.read_bytes()==(src/name).read_bytes()
                else: shutil.copy2(src/name,out)
    print('Prepared 306 tasks; reused 63, added 243. Original diffusion indices preserved.',flush=True)


def report():
    from types import SimpleNamespace
    target=B.ROOT/'results/llm-baseline-full-grid'
    diff=B.ROOT/'results/diffusion-baseline-full-grid'
    m=json.loads((target/'manifest.json').read_text())
    B.report(SimpleNamespace(output=target))
    status=json.loads((target/'status.json').read_text())
    assert status['complete'], 'Ling scoring incomplete'
    assert json.loads((target/'battle-config.json').read_text()) == json.loads((diff/'battle-config.json').read_text())
    rows=[json.loads((diff/'completions'/(t['id']+'.json')).read_text()) for t in m['trials']]
    pairs=[]
    scored=0
    for t,r in zip(m['trials'],rows):
        if not r['valid']: continue
        d=json.loads((diff/'labels'/('diffusion-'+t['id']+'.json')).read_text())
        assert d['battles']==50
        scored+=1
        a=target/'labels'/('llm-'+t['id']+'.json')
        if a.exists():
            a=json.loads(a.read_text())
            pairs.append(dict(trial=t['id'],start=t['start'],delta_pp=100*(d['wins']/50-a['wins']/50)))
    summary=dict(supported=sum(r['supported'] for r in rows),legal=sum(r['valid'] for r in rows),scored=scored,pairs=pairs)
    B.save(diff/'summary.json',summary)
    old=json.loads((B.ROOT/'results/llm-baseline-instant-pilot/status.json').read_text())
    cost=status['reported_cost_usd']-old['reported_cost_usd']
    delta=sum(sum(r['delta_pp'] for r in pairs if r['start']==s)/sum(r['start']==s for r in pairs) for s in range(3))/3
    lines=['# Full mask grid results','',f"306 tasks; 63 reused and 243 additional Ling calls. Additional API cost: ${cost:.8f}; combined cost: ${status['reported_cost_usd']:.8f}.",'',f"Ling: {status['legal_completions']}/306 legal. Diffusion: {summary['legal']}/306 legal, {summary['supported']}/306 supported. {len(pairs)} jointly scored tasks.",'',f'Equal-start-weighted diffusion minus Ling: {delta:+.2f} percentage points on jointly scored tasks. Three independent starts: descriptive only, not proof of a winner.','', 'Diffusion uses the full-regulation F1/G2 v2 ensemble, ask 0.5, guidance 2, temperature 1. All battles use the frozen behavior-cloning policy on both sides and the original 50-battle meta-opponent schedule. Policy seeds match; simulator randomness is not fixed. Invalid completions are retained without retries. Random-fill controls are team completions, not a random-move battle policy.','', '| Task | Mask sizes | Ling legal | Diffusion legal |','|---|---|---:|---:|']
    for task,ks in m['grid'].items():
        ts=[t for t in m['trials'] if t['task']==task]
        legal=sum(json.loads((target/'completions'/(t['id']+'.json')).read_text())['valid'] for t in ts)
        dl=sum(r['valid'] for t,r in zip(m['trials'],rows) if t['task']==task)
        lines.append(f'| {task} | 1–{max(ks)} | {legal}/{len(ts)} | {dl}/{len(ts)} |')
    (B.ROOT/'docs/dense-mask-grid-results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(added_cost=cost,total_cost=status['reported_cost_usd'],ling_legal=status['legal_completions'],diffusion_legal=summary['legal'],paired=len(pairs),delta_pp=delta)))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','report'],nargs='?',default='prepare')
    globals()[p.parse_args().command]()
