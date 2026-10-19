"""Build the selected Mask Lab dashboard from recorded results; no API calls."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
RUN = CODE/'results/llm-baseline-instant-pilot'
DIFF = CODE/'results/diffusion-baseline-f1g2-strict'
if (CODE/'results/diffusion-baseline-f1g2-regmb-v2/summary.json').exists():
    DIFF=CODE/'results/diffusion-baseline-f1g2-regmb-v2'
if (CODE/'results/diffusion-baseline-full-grid/summary.json').exists():
    RUN=CODE/'results/llm-baseline-full-grid'
    DIFF=CODE/'results/diffusion-baseline-full-grid'
if (CODE/'results/diffusion-baseline-meta-starters/summary.json').exists():
    RUN=CODE/'results/llm-baseline-meta-starters'
    DIFF=CODE/'results/diffusion-baseline-meta-starters'
if (CODE/'results/diffusion-baseline-meta-starters-v3/summary.json').exists():
    RUN=CODE/'results/llm-baseline-meta-starters'
    DIFF=CODE/'results/diffusion-baseline-meta-starters-v3'
m = json.loads((RUN/'manifest.json').read_text())
assert json.loads((DIFF/'manifest.json').read_text()) == m
assert json.loads((DIFF/'battle-config.json').read_text()) == json.loads((RUN/'battle-config.json').read_text())
trials, usage = [], []
for t in m['trials']:
    completion = json.loads((RUN/'completions'/(t['id']+'.json')).read_text())
    diffusion = json.loads((DIFF/'completions'/(t['id']+'.json')).read_text())
    usage.append(completion['response']['usage'])
    labels = {}
    for method, folder, key in [('original',RUN,f"start-{t['start']}"),('random',RUN,'random-'+t['id']),
                                ('ling',RUN,'llm-'+t['id']),('diffusion',DIFF,'diffusion-'+t['id'])]:
        p = folder/'labels'/(key+'.json')
        if p.exists():
            r = json.loads(p.read_text())
            assert r['battles'] == 50
            labels[method] = {k:r[k] for k in ['wins','battles','results']}
    trials.append(dict(t,valid=completion['valid'],error=completion.get('error'),
                       ling_team=completion.get('team'),diffusion=diffusion,labels=labels))
status = json.loads((RUN/'status.json').read_text())
comparisons = status['comparisons']
mean = lambda a: sum(a)/len(a)
metrics = dict(attempts=len(trials),legal=sum(t['valid'] for t in trials))
for target, field in [('deltaOriginal','delta_original'),('deltaRandom','delta_random')]:
    metrics[target] = 100*mean([mean([r[field] for r in comparisons if r['start']==s])
                                for s in sorted({r['start'] for r in comparisons})])
for u in usage:
    cached = u['prompt_tokens_details']['cached_tokens']
    reconstructed = ((u['prompt_tokens']-cached)*.021+cached*.0042+u['completion_tokens']*.0616)/1e6
    assert abs(reconstructed-u['cost']) < 1e-12, 'recorded rate no longer matches; revise cost assumptions'
smoke = sum(json.loads(p.read_text()).get('response',{}).get('usage',{}).get('cost',0)
            for p in (CODE/'results/llm-baseline-pilot/completions').glob('*.json'))
cost = dict(total=sum(u['cost'] for u in usage),smoke=smoke,
            meanInput=mean([u['prompt_tokens'] for u in usage]),
            meanOutput=mean([u['completion_tokens'] for u in usage]),
            cachedFraction=sum(u['prompt_tokens_details']['cached_tokens'] for u in usage)/sum(u['prompt_tokens'] for u in usage))
cost.update(observed=cost['total']/len(usage),cold=(cost['meanInput']*.021+cost['meanOutput']*.0616)/1e6,
            maxOutput=(cost['meanInput']*.021+4096*.0616)/1e6)
data = dict(starts=m['starts'],opponents=m['opponents'],trials=trials,metrics=metrics,cost=cost,
            diffusion=json.loads((DIFF/'summary.json').read_text()))
data['starterKind']=m.get('starter_kind','random legal teams')
data['excludedStarterIds']=m.get('exclude_starter_team_ids',[])
generation=json.loads((DIFF/'generation-config.json').read_text())
profile=json.loads((CODE/'data/diffusion-baseline-profile.json').read_text())
for key in ['target_win_rate','guidance','temperature']:
    assert generation[key]==profile[key], f'recorded {key} differs from selected baseline'
data['diffusion'].update({k:generation[k] for k in ['members','target_win_rate','guidance','temperature']})
encoded = json.dumps(data,separators=(',',':')).replace('<','\\u003c')
page=(HERE/'template.html').read_text().replace('@@DATA@@',encoded)
(HERE/('meta-results.html' if m.get('exclude_starter_team_ids') else 'index.html')).write_text(page)
print(f'Built {HERE / "index.html"}; {len(trials)} recorded tasks, verified costs and identical battle configuration.')
