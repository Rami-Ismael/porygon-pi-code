"""Verify the completed matched diffusion study without regenerating outputs."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import statistics
from repo_paths import ROOT  # also puts src/ first on sys.path
import battles
import results
import team_masks
import lossdown as L
import reasoning_experiment as E

P=ROOT/'results/diffusion-matched-20261013'
s=json.loads((P/'status.json').read_text());assert s['complete']
parent=json.loads((ROOT/'results/mercury-reasoning-20261013/off/r0/manifest.json').read_text())
config=json.loads((ROOT/'results/mercury-reasoning-20261013/off/r0/battle-config.json').read_text())
audit=json.loads((P/'preservation-audit.json').read_text())
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in audit['before'].items())
labels=collections.defaultdict(list);configs=[];rows=[];seconds=[]
for rep in range(3):
 folder=P/f'r{rep}';m=json.loads((folder/'manifest.json').read_text());assert m==parent
 assert json.loads((folder/'battle-config.json').read_text())==config
 g=json.loads((folder/'generation-config.json').read_text());assert g['seed']==m['seed']+100000*rep
 for f,h in g['source_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 for name,checkpoint in g['checkpoints'].items():assert hashlib.sha256((L.CK/(name+'.pt')).read_bytes()).hexdigest()==checkpoint['sha256']
 configs.append({k:v for k,v in g.items() if k!='seed'})
 assert len(list((folder/'completions').glob('*.json')))==140
 for t in m['trials']:
  r=json.loads((folder/'completions'/(t['id']+'.json')).read_text());assert r['supported'] and r['valid']
  assert team_masks.masked(r['team'],t['paths'])==team_masks.masked(m['starts'][t['start']],t['paths'])
  lab=json.loads((folder/'labels'/('diffusion-'+t['id']+'.json')).read_text())
  assert lab['battles']==50 and sum(x[2] for x in lab['results'])==lab['wins']
  actual=collections.Counter()
  for x in lab['results']:actual[x[1]]+=sum(x[2:5])
  assert actual==collections.Counter(battles.schedule_for(m,t['start']))
  labels[lab['cache_key']].append(lab)
  seconds.append(r['seconds'])
  rows.append(dict(trial=t['id'],replicate=rep,start=t['start'],legal=True,win_rate=lab['wins']/50))
assert configs[0]==configs[1]==configs[2]
assert all(all(v==group[0] for v in group) for group in labels.values())
for f,h in config['fingerprints'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
for f in P.glob('battle-workers/*/battle-config.json'):assert json.loads(f.read_text())==config
comparisons={}
for name in ['mercury-reasoning-20261013','reasoning-expanded-20261012']:
 old=json.loads((ROOT/'results'/name/'status.json').read_text());assert old['complete']
 for arm in old['arms']:
  baseline={(r['trial'],r['replicate']):r for r in old['rows'] if r['arm']==arm}
  legal=[];wins=[];both=[]
  for r in rows:
   c=baseline[(r['trial'],r['replicate'])];legal.append((r['start'],1-int(c['legal'])))
   if c['legal']:
    wins.append((r['start'],r['win_rate']-c['win_rate']));both.append((r,c))
  item=dict(legality=E.cluster_interval(legal,parent['seed']),battles=E.cluster_interval(wins,parent['seed']))
  assert item==s['comparisons'][name+'/'+arm]
  starts=sorted({r['start'] for r,c in both})
  item['paired_diffusion_rate']=statistics.mean(statistics.mean(r['win_rate'] for r,c in both if r['start']==start) for start in starts)
  item['paired_llm_rate']=statistics.mean(statistics.mean(c['win_rate'] for r,c in both if r['start']==start) for start in starts)
  comparisons[name+'/'+arm]=item
result=dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),preserved_files=len(audit['before']),outputs=420,supported=420,legal=420,scored=420,missing=0,invalid=0,unsupported=0,unique_panels=len(labels),duplicate_panels=420-len(labels),represented_battles=21000,unique_panel_battles=50*len(labels),all_legal_win_rate=statistics.mean(r['win_rate'] for r in rows),comparisons=comparisons,generation_seconds_sum=sum(seconds),generation_seconds_median=statistics.median(seconds),api_cost_usd=0,compute_cost_measured=False,configs_identical=True,checkpoint_hashes_verified=True,unmasked_values_preserved=True)
results.save(P/'final-audit.json',result)
lines=['# Matched Pokémon diffusion study','', '420/420 supported and legal outputs; 420 fully scored panels. 20 starters, 140 masks, three seeded repeats.','',f"All-legal descriptive win rate: {result['all_legal_win_rate']:.2%}. 414 unique panels and six reused panels. 21,000 represented battles; 20,700 unique-panel battles.",'','| Diffusion minus | Legality delta (pp) | Matched battle delta (pp) | Battle 95% interval (pp) | Pairs |','|---|---:|---:|---|---:|']
for name,x in comparisons.items():
 b=x['battles'];lines.append(f"| {name} | {100*x['legality']['mean']:.2f} | {100*b['mean']:.2f} | {100*b['low']:.2f} to {100*b['high']:.2f} | {b['pairs']} |")
lines+=['','Intervals are exploratory equal-starter bootstrap intervals (4,000 draws), without multiplicity correction. Mask/repeat outputs and cached panels are not independent samples. Battle comparisons select jointly legal outputs and each LLM setting selects a different subset.','', 'F1/G2 v3 uses constrained decoding and corpus-based or random legal Stat Point fills. It does not read opponent pastes. These compare complete systems, not equal-input architectures or reasoning interventions. Frozen visible M-B pool; no held-out, live-meta or causal claim.','',f"Preserved {result['preserved_files']} prior LLM files unchanged. Matching configurations, frozen inputs, generation source/checkpoint hashes, unmasked values and exact battle schedules verified. No missing outputs, unsupported contexts or invalid teams. Local API cost $0; machine/electricity costs not measured."]
(P/'final-report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result))
