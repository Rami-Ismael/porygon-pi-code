"""A shared complete-case cohort for the seven matched systems."""
import datetime,json
from pathlib import Path
import reasoning_experiment as E
P=E.B.ROOT/'results'
methods=[];maps={}
d=json.loads((P/'diffusion-matched-20261013/status.json').read_text())
methods.append(dict(id='diffusion',name='Pokémon diffusion',legal=d['legal'],cost=0,cost_coverage=420))
maps['diffusion']={(r['trial'],r['replicate']):r for r in d['rows']}
for folder,prefix in [('reasoning-expanded-20261012','Ling'),('mercury-reasoning-20261013','Mercury')]:
 s=json.loads((P/folder/'status.json').read_text());assert s['complete']
 for arm,a in s['arms'].items():
  key=prefix.lower()+'-'+arm
  methods.append(dict(id=key,name=prefix+' '+arm,legal=a['legal'],cost=a['cost_usd'],cost_coverage=a['cost_reported']))
  maps[key]={(r['trial'],r['replicate']):r for r in s['rows'] if r['arm']==arm}
keys=set(maps['diffusion'])
assert all(set(v)==keys and len(v)==420 for v in maps.values())
common=sorted(k for k in keys if all(v[k]['legal'] and v[k]['win_rate'] is not None for v in maps.values()))
for m in methods:
 rr=maps[m['id']]
 m['legality']=m['legal']/420
 legal_rows=[r for r in rr.values() if r['legal'] and r['win_rate'] is not None]
 m['full_set']=dict(outputs=420,legal=len(legal_rows),invalid=420-m['legal'],scored=len(legal_rows),legal_win_rate=E.average([r['win_rate'] for r in legal_rows]),represented_battles=50*len(legal_rows))
 m['battle']=E.cluster_interval([(rr[k]['start'],rr[k]['win_rate']) for k in common],20261012)
 m['delta_vs_diffusion']=E.cluster_interval([(rr[k]['start'],rr[k]['win_rate']-maps['diffusion'][k]['win_rate']) for k in common],20261012)
pairwise=json.loads((P/'diffusion-matched-20261013/final-audit.json').read_text())['comparisons']
out=dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),planned_per_method=420,common_pairs=len(common),common_starters=len({maps['diffusion'][k]['start'] for k in common}),methods=methods,pairwise=pairwise,cohort=[list(k) for k in common])
(P/'diffusion-matched-20261013/all-method-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='cohort'}))
