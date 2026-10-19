"""Explicit v3 training heuristics, separate from simulator legality."""
import copy
import json
from pathlib import Path
import subprocess


def mega_stone_users(showdown):
    code=r"""
const {TeamValidator}=require(process.argv[1]+'/dist/sim/team-validator');
const dex=TeamValidator.get('gen9championsvgc2026regmb').dex;
const norm=s=>s.toLowerCase().replace(/[^a-z0-9]/g,'');const users={};
for(const item of dex.items.all()){
 if(!item.megaStone)continue;
 if(typeof item.megaStone==='object')users[item.id]=[...new Set(Object.entries(item.megaStone).flat().map(norm))];
 else if(item.megaEvolves)users[item.id]=[norm(item.megaEvolves),norm(item.megaStone)];
 else throw Error('Unknown Mega Stone mapping: '+item.id);
}
console.log(JSON.stringify(users));
"""
    return json.loads(subprocess.check_output(['node','-e',code,str(showdown)],text=True))


def training_regulation(regulation,users):
    """Keep the vocabulary; narrow synthetic and decoder item choices by policy."""
    result=copy.deepcopy(regulation)
    for species,slot in result['slots'].items():
        allowed=lambda item:item not in users or species in users[item]
        slot['abilityItems']={a:[i for i in items if allowed(i)] for a,items in slot['abilityItems'].items()}
        slot['abilityItems']={a:items for a,items in slot['abilityItems'].items() if items}
        slot['abilities']=sorted(slot['abilityItems'])
        slot['items']=sorted({i for items in slot['abilityItems'].values() for i in items})
        if not slot['abilities']: raise ValueError('Training policy eliminates species '+species)
    result['training_policy']={
        'version':'v3',
        'mega_stones':'only exact base/formes in the mod item Mega Evolution mapping; strategic heuristic, not a legality rule',
        'mega_stone_users':users,
        'move_counts':{'1':.05,'2':.10,'3':.15,'4':.70},
        'empty_moves':'known unused move token, distinct from [MASK]; pad after real moves; at least one real move'}
    return result


def sample_moves(legal,rng,target=None,variable=True):
    if not legal: raise ValueError('No real moves available')
    count=int(rng.choice([1,2,3,4],p=[.05,.10,.15,.70])) if variable else 4
    count=min(count,len(legal))
    if target is None:return list(rng.choice(legal,count,replace=False))
    remaining=[m for m in legal if m!=target]
    return [target]+list(rng.choice(remaining,count-1,replace=False))


def encode_training_team(vocab,team):
    normalized=[dict(s,moves=[m for m in s['moves'] if m]) for s in team]
    if any(not s['moves'] for s in normalized): raise ValueError('Every Pokémon needs at least one move')
    return vocab.encode(normalized)
