"""Regulation-grounded prompt variant; never reveal masked original values."""
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def regulation():
    return json.loads((Path(__file__).resolve().parents[1]/'data/regmb-vocabulary.json').read_text())


def norm(value):return ''.join(c for c in value.lower() if c.isalnum())


def strengthen(messages,manifest,trial):
    reg=regulation();names=reg['names']
    content=json.loads(messages[1]['content'])
    context=content['masked_team'];hints={}
    for i,slot in enumerate(context):
        if slot is None:continue
        wanted={p[1] for p in trial['paths'] if p[0]==i and len(p)>1}
        if not wanted:continue
        rules=reg['slots'][norm(slot['species'])];hint={}
        if 'moves' in wanted:
            fixed={norm(m) for m in slot['moves'] if m}
            hint['available_moves']=[names.get(m,m) for m in rules['moves'] if m not in fixed]
        if 'item' in wanted or 'ability' in wanted:
            pairs={a:[it for it in items if slot['item'] is None or it==norm(slot['item'])]
                   for a,items in rules['abilityItems'].items() if slot['ability'] is None or a==norm(slot['ability'])}
            hint['ability_to_legal_items']={names.get(a,a):[names.get(it,it) for it in its] for a,its in pairs.items() if its}
        if 'nature' in wanted:hint['natures']=[names.get(n,n) for n in reg['values']['nature'] if n]
        if 'evs' in wanted:hint['stat_points']={'keys':['HP','Atk','Def','SpA','SpD','Spe'],'integer_each':[0,32],'sum_at_most':66}
        hints[str(i)]=hint
    if any(len(p)==1 for p in trial['paths']):
        content['legal_species']=[names.get(s,s) for s in reg['values']['species']]
    content['regulation_reference']={'name':'Pokémon Champions Regulation M-B','format_id':manifest['format'],
        'source':'project-pinned simulator vocabulary, not current live-season rules','slot_choices':hints}
    system=(
        'TARGET RULESET: Pokémon Champions Regulation M-B (gen9championsvgc2026regmb). '
        'Use this frozen ruleset throughout, not Scarlet/Violet VGC or another Champions regulation. '
        'Complete exactly the requested null paths to improve battle performance against the supplied pool. '
        'Use the supplied legal choices where present. Choices are per slot: also enforce team-wide Species Clause and Item Clause. '
        'Do not duplicate a species/base-species family or a nonempty held item already in any fixed slot or another fill. '
        'Champions Stat Points: HP, Atk, Def, SpA, SpD, Spe are integers from 0 through 32; their sum must be at most 66. '
        'Do not output Scarlet/Violet spreads such as 252/252/4. Stat Alignment is the nature. No Terastallization. '
        'Preserve fixed empty move slots. Every masked move gets a real legal move, distinct from all other real moves in that slot. '
        'A whole Pokémon fill needs species, item, ability, nature, four distinct real moves, and evs containing all six Stat Point keys. '
        'Legal-to-hold items may be ineffective: prefer an item that works for that species. '
        'Return only strict JSON {"fills":[{"path":[0,"item"],"value":"Leftovers"}]}, adapted to the actual requested paths. '
        'Use double-quoted keys and strings. No comments, ellipses, trailing commas, Markdown, explanations, or placeholder values. '
        'Return exactly one fill per requested path. Never include or change unmasked fields. '
        'Before answering, check JSON syntax, all requested paths, Stat Point totals, move legality, ability/item compatibility, and both clauses.'
    )
    return [{'role':'system','content':system},{'role':'user','content':json.dumps(content)}]
