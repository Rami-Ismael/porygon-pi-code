"""Every prompt version sent to the model. A released version never changes:
prompts/<version>.system.txt pins its text and test_prompt_versions.py compares.
A new idea is a new version here plus a row in docs/prompt-changelog.md."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import repo_paths  # puts src/ first on sys.path

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOTS=ROOT/'prompts'
ORIGINAL='mb-original-v1'
REGULATION='mb-regulation-v3'
VERSIONS={
    ORIGINAL:dict(date='2026-10-08',parent=None,
        change='First prompt. Names Regulation M-B, the Stat Point limits, both clauses and the answer format. No lists of what is legal.'),
    'mb-grounded-v2':dict(date='2026-10-08',parent=ORIGINAL,
        change='Instructions rewritten; legal moves, ability/item pairs and natures added for the masked fields only. Rules and formatting changed together.'),
    REGULATION:dict(date='2026-10-10',parent=ORIGINAL,
        change='First prompt word for word, followed by the Regulation M-B reference: season, battle format, Mega Evolution, every accepted species and item, Stat Points. Nothing else changes.'),
}
# Smallest request that builds every version; the system text does not depend on the task.
FIXTURE=dict(format='gen9championsvgc2026regmb',opponents=[],preserve_empty_move_slots=True,starts=[[
    dict(species='Charizard',item='Charizardite Y',ability='Blaze',nature='Timid',moves=['Heat Wave','Protect','Solar Beam','Air Slash'],
         evs=dict(HP=2,Atk=0,Def=0,SpA=32,SpD=0,Spe=32))]])
FIXTURE_TRIAL=dict(start=0,paths=[[0,'item']])


@lru_cache(maxsize=1)
def regulation_reference():
    """Official rules: champions-news.pokemon-home.com/en/page/776.html (read 2026-10-10).
    Battle format, clauses, Stat Point limits and both lists: the pinned simulator."""
    reg=json.loads((ROOT/'data/regmb-vocabulary.json').read_text());names=reg['names'];named=lambda field:[names.get(v,v) for v in reg['values'][field] if v]
    megas=[s for s in reg['values']['species'] if '-Mega' in names.get(s,s)]
    # The Mega sentences below are claims about the simulator; fail loudly if it disagrees.
    assert megas and all(len({i for items in reg['slots'][s]['abilityItems'].values() for i in items})==1 for s in megas)
    assert reg['slots']['charizard']['abilityItems']['toughclaws']==['charizarditex']
    return ' '.join([
        'REGULATION M-B REFERENCE.',
        'Every answer is checked by a Pokémon Showdown validator pinned to "[Gen 9 Champions] VGC 2026 Reg M-B".',
        'Regulation Set M-B was the Pokémon Champions Ranked Battles ruleset from 17 June 2026 to 9 September 2026.',
        'Regulation Set M-C replaced it afterwards and is not the target; neither are Regulation M-A or Scarlet/Violet rules.',
        'Battle format: Double Battle. A team has six Pokémon and each player brings four of the six to a battle. Every Pokémon is level 50.',
        'Both players see the other team\'s species, items, abilities and moves before choosing their four.',
        'Mega Evolution: a Pokémon can Mega Evolve only while holding its own Mega Stone, and a player can Mega Evolve only one time per battle.',
        'A species written with "-Mega" must hold its own Mega Stone.',
        'An ability that belongs to a Mega form is accepted on the base species only with the matching stone, for example Tough Claws on Charizard only with Charizardite X.',
        'Held items: duplicate held items are not allowed, so no two Pokémon on the team hold the same item.',
        f'These {len(named("item"))} items are the only items in this regulation: {", ".join(named("item"))}.',
        'Species: no two Pokémon on the team are the same species, and forms of one species count as the same species.',
        'Mythical and restricted Legendary Pokémon are not allowed.',
        f'These {len(named("species"))} species names are the only ones the validator accepts: {", ".join(named("species"))}.',
        'Stat Points: each of HP, Atk, Def, SpA, SpD and Spe is a whole number from 0 to 32, and the six numbers add up to 66 or less, for example 32 + 32 + 2.',
        f'Stat Alignment is one of these {len(named("nature"))} natures: {", ".join(named("nature"))}.',
        'There is no Terastallization.'])


def add_regulation(messages):
    return [dict(messages[0],content=messages[0]['content']+'\n\n'+regulation_reference()),*messages[1:]]


def messages(version,manifest,trial):
    import llm_completion
    if version not in VERSIONS:raise ValueError(f'unknown prompt version {version}')
    m=dict(manifest,prompt_version=version)
    if version==ORIGINAL:del m['prompt_version']
    return llm_completion.prompt(m,trial)


def system_text(version,manifest=None):
    m=FIXTURE if manifest is None else dict(FIXTURE,preserve_empty_move_slots=manifest.get('preserve_empty_move_slots',False))
    return messages(version,m,FIXTURE_TRIAL)[0]['content']


def system_sha256(version,manifest=None):return hashlib.sha256(system_text(version,manifest).encode()).hexdigest()


def snapshot():
    for version in VERSIONS:
        path=SNAPSHOTS/(version+'.system.txt');text=system_text(version)
        if not path.exists():path.parent.mkdir(exist_ok=True);path.write_text(text)
        elif path.read_text()!=text:raise ValueError(f'{version} no longer matches {path}; a released prompt is frozen, add a new version')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['list','show','snapshot']);p.add_argument('version',nargs='?',default=REGULATION)
    args=p.parse_args()
    if args.command=='show':print(system_text(args.version))
    elif args.command=='snapshot':snapshot()
    else:
        for version,v in VERSIONS.items():
            print(f"{version}  {v['date']}  parent={v['parent']}  sha256={system_sha256(version)[:12]}  {len(system_text(version))} chars\n    {v['change']}")
