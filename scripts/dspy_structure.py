"""DSPy request structure for the masked-completion baseline. See docs/dspy-structure-experiment.md.

The baseline sends one system string and one JSON user message, and parses the
reply with json.loads. This module keeps the same instructions and the same
input values, and changes only how they are packaged and parsed:

- each input value becomes a named, typed DSPy input field;
- the answer becomes a typed output field, `fills: list[Fill]`, whose JSON schema
  the DSPy adapter shows to the model;
- the reply is parsed by the DSPy adapter into that type.

The parsed answer then goes through the unchanged baseline checks
(`llm_completion.apply_completion` and the Showdown validator). DSPy only formats
and parses here: no DSPy LM call, retry, adapter fallback or optimization runs.
"""
import json
import re
from enum import Enum
from typing import Union

import dspy
from pydantic import BaseModel, ConfigDict, Field, StrictInt, create_model

import repo_paths  # puts src/ first on sys.path

KIND = 'dspy'
ADAPTERS = ('chat', 'json')
# 'legal-choices' replaces the generic answer type with one typed field per
# requested path whose values are limited to what the regulation allows there
# (see legal_completion_type). Inputs and instructions stay unchanged.
CONSTRAINTS = ('legal-choices',)
# Sentences that only dictate the baseline's raw-JSON reply envelope. The DSPy
# adapter replaces that envelope with its own, so these are the only words
# removed. Every other instruction is kept verbatim.
ENVELOPE = (
    'Return only JSON: {"fills":[{"path":[0,"item"],"value":"..."}]}. ',
    ' No prose or Markdown.',
    'Return only strict JSON {"fills":[{"path":[0,"item"],"value":"Leftovers"}]}, adapted to the actual requested paths. ',
    'Use double-quoted keys and strings. No comments, ellipses, trailing commas, Markdown, explanations, or placeholder values. ',
)
# Short labels for the existing user-message keys. They name each value; they
# add no rule or game knowledge that the baseline message lacks.
DESCRIPTIONS = {
    'format': 'Showdown format id',
    'opponents': 'opponent team pastes',
    'masked_team': 'six team slots; null marks a masked field or slot',
    'requested_paths': 'masked paths to fill',
    'legal_species': 'legal species names',
    'regulation_reference': 'regulation reference and legal choices per masked slot',
}
INPUT_TYPES = {'format': str, 'opponents': list[str], 'masked_team': list,
               'requested_paths': list[list[Union[int, str]]], 'legal_species': list[str],
               'regulation_reference': dict}


class _Strict(BaseModel):
    # Strict: "12" is not coerced to 12 and unknown keys are refused, so the
    # typed parse never repairs a value the baseline checks would reject.
    model_config = ConfigDict(strict=True, extra='forbid')


class StatPoints(_Strict):
    HP: int
    Atk: int
    Def: int
    SpA: int
    SpD: int
    Spe: int


class PokemonSet(_Strict):
    species: str
    item: str
    ability: str
    nature: str
    moves: list[str]
    evs: StatPoints


class Fill(_Strict):
    path: list[Union[int, str]]
    value: Union[str, StatPoints, PokemonSet]


STATS = ('HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe')
TOTAL = ('HP+Atk+Def+SpA+SpD+Spe, written out so the sum is checked: it must equal the six values '
         'added together and be at most 66, for example 32+32+2')


def _stat():
    return (StrictInt, Field(ge=0, le=32))


# Every stat 0..32 is enforced by the schema. The sum cannot be written as a
# JSON-schema rule, so the model must also state the total (schema maximum 66).
# The total is never validated or kept: the unchanged baseline check judges the
# real sum, so a wrong total cannot turn a legal spread into a rejected one.
Spread = create_model('StatPoints', __config__=ConfigDict(extra='forbid'),
                      **{k: _stat() for k in STATS},
                      total=(StrictInt, Field(description=TOTAL, json_schema_extra={'maximum': 66})))


def _enum(name, values):
    if not values:
        raise ValueError(f'no legal choice for {name}')
    return Enum(name, {f'v{i}': v for i, v in enumerate(values)}, type=str)


def legal_completion_type(masked_team, paths):
    """One field per requested path, typed with what the regulation allows there.

    Per-slot choices come from the pinned simulator vocabulary (data/regmb-vocabulary.json):
    an ability or item is limited to the species' legal ability/item pairs given the
    slot's fixed ability or item; a move to the species' learnable moves not already
    in the slot; an item excludes items fixed teammates hold (Item Clause); a whole
    slot's species excludes families already on the team (Species Clause). Choices
    that depend on the model's own other answers (two masked items, a whole slot's
    moves for the species it picks) are stated in descriptions, not enforced.
    """
    from mb_prompt import norm, regulation
    reg = regulation()
    names = reg['names']
    show = lambda ids: sorted({names.get(i, i) for i in ids if i})
    slots = [None if s is None else reg['slots'].get(norm(s['species'])) if s.get('species') else None
             for s in masked_team]
    held = {norm(s['item']) for s in masked_team if s and s.get('item')}
    families = {r['base'] for r in slots if r}
    enums = {}

    def enum(name, values):
        if name not in enums:
            enums[name] = _enum(name, values)
        return enums[name]

    nature = lambda: enum('Nature', show(reg['values']['nature']))
    whole = None
    fields = {}
    for path in paths:
        i = path[0]
        slot, rules = masked_team[i], slots[i]
        where = f'path {json.dumps(path)}'
        if len(path) == 1:
            if whole is None:
                species = [s for s in reg['values']['species'] if s and reg['slots'][s]['base'] not in families]
                whole = create_model(
                    'PokemonSet', __config__=ConfigDict(extra='forbid'),
                    species=(enum('Species', show(species)), Field(description='not the same species family as any other Pokémon on the team')),
                    item=(enum('Item', show(it for it in reg['values']['item'] if it not in held)),
                          Field(description='held by no other Pokémon on the team; a Mega Stone only on its own species')),
                    ability=(enum('Ability', show(reg['values']['ability'])), Field(description="one of the chosen species' own abilities")),
                    nature=(nature(), ...),
                    moves=(list[enum('Move', show(reg['values']['move']))],
                           Field(min_length=4, max_length=4, description='four different moves the chosen species can learn')),
                    evs=(Spread, ...))
            fields[f'slot{i}'] = (whole, Field(description=where))
        elif path[1] == 'evs':
            fields[f'slot{i}_evs'] = (Spread, Field(description=where))
        elif path[1] == 'nature':
            fields[f'slot{i}_nature'] = (nature(), Field(description=where))
        elif rules is None:  # species masked too: only regulation-wide choices
            pool = {'ability': 'ability', 'item': 'item', 'moves': 'move', 'species': 'species'}[path[1]]
            fields[f'slot{i}_{path[1]}' + (f'{path[2]}' if len(path) > 2 else '')] = (
                enum(pool.title(), show(reg['values'][pool])), Field(description=where))
        elif path[1] == 'ability':
            item = slot.get('item')
            legal = [a for a, items in rules['abilityItems'].items()
                     if items and (item is None or norm(item) in items)]
            fields[f'slot{i}_ability'] = (enum(f'Slot{i}Ability', show(legal)), Field(description=where))
        elif path[1] == 'item':
            ability = slot.get('ability')
            legal = {it for a, items in rules['abilityItems'].items()
                     if ability is None or a == norm(ability) for it in items}
            fields[f'slot{i}_item'] = (enum(f'Slot{i}Item', show(legal - held)),
                                       Field(description=where + '; different from every other item on the team'))
        elif path[1] == 'moves':
            known = {norm(m) for m in slot['moves'] if m}
            fields[f'slot{i}_move{path[2]}'] = (
                enum(f'Slot{i}Move', show(m for m in rules['moves'] if m not in known)),
                Field(description=where + '; different from the other moves of this Pokémon'))
        elif path[1] == 'species':
            fields[f'slot{i}_species'] = (enum('Species', show(s for s in reg['values']['species'] if s and reg['slots'][s]['base'] not in families)),
                                          Field(description=where))
        else:
            raise ValueError(f'no legal-choice type for path {path}')
    return create_model('Completion', __config__=ConfigDict(extra='forbid'), **fields)


def config(adapter, instructions=None, demos=(), constraints=None):
    """Manifest entry for a structured arm. Phase two adds instructions and demos."""
    if adapter not in ADAPTERS:
        raise ValueError(f'unknown DSPy adapter {adapter}; choose from {ADAPTERS}')
    if constraints is not None and constraints not in CONSTRAINTS:
        raise ValueError(f'unknown answer constraints {constraints}; choose from {CONSTRAINTS}')
    entry = dict(kind=KIND, adapter=adapter, dspy_version=dspy.__version__,
                 instructions=instructions, demos=list(demos))
    if constraints:  # absent otherwise, so phase-one manifests keep their hashes
        entry['constraints'] = constraints
    return entry


def adapter_for(structure):
    if structure.get('kind') != KIND:
        raise ValueError(f"unknown request structure {structure.get('kind')}")
    if structure['dspy_version'] != dspy.__version__:
        raise ValueError(f"installed DSPy {dspy.__version__} differs from frozen {structure['dspy_version']}")
    if structure['adapter'] == 'chat':
        # No JSON fallback: the baseline makes no second (repair) call.
        return dspy.ChatAdapter(use_json_adapter_fallback=False)
    if structure['adapter'] == 'json':
        return dspy.JSONAdapter()
    raise ValueError(f"unknown DSPy adapter {structure['adapter']}")


def strip_envelope(system):
    text = system
    for sentence in ENVELOPE:
        text = text.replace(sentence, '')
    if text == system:
        raise ValueError('baseline system text has no known reply-format sentence; add it to ENVELOPE')
    return text.strip()


def baseline_parts(manifest, trial):
    import llm_completion
    plain = {k: v for k, v in manifest.items() if k != 'structure'}
    system, user = llm_completion.prompt(plain, trial)
    return system['content'], json.loads(user['content'])


def signature(instructions, keys, answer=None):
    unknown = set(keys) - set(INPUT_TYPES)
    if unknown:
        raise ValueError(f'no structured type for baseline inputs {sorted(unknown)}')
    fields = {k: (INPUT_TYPES[k], dspy.InputField(desc=DESCRIPTIONS[k])) for k in keys}
    if answer is None:
        fields['fills'] = (list[Fill], dspy.OutputField(desc='one fill per requested path'))
    else:
        fields['fills'] = (answer, dspy.OutputField(
            desc='one value per requested path; each field name gives the slot and field it fills'))
    return dspy.make_signature(fields, instructions, signature_name='CompleteMaskedTeam')


def parts(manifest, trial):
    """Signature, inputs and adapter that carry the baseline request's content."""
    structure = manifest['structure']
    system, inputs = baseline_parts(manifest, trial)
    instructions = structure.get('instructions') or strip_envelope(system)
    answer = None
    if structure.get('constraints') == 'legal-choices':
        answer = legal_completion_type(inputs['masked_team'], trial['paths'])
    elif structure.get('constraints') is not None:
        raise ValueError(f"unknown answer constraints {structure['constraints']}")
    return signature(instructions, list(inputs), answer), inputs, adapter_for(structure)


def as_fills(completion, paths):
    """Turn the one-field-per-path answer back into the baseline {"fills": [...]} shape."""
    values = completion.model_dump(mode='json')
    fills = []
    for path, value in zip(paths, values.values()):
        if isinstance(value, dict):
            value = dict(value)
            for spread in (value, value.get('evs')):
                if isinstance(spread, dict):
                    spread.pop('total', None)  # only there to make the model add up
        fills.append({'path': list(path), 'value': value})
    return {'fills': fills}


def messages(manifest, trial):
    sig, inputs, adapter = parts(manifest, trial)
    return adapter.format(sig, list(manifest['structure'].get('demos') or []), inputs)


def raw_fills(structure, content):
    """The model's own text for the fills field, before any DSPy repair."""
    if structure['adapter'] == 'chat':
        match = re.search(r'\[\[ ## fills ## \]\]\s*(.*?)\s*(?:\[\[ ## |\Z)', content, re.S)
        return match.group(1) if match else None
    try:
        return json.dumps(json.loads(content)['fills'])
    except (ValueError, KeyError, TypeError):
        return None


def refusal_message(exc):
    """A DSPy parse failure as a short message that leads with the reason.

    DSPy's message echoes the whole answer before the reason, so the reason comes
    first or it is cut off; the list of allowed values (up to hundreds of names)
    is dropped, keeping the field, the pydantic error code and the input.
    """
    text = str(exc)
    reason = text.split('Error message:', 1)[1].strip() if 'Error message:' in text else ''
    reason = re.sub(r'Input should be .*? \[type=', '[type=', reason, flags=re.S)
    reason = re.sub(r'\s*For further information visit \S+', '', reason)
    return f'DSPy parse: {reason[:400]} | {text[:300]}' if reason else f'DSPy parse: {text[:500]}'


def parse(manifest, trial, content, diagnostics):
    """Typed parse into the baseline answer shape, {"fills": [...]}.

    `diagnostics['strict_json']` records whether the model's fills text was
    valid JSON as written, so a report can separate DSPy's tolerant JSON
    repair from the model's own output. It is set before parsing can fail.
    """
    if not isinstance(content, str):
        raise TypeError('reply has no answer text')
    sig, _, adapter = parts(manifest, trial)
    try:
        json.loads(raw_fills(manifest['structure'], content))
        diagnostics['strict_json'] = True
    except (TypeError, ValueError):
        diagnostics['strict_json'] = False
    try:
        fields = adapter.parse(sig, content)
    except dspy.AdapterParseError as exc:
        # The baseline counts a ValueError as an invalid answer.
        raise ValueError(refusal_message(exc)) from None
    if manifest['structure'].get('constraints'):
        return as_fills(fields['fills'], trial['paths'])
    return {'fills': [fill.model_dump() for fill in fields['fills']]}


def section(text, key):
    match = re.search(r'\[\[ ## ' + re.escape(key) + r' ## \]\]\n(.*?)(?=\n\n\[\[ ## |\n\nRespond |\Z)', text, re.S)
    return match.group(1) if match else None


def check_information(manifest, trial):
    """Prove the structured request carries every baseline input value unchanged."""
    _, inputs = baseline_parts(manifest, trial)
    user = messages(manifest, trial)[-1]['content']
    for key, value in inputs.items():
        text = section(user, key)
        if text is None:
            raise ValueError(f'structured request lacks input {key}')
        if (text if isinstance(value, str) else json.loads(text)) != value:
            raise ValueError(f'structured request changed input {key}')
    return sorted(inputs)
