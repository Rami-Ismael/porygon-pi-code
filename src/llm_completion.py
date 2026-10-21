"""Ask the language model to fill a masked team, and check its reply. See docs/llm-baseline.md.

A trial hides some paths of a starting team. prompt() writes the request,
complete() sends it to OpenRouter once and saves the paid reply before parsing
it, and validate_record() turns the reply into a legal team or an error.
Prompt versions (mb_prompt, prompt_versions) and the DSPy request structure
(dspy_structure) live in scripts/ and load only when a manifest names them."""
import copy
import json
import os
import time
import urllib.error
import urllib.request

from results import digest, save
import showdown_validator
from team_masks import masked, set_path, validate_shape

MODEL = "inclusionai/ling-3.0-flash-vl"


def load_manifest(args):
    m = json.loads((args.output / "manifest.json").read_text())
    if m["model"] not in (MODEL, 'inception/mercury-2.5') or m["format"] != showdown_validator.FORMAT:
        raise ValueError("model/format mismatch")
    return m


def opponents_for(manifest, start):
    excluded = manifest.get('exclude_starter_team_ids')
    return [o for o in manifest['opponents'] if not excluded or o['team_id'] != excluded[start]]


def prompt(manifest, trial):
    team = manifest["starts"][trial["start"]]
    instructions = (
        "Complete only the null fields of this Pokemon Champions Regulation M-B team to maximize "
        "win rate against the supplied opponent pool. These are Champions Stat Points, NOT Scarlet/Violet EVs: "
        "integer 0..32 each, total <=66; Stat Alignment means nature. Species Clause and Item Clause apply. "
        "Four distinct legal moves per Pokemon; legal species, ability and item combinations. No Terastallization. "
        "Return only JSON: {\"fills\":[{\"path\":[0,\"item\"],\"value\":\"...\"}]}. "
        "Return exactly one fill per requested path, using the identical path. For a whole slot return "
        "{species,item,ability,nature,moves,evs}; evs keys are HP,Atk,Def,SpA,SpD,Spe. "
        "Never change, reorder, or return unmasked fields. No prose or Markdown."
    )
    if manifest.get('preserve_empty_move_slots'):
        instructions += ' Empty-string move slots are intentionally unused: preserve them when unmasked. Fill each masked move position with a nonempty legal move.'
    messages = [{"role": "system", "content": instructions}, {"role": "user", "content": json.dumps({
        "format": showdown_validator.FORMAT, "opponents": [o["paste"] for o in opponents_for(manifest, trial['start'])],
        "masked_team": masked(team, trial["paths"]), "requested_paths": trial["paths"]})}]
    # Versions are logged in docs/prompt-changelog.md; no key means mb-original-v1.
    version = manifest.get('prompt_version')
    if version == 'mb-grounded-v2':
        from mb_prompt import strengthen
        messages = strengthen(messages,manifest,trial)
    elif version == 'mb-regulation-v3':
        from prompt_versions import add_regulation
        messages = add_regulation(messages)
    elif version is not None:
        raise ValueError(f'unknown prompt_version {version}')
    return messages


def completion_payload(manifest, trial):
    """One request definition for generation and experiment provenance checks."""
    if manifest.get('structure'):
        # Same instructions and inputs, packaged as typed DSPy fields.
        from dspy_structure import messages
    else:
        messages = prompt
    return dict(model=manifest['model'], messages=messages(manifest, trial),
                temperature=manifest['temperature'], max_tokens=manifest['max_tokens'],
                reasoning=manifest.get('reasoning', {'enabled': True}),
                provider=manifest.get('provider', {'order': ['novita'], 'allow_fallbacks': False}))


def apply_completion(team, paths, response):
    """Patches only: no reordering, missing fills, duplicate paths, or extra edits."""
    if not isinstance(response, dict) or set(response) != {"fills"}:
        raise ValueError("expected exactly {fills: [{path, value}, ...]}")
    fills = response["fills"]
    if not isinstance(fills, list) or len(fills) != len(paths):
        raise ValueError("wrong number of fills")
    remaining = [list(p) for p in paths]
    result = copy.deepcopy(team)
    for fill in fills:
        if not isinstance(fill, dict) or set(fill) != {"path", "value"}:
            raise ValueError("each fill must contain path and value")
        path = fill["path"]
        if path not in remaining or any(type(k) not in (str, int) for k in path):
            raise ValueError("duplicate or unmasked path")
        remaining.remove(path)
        if len(path)==3 and path[1]=='moves' and not fill['value']:
            raise ValueError('masked moves require a nonempty completion')
        set_path(result, path, fill["value"])
    validate_shape(result)
    for path in paths:
        if len(path)==1 and any(not move for move in result[path[0]]['moves']):
            raise ValueError('whole-slot completions require four nonempty moves')
    if masked(result, paths) != masked(team, paths):
        raise ValueError("unmasked fields changed")
    return result


def complete(args, trials_override=None):
    m = load_manifest(args)
    key = os.environ.get("OPENROUTER_API_KEY")
    import hps_generate as h
    validator = h.Validator()
    try:
        for trial_index, trial in enumerate(m["trials"] if trials_override is None else trials_override):
            if trial_index % getattr(args, 'shards', 1) != getattr(args, 'shard', 0): continue
            out = args.output / "completions" / (trial["id"] + ".json")
            payload = completion_payload(m, trial)
            request_hash = digest(payload)
            if out.exists():
                record = json.loads(out.read_text())
                if record["request_hash"] != request_hash:
                    raise ValueError("saved completion request mismatch")
                if not record['valid'] and 'error' not in record:
                    validate_record(record, m, trial, h, validator)
                    save(out, record)
                continue
            pending = out.with_suffix(".pending")
            if pending.exists():
                raise RuntimeError(f"uncertain previous API request: {pending}; inspect before retrying")
            if not key: raise RuntimeError("OPENROUTER_API_KEY is not configured; no requests sent")
            save(pending, dict(request_hash=request_hash, payload=payload, started_at=time.time()))
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                  data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {key}",
                  "Content-Type": "application/json"})
            started = time.monotonic()
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=m.get('request_timeout', 180)) as res:
                        response = json.load(res)
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode()[:2000]
                    save(out.with_suffix(".http-error.json"), dict(status=exc.code, detail=detail))
                    if exc.code == 429 and attempt < 2:
                        print("OpenRouter rate limit; retrying in 20 seconds", flush=True)
                        time.sleep(20)
                        continue
                    pending.unlink()
                    raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from None
            record = dict(request_hash=request_hash, response=response, valid=False,
                          elapsed_seconds=time.monotonic() - started)
            # Persist the paid response before parsing or validating it.
            save(out, record)
            pending.unlink()
            validate_record(record, m, trial, h, validator)
            save(out, record)
            print(trial["id"], "legal" if record["valid"] else "invalid", flush=True)
    finally:
        validator.close()


def validate_record(record, manifest, trial, h, validator):
    """Recover validation after a crash without resending the paid request."""
    try:
        choice = record['response']['choices'][0]
        if choice['finish_reason'] != 'stop': raise ValueError('truncated/non-stop completion')
        content = choice['message']['content']
        if manifest.get('structure'):
            from dspy_structure import parse
            record['structure'] = {}
            answer = parse(manifest, trial, content, record['structure'])
        else:
            answer = json.loads(content)
        team = apply_completion(manifest['starts'][trial['start']], trial['paths'], answer)
        error = validator(showdown_validator.paste(team, h))
        if error: raise ValueError(error)
        record.update(valid=True, team=team)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        record['error'] = str(exc)
