"""Battle candidate teams in exact 50-battle panels on the fast evaluator. See docs/llm-baseline.md.

Each candidate plays its starting team's frozen opponent schedule. A panel is
cached in matchup_db's llm_completion_panel table under its battle config,
candidate and schedule, and is published only once all 50 battles are attributed.

The legacy engine (pair_shard with the behaviour-cloning PPO policy on one
private Showdown server) was removed on 2026-10-19. Folders it battled still
read: recorded_backend() names their engine, and battle() moves their labels
into an archive folder before re-battling them on the fast evaluator."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import time

from llm_completion import load_manifest, opponents_for
from results import digest, save
import showdown_validator

FAST_PANELS = 12  # panels per evaluator call: 600 battles keep six workers x 24 battles busy
_fast_runner = None


def schedule_for(manifest, start):
    ids = [o['team_id'] for o in opponents_for(manifest, start)]
    if not ids: raise ValueError('empty opponent panel')
    random.Random(manifest['seed']).shuffle(ids)
    return (ids * (50 // len(ids) + 1))[:50]


def candidate_start(label):
    if label.startswith('start-'): return int(label.split('-')[1])
    return int(label.split('-')[1][1:])


def candidates_for(args, m):
    candidates = [(f"start-{i}", t) for i, t in enumerate(m["starts"])]
    for trial in m["trials"]:
        candidates.append(("random-" + trial["id"], trial["random_team"]))
        path = args.output / "completions" / (trial["id"] + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if row["valid"]: candidates.append(("llm-" + trial["id"], row["team"]))
    return candidates


def battle_paste(text):
    """Preserve neutral stats while acknowledging intentional zero investment.

    Canonical identity maps neutral natures to Serious. Showdown's challenge
    validator uniquely rejects Serious plus zero Stat Points as a user warning;
    Hardy has identical stats and neutral flavor preferences, and acknowledges
    intentional zero investment. Do not alter canonical DB/cache identity.
    """
    blocks = []
    for block in text.strip().split('\n\n'):
        total = sum(int(n) for line in block.splitlines() if line.startswith('EVs:')
                    for n in re.findall(r'\b(\d+)\s+(?:HP|Atk|Def|SpA|SpD|Spe)\b', line))
        if total == 0:
            block = '\n'.join('Hardy Nature' if line == 'Serious Nature' else line
                              for line in block.splitlines())
        blocks.append(block)
    return '\n\n'.join(blocks) + '\n'


def read_config(folder):
    path = Path(folder) / 'battle-config.json'
    return json.loads(path.read_text()) if path.exists() else None


def recorded_backend(folder):
    """'fast', 'legacy' or None: the engine a folder's saved labels came from.
    Configs written before the fast engine name none; labels without a config predate it."""
    config = read_config(folder)
    if config is not None:
        return config.get('backend', 'legacy')
    labels = Path(folder) / 'labels'
    return 'legacy' if labels.is_dir() and any(labels.glob('*.json')) else None


def archive_legacy_labels(folder, extra=()):
    """Before battling a folder the legacy engine battled, move its labels and battle
    config (plus any named summaries) into an archive folder, never deleting them, so
    every label in the folder comes from the fast evaluator. Returns the archive, or None."""
    folder = Path(folder)
    if recorded_backend(folder) != 'legacy':
        return None
    archive = folder / f"archive-legacy-battles-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    archive.mkdir()
    for name in ('labels', 'battle-config.json', *extra):
        if (folder / name).exists():
            (folder / name).rename(archive / name)
    print(f'{folder}: legacy labels moved to {archive.name}; re-battling on the fast evaluator', flush=True)
    return archive


def fast_runner(folder):
    """One persistent evaluator per process, reused by every later battle call."""
    global _fast_runner
    if _fast_runner is None or _fast_runner.closed:
        import atexit
        import matchup_db as db
        if str(db.FAST_RUNNER_DIR) not in sys.path: sys.path.append(str(db.FAST_RUNNER_DIR))
        from fast_battle_runner import FastBattleRunner
        _fast_runner = FastBattleRunner(Path(folder) / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.time_ns()}",
                                        workers=int(os.environ.get("MATCHUP_FAST_WORKERS", 6)),
                                        concurrency=int(os.environ.get("MATCHUP_FAST_CONCURRENCY", 24)),
                                        runtime=db.FAST_EVIDENCE / "runtime")
        atexit.register(_fast_runner.close)
    return _fast_runner


def fast_panels(runner, con, panels, wire, seed):
    """Battle several exact panels in one all-or-nothing evaluator call.

    Each panel row gets its own identity in a private in-memory team table that
    holds the battle exports, so a completion equal to a pool team still plays
    its mirror (the evaluator refuses a team battling its own identity).
    Returns {panel index: [(team_id, opponent_id, wins, losses, ties)]}."""
    mem = sqlite3.connect(':memory:')
    try:
        mem.execute('CREATE TABLE team (team_id INTEGER PRIMARY KEY, paste TEXT, base_key TEXT, format_id TEXT, legal INTEGER)')
        def local(team_id):
            base = con.execute('SELECT base_key FROM team WHERE team_id=?', (team_id,)).fetchone()[0]
            row = mem.execute('INSERT INTO team (paste,base_key,format_id,legal) VALUES (?,?,?,1)',
                              (wire(team_id), base, showdown_validator.FORMAT)).lastrowid
            real[row] = team_id
            return row, base
        real, owner, cols, jobs = {}, {}, {}, []
        for index, (tid, schedule) in enumerate(panels):
            row, _ = local(tid)
            owner[row] = index
            job, seen = [], {}
            for oid, n in sorted(Counter(schedule).items()):
                if oid not in cols: cols[oid] = local(oid)
                col, base = cols[oid]
                # Two opponents with one preview cannot be told apart in one job (as in jobs_for).
                if seen.get(base, col) != col:
                    jobs.append(dict(row=row, schedule=job))
                    job, seen = [], {}
                job += [col] * n
                seen[base] = col
            jobs.append(dict(row=row, schedule=job))
        raw, unattributed, _ = runner(mem, jobs, seed=seed)
    finally:
        mem.close()
    merged = defaultdict(lambda: [0, 0, 0])
    for row, col, *counts in raw:
        for i, count in enumerate(counts): merged[owner[row], real[col]][i] += count
    by_panel = defaultdict(list)
    for (index, oid), counts in sorted(merged.items()):
        by_panel[index].append((panels[index][0], oid, *counts))
    return by_panel, unattributed


def battle(args, candidates_override=None, on_result=None):
    """The exact 50-battle panels on the persistent six-worker evaluator.

    The config (and so every cache key) names the evaluator's fingerprint, so a
    fast panel never answers a lookup cached by the legacy engine. Panels are
    battled several per evaluator call and published one by one, each only once
    all 50 battles are attributed."""
    requested = getattr(args, 'battle_backend', None)
    if requested not in (None, 'fast'):
        raise ValueError(f"battle engine {requested!r} does not exist: the legacy engine (pair_shard with "
                         "the behaviour-cloning PPO policy) was removed on 2026-10-19 and only the fast "
                         "evaluator remains. Labels it recorded stay readable and are archived before a re-battle.")
    archive_legacy_labels(args.output)
    import hps_generate as h, matchup_db as db
    m = load_manifest(args)
    runner = fast_runner(args.output / 'fast-battles')
    config = dict(format=showdown_validator.FORMAT, policy="behaviour-cloning both sides", backend='fast',
                  evaluator=runner.identity, evaluator_provenance=runner.provenance,
                  seed=m["seed"], battles=50, opponents=m["opponents"], seed_limit=m["seed_limit"])
    if m.get('exclude_starter_team_ids'):
        config['exclude_starter_team_ids'] = m['exclude_starter_team_ids']
    con = db.connect()
    con.execute("CREATE TABLE IF NOT EXISTS llm_completion_panel "
                "(cache_key TEXT PRIMARY KEY, config_json TEXT NOT NULL, result_json TEXT NOT NULL)")
    con.commit()
    pid = runner.policy_id(con, db)
    candidates = candidates_for(args, m) if candidates_override is None else candidates_override
    validator = h.Validator()
    try:
        wire_texts = {}
        def wire(team_id):
            # The battle export: canonical paste, zero-point Serious written as Hardy.
            if team_id not in wire_texts:
                text = con.execute("SELECT paste FROM team WHERE team_id=?", (team_id,)).fetchone()[0]
                exported = battle_paste(text)
                if db.canonical_paste(exported) != db.canonical_paste(text):
                    raise ValueError(f'battle export changed canonical team {team_id}')
                error = validator(exported)
                if error:
                    raise ValueError(f'battle export rejected for team {team_id}: {error}')
                wire_texts[team_id] = exported
            return wire_texts[team_id]
        for o in m["opponents"]:
            stored = con.execute("SELECT paste FROM team WHERE team_id=?", (o["team_id"],)).fetchone()
            if not stored or stored[0] != o["paste"]: raise ValueError("opponent database changed")
        schedule = schedule_for(m, 0)
        panels = {'schedules_by_start': {str(i):schedule_for(m,i) for i in range(len(m['starts']))}} if m.get('exclude_starter_team_ids') else {'schedule':schedule}
        save(args.output / "battle-config.json", dict(config, policy_id=pid, **panels))
        def finish(label, result):
            save(args.output / "labels" / (label + ".json"), result)
            if on_result is not None:
                on_result(label, result)
            print(label, f"{result['wins']}/50", flush=True)
        pending = {}
        for label, team in candidates:
            schedule = schedule_for(m, candidate_start(label))
            text = showdown_validator.paste(team, h)
            error = validator(text)
            if error: raise ValueError(error)
            tid = db.add_team(con, text, "llm-completion:" + args.output.name, validator)[0]
            key = digest(dict(config=config, candidate=db.canonical_paste(text), schedule=schedule))
            cached = con.execute("SELECT result_json FROM llm_completion_panel WHERE cache_key=?", (key,)).fetchone()
            if cached:
                finish(label, json.loads(cached[0]))
            else:
                pending.setdefault(key, dict(tid=tid, schedule=schedule, labels=[]))['labels'].append(label)
        work = list(pending.items())
        size = getattr(args, 'fast_panels', None) or FAST_PANELS
        for first in range(0, len(work), size):
            batch = work[first:first + size]
            by_panel, unattributed = fast_panels(runner, con, [(p['tid'], p['schedule']) for _, p in batch],
                                                 wire, m["seed"])
            if unattributed: raise RuntimeError(f"unattributed battles: {unattributed}")
            for index, (key, p) in enumerate(batch):
                results = by_panel.get(index, [])
                n = sum(w+l+t for _, _, w,l,t in results)
                played = Counter()
                for _, oid, w, l, t in results: played[oid] += w+l+t
                if n != 50 or played != Counter(p['schedule']):
                    raise RuntimeError(f"incomplete panel {p['labels'][0]}: {n} battles against the 50-battle schedule")
                # Round-trip through JSON so a published result equals its saved copy.
                result = json.loads(json.dumps(dict(team_id=p['tid'], wins=sum(r[2] for r in results), battles=n,
                                                    results=results, cache_key=key)))
                off_diagonal = [r for r in results if r[0] != r[1]]
                if off_diagonal:
                    db.record(con, pid, off_diagonal, seed=m["seed"], note="llm-completion:" + args.output.name)
                with con:
                    con.execute("INSERT INTO llm_completion_panel VALUES (?,?,?)",
                                (key, json.dumps(config), json.dumps(result)))
                for label in p['labels']:
                    finish(label, result)
    finally:
        validator.close()
        con.close()
