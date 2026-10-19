"""Frozen, resumable Ling-only masked-completion pilot. See docs/llm-baseline.md."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from repo_paths import ROOT  # also puts src/ first on sys.path
from llm_completion import complete, load_manifest, opponents_for
from completion_study import prepare, report
from results import digest, save
import showdown_validator
BC_SHA256 = "57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b"


def schedule_for(manifest, start):
    ids = [o['team_id'] for o in opponents_for(manifest, start)]
    if not ids: raise ValueError('empty opponent panel')
    random.Random(manifest['seed']).shuffle(ids)
    return (ids * (50 // len(ids) + 1))[:50]


def candidate_start(label):
    if label.startswith('start-'): return int(label.split('-')[1])
    return int(label.split('-')[1][1:])


def isolated_showdown(source, destination):
    """Copy executable/config state; only third-party dependencies are shared.

    Separate ports alone are insufficient: Showdown writes config/*.NEW and
    logs/lastbattle.txt relative to its module root, not the process cwd.
    Copying dist (rather than symlinking it) gives each server its own root.
    """
    source, destination = Path(source), Path(destination)
    shutil.copytree(source, destination,
                    ignore=lambda folder, names: {'node_modules', '.git', 'logs'} if Path(folder) == source else set())
    (destination / 'node_modules').symlink_to(source / 'node_modules', target_is_directory=True)
    (destination / 'logs').mkdir()
    # The CLI's skipped build normally creates this directory before REPL cleanup.
    (destination / 'logs/repl').mkdir()
    return destination


def battle_paste(text):
    """Preserve neutral stats while acknowledging intentional zero investment.

    Canonical identity maps neutral natures to Serious. Showdown's challenge
    validator uniquely rejects Serious plus zero Stat Points as a user warning;
    Hardy has identical stats and neutral flavor preferences, and acknowledges
    intentional zero investment. Do not alter canonical DB/cache identity.
    """
    import re
    blocks = []
    for block in text.strip().split('\n\n'):
        total = sum(int(n) for line in block.splitlines() if line.startswith('EVs:')
                    for n in re.findall(r'\b(\d+)\s+(?:HP|Atk|Def|SpA|SpD|Spe)\b', line))
        if total == 0:
            block = '\n'.join('Hardy Nature' if line == 'Serious Nature' else line
                              for line in block.splitlines())
        blocks.append(block)
    return '\n\n'.join(blocks) + '\n'


def candidates_for(args, m):
    candidates = [(f"start-{i}", t) for i, t in enumerate(m["starts"])]
    for trial in m["trials"]:
        candidates.append(("random-" + trial["id"], trial["random_team"]))
        path = args.output / "completions" / (trial["id"] + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if row["valid"]: candidates.append(("llm-" + trial["id"], row["team"]))
    return candidates


def backend_of(config):
    return config.get('backend', 'legacy')  # configs written before the fast engine name none


def resolve_backend(requested=None):
    """Fast unless legacy is asked for by name."""
    if requested not in (None, 'legacy', 'fast'):
        raise ValueError(f"battle backend must be legacy or fast, not {requested!r}")
    return requested or 'fast'


def read_config(folder):
    path = Path(folder) / 'battle-config.json'
    return json.loads(path.read_text()) if path.exists() else None


def recorded_backend(folder):
    """The engine a folder's saved labels came from; labels without a config predate the fast engine."""
    config = read_config(folder)
    if config is not None:
        return backend_of(config)
    labels = Path(folder) / 'labels'
    return 'legacy' if labels.is_dir() and any(labels.glob('*.json')) else None


def switch_engine(folder, backend, extra=()):
    """Before battling a folder on another engine, move its labels and battle config
    (plus any named summaries) into an archive folder, never deleting them, so every
    label in the folder comes from one engine. Returns the archive, or None."""
    folder = Path(folder)
    old = recorded_backend(folder)
    if old is None or old == backend:
        return None
    archive = folder / f"archive-{old}-battles-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    archive.mkdir()
    for name in ('labels', 'battle-config.json', *extra):
        if (folder / name).exists():
            (folder / name).rename(archive / name)
    print(f'{folder}: {old} labels moved to {archive.name}; re-battling on {backend}', flush=True)
    return archive


def battle(args, candidates_override=None, on_result=None):
    """Separate exact-panel cache; never reuse aggregate cells with unknown seeds.

    Fast by default (legacy only when args.battle_backend == 'legacy'). A folder
    battled on the other engine is archived first and re-battled in full."""
    backend = resolve_backend(getattr(args, 'battle_backend', None))
    switch_engine(args.output, backend)
    if backend == 'fast':
        return fast_battle(args, candidates_override, on_result)
    import asyncio
    import importlib.util
    import torch
    from stable_baselines3 import PPO
    h, db = showdown_validator.runtime()
    m = load_manifest(args)
    bench = Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2"
    os.chdir(bench)  # vgc_bench loads data/*.json relative to its checkout.
    sys.path.insert(0, str(bench))
    # Load the project's attribution-aware worker, using the local shard module.
    spec = importlib.util.spec_from_file_location("llm_pair_worker", ROOT / "src/pair_shard.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    torch.set_num_threads(1)
    if hashlib.sha256(db.POLICY_FILE.read_bytes()).hexdigest() != BC_SHA256:
        raise ValueError("checkpoint differs from the verified behaviour-cloning model")
    policy = PPO.load(str(db.POLICY_FILE), device=torch.device("cpu")).policy
    if not policy.choose_on_teampreview:
        raise ValueError("checkpoint does not use learned team preview")
    runtime_files = [db.POLICY_FILE, ROOT / "src/pair_shard.py", ROOT / "src/shard.py"]
    runtime_files += sorted((bench / "vgc_bench").rglob("*.py"))
    runtime_files += sorted((bench / "data").rglob("*.json"))
    runtime_files += sorted((h.SHOWDOWN / "dist").rglob("*.js"))
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files}
    config = dict(format=showdown_validator.FORMAT, policy="behaviour-cloning both sides", fingerprints=fingerprints,
                  concurrency=1, seed=m["seed"], battles=50, opponents=m["opponents"],
                  seed_limit=m["seed_limit"])
    if m.get('exclude_starter_team_ids'):
        config['exclude_starter_team_ids'] = m['exclude_starter_team_ids']
    con = db.connect()
    con.execute("CREATE TABLE IF NOT EXISTS llm_completion_panel "
                "(cache_key TEXT PRIMARY KEY, config_json TEXT NOT NULL, result_json TEXT NOT NULL)")
    con.commit()
    pid = db.get_policy(con, simulator="showdown-content:" + digest(fingerprints),
                        note="Ling baseline; BC both sides; exact-panel cache stored separately")
    candidates = candidates_for(args, m) if candidates_override is None else candidates_override
    # Own an isolated server built from the same checkout used for validation.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # macOS Unix socket paths are short; its default per-user temp path is too
    # long for Showdown's logs/repl/team-validator-* sockets.
    sandbox = tempfile.TemporaryDirectory(prefix='vgc-sd-', dir='/tmp')
    server_root = isolated_showdown(h.SHOWDOWN, Path(sandbox.name) / 'server')
    # Never rebuild the frozen simulator while multiple workers are running.
    for original in sorted((h.SHOWDOWN / 'dist').rglob('*.js')):
        copied = server_root / original.relative_to(h.SHOWDOWN)
        if hashlib.sha256(copied.read_bytes()).hexdigest() != fingerprints[str(original)]:
            raise ValueError('isolated simulator differs from frozen code')
    log = (args.output / "showdown.log").open("a")
    server = subprocess.Popen(["node", "pokemon-showdown", "start", str(port), "--no-security", "--skip-build"],
                              cwd=server_root, stdout=log, stderr=subprocess.STDOUT)
    validator = h.Validator()
    try:
        wire_files = {}
        def battle_file(team_id):
            if team_id not in wire_files:
                original = Path(db.paste_file(con, team_id))
                text = original.read_text()
                wire = battle_paste(text)
                if db.canonical_paste(wire) != db.canonical_paste(text):
                    raise ValueError(f'battle export changed canonical team {team_id}')
                error = validator(wire)
                if error:
                    raise ValueError(f'battle export rejected for team {team_id}: {error}')
                path = original
                if wire != text:
                    path = args.output / 'wire-teams' / (digest(wire) + '.txt')
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(wire)
                wire_files[team_id] = str(path)
            return wire_files[team_id]
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2): break
            except OSError:
                if server.poll() is not None: raise RuntimeError("Showdown failed; see showdown.log")
                time.sleep(.1)
        else: raise RuntimeError("Showdown startup timeout")
        opp_ids = [o["team_id"] for o in m["opponents"]]
        for o in m["opponents"]:
            stored = con.execute("SELECT paste FROM team WHERE team_id=?", (o["team_id"],)).fetchone()
            if not stored or stored[0] != o["paste"]: raise ValueError("opponent database changed")
        schedule = schedule_for(m, 0)
        panels = {'schedules_by_start': {str(i):schedule_for(m,i) for i in range(len(m['starts']))}} if m.get('exclude_starter_team_ids') else {'schedule':schedule}
        save(args.output / "battle-config.json", dict(config, policy_id=pid, **panels))
        for label, team in candidates:
            schedule = schedule_for(m, candidate_start(label))
            text = showdown_validator.paste(team, h)
            error = validator(text)
            if error: raise ValueError(error)
            tid = db.add_team(con, text, "llm-completion:" + args.output.name, validator)[0]
            key = digest(dict(config=config, candidate=db.canonical_paste(text), schedule=schedule))
            cached = con.execute("SELECT result_json FROM llm_completion_panel WHERE cache_key=?", (key,)).fetchone()
            if cached:
                result = json.loads(cached[0])
            else:
                results, unattributed = [], 0
                # jobs_for splits preview collisions while preserving common panels.
                from collections import Counter
                jobs = db.jobs_for(con, [(tid, oid, n) for oid, n in Counter(schedule).items()])
                for job in jobs:
                    cols = sorted(set(job["schedule"]))
                    info = {c: con.execute("SELECT species_key,base_key FROM team WHERE team_id=?", (c,)).fetchone() for c in cols}
                    job_spec = dict(row=tid, row_file=battle_file(tid),
                                    schedule=[battle_file(c) for c in job["schedule"]],
                                    key2col={info[c][0]: c for c in cols}, base2col={info[c][1]: c for c in cols})
                    try:
                        raw = asyncio.run(asyncio.wait_for(worker.run_job(policy, job_spec, 1, port, m["seed"]), timeout=600))
                    except TimeoutError as exc:
                        raise TimeoutError(f'panel {label}, team {tid}: simulator exceeded 600 seconds; see {args.output / "showdown.log"}') from exc
                    unattributed += raw["unattributed"]
                    results += [(tid, int(c), *counts) for c, counts in raw["by_col"].items()]
                n = sum(w+l+t for _, _, w,l,t in results)
                if unattributed or n != 50: raise RuntimeError(f"incomplete panel: {n}, unattributed {unattributed}")
                result = dict(team_id=tid, wins=sum(r[2] for r in results), battles=n,
                              results=results, cache_key=key)
                # The aggregate matrix excludes diagonal cells. A completion may
                # equal another pool team; retain its mirror outcomes in the exact
                # panel without changing the shared opponent schedule.
                off_diagonal = [r for r in results if r[0] != r[1]]
                if off_diagonal:
                    db.record(con, pid, off_diagonal, seed=m["seed"], note="llm-completion:" + args.output.name)
                with con:
                    con.execute("INSERT INTO llm_completion_panel VALUES (?,?,?)",
                                (key, json.dumps(config), json.dumps(result)))
            save(args.output / "labels" / (label + ".json"), result)
            if on_result is not None:
                on_result(label, result)
            print(label, f"{result['wins']}/50", flush=True)
    finally:
        validator.close()
        server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill(); server.wait()
        log.close()
        con.close()
        sandbox.cleanup()


FAST_PANELS = 12  # panels per evaluator call: 600 battles keep six workers x 24 battles busy
_fast_runner = None


def fast_runner(folder):
    """One persistent evaluator per process, reused by every later fast_battle call."""
    global _fast_runner
    if _fast_runner is None or _fast_runner.closed:
        import atexit
        _, db = showdown_validator.runtime()
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
    from collections import Counter, defaultdict
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


def fast_battle(args, candidates_override=None, on_result=None):
    """The same exact 50-battle panels on the persistent six-worker evaluator.

    A separate battle identity: the config (and so every cache key) names the
    evaluator's fingerprint instead of concurrency=1 and the pair_shard files,
    so a fast panel never answers a legacy lookup and publish() refuses to mix
    the two in one study folder. Panels are battled several per evaluator call
    and published one by one, each only once all 50 battles are attributed."""
    from collections import Counter
    h, db = showdown_validator.runtime()
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
            # The legacy battle export: canonical paste, zero-point Serious written as Hardy.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "complete", "battle", "report"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/llm-baseline-instant-pilot")
    parser.add_argument("--teams", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--battle-backend', choices=['legacy', 'fast'],
                        help='battle: fast (default) = six persistent workers, a separate battle identity that never '
                             'reuses legacy panels; legacy = pair_shard on one private server. Labels from the '
                             'other engine are archived and re-battled')
    parser.add_argument('--fast-panels', type=int, default=FAST_PANELS, help='battle: panels per fast evaluator call')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.teams < 1: parser.error("--teams must be positive")
    if not 0 <= args.shard < args.shards: parser.error('invalid shard')
    args.output.mkdir(parents=True, exist_ok=True)
    lock_name = '.' + args.command + (f'-{args.shard}-of-{args.shards}' if args.command=='complete' else '') + '.lock'
    with (args.output / lock_name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.command](args)


if __name__ == "__main__":
    main()
