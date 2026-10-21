"""The masked-completion study: build its frozen manifest, and report its battle results.

prepare() samples legal starting teams, masks them per GRID task and size, and
gives every trial a random-fill control. report() pairs each language-model
completion with its starting team and its random fill. See docs/llm-baseline.md."""
import copy
import hashlib
import json
from pathlib import Path
import random
import sqlite3

from llm_completion import MODEL, load_manifest, prompt
from results import save
import showdown_validator
from team_masks import GRID, masked, paths_for, set_path, validate_shape


def random_fill(team, paths, tables, rng, h, validator):
    from corpus import norm, NATURES
    for _ in range(1000):
        result = copy.deepcopy(team)
        # Whole slots are sampled against all fixed slots, including later slots.
        whole = {p[0] for p in paths if len(p) == 1}
        bases = {tables["base_of"][norm(s["species"])] for i, s in enumerate(team) if i not in whole}
        items = {norm(s["item"]) for i, s in enumerate(team) if i not in whole}
        for path in paths:
            s = path[0]
            if len(path) == 1:
                value = h.sample_slot(tables, rng, bases, items)
                if value is None:
                    break
            else:
                sp = norm(team[s]["species"])
                field = path[1]
                if field == "evs": value = h.sample_spread(rng)
                elif field == "nature": value = str(rng.choice(NATURES))
                elif field == "moves": value = str(rng.choice(h.TRUE[sp]))
                elif field == "ability": value = str(rng.choice(tables["abil"][sp]))
                else: value = str(rng.choice(tables["items"]))
            set_path(result, path, value)
        else:
            validate_shape(result)
            text = showdown_validator.paste(result, h)
            if validator(text) is None:
                assert masked(result, paths) == masked(team, paths)
                return result
    raise RuntimeError("random fill failed after 1000 attempts; no fallback to original")


def prepare(args):
    import numpy as np
    import hps_generate as h, matchup_db as db
    destination = args.output / "manifest.json"
    if destination.exists():
        raise ValueError("manifest already exists; use another directory or resume")
    tables = h.build_tables()
    con = sqlite3.connect(f"file:{db.DEFAULT_DB}?mode=ro", uri=True)
    opponents = [dict(team_id=r[0], paste=r[1]) for r in con.execute(
        "SELECT t.team_id,t.paste FROM team t JOIN team_set s USING(team_id) "
        "WHERE s.set_name='top50' AND t.format_id=? AND t.legal=1 ORDER BY t.team_id", (showdown_validator.FORMAT,))]
    con.close()
    if not opponents:
        raise ValueError("no legal meta opponents")
    rng = np.random.default_rng(args.seed)
    validator = h.Validator()
    starts, trials, seen = [], [], {db.canon_hash(db.canonical_paste(o["paste"])) for o in opponents}
    try:
        for o in opponents:
            error = validator(o["paste"])
            if error: raise ValueError(f"invalid opponent {o['team_id']}: {error}")
        for _ in range(10000):
            team = h.one_team(tables, rng)
            if team is None: continue
            text = showdown_validator.paste(team, h)
            key = db.canon_hash(db.canonical_paste(text))
            if key in seen or validator(text) is not None: continue
            validate_shape(team)
            seen.add(key)
            starts.append(team)
            if len(starts) == args.teams: break
        if len(starts) != args.teams: raise ValueError("could not sample starting teams")
        for i, team in enumerate(starts):
            for task, sizes in GRID.items():
                # Nested masks make k comparisons interpretable within a task.
                order = paths_for(task)
                random.Random(f"{args.seed}:{i}:{task}").shuffle(order)
                for k in sizes:
                    paths = order[:k]
                    control = random_fill(team, paths, tables, rng, h, validator)
                    trials.append(dict(id=f"s{i}-{task}-{k}", start=i, task=task, k=k,
                                       paths=paths, random_team=control))
    finally:
        validator.close()
    manifest = dict(schema=1, model=MODEL, format=showdown_validator.FORMAT, seed=args.seed, grid=GRID,
                    starts=starts, trials=trials, opponents=opponents, battles=50,
                    temperature=0.7, max_tokens=4096, reasoning={"enabled": False}, replicates=1,
                    sampler="hierarchical product, legal tables; items limited to corpus vocabulary; not uniform over teams",
                    seed_limit="shared policy RNG seeds; Showdown RNG is not seeded by existing runner",
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(destination, manifest)
    save(args.output / "example-request.json", {"model": MODEL, "messages": prompt(manifest, trials[0])})
    print(json.dumps(dict(prepared=str(destination), teams=len(starts), trials=len(trials),
                          opponents=len(opponents), planned_battles=(len(starts)+2*len(trials))*50)))


def report(args):
    m = load_manifest(args)
    records = [json.loads(p.read_text()) for p in (args.output / "completions").glob("*.json")
               if not p.name.endswith(".http-error.json")]
    status = dict(model=MODEL, format=showdown_validator.FORMAT, starting_teams=len(m["starts"]),
                  planned_calls=len(m["trials"]), completed_calls=len(records),
                  legal_completions=sum(r["valid"] for r in records),
                  actual_opponents=len(m["opponents"]),
                  reported_cost_usd=sum(r["response"].get("usage", {}).get("cost", 0) or 0 for r in records),
                  interpretation="pilot only; no individual team claim without separate 192-battle confirmation")
    labels = {p.stem: json.loads(p.read_text()) for p in (args.output / "labels").glob("*.json")}
    comparisons = []
    for trial in m["trials"]:
        keys = [f"start-{trial['start']}", "random-" + trial["id"], "llm-" + trial["id"]]
        if not all(k in labels for k in keys): continue
        start, control, llm = [labels[k]["wins"] / labels[k]["battles"] for k in keys]
        comparisons.append(dict(trial=trial["id"], start=trial["start"], task=trial["task"], k=trial["k"],
                                original=start, random=control, llm=llm,
                                delta_original=llm-start, delta_random=llm-control))
    summaries = []
    for task, sizes in m['grid'].items():
        for k in sizes:
            rows = [r for r in comparisons if r["task"] == task and r["k"] == k]
            if not rows:
                summaries.append(dict(task=task, k=k, n=0))
                continue
            summaries.append(dict(task=task, k=k, n=len(rows), **{
                field: sum(r[field] for r in rows)/len(rows)
                for field in ("original", "random", "llm", "delta_original", "delta_random")}))
    status.update(scored_panels=len(labels), paired_comparisons=len(comparisons), summaries=summaries,
                  comparisons=comparisons, complete=(len(records) == len(m["trials"])
                  and len(comparisons) == sum(r["valid"] for r in records)
                  and all(f"start-{i}" in labels for i in range(len(m["starts"])))
                  and all("random-" + t["id"] in labels for t in m["trials"])))
    unique_panels = {r["cache_key"]: r for r in labels.values()}
    status["unique_cached_panels"] = len(unique_panels)
    status["unique_battles"] = sum(r["battles"] for r in unique_panels.values())
    save(args.output / "status.json", status)
    lines = ["# Ling masked-completion pilot", "",
             f"Model: `{MODEL}`. Format: Champions M-B. Behaviour-cloning policy on both sides.", "",
             f"{len(m['starts'])} random legal starting teams; {len(m['trials'])} planned completions; "
             f"{len(records)} returned; {status['legal_completions']} legal; {len(comparisons)} scored comparisons.",
             f"{status['unique_battles']} battles in {len(unique_panels)} distinct cached panels "
             f"({len(labels)} candidate labels; identical candidates reuse their panel).",
             f"Reported API cost: ${status['reported_cost_usd']:.6f} (excludes separate setup smoke).", "",
             "Each candidate plays a frozen 50-battle panel, with repeated opponents as needed. "
             "The schedule matches across methods for each starter. Policy RNG seeds match; Showdown RNG is not seeded. "
             f"This experiment has {len(m['starts'])} starting teams and does not establish held-out generalization. "
             "The supplied meta pool is visible in the prompt, so this measures performance against that pool, "
             "not generalization to unseen threats. Invalid completions are failures, never repaired or silently replaced. "
             "Win-rate comparisons below are conditional on legal completions with complete battle panels.", "",
             "| Task | k | Legal/scored pairs | Original | Random fill | Ling | Δ original (pp) | Δ random (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if not row["n"]:
            lines.append(f"| {row['task']} | {row['k']} | 0 | — | — | — | — | — |")
            continue
        lines.append(f"| {row['task']} | {row['k']} | {row['n']} | {row['original']:.1%} | "
                     f"{row['random']:.1%} | {row['llm']:.1%} | {100*row['delta_original']:+.1f} | "
                     f"{100*row['delta_random']:+.1f} |")
    if comparisons:
        lines += ["", "Aggregate paired differences (each starting team weighted equally):"]
        import numpy as np
        for field in ("delta_original", "delta_random"):
            groups = sorted({r["start"] for r in comparisons})
            values = np.array([np.mean([r[field] for r in comparisons if r["start"] == s]) for s in groups])
            draws = np.random.default_rng(m["seed"]).choice(values, (4000, len(values))).mean(axis=1)
            lo, hi = np.quantile(draws, [.025, .975])
            lines += [f"- {field}: {100*values.mean():+.1f} percentage points; exploratory 95% "
                      f"starting-team bootstrap interval [{100*lo:+.1f}, {100*hi:+.1f}]."]
        lines += ["", f"There are {len(m['starts'])} starter clusters; the intervals are exploratory. "
                  "No single candidate is presented as a validated result; that requires a fresh 192-battle panel."]
    errors = [r.get("error", "unknown") for r in records if not r["valid"]]
    if errors:
        from collections import Counter
        lines += ["", "Validation failures:"] + [f"- {n} × {error}" for error, n in Counter(errors).items()]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in status.items() if k not in ("summaries", "comparisons")}))
