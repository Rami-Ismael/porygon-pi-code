"""Fresh-battle, fixed-budget DiBO versus historical CEM comparison.

Both arms receive the same frozen observations. Search labels use only this
arm's fresh per-opponent outcomes, never the shared matrix's aggregate/cache.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dibo_v4 as DB
import dibo_core_v4 as DC
import cem_v3 as CV
import foundation as FD
import lossdown as L
import matchup_db as M
import heldout as H
import pool
import top50
from corpus import norm
from hps_generate import Validator


def config(smoke=False):
    result = dict(rounds=10, propose=12800, battle=128, buffer=1000, initial_limit=5000,
                  ensemble=5, width=32, hidden=256, prior_epochs=50, predictor_epochs=50,
                  rtb_steps=50, batch=64, rtb_batch=256, rtb_microbatch=128, sample_batch=128, local_steps=10,
                  local_neighbors=4, attempt_factor=12, per_cell=1, seed=5050, threads=4,
                  final_n=16, final_battles=192, lr=1e-4, predictor_lr=1e-3, rtb_lr=1e-4,
                  beta=30., gamma=1., cem_propose=512, cem_prior_epochs=40)
    if smoke:
        result.update(rounds=1, propose=4, battle=2, initial_limit=64, prior_epochs=1,
                      predictor_epochs=2, rtb_steps=2, batch=8, rtb_batch=2, sample_batch=4,
                      local_steps=1, local_neighbors=2, final_n=2, final_battles=8)
        result.update(rtb_microbatch=2, cem_propose=4, cem_prior_epochs=1)
    return result


def digest_file(path):
    return DB.Runtime.file_hash(path)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    DB.atomic_json(path, value)


def protocol(directory, runtime):
    path = directory / "protocol.json"
    signature = runtime.signature
    expected = dict(config=runtime.cfg, runtime_signature=signature,
                    comparison_source_file=str(Path(__file__).resolve()),
                    comparison_source=digest_file(Path(__file__)), cem_source=digest_file(Path(CV.__file__)),
                    loaded_source_paths={"cem": CV.__file__, "diffusion": DB.__file__, "core": DC.__file__, "evaluator": M.__file__},
                    ridge_source=digest_file(FD.RIDGE), search_columns=signature["meta"],
                    arms=["cem", "dibo"], stopping="fixed budget; diversity guardrail recorded, not used to shorten either arm",
                    primary="Fresh heldout score of highest-search-score finalist, selected before confirmation",
                    secondary="Fresh top50 and heldout mean across the 16 locked finalists; composition count, effective compositions, largest core share and within-roster build variation on equal-sized outputs",
                    initial_reference=str(HERE.parent/'results/dibo_vs_cem_full_20261012/protocol.json'),
                    initial_reference_sha256=digest_file(HERE.parent/'results/dibo_vs_cem_full_20261012/protocol.json'),
                    proposal_counts=dict(cem=runtime.cfg['cem_propose'], dibo=runtime.cfg['propose']),
                    method_revision='v4: ownership-preserving proxy, standardized weighting/rewards, prioritized growing replay, scaled RTB, 100x proposals and 10 exact-acceptance discrete local steps',
                    remaining_adaptations=['categorical fixed-order denoiser; not Gaussian diffusion', 'beta=30 on standardized rewards fixed before rerun; not transplanted continuous-domain beta', '1000-row buffer', 'local candidates proposed by conditional/UCB score then accepted by exact full density', 'Stat Points remain rendered rather than optimized', 'same previously inspected heldout opponent collection with fresh seeds; not a new untouched population'],
                    physical_battles_per_arm=dict(search=runtime.cfg["rounds"] * runtime.cfg["battle"] * len(signature["meta"]),
                                                 confirmation=runtime.cfg["final_n"] * runtime.cfg["final_battles"],
                                                 heldout=runtime.cfg["final_n"] * runtime.cfg["final_battles"]),
                    repeats=1,
                    changes_from_historical_cem=["common frozen initial labels (including all meta anchors)",
                        "ridge refit on those labels instead of an extra private baseline label pool",
                        "same unseen/distinct candidate rule as DiBO", "no early stopping before the matched budget",
                        "item-less slots parsed separately", "fresh labels instead of cached/pooled scores"])
    expected = json.loads(json.dumps(expected))
    if path.exists():
        existing = read(path)
        if {k: existing[k] for k in expected} != expected:
            raise ValueError("comparison protocol or implementation changed; use a new output directory")
        return existing
    # Reuse ONLY the previous run's frozen starting pool, never its new labels.
    source = read(expected['initial_reference'])
    initial = source['initial'][:runtime.cfg['initial_limit']]
    expected.update(initial=initial, initial_hash=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest(),
                    created_at=time.time(), shared_setup=dict(planned_battles=0,attributed_battles=0,reused_frozen_initial=True),
                    historical_cem=source['historical_cem'])
    archive=directory/'source';archive.mkdir(exist_ok=True)
    for source_path in [Path(__file__),Path(DB.__file__),Path(DC.__file__)]:
        (archive/source_path.name).write_bytes(source_path.read_bytes())
    save(directory/'shared_setup.json',expected['shared_setup'])
    save(path,expected)
    return expected


def initial_out(proto):
    return dict(initial_hash=proto["initial_hash"], rounds={}, created_at=time.time())


def rows_for(out, proto, generation, buffer=None):
    rows = list(proto["initial"])
    for g, record in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])):
        if int(g) < generation:
            rows.extend(record.get("evaluated", []))
    return sorted(rows, key=lambda r: -r["y"])[:buffer] if buffer else rows


def cem_propose(runtime, rows, generation, directory, seen):
    cfg = dict(runtime.cfg, propose=runtime.cfg['cem_propose'], prior_epochs=runtime.cfg['cem_prior_epochs'],sample_batch=32)
    CV.TAG = "cem_matched_" + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:10]
    CV.SEED = cfg["seed"]
    CV.FT_EPOCHS = cfg["prior_epochs"]
    elites = None
    if generation:
        qualified = [r for r in rows if r["y"] >= CV.CUT]
        if len(qualified) < CV.ELITE_MIN:
            qualified = sorted(rows, key=lambda r: -r["y"])[:CV.ELITE_MIN]
        elites = [(np.asarray(r["x"], np.int64), r["y"], r["src"]) for r in qualified]
    runtime.generator.f = CV.members(runtime.V, runtime.generator.tables, elites, generation)
    base_ridge = FD.RIDGE
    ridge = CV.Ridge()
    features = np.stack([ridge.vec(DB.parse_slots(r["paste"])) for r in rows])
    y = np.asarray([r["y"] for r in rows])
    weights = np.asarray([1. if r["src"] == "initial" else CV.LOOP_W for r in rows])
    w = np.linalg.solve(features.T @ (features * weights[:, None]) + np.diag(ridge.lam), features.T @ (weights * y))
    z = np.load(base_ridge)
    ridge_path = directory / f"cem_ridge_round{generation}.npz"
    np.savez(ridge_path, w=w, keys=z["keys"], pairs=z["pairs"], lam=z["lam"], rho=z["rho"], n=len(rows))
    FD.RIDGE = ridge_path
    try:
        surrogate = FD.Surrogate(runtime.V)
    finally:
        FD.RIDGE = base_ridge
    del features
    seed = cfg["seed"] + generation * 100
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keep, keys, tried = [], set(seen), 0
    raw_accepted = []
    limit = cfg["propose"] * cfg["attempt_factor"]
    while len(keep) < cfg["propose"] and tried < limit:
        n = min(cfg["sample_batch"], limit - tried)
        for row in runtime.generator.sample(n, .5, 2., 1., 1., surrogate, surrogate.lam).cpu().numpy():
            tried += 1
            paste, slots = runtime.generator.to_paste(row, rng, repair=True)
            if runtime.validator(paste) is not None or H.is_heldout_text(paste):
                continue
            key = H.grid_key_text(paste)
            if len(raw_accepted)<512:
                raw_accepted.append(dict(paste=paste,key=key))
            if key in keys:
                continue
            keys.add(key)
            slots = DB.parse_slots(paste)
            x = runtime.V.encode(slots).tolist()
            keep.append(dict(paste=paste, key=key, x=x, mu=float(surrogate.score(slots))))
            if len(keep) == cfg["propose"]:
                break
    if len(keep) < cfg["battle"]:
        raise RuntimeError("CEM cannot fill the shared distinct/unseen proposal budget")
    selected = sorted(keep, key=lambda r: -r["mu"])[:cfg["battle"]]
    return dict(proposals=keep, selected=selected, tried=tried, raw_accepted_sample=raw_accepted,
                elites=len(elites) if elites is not None else 0, observation_count=len(rows))


def fresh_evaluate(runtime, selected, directory, arm, generation, proto):
    """Never cache a fresh comparison score. Save each completed microbatch once."""
    columns = M.set_members(runtime.con, "top50")
    weights = dict(proto["search_columns"])
    meta_keys = {r["key"] for r in proto["initial"] if r["team_id"] in columns}
    if any(r["key"] in meta_keys or H.is_heldout_text(r["paste"]) for r in selected):
        raise ValueError("a scored candidate overlaps the meta anchors or heldout")
    results = []
    for start in range(0, len(selected), 16):
        chunk = selected[start:start + 16]
        receipt = directory / f"{arm}_round{generation}_battles_{start}.json"
        if receipt.with_suffix(".incomplete.json").exists():
            raise RuntimeError("an incomplete fresh batch is recorded; resolve it before spending more battles")
        if receipt.exists():
            old = read(receipt)
            if old["keys"] != [r["key"] for r in chunk]:
                raise ValueError("fresh battle receipt belongs to other candidates")
            results.extend(old["evaluated"])
            continue
        ids = [M.add_team(runtime.con, r["paste"], f"matched:{directory.name}:{arm}:g{generation}", runtime.validator)[0] for r in chunk]
        if set(ids) & set(columns) or len(set(ids)) != len(ids):
            raise ValueError("comparison needs distinct non-meta candidates")
        need = [(tid, col, 1) for tid in ids for col in columns]
        jobs = M.jobs_for(runtime.con, need)
        raw, unattributed, seconds = M.run_jobs(runtime.con, jobs, seed=runtime.cfg["seed"] + generation)
        cells = {(row, col): (win, loss, tie) for row, col, win, loss, tie in raw}
        if unattributed or len(cells) != len(need) or any(sum(cells.get((row, col), ())) != 1 for row, col, _ in need):
            # Do not silently rerun partial batches and call the physical budgets matched.
            save(receipt.with_suffix(".incomplete.json"), dict(raw=raw, unattributed=unattributed, seconds=seconds, expected=len(need)))
            raise RuntimeError("incomplete fresh battle batch; inspect receipt before resuming")
        evaluated = []
        for tid, candidate in zip(ids, chunk):
            y = sum(weights[col] * cells[tid, col][0] for col in columns) / sum(weights.values())
            evaluated.append(dict(paste=candidate["paste"], x=candidate["x"], key=candidate["key"], y=y,
                                  team_id=tid, src=f"round{generation}", battles=len(columns)))
        value = dict(keys=[r["key"] for r in chunk], evaluated=evaluated, raw=raw,
                     physical_battles=len(need), seconds=seconds, unattributed=0)
        save(receipt, value)
        # Add real outcomes to the project matrix for future work. Scores above
        # come from this raw batch, so pooled cells never enter either arm here.
        M.record(runtime.con, 1, raw, seconds=seconds, seed=runtime.cfg["seed"] + generation,
                 note=f"matched:{directory.name}:{arm}:g{generation}:{start}")
        results.extend(evaluated)
        print(f"{arm} round {generation}: fresh battles {start + len(chunk)}/{len(selected)} candidates", flush=True)
    return results


def search_arm(directory, arm, generation, runtime, proto):
    path = directory / f"{arm}.json"
    out = read(path) if path.exists() else initial_out(proto)
    if out["initial_hash"] != proto["initial_hash"]:
        raise ValueError("initial observations changed")
    result = out["rounds"].setdefault(str(generation), {})
    if "evaluated" in result:
        return
    before = time.time()
    if "selected" not in result:
        rows = rows_for(out, proto, generation, runtime.cfg["buffer"] if arm == "dibo" else None)
        seen = {r["key"] for r in proto["initial"]}
        seen.update(r["key"] for record in out["rounds"].values() for r in record.get("evaluated", []))
        print(f"{arm} round {generation}: training on {len(rows)} frozen arm-specific observations", flush=True)
        if arm == "dibo":
            prior, posterior, ensemble, training = runtime.models(rows, generation)
            result.update(runtime.propose(prior, posterior, ensemble, generation, seen))
            result.update(training=training, observation_count=len(rows))
            del prior, posterior, ensemble
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        else:
            result.update(cem_propose(runtime, rows, generation, directory, seen))
        result["model_and_proposal_seconds"] = time.time() - before
        save(path, out)
        runtime.render_cache.clear()
    result["evaluated"] = fresh_evaluate(runtime, result["selected"], directory, arm, generation, proto)
    y = np.asarray([r["y"] for r in result["evaluated"]])
    result.update(mean=float(y.mean()), se=float(y.std(ddof=1) / math.sqrt(len(y))), best=float(y.max()),
                  physical_battles=sum(r["battles"] for r in result["evaluated"]),
                  species_sets=len({tuple(sorted(norm(s["species"]) for s in DB.parse_slots(r["paste"]))) for r in result["proposals"]}),
                  finished_at=time.time())
    # Record the original guardrail without shortening one arm's budget.
    if generation > 0:
        first = out["rounds"]["0"]["species_sets"]
        result["historical_diversity_stop"] = result["species_sets"] < first / 2
    save(path, out)
    print(f"{arm} round {generation} DONE: mean={result['mean']:.3f} best={result['best']:.3f}; {result['physical_battles']} fresh battles", flush=True)


def final_arm(directory, arm, proto):
    path = directory / f"{arm}.json"
    out = read(path)
    cfg = proto["config"]
    if len(out["rounds"]) != cfg["rounds"] or any("evaluated" not in r for r in out["rounds"].values()):
        raise ValueError("finish the entire search budget before final selection")
    if "final_selected" not in out:
        records = [dict(r, log_target=r["y"]) for record in out["rounds"].values() for r in record["evaluated"]]
        out["final_selected"] = DC.unique_top(records, cfg["final_n"])
        save(path, out)
    if len(out["final_selected"]) != cfg["final_n"]:
        raise ValueError("not enough finalists for the predeclared final budget")
    paths = []
    for i, candidate in enumerate(out["final_selected"]):
        target = directory / f"{arm}_final_{i:02d}.txt"
        # Use exactly the canonical representation used by search battles in
        # matchup_db.paste_file, including slot/move ordering and neutral nature.
        target.write_text(M.canonical_paste(candidate["paste"]))
        paths.append(str(target))
    for name in ("confirmation", "heldout"):
        if name in out:
            continue
        if (directory / f"{arm}_{name}_incomplete.json").exists():
            raise RuntimeError("an incomplete final stage is recorded; resolve it before spending more battles")
        opponents = top50.files() if name == "confirmation" else H.files()
        rng = np.random.default_rng(cfg["seed"] + (100000 if name == "confirmation" else 200000))
        schedule = [opponents[i] for i in rng.integers(len(opponents), size=cfg["final_battles"])]
        results = H.evaluate(paths, battles=cfg["final_battles"], conc=50,
                             opp_schedule=schedule, seed=cfg["seed"] + 90000) if name == "heldout" else \
                  pool.score([(f, opponents) for f in paths], battles=cfg["final_battles"], conc=50,
                             opp_schedule=schedule, seed=cfg["seed"] + 90000)
        if any(results.get(f, {}).get("battles") != cfg["final_battles"] for f in paths):
            save(directory / f"{arm}_{name}_incomplete.json", results)
            raise RuntimeError("final stage incomplete; do not silently add another physical budget")
        out[name] = [results[f] for f in paths]
        out[name + "_opponent_schedule"] = schedule
        save(path, out)
        print(f"{arm} {name} DONE: {sum(r['battles'] for r in out[name])} fresh battles", flush=True)


def report(directory):
    proto = read(directory / "protocol.json")
    arms = {arm: read(directory / f"{arm}.json") for arm in proto["arms"] if (directory / f"{arm}.json").exists()}
    rows = []
    for arm, out in arms.items():
        evaluated = [r for record in out["rounds"].values() for r in record.get("evaluated", [])]
        rows.append(dict(method=arm, rounds=sum("evaluated" in r for r in out["rounds"].values()),
                         search_battles=sum(r["battles"] for r in evaluated),
                         search_best=max((r["y"] for r in evaluated), default=None),
                         confirmation_battles=sum(r["battles"] for r in out.get("confirmation", [])),
                         heldout_battles=sum(r["battles"] for r in out.get("heldout", []))))
    complete = len(arms) == 2 and all(all(name in out for name in ("confirmation", "heldout")) for out in arms.values())
    summary = dict(complete=complete, budgets=rows, expected=proto["physical_battles_per_arm"])
    if complete:
        for row in rows:
            for key, expected_key in (("search_battles", "search"), ("confirmation_battles", "confirmation"), ("heldout_battles", "heldout")):
                if row[key] != proto["physical_battles_per_arm"][expected_key]:
                    raise ValueError("physical budgets did not match the preregistration")
        for stage in ("confirmation", "heldout"):
            a, b = arms["dibo"], arms["cem"]
            if a[stage + "_opponent_schedule"] != b[stage + "_opponent_schedule"]:
                raise ValueError("final opponent schedules differ")
            values = {arm: np.asarray([r["win_rate"] for r in out[stage]]) for arm, out in arms.items()}
            rng = np.random.default_rng(381)
            boot = np.array([rng.choice(values["dibo"], len(values["dibo"]), replace=True).mean() -
                             rng.choice(values["cem"], len(values["cem"]), replace=True).mean() for _ in range(10000)])
            summary[stage] = dict(cem_first=float(values["cem"][0]), dibo_first=float(values["dibo"][0]),
                                 cem_mean=float(values["cem"].mean()), dibo_mean=float(values["dibo"].mean()),
                                 mean_difference=float(values["dibo"].mean() - values["cem"].mean()),
                                 team_cluster_bootstrap_95=[float(v) for v in np.quantile(boot, [.025, .975])])
        score = summary["heldout"]
        summary["descriptive_primary_winner"] = "DiBO" if score["dibo_first"] > score["cem_first"] else "CEM" if score["cem_first"] > score["dibo_first"] else "tie"
    save(directory / "summary.json", summary)
    lines = ["# DiBO versus CEM — matched fresh-battle experiment", "", f"Complete: **{complete}**", "",
             "| Method | Finished rounds | Fresh search battles | Fresh confirmation | Fresh heldout | Best search score |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['rounds']} | {row['search_battles']} | {row['confirmation_battles']} | {row['heldout_battles']} | {row['search_best']} |")
    if complete:
        for stage in ("confirmation", "heldout"):
            s = summary[stage]
            lines += ["", f"## {stage}", "", f"First search-selected finalist: DiBO {s['dibo_first']:.3f}; CEM {s['cem_first']:.3f}.",
                      f"Mean of locked finalists: DiBO {s['dibo_mean']:.3f}; CEM {s['cem_mean']:.3f}.",
                      f"DiBO − CEM mean difference: {s['mean_difference']:+.3f}; team-cluster bootstrap 95% interval {s['team_cluster_bootstrap_95']}."]
        lines += ["", "This is one seed and one categorical adaptation. Team-cluster intervals describe the selected candidate sets, not variability over independent training runs. The primary finalist was chosen by search score before either confirmation stage; heldout results never selected a winner within an arm."]
    lines += ["", "The historical CEM diversity stop is logged but disabled for this fixed-budget repeat. Both methods receive the same starting labels, distinct/unseen rule, foundation, battle policy and opponent schedules. The common-label ridge refit differs from the historical private 44,887-label initialization. Historical CEM's three-round numbers are context, not a randomized matched replicate.", "", "Protocol and complete raw receipts are beside this report."]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "report":
            report(directory)
            return
        cfg = config(args.smoke)
        runtime = DB.Runtime(cfg, directory / "dibo.json")
        try:
            proto = protocol(directory, runtime)
            for generation in range(cfg["rounds"]):
                for arm in ("cem", "dibo") if generation % 2 == 0 else ("dibo", "cem"):
                    search_arm(directory, arm, generation, runtime, proto)
                report(directory)
            # Lock finalists in both arms before accessing any heldout results.
            for arm in ("cem", "dibo"):
                path = directory / f"{arm}.json"
                out = read(path)
                if "final_selected" not in out:
                    values = [dict(r, log_target=r["y"]) for result in out["rounds"].values() for r in result["evaluated"]]
                    out["final_selected"] = DC.unique_top(values, cfg["final_n"])
                    save(path, out)
            for arm in ("cem", "dibo"):
                final_arm(directory, arm, proto)
            report(directory)
        finally:
            runtime.close()


if __name__ == "__main__":
    main()
