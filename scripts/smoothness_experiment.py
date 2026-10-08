"""Reproducible legal-move sensitivity experiment; see docs/smoothness-experiment.md.

Preparation never battles. Battle results are checkpointed per team and replicate.
The main estimator is E[delta**2], corrected using independent replicate panels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmb"
DEFAULT_RUNTIME = Path("/tmp/vgc-pilot")
SCHEMA = 1
GROUP_LABELS = {"random": "Random legal", "tournament": "Frozen top placements"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def seed_for(seed, *parts):
    return int.from_bytes(hashlib.sha256(json.dumps([seed, *parts]).encode()).digest()[:4], "big")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def normalized_paste(text):
    """Identity preserves roster/move order, which can affect the fixed policy."""
    return "\n".join(line.strip().lower() for line in text.splitlines() if line.strip())


def paste_key(text):
    return hashlib.sha256(normalized_paste(text).encode()).hexdigest()


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def move_slots(text):
    """Read only headers and move positions; never round-trip away IVs or other fields."""
    lines = text.splitlines(keepends=True)
    slots = []
    for line_number, line in enumerate(lines):
        clean = line.strip()
        if not clean:
            continue
        if clean.startswith("- "):
            if not slots:
                raise ValueError("move precedes team header")
            slots[-1]["moves"].append((line_number, clean[2:].strip()))
        elif ":" not in clean and not clean.endswith(" Nature"):
            header = clean.split(" @ ")[0]
            header = re.sub(r"\s+\((M|F)\)$", "", header)
            nickname = re.search(r"\(([^()]+)\)$", header)
            species = nickname.group(1) if nickname else header
            slots.append(dict(species=species, moves=[]))
    if len(slots) != 6 or any(not slot["moves"] for slot in slots):
        raise ValueError("expected six Pokemon with moves")
    return lines, slots


def assert_move_edit(original, edited):
    before, slots = move_slots(original)
    after, _ = move_slots(edited)
    if len(before) != len(after):
        raise ValueError("edit changed line count")
    changes = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    move_lines = {i for slot in slots for i, _ in slot["moves"]}
    if len(changes) != 1 or changes[0] not in move_lines:
        raise ValueError("expected exactly one replaced move line")
    if norm(before[changes[0]]) == norm(after[changes[0]]):
        raise ValueError("semantic no-op")


class Validator:
    def __init__(self, showdown, node):
        self.process = subprocess.Popen(
            [node, "validate-teams-batch.js"], cwd=showdown,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )

    def __call__(self, text):
        self.process.stdin.write(json.dumps(dict(format=FORMAT, team=text)) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Showdown validator exited unexpectedly")
        result = json.loads(line)
        return result["valid"], result["errors"]

    def close(self):
        self.process.stdin.close()
        if self.process.wait(timeout=15):
            raise RuntimeError("Showdown validator failed")


def sample_neighbors(text, learnsets, count, rng, validator):
    lines, slots = move_slots(text)
    positions = []
    for slot_index, slot in enumerate(slots):
        species = norm(slot["species"])
        if species not in learnsets:
            raise ValueError(f"species missing from validated move table: {species}")
        used = {norm(move) for _, move in slot["moves"]}
        alternatives = sorted(set(learnsets[species]) - used)
        if alternatives:
            positions.extend((slot_index, line, move, alternatives) for line, move in slot["moves"])
    if not positions:
        raise ValueError("team has no eligible move replacements")
    records, seen = [], set()
    for attempt in range(max(1000, count * 200)):
        slot_index, line, old_move, alternatives = positions[int(rng.integers(len(positions)))]
        new_move = alternatives[int(rng.integers(len(alternatives)))]
        candidate = lines.copy()
        ending = "\r\n" if lines[line].endswith("\r\n") else "\n" if lines[line].endswith("\n") else ""
        candidate[line] = f"- {new_move}{ending}"
        paste = "".join(candidate)
        key = paste_key(paste)
        if key in seen:
            continue
        valid, _ = validator(paste)
        if not valid:
            continue
        assert_move_edit(text, paste)
        seen.add(key)
        records.append(dict(paste=paste, slot=slot_index, species=slots[slot_index]["species"],
                            old_move=old_move, new_move=new_move, proposal_attempt=attempt + 1))
        if len(records) == count:
            return records
    raise RuntimeError(f"could only generate {len(records)}/{count} distinct legal neighbors")


def sample_random_records(path, count, seed):
    """Uniform sample of rows in the frozen HPS corpus, not of all legal teams."""
    rng = np.random.default_rng(seed)
    reservoir = []
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            if index < count:
                reservoir.append((index, line))
            else:
                replacement = int(rng.integers(index + 1))
                if replacement < count:
                    reservoir[replacement] = (index, line)
    if len(reservoir) != count:
        raise ValueError("insufficient random-team records")
    return [(index, json.loads(line)) for index, line in sorted(reservoir)]


def runtime_fingerprints(runtime, checkpoint):
    bench = runtime / "vgc-bench"
    showdown = bench / "pokemon-showdown"
    paths = [Path(checkpoint), ROOT / "src/shard.py", showdown / "validate-teams-batch.js"]
    paths += sorted((bench / "vgc_bench").rglob("*.py"))
    paths += sorted((bench / "data").rglob("*.json"))
    paths += sorted((showdown / "dist").rglob("*.js"))
    if not (showdown / "dist/sim/battle.js").exists():
        raise FileNotFoundError("built Showdown runtime not found")
    return {str(path.resolve()): digest(path) for path in paths}


def prepare(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"use a fresh output directory: {output}")
    if not 1 <= args.anchors <= 50 or args.neighbors < 1 or args.replicates < 2:
        raise ValueError("anchors must be 1..50; neighbors >=1; replicates >=2")
    if args.battles < 50 or args.battles % 50:
        raise ValueError("battles per replicate must be a positive multiple of 50")
    runtime = args.runtime.resolve()
    showdown = runtime / "vgc-bench/pokemon-showdown"
    metadata = read_json(ROOT / "results/top50_evs.json")
    if len(metadata) != 50 or len({row["id"] for row in metadata}) != 50:
        raise ValueError("expected the fifty frozen tournament entries")
    opponent_sources = []
    for row in metadata:
        paths = [ROOT / "teams/reg_mb" / f"{row['id']}.txt",
                 ROOT / "teams/reg_mb/featured" / f"{row['id']}.txt"]
        matches = [p for p in paths if p.exists()]
        if len(matches) != 1:
            raise ValueError(f"could not uniquely resolve {row['id']}")
        opponent_sources.append(matches[0])
    fingerprints = runtime_fingerprints(runtime, args.checkpoint)
    random_source = ROOT / "teams/hps_reg_mb_100k.jsonl"
    learnset_path = runtime / "learnset_true.json"
    learnsets = read_json(learnset_path)
    random_records = sample_random_records(random_source, args.anchors, args.seed)
    output.mkdir(parents=True)
    (output / "teams").mkdir()
    (output / "opponents").mkdir()
    validator = Validator(showdown, args.node)
    manifest = dict(schema=SCHEMA, format=FORMAT, seed=args.seed, anchors_per_group=args.anchors,
                    neighbors=args.neighbors, replicates=args.replicates, battles_per_replicate=args.battles,
                    random_sampling="uniform rows from frozen hierarchical-product-sampling corpus",
                    edit_sampling="uniform eligible move position then uniform alternative; reject illegal/no-op/duplicate",
                    runtime=str(runtime), checkpoint=str(Path(args.checkpoint).resolve()),
                    node=str(Path(args.node).resolve()), battle_python=sys.executable,
                    runtime_fingerprints=fingerprints, anchors=[], opponents=[], candidates=[],
                    source_fingerprints={str(p.resolve()): digest(p) for p in
                                         [random_source, learnset_path, ROOT / "results/top50_evs.json"]},
                    code_fingerprints={str(p): digest(p) for p in [Path(__file__).resolve(),
                                       ROOT / "scripts/smoothness_worker.py", ROOT / "scripts/ruggedness.py"]},
                    seed_limit="simulator RNG and concurrent action scheduling are not exactly replayable; raw outcomes are saved",
                    objective="probability of win; ties count as non-wins; fixed BC policy on both sides")
    manifest["placement_cohort_note"] = (
        "The user retained the frozen top-50 snapshot on 2026-10-01. It includes six ranked-season "
        "and four Showdown-ladder entries; this is not a strictly tournament-only cohort. "
        "The internal group key 'tournament' denotes this frozen top-placement cohort.")
    try:
        for row, path in zip(metadata, opponent_sources):
            text = path.read_text()
            valid, errors = validator(text)
            if not valid:
                raise ValueError(f"illegal frozen opponent {row['id']}: {errors}")
            destination = output / "opponents" / path.name
            destination.write_text(text)
            manifest["opponents"].append(dict(id=row["id"], file=str(destination), sha256=digest(destination),
                                                identity=paste_key(text), placement=row))
        sources = [("random", f"hps-row-{i}", row["team"], dict(row=i, key=row["key"]))
                   for i, row in random_records]
        sources += [("tournament", row["id"], path.read_text(), row)
                    for row, path in zip(metadata[:args.anchors], opponent_sources[:args.anchors])]
        for index, (group, source_id, text, provenance) in enumerate(sources):
            valid, errors = validator(text)
            if not valid:
                raise ValueError(f"illegal original {source_id}: {errors}")
            identity = paste_key(text)
            anchor_id = f"{group}-{index % args.anchors:03d}"
            anchor = dict(id=anchor_id, group=group, source_id=source_id, cluster=identity,
                          provenance=provenance, candidates=[])
            neighbors = sample_neighbors(text, learnsets, args.neighbors,
                                         np.random.default_rng(seed_for(args.seed, group, identity)), validator)
            entries = [("base", text, None), ("null", text, None)]
            entries += [(f"edit-{i:03d}", row["paste"], {k: v for k, v in row.items() if k != "paste"})
                        for i, row in enumerate(neighbors)]
            for kind, paste, edit in entries:
                candidate_id = f"{anchor_id}-{kind}"
                path = output / "teams" / f"{candidate_id}.txt"
                path.write_text(paste)
                record = dict(id=candidate_id, anchor=anchor_id, kind=kind, file=str(path),
                              sha256=digest(path), edit=edit)
                manifest["candidates"].append(record)
                anchor["candidates"].append(candidate_id)
            manifest["anchors"].append(anchor)
        for group in ("random", "tournament"):
            keys = [a["cluster"] for a in manifest["anchors"] if a["group"] == group]
            manifest.setdefault("distinct_anchor_clusters", {})[group] = len(set(keys))
        manifest["planned_battles"] = len(manifest["candidates"]) * args.replicates * args.battles
        atomic_json(output / "manifest.json", manifest)
    finally:
        validator.close()
    print(json.dumps(dict(prepared=str(output), battles=manifest["planned_battles"],
                          clusters=manifest["distinct_anchor_clusters"])), flush=True)
    return manifest


def verify_manifest(directory, runtime=False):
    manifest = read_json(directory / "manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("unsupported manifest schema")
    hashes = {row["file"]: row["sha256"] for row in manifest["candidates"] + manifest["opponents"]}
    if runtime:
        hashes.update(manifest["runtime_fingerprints"])
        hashes.update(manifest["code_fingerprints"])
    for file, expected in hashes.items():
        if digest(file) != expected:
            raise ValueError(f"frozen input changed: {file}")
    return manifest


def job_list(manifest):
    jobs = []
    for candidate in manifest["candidates"]:
        for replicate in range(manifest["replicates"]):
            job_id = f"{candidate['id']}-r{replicate:02d}"
            jobs.append(dict(id=job_id, candidate=candidate["id"], file=candidate["file"],
                             replicate=replicate, seed=seed_for(manifest["seed"], job_id),
                             battles=manifest["battles_per_replicate"]))
    return jobs


def checked_result(row, job, manifest_hash):
    if row.get("job") != job or row.get("manifest_sha256") != manifest_hash:
        raise ValueError("battle result does not match frozen job")
    if type(row.get("wins")) is not int or type(row.get("battles")) is not int:
        raise ValueError("noninteger battle counts")
    if row["battles"] != job["battles"] or not 0 <= row["wins"] <= row["battles"]:
        raise ValueError("partial or invalid battle result")
    return row


def battle(args):
    import fcntl
    directory = args.output.resolve()
    with (directory / "battle.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_manifest(directory, runtime=True)
        manifest_hash = digest(directory / "manifest.json")
        labels = directory / "labels"
        labels.mkdir(exist_ok=True)
        jobs = job_list(manifest)
        pending = []
        for job in jobs:
            path = labels / f"{job['id']}.json"
            if path.exists():
                checked_result(read_json(path), job, manifest_hash)
            else:
                pending.append(job)
        if not pending:
            print("All battle jobs already complete.", flush=True)
            return
        ports = list(dict.fromkeys(args.ports))
        for port in ports:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                pass
        run = directory / f"attempt-{time.time_ns()}"
        run.mkdir()
        workers, streams = [], []
        try:
            for index, port in enumerate(ports[:len(pending)]):
                assigned = pending[index::min(len(ports), len(pending))]
                spec = run / f"worker-{index}.json"
                atomic_json(spec, dict(directory=str(directory), jobs=assigned, port=port,
                                      concurrency=args.concurrency, timeout=args.job_timeout))
                stream = (run / f"worker-{index}.log").open("w")
                streams.append(stream)
                env = dict(os.environ, PYTHONPATH=f"{manifest['runtime']}/vgc-bench:{ROOT / 'src'}",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                workers.append(subprocess.Popen(
                    [manifest["battle_python"], str(ROOT / "scripts/smoothness_worker.py"), str(spec)],
                    cwd=f"{manifest['runtime']}/vgc-bench", env=env, stdout=stream, stderr=stream))
            last_report = 0
            while any(worker.poll() is None for worker in workers):
                if any(worker.poll() not in (None, 0) for worker in workers):
                    raise RuntimeError(f"battle worker failed; logs in {run}")
                if time.monotonic() - last_report > 20:
                    print(f"battle jobs: {len(list(labels.glob('*.json')))}/{len(jobs)}", flush=True)
                    last_report = time.monotonic()
                time.sleep(1)
            if any(worker.returncode for worker in workers):
                raise RuntimeError(f"battle worker failed; logs in {run}")
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
                    try:
                        worker.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
            for stream in streams:
                stream.close()
        for job in jobs:
            checked_result(read_json(labels / f"{job['id']}.json"), job, manifest_hash)
        print(f"Completed {manifest['planned_battles']} battles.", flush=True)


def corrected_mean_square(deltas):
    """Unbiased delta^2 under independent, identically designed replicate panels.

    Equivalent to the mean product over distinct replicate pairs. Keep negatives.
    """
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite replicate deltas")
    return float(values.mean() ** 2 - values.var(ddof=1) / len(values))


def wilson(wins, battles, z=1.959963984540054):
    probability = wins / battles
    denominator = 1 + z * z / battles
    center = (probability + z * z / (2 * battles)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / battles + z * z / (4 * battles**2)) / denominator
    return max(0., center - radius), min(1., center + radius)


def difference_interval(base_wins, child_wins, battles):
    """Newcombe/Wilson score interval; pointwise, approximate for fixed strata."""
    base, child = base_wins / battles, child_wins / battles
    base_low, base_high = wilson(base_wins, battles)
    child_low, child_high = wilson(child_wins, battles)
    delta = child - base
    return [delta - math.hypot(child - child_low, base_high - base),
            delta + math.hypot(child_high - child, base - base_low)]


def cluster_bootstrap(records, field, seed, iterations=4000):
    """Resample identities, carrying all placement entries and their weights."""
    clusters = defaultdict(list)
    for record in records:
        clusters[record["cluster"]].append(record[field])
    values = list(clusters.values())
    sums = np.array([sum(v) for v in values])
    sizes = np.array([len(v) for v in values])
    estimate = float(sums.sum() / sizes.sum())
    if len(values) < 2:
        return dict(mean=estimate, ci95=None, clusters=len(values), bootstrap=[])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    draws = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    return dict(mean=estimate, ci95=np.quantile(draws, [.025, .975]).tolist(),
                clusters=len(values), bootstrap=draws.tolist())


def benchmarks(args):
    from ruggedness import hartmann6, pest_control
    directory = args.output.resolve()
    manifest = verify_manifest(directory)
    path = directory / "benchmarks.json"
    if path.exists():
        if read_json(path)["manifest_sha256"] != digest(directory / "manifest.json"):
            raise ValueError("benchmark manifest mismatch")
        return
    seed = manifest["seed"]
    anchor_count, neighbor_count, repeats = manifest["anchors_per_group"], manifest["neighbors"], manifest["replicates"]
    output = dict(manifest_sha256=digest(directory / "manifest.json"), conditions=[])
    for domain, steps in [("pest_control", [1, 2, 3]), ("hartmann6", [.001, .01, .05])]:
        # Domain range, fixed before observing effects: Pest costs in [0,50],
        # Hartmann in [-sum(alpha),0]=[-8.4,0]. These are bounds, not fitted ranges.
        scale = 50. if domain == "pest_control" else 8.4
        for step in steps:
            records, edges = [], []
            for anchor in range(anchor_count):
                rng = np.random.default_rng(seed_for(seed, domain, anchor))
                original = rng.integers(0, 5, 25) if domain == "pest_control" else rng.uniform(.05, .95, 6)
                def evaluate(point, label):
                    return [pest_control(point, np.random.default_rng(seed_for(seed, domain, anchor, str(step), label, r)))
                            if domain == "pest_control" else hartmann6(point) for r in range(repeats)]
                baseline = np.array(evaluate(original, "base"))
                null = np.array(evaluate(original, "null")) - baseline
                squares = []
                for edit in range(neighbor_count):
                    child = original.copy()
                    if domain == "pest_control":
                        for coordinate in rng.choice(25, int(step), replace=False):
                            child[coordinate] = (child[coordinate] + rng.integers(1, 5)) % 5
                    else:
                        coordinate = int(rng.integers(6))
                        child[coordinate] += step * (-1 if rng.random() < .5 else 1)
                    changed = np.array(evaluate(child, edit))
                    deltas = changed - baseline
                    square = corrected_mean_square(deltas)
                    squares.append(square)
                    edges.append(dict(anchor=anchor, original=original.tolist(), edited=child.tolist(),
                                      baseline=baseline.tolist(), outcomes=changed.tolist(), delta=float(deltas.mean())))
                records.append(dict(cluster=str(anchor), corrected_mse=float(np.mean(squares)) / scale**2,
                                    null_mse=corrected_mean_square(null) / scale**2))
            statistics = cluster_bootstrap(records, "corrected_mse", seed_for(seed, domain, str(step)))
            statistics.pop("bootstrap")
            output["conditions"].append(dict(domain=domain, step=step, fixed_scale=scale,
                                               summary=statistics, anchors=records, edges=edges))
    atomic_json(path, output)
    print(f"Benchmarks saved: {path}", flush=True)


def analyse(args):
    directory = args.output.resolve()
    manifest = verify_manifest(directory)
    manifest_hash = digest(directory / "manifest.json")
    results = defaultdict(dict)
    for job in job_list(manifest):
        row = checked_result(read_json(directory / "labels" / f"{job['id']}.json"), job, manifest_hash)
        results[job["candidate"]][job["replicate"]] = row
    edges, anchors = [], []
    count = manifest["battles_per_replicate"]
    candidates = {row["id"]: row for row in manifest["candidates"]}
    for anchor in manifest["anchors"]:
        original = np.array([results[f"{anchor['id']}-base"][r]["wins"]
                             for r in range(manifest["replicates"])])
        local = []
        for candidate_id in anchor["candidates"]:
            candidate = candidates[candidate_id]
            if candidate["kind"] == "base":
                continue
            changed = np.array([results[candidate_id][r]["wins"] for r in range(manifest["replicates"])])
            deltas = (changed - original) / count
            edge = dict(anchor=anchor["id"], group=anchor["group"], cluster=anchor["cluster"],
                        candidate=candidate_id, kind=candidate["kind"], edit=candidate["edit"],
                        delta=float(deltas.mean()), replicate_deltas=deltas.tolist(),
                        corrected_mse=corrected_mean_square(deltas),
                        ci95=difference_interval(int(original.sum()), int(changed.sum()), count * len(deltas)))
            edges.append(edge)
            local.append(edge)
        edits = [row for row in local if row["kind"] != "null"]
        anchors.append(dict(id=anchor["id"], group=anchor["group"], cluster=anchor["cluster"],
                            original_win_rate=float(original.mean() / count),
                            corrected_mse=float(np.mean([row["corrected_mse"] for row in edits])),
                            observed_mean_absolute_delta=float(np.mean([abs(row["delta"]) for row in edits])),
                            null_mse=next(row["corrected_mse"] for row in local if row["kind"] == "null")))
    summary = {}
    bootstrap_draws = {}
    for group in ("random", "tournament"):
        group_rows = [row for row in anchors if row["group"] == group]
        statistics = cluster_bootstrap(group_rows, "corrected_mse", seed_for(manifest["seed"], group))
        bootstrap_draws[group] = statistics.pop("bootstrap")
        null = cluster_bootstrap(group_rows, "null_mse", seed_for(manifest["seed"], group, "null"))
        null.pop("bootstrap")
        summary[group] = dict(**statistics, null_control=null,
                              mean_original_win_rate=float(np.mean([row["original_win_rate"] for row in group_rows])))
    difference = summary["tournament"]["mean"] - summary["random"]["mean"]
    draws = np.array(bootstrap_draws["tournament"]) - np.array(bootstrap_draws["random"])
    summary["tournament_minus_random"] = dict(mean=difference,
        ci95=np.quantile(draws, [.025, .975]).tolist() if len(draws) else None)
    report = dict(manifest_sha256=manifest_hash, complete=True, battles=manifest["planned_battles"],
                  units="win-probability squared (multiply by 10000 for percentage-points squared)",
                  summary=summary, anchors=anchors, edges=edges,
                  interpretation="local move sensitivity under the frozen evaluator; not proof of mathematical nonsmoothness",
                  uncertainty="95% percentile bootstrap over original identities, retaining duplicate placement weights; "
                  "edit intervals are pointwise approximate Newcombe score intervals, not multiplicity-adjusted")
    atomic_json(directory / "analysis.json", report)
    with (directory / "edits.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "anchor", "candidate", "kind", "delta_percentage_points", "ci95_low", "ci95_high", "corrected_mse_pp2", "edit"])
        for row in edges:
            writer.writerow([row["group"], row["anchor"], row["candidate"], row["kind"], row["delta"] * 100,
                             row["ci95"][0] * 100, row["ci95"][1] * 100, row["corrected_mse"] * 10000,
                             json.dumps(row["edit"])])
    plot(directory, manifest, report)
    write_report(directory, manifest, report)
    print(json.dumps(summary, indent=2), flush=True)


def plot(directory, manifest, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"random": "#007c91", "tournament": "#c06425"}
    for position, group in enumerate(colors):
        for kind, style in [("edit", "-"), ("null", "--")]:
            values = sorted(row["delta"] * 100 for row in report["edges"]
                            if row["group"] == group and (row["kind"] == "null") == (kind == "null"))
            axes[0, 0].step(values, np.arange(1, len(values) + 1) / len(values), where="post",
                            color=colors[group], linestyle=style, label=f"{GROUP_LABELS[group]}: {kind}")
        points = [row["corrected_mse"] * 10000 for row in report["anchors"] if row["group"] == group]
        jitter = np.random.default_rng(0).uniform(-.13, .13, len(points))
        axes[0, 1].scatter(position + jitter, points, color=colors[group], alpha=.5, s=18)
        statistics = report["summary"][group]
        axes[0, 1].plot(position, statistics["mean"] * 10000, "ko")
        if statistics["ci95"]:
            axes[0, 1].vlines(position, *np.array(statistics["ci95"]) * 10000, color="black", linewidth=2)
    axes[0, 0].axvline(0, color="gray", linewidth=.7)
    axes[0, 0].set(xlabel="Edited − original win rate (percentage points)", ylabel="Empirical cumulative fraction",
                    title="Observed changes, including battle noise")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(xticks=[0, 1], xticklabels=["Random legal", "Frozen top placements"],
                    ylabel="Noise-corrected mean squared change (pp²)", title="Original-team clusters; group 95% intervals")
    axes[0, 1].axhline(0, color="gray", linewidth=.7)
    if (directory / "benchmarks.json").exists():
        bench = read_json(directory / "benchmarks.json")
        if bench["manifest_sha256"] != report["manifest_sha256"]:
            raise ValueError("benchmark manifest mismatch")
        primary = [(GROUP_LABELS[group], report["summary"][group]) for group in colors]
        primary += [(row["domain"], row["summary"]) for row in bench["conditions"]
                    if (row["domain"], row["step"]) in [("pest_control", 1), ("hartmann6", .01)]]
        for position, (name, statistics) in enumerate(primary):
            axes[1, 0].plot(position, statistics["mean"], "o", color="#35456b")
            if statistics["ci95"]:
                axes[1, 0].vlines(position, *statistics["ci95"], color="#35456b")
        axes[1, 0].set(xticks=range(len(primary)), xticklabels=[name.replace("_", " ") for name, _ in primary],
                        ylabel="Corrected mean squared change / fixed range²",
                        title="Declared scales and neighborhoods; descriptive comparison")
        axes[1, 0].set_yscale("symlog", linthresh=1e-7)
        axes[1, 0].axhline(0, color="gray", linewidth=.7)
        for domain, marker in [("pest_control", "o"), ("hartmann6", "s")]:
            rows = [row for row in bench["conditions"] if row["domain"] == domain]
            steps = [row["step"] for row in rows]
            axes[1, 1].plot(steps, [row["summary"]["mean"] for row in rows], marker=marker, label=domain)
        axes[1, 1].set(xscale="log", yscale="log", xlabel="Step (stage count or coordinate displacement)",
                        ylabel="Corrected mean squared change / fixed range²", title="Benchmark scale check; different distance units")
        axes[1, 1].legend(fontsize=8)
    else:
        for axis in axes[1]:
            axis.text(.5, .5, "Run benchmarks to fill this panel", ha="center")
            axis.set_axis_off()
    figure.suptitle(f"Legal-move sensitivity · {manifest['anchors_per_group']} + {manifest['anchors_per_group']} original teams\n"
                   f"{manifest['neighbors']} edits/team · {manifest['replicates']} × {manifest['battles_per_replicate']} battles/candidate")
    figure.savefig(directory / "smoothness.png", dpi=180)
    figure.savefig(directory / "smoothness.pdf")
    plt.close(figure)


def write_report(directory, manifest, report):
    lines = ["# Measured legal-move sensitivity", "",
             f"Completed {report['battles']:,} real simulator battles. "
             f"{manifest['anchors_per_group']} random and {manifest['anchors_per_group']} top-placement entries; "
             f"{manifest['neighbors']} move replacements per original; "
             f"{manifest['replicates']} independent panels of {manifest['battles_per_replicate']} battles per candidate.", "",
             "| Group | Original win rate | Corrected mean squared change (pp²) | 95% cluster interval (pp²) |", "|---|---:|---:|---|" ]
    for group in ("random", "tournament"):
        row = report["summary"][group]
        interval = "unavailable" if row["ci95"] is None else ", ".join(f"{x * 10000:.3f}" for x in row["ci95"])
        lines.append(f"| {GROUP_LABELS[group]} | {row['mean_original_win_rate']:.3%} | {row['mean'] * 10000:.3f} | {interval} |")
    contrast = report["summary"]["tournament_minus_random"]
    lines += ["", f"Top placements minus random corrected mean squared change: {contrast['mean'] * 10000:.3f} pp²; "
              f"95% interval: {None if contrast['ci95'] is None else [round(v * 10000, 3) for v in contrast['ci95']]}.", "",
              "![Measured sensitivity](smoothness.png)", "",
              "This measures sensitivity to move replacements under one fixed policy and opponent pool. "
              "It does not establish mathematical non-smoothness or performance under optimal play. "
              "Negative noise-corrected squared estimates are possible and are deliberately retained. "
              "The observed-change distribution still includes battle noise.", "",
              "Random teams are sampled from the frozen HPS corpus, whose distribution is not uniform over all legal teams. "
              "The top-placement set is the existing snapshot, not a new sample of independent tournaments. "
              "As requested, the full snapshot retains six ranked-season and four Showdown-ladder entries; "
              "it must not be described as strictly tournament-only. "
              "Duplicate identities are clustered; repeated opponent entries retain their frozen weight. "
              "All originals, including tournament teams, face exactly the same pool (including self matchups).", "",
              "Intervals describe resampling original-team clusters from these empirical groups. "
              "They do not account for new opponents, policy changes, tournament selection bias, or all possible legal edits. "
              "Edit-level intervals are pointwise approximate score intervals; no family-wise discovery claim is made.", "",
              "The benchmark comparison uses fixed objective bounds: win probability 1, Pest Control 50, Hartmann-6 8.4. "
              "Different edit sizes, dimensions, and reference regions prevent interpreting it as a universal ranking of smoothness.", "",
              "See analysis.json for all estimates, edits.csv for figure-ready differences and intervals, "
              "labels/ for original counts, benchmarks.json for benchmark inputs/outcomes, and manifest.json for provenance."]
    if manifest["anchors_per_group"] < 50:
        lines.insert(2, "**Pilot run: this is not the full 50-versus-50 experiment.**\n")
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=["prepare", "battle", "benchmarks", "analyse", "all"])
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    result.add_argument("--checkpoint", default="/tmp/bc_100.zip")
    result.add_argument("--node", default="/usr/local/bin/node")
    result.add_argument("--anchors", type=int, default=50)
    result.add_argument("--neighbors", type=int, default=8)
    result.add_argument("--replicates", type=int, default=4)
    result.add_argument("--battles", type=int, default=100, help="per replicate; multiple of 50")
    result.add_argument("--seed", type=int, default=20261001)
    result.add_argument("--ports", type=int, nargs="+", default=[8123])
    result.add_argument("--concurrency", type=int, default=16)
    result.add_argument("--job-timeout", type=int, default=600)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.command == "all":
        if not (arguments.output / "manifest.json").exists():
            prepare(arguments)
        benchmarks(arguments)
        battle(arguments)
        analyse(arguments)
    else:
        globals()[arguments.command](arguments)
