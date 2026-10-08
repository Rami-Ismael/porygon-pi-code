"""Capture the original checkpoint before CEM fine-tuning; no generator training.

Uses 512 proposals plus an independent 128-team, 192-battle unranked sample per
original run seed. Results are separate from the completed temperature campaign.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from temperature_experiment import (BattleEngine, Config, atomic_json, digest,
    fingerprints, load_or_create, opponent_schedule, phase_seed)
from report_team_diversity import load_reference
from diversity_metrics import Reference, STATS, measure
from start_temperature_servers import RUNTIME, SOURCE, NODE, link_or_copy, listening


class BaselineEngine(BattleEngine):
    def baseline(self):
        E = self.E
        model = E.H.TeamDiffusionHPS(self.V).to(E.D.DEV)
        model.load_state_dict(E.torch.load(self.c.checkpoint, map_location=E.D.DEV)["sd"])
        model.eval()
        features, coefficients = E.GG.fit_ridge(self.initial_teams, self.initial_y, self.corpus)
        guide = E.GG.Guide(features, coefficients, self.V, E.D.DEV)
        return model, features, coefficients, guide


def run_baseline(config, output, engine, inputs, source_sha256):
    ref = engine.diversity_reference
    manifest = json.loads(json.dumps(dict(protocol="diversity-baseline-v1", config=asdict(config),
        inputs=inputs, source_campaign_sha256=source_sha256, opponents=engine.opponents,
        reference=dict(metadata=ref.metadata, normalized_records=ref.snapshot),
        generator_refit_steps=0,
        expected_battles=len(config.seeds) * config.final_sample * config.final_battles)))
    out = load_or_create(output, manifest)
    fitted = None
    for seed in config.seeds:
        row = out["runs"].setdefault(str(seed), dict(seed=seed))
        directory = output.parent / (output.stem + "_teams") / str(seed)
        for phase, n, random_phase in (("proposal", config.propose, "proposal"),
                                        ("holdout", config.final_sample, "holdout_proposals")):
            if phase not in row:
                if fitted is None:
                    fitted = engine.baseline()
                row[phase] = engine.propose(fitted, n, directory / phase,
                                            phase_seed(seed, 0, random_phase))
                row[phase]["sampling_seed"] = phase_seed(seed, 0, random_phase)
                atomic_json(output, out)
                print(f"BASELINE_PROPOSALS seed={seed} phase={phase} n={n}", flush=True)
        if "scores" not in row:
            schedule = opponent_schedule(engine.opponents, config.final_battles,
                                         phase_seed(seed, 0, "holdout_opponents"))
            battle_seed = phase_seed(seed, 0, "holdout_battles")
            row["scores"] = engine.score(row["holdout"]["pastes"], directory / "battles",
                                         config.final_battles, schedule, battle_seed)
            row["opponent_schedule"], row["battle_seed"] = schedule, battle_seed
            atomic_json(output, out)
            print(f"BASELINE_SCORED seed={seed} teams={len(row['scores'])}", flush=True)
    verify_baseline(out, engine.parse)
    out["status"] = "complete"
    atomic_json(output, out)
    return out


def verify_baseline(out, parse):
    config = out["manifest"]["config"]
    frozen = out["manifest"]["reference"]
    teams = [[dict(species=s[0], ability=s[1], item=s[2], nature=s[3],
                   moves=[m for m in s[4:8] if m], evs=dict(zip(STATS, s[8:])))
              for s in t] for t in frozen["normalized_records"]]
    metadata = frozen["metadata"]
    reference = Reference(teams, scope=metadata["scope"], format_id=metadata["format"],
                          provenance=metadata["provenance"])
    if reference.metadata != metadata:
        raise ValueError("frozen reference metadata does not match its records")
    if set(out["runs"]) != {str(seed) for seed in config["seeds"]}:
        raise ValueError("incomplete baseline seeds")
    if out["manifest"]["generator_refit_steps"] != 0:
        raise ValueError("baseline must have no generator fine-tuning")
    total = 0
    for seed, row in out["runs"].items():
        if row["seed"] != int(seed):
            raise ValueError("seed identity mismatch")
        for phase, n in (("proposal", config["propose"]), ("holdout", config["final_sample"])):
            sample = row[phase]
            if len(sample["pastes"]) != n:
                raise ValueError("baseline sample size mismatch")
            metrics = sample["diversity"]["team_diversity"]
            if metrics["status"] != "ok" or metrics["sampling"]["accepted"] != n:
                raise ValueError("baseline measurement incomplete")
            recomputed = measure(parse(sample["pastes"]), reference, expected_n=n,
                                 attempts=metrics["sampling"]["attempts"])
            recomputed["sampling"]["attempt_limit"] = n * 8
            if recomputed != metrics:
                raise ValueError("saved baseline metrics do not match the draws")
            random_phase = "proposal" if phase == "proposal" else "holdout_proposals"
            if sample["sampling_seed"] != phase_seed(int(seed), 0, random_phase):
                raise ValueError("baseline sampling seed mismatch")
        scores = row["scores"]
        if len(scores) != config["final_sample"] or any(
            type(s["wins"]) is not int or not 0 <= s["wins"] <= config["final_battles"]
            or s["battles"] != config["final_battles"] for s in scores):
            raise ValueError("baseline battle budget incomplete")
        total += sum(s["battles"] for s in scores)
        if row["opponent_schedule"] != opponent_schedule(out["manifest"]["opponents"],
            config["final_battles"], phase_seed(int(seed), 0, "holdout_opponents")):
            raise ValueError("baseline opponent schedule mismatch")
        if row["battle_seed"] != phase_seed(int(seed), 0, "holdout_battles"):
            raise ValueError("baseline battle seed mismatch")
    if total != out["manifest"]["expected_battles"]:
        raise ValueError("total baseline battle budget mismatch")


@contextmanager
def private_servers(ports):
    processes = []
    try:
        for port in ports:
            if listening(port):
                raise RuntimeError(f"private baseline port is occupied: {port}")
            root = RUNTIME / "servers" / "diversity-baseline" / str(port)
            if not root.exists():
                excluded = {".git", "node_modules", "config", "databases", "logs"}
                shutil.copytree(SOURCE, root, copy_function=link_or_copy,
                    ignore=lambda directory, names: [name for name in names
                        if Path(directory) == SOURCE and name in excluded])
                (root / "node_modules").symlink_to(SOURCE / "node_modules", target_is_directory=True)
                for directory in ("config", "databases"):
                    if (SOURCE / directory).exists():
                        shutil.copytree(SOURCE / directory, root / directory)
                (root / "logs/repl").mkdir(parents=True, exist_ok=True)
                with (root / "config/config.js").open("a") as file:
                    file.write('\nexports.bindaddress = "127.0.0.1";\n')
            source_files = {p.relative_to(SOURCE): digest(p) for p in (SOURCE / "dist").rglob("*.js")}
            copies = {p.relative_to(root): digest(p) for p in (root / "dist").rglob("*.js")}
            if source_files != copies:
                raise RuntimeError("private simulator differs from source")
            with (RUNTIME / f"diversity-baseline-{port}.log").open("ab") as log:
                process = subprocess.Popen([NODE, "pokemon-showdown", "start", str(port), "--no-security"],
                    cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(f"private simulator {port} exited")
                if listening(port):
                    break
                time.sleep(.1)
            else:
                raise RuntimeError(f"private simulator {port} not ready")
            print(f"BASELINE_SERVER port={port} pid={process.pid}", flush=True)
        yield
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, default=REPO / "results/temperature_comparison_parallel.json")
    cli.add_argument("--training-manifest", type=Path, default=REPO / "results/temperature_p0.training.json")
    cli.add_argument("--output", type=Path, default=REPO / "results/diversity_baseline.json")
    cli.add_argument("--ports", default="8150,8151,8152,8153")
    args = cli.parse_args()
    if args.output.resolve() in (args.source.resolve(), args.training_manifest.resolve()):
        raise ValueError("baseline output must be separate")
    data = json.loads(args.source.read_text())
    if data["status"] != "complete":
        raise ValueError("expected completed source campaign")
    settings = dict(data["manifest"]["config"])
    settings["ports"] = tuple(int(x) for x in args.ports.split(","))
    settings["seeds"] = tuple(settings["seeds"])
    config = Config(**settings)
    config.validate()
    reference, _ = load_reference(args.training_manifest, settings, data["manifest"]["inputs"])
    # All historical non-source inputs, including opponents and battle policy,
    # must match; code changes are fingerprinted explicitly in the new output.
    for file, expected in data["manifest"]["inputs"].items():
        if file == "runtime_versions":
            if {name: importlib.metadata.version(name) for name in expected} != expected:
                raise ValueError("runtime packages differ from the historical campaign")
            continue
        if file.startswith(str(REPO / "src") + "/"):
            continue
        if digest(file) != expected:
            raise ValueError(f"historical experimental input changed: {file}")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    with private_servers(config.ports):
        engine = BaselineEngine(config)
        try:
            if engine.opponents != data["manifest"]["opponents"]:
                raise ValueError("opponent pool changed")
            engine.diversity_reference = reference
            inputs = fingerprints(config, engine.opponents)
            inputs["runtime_versions"] = {name: importlib.metadata.version(name)
                for name in data["manifest"]["inputs"]["runtime_versions"]}
            inputs[str(Path(__file__).resolve())] = digest(__file__)
            inputs[str(args.training_manifest.resolve())] = digest(args.training_manifest)
            run_baseline(config, args.output, engine, inputs, digest(args.source))
        finally:
            engine.close()
    print(f"BASELINE_COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
