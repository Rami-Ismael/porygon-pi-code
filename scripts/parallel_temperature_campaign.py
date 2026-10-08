"""Collect independent seed runs without changing their treatment or budgets.

The original process completes seed 101 and is then interrupted before spending
another full generation on seed 202. Seeds 202/303 run in independent processes
and simulator pools. Raw result files remain intact; the combined artifact has
explicit source and per-seed server provenance and is not a legacy resume file.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from temperature_experiment import atomic_json

RUNTIME = Path("/Users/ramiismael/.local/share/vgc-pilot-runtime")
PYTHON = RUNTIME / "venv/bin/python"
PORTS = {101: [8130, 8131, 8132, 8133], 202: [8134, 8135, 8136, 8137],
         303: [8138, 8139, 8140, 8141]}


def live_command(pid):
    p = subprocess.run(["ps", "-p", str(pid), "-o", "stat=,command="], capture_output=True, text=True)
    parts = p.stdout.strip().split(None, 1)
    return parts[1] if len(parts) == 2 and "Z" not in parts[0] else None


def seed_complete(data, seed):
    arms = data["manifest"]["arms"]
    return all("final" in data.get("runs", {}).get(f"seed{seed}/{arm}", {}) for arm in arms)


def read_source(path):
    """Bind the version to the opened file, even if its path is replaced mid-read."""
    with path.open("rb") as stream:
        data = json.load(stream)
        return data, os.fstat(stream.fileno()).st_mtime_ns


def merge(base_manifest, sources):
    """Reject treatment/input drift; copy observed records with exact provenance."""
    control = {k: v for k, v in base_manifest["config"].items() if k not in ("seeds", "ports")}
    out = dict(status="running", manifest=copy.deepcopy(base_manifest), runs={})
    out["manifest"]["schema"] = 2
    out["manifest"]["config"].pop("ports", None)
    execution = dict(kind="independent_seed_processes", ports_by_seed={}, sources={})
    out["manifest"]["execution"] = execution
    for seed, (path, data) in sources.items():
        manifest = data["manifest"]
        candidate = {k: v for k, v in manifest["config"].items() if k not in ("seeds", "ports")}
        if candidate != control:
            raise ValueError(f"experimental controls changed for seed {seed}")
        for key in ("inputs", "opponents", "arms"):
            if manifest[key] != base_manifest[key]:
                raise ValueError(f"{key} changed for seed {seed}")
        if seed not in manifest["config"]["seeds"]:
            raise ValueError(f"source did not declare seed {seed}")
        if manifest["config"]["ports"] != PORTS[seed]:
            raise ValueError(f"unexpected simulator pool for seed {seed}")
        execution["ports_by_seed"][str(seed)] = PORTS[seed]
        execution["sources"][str(seed)] = dict(path=str(path),
            manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
        for arm in base_manifest["arms"]:
            key = f"seed{seed}/{arm}"
            if key in data.get("runs", {}):
                out["runs"][key] = copy.deepcopy(data["runs"][key])
    # Independent workers write the last finalist record, then their terminal
    # status. Wait for that second write before hashing their immutable files.
    # The original multi-seed controller is instead stopped by main after seed101.
    initial_seed = base_manifest["config"]["seeds"][0]
    if (set(sources) == set(base_manifest["config"]["seeds"]) and
        all(seed_complete(data, seed) for seed, (_, data) in sources.items()) and
        all(seed == initial_seed or data.get("status") == "complete"
            for seed, (_, data) in sources.items())):
        out["status"] = "complete"
        for seed, (path, _) in sources.items():
            execution["sources"][str(seed)]["raw_result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def launch(seed, config, output):
    command = [str(PYTHON), "-u", str(REPO / "src/entropyloop.py"), "compare", "run",
               "--seeds", str(seed), "--ports", ",".join(map(str, PORTS[seed]))]
    for name in ("generations", "anneal_generations", "propose", "battle", "battles", "final_sample",
                 "final_battles", "finalists", "epochs", "batch", "adaptive_ess", "checkpoint", "labels"):
        command.extend(["--" + name.replace("_", "-"), str(config[name])])
    command.extend(["--output", str(output)])
    log = (RUNTIME / f"temperature-seed{seed}.log").open("ab")
    process = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"SEED_STARTED {seed} pid={process.pid}", flush=True)
    return process.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--adopt-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state_path = RUNTIME / "temperature-parallel-state.json"
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = json.loads(args.initial.read_text())
        base = initial["manifest"]
        if base["config"]["seeds"] != [101, 202, 303]:
            raise ValueError("this continuation expects the registered three-seed study")
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(jobs={})
        state.update(manager_pid=os.getpid(), output=str(args.output), initial=str(args.initial),
                     adopted_pid=args.adopt_pid, status="running")
        paths = {101: args.initial, **{s: REPO / f"results/temperature_seed{s}.json" for s in (202, 303)}}
        for seed in (202, 303):
            saved = json.loads(paths[seed].read_text()) if paths[seed].exists() else None
            previous = state["jobs"].get(str(seed), {})
            command = live_command(previous.get("pid", 0)) if previous.get("pid") else None
            if saved and saved.get("status") == "complete":
                continue
            if command:
                if not command.endswith(str(paths[seed])):
                    raise RuntimeError(f"stored PID belongs to another process for seed {seed}")
            else:
                previous = dict(pid=launch(seed, base["config"], paths[seed]), output=str(paths[seed]))
            state["jobs"][str(seed)] = previous
            atomic_json(state_path, state)
        previous_signature = None
        while True:
            snapshots = {s: (p, *read_source(p)) for s, p in paths.items() if p.exists()}
            sources = {s: (p, data) for s, (p, data, _) in snapshots.items()}
            first = sources[101][1]
            command = live_command(args.adopt_pid)
            if seed_complete(first, 101):
                if command:
                    if "entropyloop.py compare run" not in command or not command.endswith(str(args.initial)):
                        raise RuntimeError("adopted PID belongs to an unexpected process")
                    if not state.get("initial_interrupt_sent"):
                        os.kill(args.adopt_pid, signal.SIGINT)
                        state["initial_interrupt_sent"] = True
                        print("INITIAL_SEED_COMPLETE: stopping the sequential controller before duplicate seed runs", flush=True)
                else:
                    state["initial_stopped_after_seed101"] = True
            elif not command:
                raise RuntimeError("initial seed process stopped before all its arms completed")
            for seed in (202, 303):
                if seed not in sources or not seed_complete(sources[seed][1], seed):
                    if not live_command(state["jobs"][str(seed)]["pid"]):
                        raise RuntimeError(f"seed {seed} stopped before completion; inspect its log and resume")
            signature = (tuple((str(p), version) for p, _, version in snapshots.values()),
                         state.get("initial_stopped_after_seed101", False))
            if signature != previous_signature:
                out = merge(base, sources)
                out["manifest"]["execution"]["initial_controller_stopped_after_seed101"] = state.get("initial_stopped_after_seed101", False)
                # Never publish terminal completeness while the sequential controller can still write.
                if not state.get("initial_stopped_after_seed101"):
                    out["status"] = "running"
                atomic_json(args.output, out)
                done = sum("final" in r for r in out["runs"].values())
                gens = sum("scores" in cell for r in out["runs"].values() for cell in r["generations"].values())
                print(f"PROGRESS {done}/15 runs, {gens}/165 generations", flush=True)
                previous_signature = signature
                if out["status"] == "complete":
                    break
            atomic_json(state_path, state)
            time.sleep(3)
        for name in ("check_temperature_results.py", "analyze_temperature_comparison.py"):
            subprocess.run([str(PYTHON), str(REPO / "scripts" / name), str(args.output)], check=True)
        state["status"] = "verified_and_analyzed"
        atomic_json(state_path, state)
        print("TEMPERATURE_VERIFIED_AND_ANALYZED", flush=True)


if __name__ == "__main__":
    main()
