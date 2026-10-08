"""Own local servers and run the frozen study through analysis with durable status.

This launcher does not change the scientific protocol or its fingerprinted code.
Restarting it resumes completed panels through smoothness_experiment.py.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

from smoothness_experiment import atomic_json, digest, job_list, verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def listening(port):
    with socket.socket() as client:
        client.settimeout(.2)
        return client.connect_ex(("127.0.0.1", port)) == 0


def stop(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(args):
    directory = args.output.resolve()
    with (directory / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_manifest(directory, runtime=True)
        if any(listening(port) for port in args.ports):
            raise RuntimeError("A requested dedicated port is occupied; refusing to reuse or stop its server")
        run_directory = directory / f"supervision-{time.time_ns()}"
        run_directory.mkdir()
        jobs = {job["id"] for job in job_list(manifest)}
        def completed():
            return len({p.stem for p in (directory / "labels").glob("*.json")} & jobs)
        initial, started = completed(), time.monotonic()
        state = dict(status="starting", started_at=timestamp(), updated_at=timestamp(),
                     supervisor_pid=os.getpid(), ports=args.ports, log_directory=str(run_directory),
                     manifest_sha256=digest(directory / "manifest.json"),
                     planned_panels=len(jobs), planned_battles=manifest["planned_battles"],
                     completed_panels=initial, completed_battles=initial * manifest["battles_per_replicate"],
                     server_pids=[], battle_pid=None, error=None)
        def publish(stage=None):
            if stage:
                state["status"] = stage
            count = completed()
            elapsed = time.monotonic() - started
            rate = (count - initial) * manifest["battles_per_replicate"] / max(elapsed, 1)
            state.update(updated_at=timestamp(), completed_panels=count,
                         completed_battles=count * manifest["battles_per_replicate"],
                         elapsed_seconds=round(elapsed), battles_per_second=round(rate, 2),
                         estimated_seconds_remaining=round((len(jobs) - count) * manifest["battles_per_replicate"] / rate)
                         if rate > 0 else None)
            atomic_json(directory / "run-status.json", state)
            percent = count / len(jobs) * 100
            note = ("# Full smoothness study: " + state["status"] + "\n\n"
                    f"**{state['completed_battles']:,} / {state['planned_battles']:,} battles ({percent:.1f}%)**\n\n"
                    "50 random legal originals and the frozen 50 top-placement entries; eight move edits "
                    "per original; unchanged-team controls; four panels of 100 battles per candidate.\n\n"
                    f"Updated: {state['updated_at']} (UTC). Progress is saved after every complete panel.\n\n"
                    "The launcher automatically runs analysis after all panels finish and stops only its own servers.\n\n"
                    "[Run status](run-status.json) · [Frozen protocol](../../docs/smoothness-experiment.md)\n")
            if state["status"] == "complete":
                note += "\n[Measured report](report.md) · [Figure](smoothness.png) · [Analysis](analysis.json)\n"
            if state["error"]:
                note += f"\nRun stopped: {state['error']}\n"
            (directory / "run-status.md").write_text(note)
        servers, streams, children = [], [], []
        publish()
        try:
            # Keep the machine awake while computing; the display can still sleep.
            caffeine = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                                         start_new_session=True)
            children.append(caffeine)
            showdown = Path(manifest["runtime"]) / "vgc-bench/pokemon-showdown"
            for port in args.ports:
                stream = (run_directory / f"server-{port}.log").open("w")
                streams.append(stream)
                server = subprocess.Popen([manifest["node"], "pokemon-showdown", "start", str(port), "--no-security"],
                                          cwd=showdown, stdout=stream, stderr=stream, start_new_session=True)
                servers.append(server)
            state["server_pids"] = [p.pid for p in servers]
            deadline = time.monotonic() + 60
            while not all(listening(port) for port in args.ports):
                if any(p.poll() is not None for p in servers) or time.monotonic() > deadline:
                    raise RuntimeError("Dedicated Showdown server startup failed; inspect server logs")
                time.sleep(1)
            command = [manifest["battle_python"], str(ROOT / "scripts/smoothness_experiment.py"), "battle",
                       "--output", str(directory), "--ports", *map(str, args.ports)]
            stream = (run_directory / "battle.log").open("w")
            streams.append(stream)
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=stream, start_new_session=True)
            children.append(process)
            state["battle_pid"] = process.pid
            publish("battling")
            while process.poll() is None:
                if any(server.poll() is not None for server in servers):
                    raise RuntimeError("Dedicated Showdown server exited during evaluation")
                time.sleep(20)
                publish()
            if process.returncode:
                raise RuntimeError(f"Battle runner exited {process.returncode}; inspect {run_directory / 'battle.log'}")
            verify_manifest(directory, runtime=True)
            publish("analysing")
            with (run_directory / "analysis.log").open("w") as stream:
                subprocess.run([manifest["battle_python"], str(ROOT / "scripts/smoothness_experiment.py"),
                                "analyse", "--output", str(directory)], cwd=ROOT, stdout=stream,
                               stderr=stream, check=True)
            report = json.loads((directory / "analysis.json").read_text())
            if not report["complete"] or report["battles"] != manifest["planned_battles"]:
                raise RuntimeError("Final analysis did not confirm the planned battle count")
            verification_path = directory / "verification.json"
            verification = json.loads(verification_path.read_text())
            atomic_json(run_directory / "verification-before-run.json", verification)
            verification.update(status="full_pokemon_battles_and_analysis_complete",
                                completed_pokemon_battles=report["battles"], completed_panels=len(jobs),
                                completed_at=timestamp(), final_manifest_integrity="passed",
                                report_and_figure_generated=True, full_figure_visually_inspected=False)
            atomic_json(verification_path, verification)
            publish("complete")
        except BaseException as error:
            state["error"] = f"{type(error).__name__}: {error}"
            (run_directory / "error.log").write_text(traceback.format_exc())
            publish("failed")
            raise
        finally:
            for process in reversed(children + servers):
                stop(process)
            for stream in streams:
                stream.close()
            state["owned_processes_stopped"] = True
            publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ports", type=int, nargs="+", default=[8170, 8171, 8172, 8173])
    arguments = parser.parse_args()
    if len(set(arguments.ports)) != len(arguments.ports):
        parser.error("ports must be distinct")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit("Terminated"))
    run(arguments)
