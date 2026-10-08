"""Fill only missing member-level edits; reuse the frozen pilot's original panels.

The added edits support exploration. They do not replace the original random-edit
sample used for the cohort and benchmark comparison.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
import smoothness_experiment as S


def prepare(parent, output):
    parent, output = parent.resolve(), output.resolve()
    if output.exists():
        manifest = S.verify_manifest(output, runtime=True)
        if manifest.get("parent_manifest_sha256") != S.digest(parent / "manifest.json"):
            raise ValueError("coverage directory belongs to another parent")
        return manifest
    original = S.verify_manifest(parent)
    analysis = S.read_json(parent / "analysis.json")
    if not analysis["complete"] or analysis["manifest_sha256"] != S.digest(parent / "manifest.json"):
        raise ValueError("a completed, matching parent analysis is required")
    for filename, expected in original["runtime_fingerprints"].items():
        if S.digest(filename) != expected:
            raise ValueError(f"parent runtime changed: {filename}; do not reuse its baseline")
    manifest = copy.deepcopy(original)
    manifest.update(parent_directory=str(parent), parent_manifest_sha256=S.digest(parent / "manifest.json"),
                    parent_analysis_sha256=S.digest(parent / "analysis.json"),
                    purpose="additional member coverage; excluded from primary cohort statistics",
                    edit_sampling="one uniformly sampled eligible move position and alternative per previously uncovered member; validate and reject invalid proposals",
                    seed=2026100101, candidates=[], planned_battles=0)
    manifest["code_fingerprints"] = {str(p): S.digest(p) for p in [Path(__file__).resolve(), S.ROOT / "scripts/smoothness_experiment.py", S.ROOT / "scripts/smoothness_worker.py"]}
    output.mkdir(parents=True)
    (output / "teams").mkdir()
    learnsets = S.read_json(Path(original["runtime"]) / "learnset_true.json")
    validator = S.Validator(Path(original["runtime"]) / "vgc-bench/pokemon-showdown", original["node"])
    try:
        for anchor in manifest["anchors"]:
            base = next(c for c in original["candidates"] if c["anchor"] == anchor["id"] and c["kind"] == "base")
            text = Path(base["file"]).read_text()
            lines, slots = S.move_slots(text)
            covered = {e["edit"]["slot"] for e in analysis["edges"] if e["anchor"] == anchor["id"] and e["edit"]}
            anchor["candidates"] = []
            for slot_index, slot in enumerate(slots):
                if slot_index in covered:
                    continue
                alternatives = sorted(set(learnsets[S.norm(slot["species"])]) - {S.norm(m) for _, m in slot["moves"]})
                if not alternatives:
                    raise ValueError(f"no legal alternatives: {anchor['id']} / {slot_index}")
                rng = np.random.default_rng(S.seed_for(manifest["seed"], anchor["id"], slot_index))
                for attempt in range(1000):
                    line, old = slot["moves"][int(rng.integers(len(slot["moves"])))]
                    new = alternatives[int(rng.integers(len(alternatives)))]
                    changed = lines.copy()
                    ending = "\r\n" if lines[line].endswith("\r\n") else "\n" if lines[line].endswith("\n") else ""
                    changed[line] = f"- {new}{ending}"
                    paste = "".join(changed)
                    if validator(paste)[0]:
                        S.assert_move_edit(text, paste)
                        break
                else:
                    raise ValueError(f"no valid replacement found: {anchor['id']} / {slot_index}")
                candidate_id = f"{anchor['id']}-cover-{slot_index:02d}"
                path = output / "teams" / f"{candidate_id}.txt"
                path.write_text(paste)
                candidate = dict(id=candidate_id, anchor=anchor["id"], kind="coverage",
                                 file=str(path), sha256=S.digest(path),
                                 edit=dict(slot=slot_index, species=slot["species"], old_move=old,
                                           new_move=new, proposal_attempt=attempt + 1))
                manifest["candidates"].append(candidate)
                anchor["candidates"].append(candidate_id)
    finally:
        validator.close()
    manifest["planned_battles"] = len(manifest["candidates"]) * manifest["replicates"] * manifest["battles_per_replicate"]
    if manifest["planned_battles"] > 6000:
        raise ValueError("this tool is bounded to the eight-team visual, at most 6,000 additional battles")
    S.atomic_json(output / "manifest.json", manifest)
    return manifest


def analyse(output):
    manifest = S.verify_manifest(output)
    parent = Path(manifest["parent_directory"])
    if S.digest(parent / "manifest.json") != manifest["parent_manifest_sha256"] or S.digest(parent / "analysis.json") != manifest["parent_analysis_sha256"]:
        raise ValueError("parent provenance changed")
    originals = S.verify_manifest(parent)
    parent_hash = manifest["parent_manifest_sha256"]
    parent_jobs = {j["id"]: j for j in S.job_list(originals)}
    rows = {}
    for job in S.job_list(manifest):
        rows[job["id"]] = S.checked_result(S.read_json(output / "labels" / (job["id"] + ".json")), job, S.digest(output / "manifest.json"))
    edges = []
    for candidate in manifest["candidates"]:
        baseline, edited = [], []
        for replicate in range(manifest["replicates"]):
            base_id = f"{candidate['anchor']}-base-r{replicate:02d}"
            baseline.append(S.checked_result(S.read_json(parent / "labels" / (base_id + ".json")), parent_jobs[base_id], parent_hash)["wins"])
            edited.append(rows[f"{candidate['id']}-r{replicate:02d}"]["wins"])
        deltas = (np.array(edited) - np.array(baseline)) / manifest["battles_per_replicate"]
        edges.append(dict(anchor=candidate["anchor"], candidate=candidate["id"], kind="coverage", edit=candidate["edit"],
                          delta=float(deltas.mean()), replicate_deltas=deltas.tolist(), corrected_mse=S.corrected_mean_square(deltas),
                          ci95=S.difference_interval(sum(baseline), sum(edited), manifest["battles_per_replicate"] * manifest["replicates"]),
                          baseline_experiment=parent_hash, baseline_candidate=candidate["anchor"] + "-base",
                          primary_sample=False))
    result = dict(manifest_sha256=S.digest(output / "manifest.json"), parent_manifest_sha256=parent_hash,
                  complete=True, battles=sum(r["battles"] for r in rows.values()), edges=edges,
                  note="Added member coverage only. Original primary group and benchmark summaries remain unchanged.")
    S.atomic_json(output / "analysis.json", result)
    print(json.dumps({"complete": True, "additional_edits": len(edges), "additional_battles": result["battles"]}), flush=True)


def run(args):
    manifest = prepare(args.parent, args.output)
    print(json.dumps({"additional_candidates": len(manifest["candidates"]), "planned_additional_battles": manifest["planned_battles"], "baseline_battles_reused": True}), flush=True)
    if args.prepare_only:
        return
    if all((args.output / "labels" / (j["id"] + ".json")).exists() for j in S.job_list(manifest)):
        analyse(args.output)
        return
    servers, streams = [], []
    try:
        for port in args.ports:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"port {port} belongs to an existing server; choose an unused port")
            stream = (args.output / f"server-{port}.log").open("a")
            streams.append(stream)
            server = subprocess.Popen([manifest["node"], "pokemon-showdown", "start", str(port), "--no-security"],
                                      cwd=Path(manifest["runtime"]) / "vgc-bench/pokemon-showdown",
                                      stdout=stream, stderr=stream, start_new_session=True)
            servers.append(server)
        deadline = time.monotonic() + 45
        for port in args.ports:
            while True:
                if any(s.poll() is not None for s in servers):
                    raise RuntimeError("coverage simulator exited; see server log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("coverage simulator did not start")
                    time.sleep(.25)
        S.battle(argparse.Namespace(output=args.output, ports=args.ports, concurrency=8, job_timeout=600))
        analyse(args.output)
    finally:
        for server in servers:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ports", nargs="+", type=int, default=[8180, 8181])
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())
