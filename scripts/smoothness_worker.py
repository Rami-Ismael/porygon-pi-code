"""One local battle worker; commits only complete evaluation panels."""
import asyncio
from pathlib import Path
import sys

import torch
from stable_baselines3 import PPO

from smoothness_experiment import atomic_json, digest, read_json, verify_manifest
from shard import score_one


def main():
    spec = read_json(sys.argv[1])
    directory = Path(spec["directory"])
    manifest = verify_manifest(directory)
    manifest_hash = digest(directory / "manifest.json")
    torch.set_num_threads(1)
    policy = PPO.load(manifest["checkpoint"], device=torch.device("cpu")).policy
    opponents = [row["file"] for row in manifest["opponents"]]
    for index, job in enumerate(spec["jobs"]):
        output = directory / "labels" / f"{job['id']}.json"
        if output.exists():
            raise FileExistsError(f"another worker completed {job['id']}")
        # Exactly equal weight for all 50 frozen entries in every panel.
        result = asyncio.run(asyncio.wait_for(
            score_one(policy, job["file"], opponents, job["battles"], spec["concurrency"],
                      spec["port"], opp_schedule=opponents, seed=job["seed"]),
            timeout=spec["timeout"]))
        wins, finished, margin, turns = result
        if finished != job["battles"] or not 0 <= wins <= finished:
            raise RuntimeError(f"incomplete battle panel: {job['id']} {finished}/{job['battles']}")
        atomic_json(output, dict(job=job, manifest_sha256=manifest_hash, wins=wins,
                                battles=finished, margin=margin, turns=turns))
        print(f"{index + 1}/{len(spec['jobs'])} {job['id']} {wins}/{finished}", flush=True)


if __name__ == "__main__":
    main()
