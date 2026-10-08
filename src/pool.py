"""A pool of Showdown servers, and a sharded battle driver that uses all of them.

Showdown's server is single-threaded Node, so many workers behind ONE server queue
behind one core. Measured on this machine: 4 workers + 1 server = 19.4 battles/sec;
4 workers each with their OWN server = 38.3; 7 workers at concurrency 50 = 63.2.
That is a 3.3x speedup and it comes entirely from removing the queue.

Start the servers once:
    cd <repo>/pokemon-showdown
    for p in 8123 8124 8125 8126 8127 8128 8129; do
        node pokemon-showdown start $p --no-security &  done
"""
import json, os, socket, subprocess, sys, time
from pathlib import Path

CANDIDATE_PORTS = [8123, 8124, 8125, 8126, 8127, 8128, 8129]
REPO = "/tmp/vgc-pilot/vgc-bench"
PY = "/tmp/vgc-pilot/.venv/bin/python"
SHARD = "/tmp/vgc-pilot/src/shard.py"

def live_ports(ports=CANDIDATE_PORTS):
    """Which Showdown servers are actually listening."""
    up = []
    for p in ports:
        s = socket.socket(); s.settimeout(0.25)
        try:
            s.connect(("127.0.0.1", p)); up.append(p)
        except OSError:
            pass
        finally:
            s.close()
    return up

# Shard scratch files are namespaced by PID. They used to be fixed paths
# (/tmp/vgc-pilot/_shard_0.json), so two battle runs at once silently overwrote each
# other's specs and read each other's results -- measured 2026-08-27, when a smoke test
# and a labelling run collided. Correctness, not tidiness.
RUN = os.getpid()

def score(jobs, battles=24, conc=50, ports=None, quiet=False, opp_schedule=None, seed=None):
    """Score a list of (team_file, [opponent_files]) jobs across every live server.

    `battles` is now the TOTAL battles for that team; the opponent draws a fresh
    team from the list each battle, so pass the whole opponent pool per job.

    `opp_schedule` (list of team files) + `seed` opt into common random numbers:
    every candidate faces the identical ordered opponent schedule and the shard
    reseeds its RNGs per candidate (added 2026-09-02 for the TRS port). Default
    None/None keeps the historical unseeded uniform draw.

    Returns {team_file: {"wins": w, "battles": n, "win_rate": r, "se": s}}.
    Work is sharded by team, so each worker owns whole teams and no result is split.
    """
    ports = ports or live_ports()
    if not ports:
        raise RuntimeError("no Showdown server is listening; start one on 8123")
    k = min(len(ports), len(jobs))
    shards = [[] for _ in range(k)]
    for i, j in enumerate(jobs):
        shards[i % k].append(j)
    procs, outs = [], []
    t0 = time.perf_counter()
    for i, sh in enumerate(shards):
        spec = f"/tmp/vgc-pilot/_shard_{RUN}_{i}.json"
        out = f"/tmp/vgc-pilot/_shard_{RUN}_{i}.out.json"
        json.dump({"jobs": sh, "battles": battles, "conc": conc, "port": ports[i],
                   "opp_schedule": opp_schedule, "seed": seed},
                  open(spec, "w"))
        env = dict(os.environ, PYTHONPATH=f"{REPO}:/tmp/vgc-pilot/src")
        procs.append(subprocess.Popen([PY, SHARD, spec, out], cwd=REPO, env=env,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        outs.append(out)
    res, failed = {}, []
    for p, out in zip(procs, outs):
        _, err = p.communicate()
        if p.returncode != 0 or not Path(out).exists():
            failed.append((p.returncode, (err or b"").decode()[-400:]))
            continue
        res.update(json.load(open(out)))
    dt = time.perf_counter() - t0
    n = sum(v["battles"] for v in res.values())
    if not quiet:
        print(f"  {len(ports)} servers · {n} battles · {dt:.1f}s · {n/max(dt,1e-9):.1f}/sec", flush=True)
    if failed:
        print(f"  WARNING {len(failed)} shard(s) failed: {failed[0][1][:200]}", flush=True)
    return res
