"""Run a matchup-matrix battle stage and survive shards that stall (2026-10-04).

Observed: after ~20 min, all but one pair_shard process stop finishing jobs at the same moment while
the Showdown servers stay healthy. matchup_db.run_jobs waits on every shard, so one stall blocks the
run forever, and killing a shard throws away its finished jobs. This supervisor watches the shard
result files (pair_shard rewrites its .out.json after every finished job); when any shard has not
finished a job for STALL seconds it stops the run, records every finished job into the matrix
itself (the driver never recorded them), and restarts the stage - matchup_db.ensure then battles
only the cells still missing. Loops until the stage exits cleanly.

    python battle_supervisor.py <script.py> <stage> [--then <stage> ...]
    e.g. python battle_supervisor.py random_fill_baseline.py battle --then report
"""
import glob, json, os, signal, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matchup_db as MDB

PY = "/tmp/vgc-pilot/.venv/bin/python"
STALL, POLL, POLICY = 480, 30, 1


def shard_files(pid):
    return sorted(f for f in glob.glob(f"/tmp/vgc-pilot/_mshard_{pid}_*.json") if not f.endswith(".out.json"))


def recover(pid, con):
    res, un, jobs = [], 0, 0
    for f in shard_files(pid):
        o = f.replace(".json", ".out.json")
        if not os.path.exists(o): continue
        try: done = json.load(open(o))
        except json.JSONDecodeError: continue           # caught mid-write: lose only that shard's file
        for job in done:
            jobs += 1; un += job["unattributed"]
            for c, (w, l, t) in job["by_col"].items(): res.append((job["row"], int(c), w, l, t))
    if res:
        MDB.record(con, POLICY, res, note=f"supervisor recovery of stalled run {pid}", unattributed=un)
    return jobs, sum(w + l + t for *_, w, l, t in res)


def kill_tree(pid):
    for f in shard_files(pid):
        subprocess.run(["pkill", "-f", Path(f).name], check=False)
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass


def main():
    script, stage = sys.argv[1], sys.argv[2]
    then = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--then"]
    con = MDB.connect(); attempt = 0
    while True:
        attempt += 1
        p = subprocess.Popen([PY, script, stage], stdout=open(f"/tmp/vgc-pilot/supervisor_{stage}.log", "a"),
                             stderr=subprocess.STDOUT)
        print(f"[attempt {attempt}] {script} {stage} pid {p.pid}", flush=True)
        stalled = False
        while p.poll() is None:
            time.sleep(POLL)
            now = time.time()
            for f in shard_files(p.pid):
                o = f.replace(".json", ".out.json")
                try:
                    spec_jobs = len(json.load(open(f))["jobs"])
                    done = len(json.load(open(o))) if os.path.exists(o) else 0
                except (json.JSONDecodeError, OSError):
                    continue                            # file caught mid-write; check again next poll
                ref = os.path.getmtime(o) if os.path.exists(o) else os.path.getmtime(f)
                if done < spec_jobs and now - ref > STALL:
                    stalled = True; break
            if stalled: break
        if not stalled and p.returncode == 0:
            print(f"[attempt {attempt}] finished cleanly", flush=True)
            break
        kill_tree(p.pid); time.sleep(3)
        jobs, n = recover(p.pid, con)
        print(f"[attempt {attempt}] {'stalled' if stalled else f'exit {p.returncode}'} · recovered {jobs} jobs / {n} battles · restarting",
              flush=True)
    for st in then:
        subprocess.run([PY, script, st], stdout=open(f"/tmp/vgc-pilot/supervisor_{st}.out", "w"), stderr=subprocess.STDOUT)
        print(f"ran {st} -> /tmp/vgc-pilot/supervisor_{st}.out", flush=True)
    print("SUPERVISOR_DONE", flush=True)


if __name__ == "__main__":
    main()
