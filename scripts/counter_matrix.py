"""Who is the best counter depends on the pool.  Battles 24 candidate teams against each of
the 50 top-placing Reg M-B pool teams ONE AT A TIME, so a pool of any size can be scored by
averaging columns.  Candidates: the 16 search-found teams of results/rebattle_top.json and
8 real Reg M-B teams from outside the pool (the first 8 ruggedness meta anchors).
Behaviour-cloning policy both sides; BATTLES battles per (candidate, opponent) cell.

Run:  /tmp/vgc-pilot/.venv/bin/python scripts/counter_matrix.py
"""
import json, os, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "results"
BATTLES = int(os.environ.get("CM_BATTLES", 48))
D = Path("/tmp/vgc-pilot/proposals/counter_matrix")

def main():
    sys.path.insert(0, "/tmp/vgc-pilot/src"); import pool
    from corpus import parse_team
    top = json.load(open(RES / "top50_evs.json"))
    rt = Path("/tmp/vgc-pilot/vgc-bench/teams/reg_mb")
    opps = []
    for t in top:
        for c in (rt / f"{t['id']}.txt", rt / "featured" / f"{t['id']}.txt"):
            if c.exists(): opps.append({**t, "file": str(c)}); break
    cands = [{"kind": "search", "file": f"/tmp/vgc-pilot/proposals/ruggedness2/s{k:02d}.txt", "id": f"S{k+1:02d}"}
             for k in range(16)]
    names = json.load(open(RES / "ruggedness.json"))["vgc"]["anchors"]
    cands += [{"kind": "meta", "file": f"/tmp/vgc-pilot/proposals/ruggedness/a{k:02d}.txt",
               "id": names[k]["name"].replace(".txt", "")} for k in range(8)]
    D.mkdir(parents=True, exist_ok=True)
    jobs, cell = [], {}
    for i, c in enumerate(cands):
        for j, o in enumerate(opps):
            f = D / f"c{i:02d}_o{j:02d}.txt"; shutil.copyfile(c["file"], f)
            jobs.append((str(f), [o["file"]])); cell[str(f)] = (i, j)
    print(f"{len(cands)} candidates x {len(opps)} opponents x {BATTLES} = {len(jobs)*BATTLES:,} battles", flush=True)
    t0 = time.time()
    r = pool.score(jobs, battles=BATTLES, conc=50)
    W = [[None] * len(opps) for _ in cands]
    for f, v in r.items():
        i, j = cell[f]; W[i][j] = {"wins": v["wins"], "n": v["battles"]}
    for c in cands: c["species"] = [s["species"] for s in parse_team(c["file"])]
    for o in opps: o["species"] = [s["species"] for s in parse_team(o["file"])]
    json.dump({"battles_per_cell": BATTLES, "candidates": cands, "opponents": opps, "cells": W,
               "runtime_s": round(time.time() - t0)}, open(RES / "counter_matrix.json", "w"), indent=1)
    print("wrote results/counter_matrix.json", flush=True)

if __name__ == "__main__":
    main()
