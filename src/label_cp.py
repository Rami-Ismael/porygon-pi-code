"""Label the 1,600 copy-paste proposals from the 2026-08-27 filter-rate sweep.

filterrate.py battled them but only kept per-arm aggregates, so the per-team labels
(the only teams in the project that reliably score above 0.5) were lost. This
re-battles every file under /tmp/vgc-pilot/filterrate/cp_rho*/ against the top-50
meta, 24 battles each, and writes /tmp/vgc-pilot/cp_labels.json in the same
{path: {wins, battles, win_rate, se}} shape as ladder_labels.json / strata_labels.json.

Interruption-safe: results are flushed after every chunk of teams and already-labelled
files are skipped on restart, so a killed run resumes where it stopped.

    python label_cp.py            # ~38k battles, ~13 min on 7 servers
"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, "/tmp/vgc-pilot/src")
import pool
from loop import opponents

OUT = "/tmp/vgc-pilot/cp_labels.json"
ROOT = Path("/tmp/vgc-pilot/filterrate")
BATTLES = 24
CHUNK = 200            # teams per flush; one chunk = ~4,800 battles, ~1.5 min

def main():
    files = sorted(str(p) for d in sorted(ROOT.glob("cp_rho*")) for p in d.glob("*.txt"))
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [f for f in files if f not in done]
    opp = opponents()
    print(f"{len(files)} proposal files · {len(done)} already labelled · {len(todo)} to do · "
          f"{len(opp)} opponents · {BATTLES} battles each", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        res = pool.score([(f, opp) for f in chunk], battles=BATTLES, conc=50, quiet=True)
        for f in chunk:
            if f in res and res[f]["battles"] > 0:
                done[f] = {k: res[f][k] for k in ("wins", "battles", "win_rate", "se")}
                done[f]["arm"] = Path(f).parent.name
        tmp = OUT + ".tmp"
        json.dump(done, open(tmp, "w"), indent=1); os.replace(tmp, OUT)   # atomic flush
        got = [done[f]["win_rate"] for f in chunk if f in done]
        print(f"  {len(done)}/{len(files)} labelled · chunk mean {sum(got)/max(len(got),1):.3f} · "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    missing = [f for f in files if f not in done]
    if missing: print(f"WARNING {len(missing)} files never returned a result", flush=True)
    print("LABEL_CP_DONE", flush=True)

if __name__ == "__main__":
    main()
