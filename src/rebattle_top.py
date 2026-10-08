"""Re-battle the best teams the guidance campaigns found, at label noise that
can actually rank them.

The 24-battle labels that drive the loop carry SE ~0.10 at the top, so the
run's best teams (labelled 0.67-0.79) are indistinguishable from each other and
from lucky 0.5s. This takes the top TOP_N distinct teams across every gradloop
and gradguide cell, re-battles each at BATTLES vs the same top-50 meta pool
(SE ~0.035), and prints the ranked list with species - the campaign's actual
deliverable: candidate counter-teams to the Reg M-B meta with trustworthy
win rates. The 200-team real-corpus anchor mean under this evaluation is 0.4625.

    python rebattle_top.py
"""
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import parse_team_text, norm
import activesearch as A

RESULTS = "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/rebattle_top.json"
SRC = ["/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/gradloop.json",
       "/Users/ramiismael/Documents/code/vgc-team-generator-pilot/results/gradguide.json"]
TOP_N = 16
BATTLES = 192
OUT = Path("/tmp/vgc-pilot/rebattle_top")


def harvest():
    rows = []
    for path in SRC:
        r = json.load(open(path))
        for tag, g in (r.get("gens") or r.get("cells")).items():
            for p, y in zip(g["pastes"], g["y"]):
                rows.append((float(y), tag, p))
    rows.sort(key=lambda t: -t[0])
    seen, top = set(), []
    for y, tag, p in rows:
        k = p.strip()
        if k in seen:
            continue
        seen.add(k); top.append((y, tag, p))
        if len(top) >= TOP_N:
            break
    return top


def main():
    top = harvest()
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (y, tag, p) in enumerate(top):
        f = OUT / f"{i:02d}.txt"
        f.write_text(p)
        files.append(str(f))
    opp = A.opponents()
    print(f"re-battling {len(files)} teams x {BATTLES} battles "
          f"vs {len(opp)} meta opponents …", flush=True)
    t0 = time.perf_counter()
    res = A.pool.score([(f, opp) for f in files], battles=BATTLES, conc=50)
    out = []
    for i, (y24, tag, p) in enumerate(top):
        r = res.get(files[i])
        if not r or not r["battles"]:
            continue
        team = parse_team_text(p)
        out.append(dict(rank=0, label24=y24, source=tag,
                        win_rate=r["win_rate"], se=r["se"], battles=r["battles"],
                        species=[norm(s["species"]) for s in team], paste=p))
    out.sort(key=lambda d: -d["win_rate"])
    for i, d in enumerate(out):
        d["rank"] = i + 1
    json.dump(dict(battles=BATTLES, anchor_real_mean=0.4625, teams=out),
              open(RESULTS, "w"), indent=1)
    print(f"  done in {(time.perf_counter()-t0)/60:.1f} min -> {RESULTS}\n", flush=True)
    print(f"{'#':>2s} {'wr':>7s} {'se':>6s} {'label24':>8s} {'source':14s} species", flush=True)
    for d in out:
        print(f"{d['rank']:2d} {d['win_rate']:7.4f} {d['se']:6.4f} {d['label24']:8.3f} "
              f"{d['source']:14s} {', '.join(d['species'])}", flush=True)
    print("\nREBATTLE_DONE", flush=True)


if __name__ == "__main__":
    main()
