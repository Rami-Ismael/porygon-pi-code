"""The clean version of the Protect question: same team, same species, same items,
same spreads -- only the move changes.

protect_cp.py holds the number of slot copies fixed but still swaps whole candidates,
so which archetype of base slot survives differs between its arms. Here nothing moves
except Protect itself: every Protect on a real corpus team is replaced by the move
that species most often carries in the corpus and does not already have. Paired: the
same base team is battled with and without its Protects.

    python src/protect_swap.py gen
    python src/protect_swap.py battle
    python src/protect_swap.py report
"""
import json, os, sys, time, collections, math
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import load_corpus, legal_moves, norm
from encode import canon
from propose import Validator, team_to_text

ROOT = Path("/tmp/vgc-pilot/protect_swap")
OUT  = Path.home()/"Documents/code/vgc-team-generator-pilot/results/protect_swap_labels.json"
BATTLES, CHUNK, N_PAIRS = 24, 200, 300

def gen():
    teams, _ = load_corpus()
    teams = [canon(t) for t in teams if len(t) == 6]
    # what does each species usually run, in this corpus?
    freq = collections.defaultdict(collections.Counter)
    for t in teams:
        for s in t: freq[norm(s["species"])].update(s["moves"])
    rng, val = np.random.default_rng(0), Validator()
    ROOT.mkdir(parents=True, exist_ok=True)
    for f in ROOT.glob("*.txt"): f.unlink()

    def strip(team):
        """Replace every Protect with that species' most common other move."""
        out, swapped = [], 0
        for s in team:
            s = dict(s)
            if "Protect" in s["moves"]:
                have, sp = set(s["moves"]), norm(s["species"])
                legal = {norm(m) for m in legal_moves(s["species"])}
                pick = next((m for m, _ in freq[sp].most_common()
                             if m not in have and norm(m) in legal), None)
                if pick is None: return None, 0
                s["moves"] = [pick if m == "Protect" else m for m in s["moves"]]
                swapped += 1
            out.append(s)
        return out, swapped

    meta, made, tries = {}, 0, 0
    order = rng.permutation(len(teams))
    while made < N_PAIRS and tries < len(teams):
        base = teams[int(order[tries])]; tries += 1
        if sum(1 for s in base if "Protect" in s["moves"]) < 3: continue
        alt, n = strip(base)
        if alt is None: continue
        tb, ta = team_to_text(base), team_to_text(alt)
        if val(tb) is not None or val(ta) is not None: continue
        for tag, txt, np_ in (("keep", tb, n), ("strip", ta, 0)):
            p = ROOT/f"{tag}_{made:05d}.txt"; p.write_text(txt)
            meta[str(p)] = {"arm": tag, "pair": made, "protect": np_, "n_swapped": n}
        made += 1
    val.close()
    json.dump(meta, open(ROOT/"meta.json", "w"), indent=1)
    print(f"{made} paired teams from {tries} corpus teams examined "
          f"({2*made} files, mean {np.mean([v['n_swapped'] for v in meta.values() if v['arm']=='keep']):.2f} "
          f"Protects removed per team)")

def battle():
    import pool
    from loop import opponents
    meta = json.load(open(ROOT/"meta.json"))
    done = json.load(open(OUT)) if OUT.exists() else {}
    todo = [f for f in sorted(meta) if f not in done]
    opp = opponents()
    print(f"{len(meta)} teams · {len(todo)} to do · {len(opp)} opponents · {BATTLES} each", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i+CHUNK]
        res = pool.score([(f, opp) for f in chunk], battles=BATTLES, conc=50, quiet=True)
        for f in chunk:
            if f in res and res[f]["battles"] > 0:
                done[f] = {k: res[f][k] for k in ("wins", "battles", "win_rate")} | meta[f]
        tmp = str(OUT)+".tmp"; json.dump(done, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
        got = [done[f]["win_rate"] for f in chunk if f in done]
        print(f"  {len(done)}/{len(meta)} · chunk mean {sum(got)/max(len(got),1):.3f} · "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    print("PROTECT_SWAP_DONE", flush=True)

def report():
    d = json.load(open(OUT))
    by = collections.defaultdict(dict)
    for v in d.values(): by[v["pair"]][v["arm"]] = v
    pairs = [(p["keep"]["win_rate"], p["strip"]["win_rate"], p["keep"]["n_swapped"])
             for p in by.values() if "keep" in p and "strip" in p]
    k = np.array([a for a, _, _ in pairs]); s = np.array([b for _, b, _ in pairs])
    d_ = k - s
    se = d_.std(ddof=1)/math.sqrt(len(d_))
    print(f"paired teams: {len(pairs)}   battles: {2*len(pairs)*BATTLES}")
    print(f"  with Protect     {k.mean():.4f} ± {k.std(ddof=1)/math.sqrt(len(k)):.4f}")
    print(f"  Protect swapped  {s.mean():.4f} ± {s.std(ddof=1)/math.sqrt(len(s)):.4f}")
    print(f"  PAIRED difference {d_.mean():+.4f}  95% [{d_.mean()-1.96*se:+.4f}, {d_.mean()+1.96*se:+.4f}]")
    print(f"  teams hurt by removal: {(d_>0).mean():.1%}")
    print("\nby number of Protects removed")
    g = collections.defaultdict(list)
    for a, b, n in pairs: g[n].append(a-b)
    for n in sorted(g):
        a = np.array(g[n]); e = a.std(ddof=1)/math.sqrt(len(a)) if len(a) > 1 else float("nan")
        print(f"  {n} removed  n={len(a):<4} diff {a.mean():+.4f} ± {e:.4f}")

if __name__ == "__main__":
    {"gen": gen, "battle": battle, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "gen"]()
