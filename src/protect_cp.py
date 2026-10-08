"""Is Protect worth anything among COMPETENT teams?

The 2,000 battle-labelled random-legal teams sit at 1.5% win rate, a floor where no
team property can show. This builds the same test in the copy-paste band (~0.44)
where the pilot's teams actually win, and adds a matched intervention so the answer
is causal rather than observational.

Arms, all built by pasting whole candidate slots between real Reg M-B teams:
  real     untouched corpus teams -- the 0.47 anchor
  free     one random slot replaced by a random donor slot; the natural copy-paste
           population, binned by Protect count afterwards. This is the observational
           test, the direct analogue of the run on the random-legal pool.
  strip0   base carries exactly 3 Protects; all three Protect slots are replaced by
           donor slots WITHOUT Protect       -> 0 Protect, 3 slot copies
  neutral  base carries exactly 3 Protects; three random slots replaced by random
           donors                            -> ~3 Protect, 3 slot copies
  keep6    base carries exactly 3 Protects; all three non-Protect slots replaced by
           donor slots WITH Protect          -> 6 Protect, 3 slot copies

strip0 / neutral / keep6 share the same base distribution and the same number of
slot copies, so the copy-paste damage is held fixed and only the Protect polarity of
the swap moves. Anything separating them is Protect.

    python src/protect_cp.py gen     # write the teams (Showdown-validated)
    python src/protect_cp.py battle  # 24 battles each vs the top-50 meta
    python src/protect_cp.py report
"""
import json, os, sys, time, collections, math
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
from corpus import load_corpus
from encode import canon
from propose import Validator, team_to_text

ROOT   = Path("/tmp/vgc-pilot/protect_cp")
OUT    = Path.home()/"Documents/code/vgc-team-generator-pilot/results/protect_cp_labels.json"
BATTLES, CHUNK = 24, 200
N = {"real": 150, "free": 600, "strip0": 300, "neutral": 300, "keep6": 300}

def nprot(team):
    return sum(1 for s in team if "Protect" in s["moves"])

def has_prot(slot):
    return "Protect" in slot["moves"]

def gen():
    teams, _ = load_corpus()
    teams = [canon(t) for t in teams if len(t) == 6]
    base3 = [t for t in teams if nprot(t) == 3]
    print(f"corpus {len(teams)} teams · {len(base3)} carry exactly 3 Protect")
    rng, val = np.random.default_rng(0), Validator()
    ROOT.mkdir(parents=True, exist_ok=True)
    for f in ROOT.glob("*.txt"): f.unlink()

    def donor_slot(want):
        """A slot from a random corpus team with the requested Protect polarity."""
        for _ in range(200):
            d = teams[int(rng.integers(0, len(teams)))]
            idx = [i for i in range(6) if has_prot(d[i]) == want]
            if idx: return dict(d[int(rng.choice(idx))])
        return None

    def build(arm):
        if arm == "real":
            return [dict(s) for s in teams[int(rng.integers(0, len(teams)))]]
        if arm == "free":
            b = [dict(s) for s in teams[int(rng.integers(0, len(teams)))]]
            d = teams[int(rng.integers(0, len(teams)))]
            b[int(rng.integers(0, 6))] = dict(d[int(rng.integers(0, 6))])
            return b
        b = [dict(s) for s in base3[int(rng.integers(0, len(base3)))]]
        if arm == "strip0":   slots, want = [i for i in range(6) if has_prot(b[i])],     False
        elif arm == "keep6":  slots, want = [i for i in range(6) if not has_prot(b[i])], True
        else:                 slots, want = list(rng.choice(6, 3, replace=False)),       None
        for i in slots:
            s = donor_slot(bool(rng.integers(0, 2)) if want is None else want)
            if s is None: return None
            b[int(i)] = s
        return b

    made, meta = collections.Counter(), {}
    for arm, want in N.items():
        tries = 0
        while made[arm] < want and tries < want * 60:
            tries += 1
            t = build(arm)
            if t is None: continue
            txt = team_to_text(t)
            if val(txt) is not None: continue
            p = ROOT/f"{arm}_{made[arm]:05d}.txt"
            p.write_text(txt)
            meta[str(p)] = {"arm": arm, "protect": nprot(t)}
            made[arm] += 1
        print(f"  {arm:8} {made[arm]:>4}/{want}  ({made[arm]/max(tries,1):.1%} of attempts valid)")
    val.close()
    json.dump(meta, open(ROOT/"meta.json", "w"), indent=1)
    d = collections.defaultdict(collections.Counter)
    for v in meta.values(): d[v["arm"]][v["protect"]] += 1
    print("\nProtect count by arm")
    for a in N:
        print(f"  {a:8} " + "  ".join(f"{k}:{v}" for k, v in sorted(d[a].items())))

def battle():
    import pool
    from loop import opponents
    meta = json.load(open(ROOT/"meta.json"))
    done = json.load(open(OUT)) if OUT.exists() else {}
    todo = [f for f in sorted(meta) if f not in done]
    opp = opponents()
    print(f"{len(meta)} teams · {len(done)} labelled · {len(todo)} to do · "
          f"{len(opp)} opponents · {BATTLES} battles each", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i+CHUNK]
        res = pool.score([(f, opp) for f in chunk], battles=BATTLES, conc=50, quiet=True)
        for f in chunk:
            if f in res and res[f]["battles"] > 0:
                done[f] = {k: res[f][k] for k in ("wins", "battles", "win_rate")} | meta[f]
        tmp = str(OUT)+".tmp"
        json.dump(done, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
        got = [done[f]["win_rate"] for f in chunk if f in done]
        print(f"  {len(done)}/{len(meta)} · chunk mean {sum(got)/max(len(got),1):.3f} · "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    print("PROTECT_CP_DONE", flush=True)

def _row(name, wrs):
    if not wrs: return f"{name:>22} {'-':>7}"
    a = np.array(wrs)
    se = a.std(ddof=1)/math.sqrt(len(a)) if len(a) > 1 else float("nan")
    return f"{name:>22} {len(a):>6} {a.mean():>9.4f} {se:>8.4f}  [{a.mean()-1.96*se:.4f}, {a.mean()+1.96*se:.4f}]"

def report():
    d = json.load(open(OUT))
    print(f"{'arm':>22} {'teams':>6} {'win rate':>9} {'se':>8}  95% interval")
    print("-"*70)
    for a in N:
        print(_row(a, [v["win_rate"] for v in d.values() if v["arm"] == a]))
    print("\nIntervention: same base pool, same 3 slot copies, only Protect polarity moves")
    g = {a: np.array([v["win_rate"] for v in d.values() if v["arm"] == a])
         for a in ("strip0", "neutral", "keep6")}
    for x, y in (("strip0", "keep6"), ("strip0", "neutral"), ("neutral", "keep6")):
        if not len(g[x]) or not len(g[y]): continue
        se = math.sqrt(g[x].var(ddof=1)/len(g[x]) + g[y].var(ddof=1)/len(g[y]))
        diff = g[y].mean() - g[x].mean()
        print(f"  {y} - {x}: {diff:+.4f}  95% [{diff-1.96*se:+.4f}, {diff+1.96*se:+.4f}]")

    print("\nObservational: the free copy-paste arm, binned by Protect count")
    free = [v for v in d.values() if v["arm"] == "free"]
    b = collections.defaultdict(list)
    for v in free: b[v["protect"]].append(v["win_rate"])
    for k in sorted(b): print(_row(f"{k} Protect", b[k]))
    for k in (2, 3):
        p = [v["win_rate"] for v in free if v["protect"] >= k]
        f = [v["win_rate"] for v in free if v["protect"] <  k]
        if not p or not f: continue
        p, f = np.array(p), np.array(f)
        se = math.sqrt(p.var(ddof=1)/len(p) + f.var(ddof=1)/len(f))
        diff = p.mean() - f.mean()
        print(f"  >= {k} Protect minus < {k}: {diff:+.4f}  95% [{diff-1.96*se:+.4f}, {diff+1.96*se:+.4f}]"
              f"   (n {len(p)} vs {len(f)})")

if __name__ == "__main__":
    {"gen": gen, "battle": battle, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "gen"]()
