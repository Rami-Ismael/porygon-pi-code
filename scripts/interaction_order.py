"""Interaction order of win rate under four simultaneous edits, against benchmark landscapes.
Pre-registration: docs/interaction-order.md.

Stages (each caches to results/interaction_order_parts/):
  prep     benchmark cubes + VGC cube files (validator only, no battles)
  battle   VGC battles, two replicates per cube
  analyse  Walsh shares, thresholds, mutant cycles -> results/interaction_order.json

Run:  /tmp/vgc-pilot/.venv/bin/python scripts/interaction_order.py prep|battle|analyse
"""
import copy, itertools, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TAG = os.environ.get("OPP_TAG", "")      # "_opp50": parts and output re-battled vs all 50 (2026-09-30)
RES = HERE.parent / "results"; PARTS = RES / f"interaction_order{TAG}_parts"; PARTS.mkdir(parents=True, exist_ok=True)
PROP = Path("/tmp/vgc-pilot/proposals/interaction_order")
K, N_VGC, N_BENCH, CLIMB, BOOT = 4, 16, 24, 100, 2000
BATTLES = {"primary": 384, "weather": 768}
REP_SEEDS = (101, 202)
WEATHER_ANCHORS = ["MB522", "MB489", "MB444"]
WEATHER_CORE = ["pelipper", "charizardmegay", "venusaur", "archaludon"]      # factor order in weather cubes
WEATHER_ABIL = {"drought", "drizzle", "sandstream", "snowwarning", "chlorophyll", "swiftswim", "sandrush",
                "slushrush", "solarpower", "raindish", "dryskin", "hydration", "leafguard", "flowergift",
                "protosynthesis", "forecast", "icebody", "sandveil", "snowcloak", "sandforce", "megasol",
                "cloudnine", "airlock", "orichalcumpulse", "harvest"}
WEATHER_MOVES = {"raindance", "sunnyday", "sandstorm", "snowscape", "weatherball", "solarbeam", "solarblade",
                 "electroshot", "thunder", "hurricane", "blizzard", "growth", "synthesis", "morningsun",
                 "moonlight", "hydrosteam", "auroraveil"}
CUBE = np.array(list(itertools.product([0, 1], repeat=K)))                    # 16 × 4, 0 = edit off
SUBSETS = [S for r in range(K + 1) for S in itertools.combinations(range(K), r)]
CHI = np.array([[np.prod([1 - 2 * (1 - b[i]) for i in S]) if S else 1 for b in CUBE] for S in SUBSETS])  # ±1, on = +1

# ---------------------------------------------------------------- landscapes
def load_run1_module():
    src = (HERE / "ruggedness.py").read_text().split("def main():")[0]
    g = {"__name__": "rug1", "__file__": str(HERE / "ruggedness.py")}
    exec(compile(src, "ruggedness.py", "exec"), g)
    return g

LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
def make_benchmarks(rng):
    n = 25
    J = np.triu(rng.normal(size=(n, n)) * (rng.random((n, n)) < 0.2), 1); h = rng.normal(size=n)
    cl = np.array([rng.choice(n, 3, replace=False) for _ in range(106)]); sg = rng.integers(0, 2, (106, 3)); w = rng.random(106)
    nb = [np.concatenate([[i], rng.choice([j for j in range(n) if j != i], 4, replace=False)]) for i in range(n)]
    tab = rng.random((n, 32))
    def spin(x): s = 2 * x - 1; return float(s @ J @ s + h @ s)
    def maxsat(x): return float(w @ np.any(x[cl] == sg, axis=1))
    def nk(x): return float(np.mean([tab[i, int("".join(map(str, x[nb[i]])), 2)] for i in range(n)]))
    def sphere(x): return float(np.sum(LEVELS[x] ** 2))
    pest = load_run1_module()["pest_control"]
    # (f(x, sim_seed), levels, maximise?)
    return {"sphere": (lambda x, s: sphere(x), 5, False), "spin_glass": (lambda x, s: spin(x), 2, False),
            "max3sat": (lambda x, s: maxsat(x), 2, True), "nk_k4": (lambda x, s: nk(x), 2, True),
            "pest_control": (lambda x, s: pest(x, np.random.default_rng(s)), 5, False)}

def bench_cubes(rng):
    out = {}
    sim = iter(range(10**7))
    for name, (f, lv, maxi) in make_benchmarks(rng).items():
        out[name] = {}
        for start in ("random", "search"):
            cubes = []
            for _ in range(N_BENCH):
                x = rng.integers(0, lv, 25)
                if start == "search":
                    fx = f(x, 9000 + next(sim))
                    for _ in range(CLIMB):
                        j = int(rng.integers(0, 25)); y = x.copy(); y[j] = rng.choice([v for v in range(lv) if v != x[j]])
                        fy = f(y, 9000 + next(sim))
                        if (fy > fx) if maxi else (fy < fx): x, fx = y, fy
                coords = rng.choice(25, K, replace=False)
                vals = [rng.choice([v for v in range(lv) if v != x[c]]) for c in coords]
                reps = []
                for _ in range(2):
                    seed = 5000 + next(sim)            # one simulation seed per replicate, shared by its 16 points
                    ys = []
                    for b in CUBE:
                        y = x.copy()
                        for c, v, on in zip(coords, vals, b):
                            if on: y[c] = v
                        ys.append(f(y, seed))
                    reps.append(ys)
                cubes.append({"reps": reps})
            out[name][start] = cubes
        print("benchmark", name, flush=True)
    return out

# ---------------------------------------------------------------- VGC cubes
def mutation_module():
    src = (HERE / "mutation_closure_test.py").read_text()
    head = src.split("# ---------------------------------------------------------------- run")[0]
    g = {"__name__": "mut", "__file__": str(HERE / "mutation_closure_test.py")}
    sys.argv = [sys.argv[0], "1"]
    exec(compile(head, "mutation_closure_test.py", "exec"), g)
    return g

def write_cube(g, val, base, slots, donors, stem):
    files = []
    for b in CUBE:
        t = copy.deepcopy(base)
        for sl, d, on in zip(slots, donors, b):
            if on: t[sl] = copy.deepcopy(d)
        txt = g["team_to_text"](t)
        err = val(txt)
        if err is not None: return None, err
        files.append(txt)
    out = []
    for b, txt in zip(CUBE, files):
        p = PROP / f"{stem}_{''.join(map(str, b))}.txt"; p.write_text(txt); out.append(str(p))
    return out, None

def draw_donors(g, rng, base, slots, pool):
    norm = g["norm"]; dexnum = g["dexnum"]
    used_num = {dexnum(s["species"]) for s in base}
    used_items = {norm(s["item"]) for s in base if s["item"]}
    donors = []
    for _ in slots:
        cand = [d for d in pool if dexnum(d["species"]) not in used_num
                and (not d["item"] or norm(d["item"]) not in used_items)]
        if not cand: return None
        d = cand[int(rng.integers(0, len(cand)))]
        donors.append(d); used_num.add(dexnum(d["species"]))
        if d["item"]: used_items.add(norm(d["item"]))
    return donors

def weather_free(g, d):
    n = g["norm"]
    return (n(d["ability"]) not in WEATHER_ABIL and not ({n(m) for m in d["moves"]} & WEATHER_MOVES)
            and n(d["species"]) not in {"charizardmegay", "charizard", "pelipper", "venusaur", "venusaurmega",
                                        "torkoal", "politoed", "ninetales", "tyranitar", "hippowdon"})

def slot_key(g, s):
    n = g["norm"]; sp = n(s["species"])
    if sp == "charizard" and n(s["item"]) == "charizarditey": return "charizardmegay"
    return sp

def prep_vgc():
    g = mutation_module()
    from corpus import parse_team
    PROP.mkdir(parents=True, exist_ok=True)
    for f in PROP.glob("*.txt"): f.unlink()
    val = g["Validator"](); rng = np.random.default_rng(1)
    sys.path.insert(0, "/tmp/vgc-pilot/src"); import top50
    rt = "/tmp/vgc-pilot/vgc-bench/teams/reg_mb"
    top = top50.ids(); top50.files()   # raises if one is missing
    pool_all = g["ALL_SLOTS"]; pool_neutral = [d for d in pool_all if weather_free(g, d)]
    plan = {"primary": [], "weather": []}
    for tid in rng.permutation([t for t in top if t not in WEATHER_ANCHORS]):
        if len(plan["primary"]) == N_VGC: break
        base = parse_team(top50.resolve(tid))
        for attempt in range(200):
            slots = sorted(rng.choice(6, K, replace=False).tolist())
            donors = draw_donors(g, rng, base, slots, pool_all)
            if donors is None: continue
            files, err = write_cube(g, val, base, slots, donors, f"p_{tid}")
            if files: break
        else:
            print(f"{tid}: no legal cube", flush=True); continue
        plan["primary"].append({"anchor": str(tid), "slots": slots, "files": files,
                                "out": [base[s]["species"] for s in slots], "in": [d["species"] for d in donors],
                                "team": [s["species"] + ("|" + s["item"] if s["item"] else "") for s in base],
                                "donor_items": [d["item"] for d in donors], "attempts": attempt + 1})
        print(f"primary {tid}: slots {slots} -> {[d['species'] for d in donors]} ({attempt+1} tries)", flush=True)
    for tid in WEATHER_ANCHORS:
        base = parse_team(f"{rt}/{tid}.txt")
        keys = [slot_key(g, s) for s in base]
        slots = [keys.index(k) for k in WEATHER_CORE]
        for attempt in range(200):
            donors = draw_donors(g, rng, base, slots, pool_neutral)
            if donors is None: continue
            files, err = write_cube(g, val, base, slots, donors, f"w_{tid}")
            if files: break
        else:
            raise SystemExit(f"{tid}: no legal weather cube")
        plan["weather"].append({"anchor": tid, "slots": slots, "files": files,
                                "out": [base[s]["species"] for s in slots], "in": [d["species"] for d in donors],
                                "team": [s["species"] + ("|" + s["item"] if s["item"] else "") for s in base],
                                "donor_items": [d["item"] for d in donors], "attempts": attempt + 1})
        print(f"weather {tid}: {[base[s]['species'] for s in slots]} -> {[d['species'] for d in donors]}", flush=True)
    val.close()
    json.dump(plan, open(PARTS / "vgc_plan.json", "w"), indent=1)

# ---------------------------------------------------------------- battles
def battle():
    plan = json.load(open(PARTS / "vgc_plan.json"))
    sys.path.insert(0, "/tmp/vgc-pilot/src"); import pool, top50
    opp = top50.files()      # all 50; raises if one is missing
    out = PARTS / "vgc_battles.json"
    res = json.load(open(out)) if out.exists() else {}
    for grp in ("weather", "primary"):
        files = [f for c in plan[grp] for f in c["files"]]
        for seed in REP_SEEDS:
            key = f"{grp}_{seed}"
            if key in res: continue
            t0 = time.time()
            res[key] = pool.score([(f, opp) for f in files], battles=BATTLES[grp], conc=50,
                                  opp_schedule=opp[:], seed=seed)
            json.dump(res, open(out, "w")); print(f"{key}: {len(files)} teams in {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- statistics
ORDER = np.array([len(S) for S in SUBSETS])

def coef(y): return CHI @ np.asarray(y) / len(CUBE)                      # a_S for every subset

def sq(cube):
    a1, a2 = coef(cube["reps"][0]), coef(cube["reps"][1])
    return a1 * a2                                                          # unbiased a_S² per subset

def shares(cubes):
    p = np.sum([sq(c) for c in cubes], axis=0)
    tot = p[ORDER > 0].sum()
    return {"o1": p[ORDER == 1].sum() / tot, "o2": p[ORDER == 2].sum() / tot,
            "o3plus": p[ORDER >= 3].sum() / tot, "total_var": tot / len(cubes)}

def with_ci(cubes, rng):
    s = shares(cubes)
    bs = [shares([cubes[i] for i in rng.integers(0, len(cubes), len(cubes))]) for _ in range(BOOT)]
    return {k: {"est": float(s[k]), "ci95": [float(x) for x in np.percentile([b[k] for b in bs], [2.5, 97.5])],
                "boot": [float(b[k]) for b in bs]} for k in s}

def cycles(y, trio):
    """0/1 coding, other factor off: ε_ij for the three pairs, ε_ijk for the trio."""
    idx = {tuple(b): i for i, b in enumerate(CUBE)}
    def f(on):
        b = [0] * K
        for i in on: b[i] = 1
        return y[idx[tuple(b)]]
    e3 = sum((-1) ** (3 - r) * f(sub) for r in range(4) for sub in itertools.combinations(trio, r))
    e2 = {f"{i}{j}": f((i, j)) - f((i,)) - f((j,)) + f(()) for i, j in itertools.combinations(trio, 2)}
    return float(e3), {k: float(v) for k, v in e2.items()}

def vgc_cubes(plan, bt, grp):
    out = []
    for c in plan[grp]:
        reps = [[bt[f"{grp}_{s}"][f]["win_rate"] for f in c["files"]] for s in REP_SEEDS]
        n = [[bt[f"{grp}_{s}"][f]["battles"] for f in c["files"]] for s in REP_SEEDS]
        out.append({**c, "reps": reps, "battles": n})
    return out

def cycle_report(c, trio):
    r1, r2 = (np.array(r) for r in c["reps"]); pooled = (r1 + r2) / 2
    n = np.array(c["battles"][0]) + np.array(c["battles"][1])
    e3_1, _ = cycles(r1, trio); e3_2, e2_2 = cycles(r2, trio); e3_p, e2_p = cycles(pooled, trio)
    se_pooled = float(np.sqrt(8 * np.mean(pooled * (1 - pooled) / n)))    # independent-noise SE, conservative under CRN
    se_rep = float(np.sqrt(8 * np.mean(r2 * (1 - r2) / np.array(c["battles"][1]))))
    return {"anchor": c["anchor"], "trio": [c["out"][i] + " → " + c["in"][i] for i in trio],
            "trio_idx": list(trio), "team": c["team"], "slots": c["slots"], "donors": c["in"],
            "eps3_rep1": e3_1, "eps3_rep2": e3_2, "eps3_pooled": e3_p, "se_rep2": se_rep, "se_pooled": se_pooled,
            "eps2_rep2": e2_2, "eps2_pooled": e2_p,
            "points_pooled": {"".join(map(str, b)): float(v) for b, v in zip(CUBE, pooled)}}

def analyse():
    rng = np.random.default_rng(7)
    plan = json.load(open(PARTS / "vgc_plan.json")); bt = json.load(open(PARTS / "vgc_battles.json"))
    bench = json.load(open(PARTS / "benchmarks.json"))
    V = {g: vgc_cubes(plan, bt, g) for g in ("primary", "weather")}
    out = {"design": {"k": K, "vgc_anchors": {g: len(v) for g, v in V.items()}, "battles_per_rep": BATTLES,
                      "rep_seeds": REP_SEEDS, "bench_anchors": N_BENCH, "climb": CLIMB, "boot": BOOT},
           "vgc": {}, "benchmarks": {}}
    for g, cubes in V.items():
        out["vgc"][g] = with_ci(cubes, rng)
        out["vgc"][g]["anchor_mean_win_rate"] = float(np.mean([c["reps"][0][0] for c in cubes] + [c["reps"][1][0] for c in cubes]))
    for name, starts in bench.items():
        out["benchmarks"][name] = {s: with_ci(cubes, rng) for s, cubes in starts.items()}
    vp = np.array(out["vgc"]["primary"]["o3plus"]["boot"]); tv = np.array(out["vgc"]["primary"]["total_var"]["boot"])
    out["signal_ci95"] = [float(x) for x in np.percentile(tv, [2.5, 97.5])]
    out["T1"] = bool(out["vgc"]["primary"]["o3plus"]["ci95"][0] > 0)
    out["vs"] = {}
    for name in bench:
        d = vp - np.array(out["benchmarks"][name]["random"]["o3plus"]["boot"])
        out["vs"][name] = [float(x) for x in np.percentile(d, [2.5, 97.5])]
    out["T2"] = bool(out["vs"]["pest_control"][0] > 0)
    out["sanity"] = {"sphere_o1_ge_0.99": all(out["benchmarks"]["sphere"][s]["o1"]["est"] >= 0.99 for s in ("random", "search")),
                     "spin_o3_zero": all(abs(out["benchmarks"]["spin_glass"][s]["o3plus"]["est"]) < 1e-9 for s in ("random", "search"))}
    out["signal_ok"] = bool(out["signal_ci95"][0] > 0)
    out["verdict"] = ("no measurable variation under four species edits" if not out["signal_ok"] else
                      "more higher-order interaction than Pest Control under four species edits" if out["T1"] and out["T2"] else
                      "measurable higher-order interaction, not more than Pest Control" if out["T1"] else
                      "no measurable interaction beyond pairs under four species edits")
    # mutant cycles
    trios = [t for t in itertools.combinations(range(K), 3)]
    cand = [(abs(cycle_report(c, t)["eps3_rep1"]), ci, t) for ci, c in enumerate(V["primary"]) for t in trios]
    cand.sort(reverse=True)
    picked, seen = [], set()
    for _, ci, t in cand:
        if ci in seen: continue
        picked.append(cycle_report(V["primary"][ci], t)); seen.add(ci)
        if len(picked) == 2: break
    # NOT pre-registered: empirical SE of a pooled ε3 under common random numbers, from replicate
    # disagreement over every anchor × trio of a group: SE = sd(ε3_rep1 − ε3_rep2) / 2
    emp = {}
    for g, cubes in V.items():
        d = [cycles(np.array(c["reps"][0]), t)[0] - cycles(np.array(c["reps"][1]), t)[0] for c in cubes for t in trios]
        emp[g] = float(np.std(d, ddof=1) / 2)
    out["eps3_se_empirical_posthoc"] = emp
    for cr in picked: cr["se_empirical_posthoc"] = emp["primary"]
    wcy = [cycle_report(c, (0, 1, 2)) for c in V["weather"]]
    for cr in wcy: cr["se_empirical_posthoc"] = emp["weather"]
    out["cycles"] = {"weather": wcy, "primary_selected": picked,
                     "primary_all_eps3_rep2": [cycle_report(c, t)["eps3_rep2"] for c in V["primary"] for t in trios]}
    for g in out["vgc"]:
        for k in ("o1", "o2", "o3plus", "total_var"): out["vgc"][g][k].pop("boot")
    for name in out["benchmarks"]:
        for s in out["benchmarks"][name]:
            for k in ("o1", "o2", "o3plus", "total_var"): out["benchmarks"][name][s][k].pop("boot")
    json.dump(out, open(RES / f"interaction_order{TAG}.json", "w"), indent=1)
    def line(lbl, d): print(f"{lbl:28s} o1 {d['o1']['est']:.3f}  o2 {d['o2']['est']:.3f}  o3+ {d['o3plus']['est']:.3f} "
                            f"[{d['o3plus']['ci95'][0]:+.3f},{d['o3plus']['ci95'][1]:+.3f}]")
    for g in V: line("vgc " + g, out["vgc"][g])
    for name in bench:
        for s in bench[name]: line(f"{name} {s}", out["benchmarks"][name][s])
    print("signal", out["signal_ci95"], "T1", out["T1"], "T2", out["T2"], "sanity", out["sanity"])
    print("verdict:", out["verdict"])

if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "prep":
        json.dump(bench_cubes(np.random.default_rng(3)), open(PARTS / "benchmarks.json", "w")); prep_vgc()
    elif stage == "battle": battle()
    elif stage == "analyse": analyse()
