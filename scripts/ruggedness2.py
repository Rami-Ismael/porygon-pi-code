"""Ruggedness per edit type, from meta and search-found teams, against Pest Control and a
gridded sphere.  Pre-registration: docs/ruggedness2.md (extends docs/ruggedness.md, run 1).

Stages (each caches to results/ruggedness2_parts/):
  prep     benchmarks + VGC edit files (validator only, no battles)
  battle   VGC battles for the new teams; needs run 1's results/ruggedness.json
  analyse  statistics + thresholds -> results/ruggedness2.json

Run:  /tmp/vgc-pilot/.venv/bin/python scripts/ruggedness2.py prep|battle|analyse
"""
import copy, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TAG = os.environ.get("OPP_TAG", "")      # "_opp50": parts, run-1 input and output re-battled vs all 50 (2026-09-30)
RES = HERE.parent / "results"; PARTS = RES / f"ruggedness2{TAG}_parts"; PARTS.mkdir(parents=True, exist_ok=True)
RUN1 = RES / f"ruggedness{TAG}.json"
PROP = Path("/tmp/vgc-pilot/proposals/ruggedness2")
N_RAND, N_SEARCH, N_EDIT, N_REP, CLIMB = 24, 16, 15, 3, 100
BATTLES, PER_OP, SEED, BOOT = 384, 3, 1, 2000
OPS = ["A_move_swap", "B_ability_swap", "C_item_swap", "D_alignment_swap", "E_spread_copy", "F_candidate_copy"]
LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

# ---------------------------------------------------------------- benchmarks
def load_run1_module():
    src = (HERE / "ruggedness.py").read_text().split("def main():")[0]
    g = {"__name__": "rug1", "__file__": str(HERE / "ruggedness.py")}
    exec(compile(src, "ruggedness.py", "exec"), g)
    return g

def sphere(x): return float(np.sum(LEVELS[x] ** 2))
def sphere_step(x, j):
    y = x.copy()
    if y[j] == 0: y[j] = 1
    elif y[j] == 4: y[j] = 3
    else: y[j] += 1 if RNG.random() < .5 else -1
    return y
def other_level(x, j):
    y = x.copy(); y[j] = RNG.choice([v for v in range(5) if v != x[j]]); return y

def climb(x, ev):
    """First-improvement hill climb, CLIMB single-edit proposals, minimising (noisy) f."""
    fx = ev(x)
    for _ in range(CLIMB):
        j = int(RNG.integers(0, x.size)); y = other_level(x, j); fy = ev(y)
        if fy < fx: x, fx = y, fy
    return x

def bench_landscape(ev, edit, reps_noisy):
    out = {}
    for start, n in (("random", N_RAND), ("search", N_SEARCH)):
        anchors = []
        for _ in range(n):
            x = RNG.integers(0, 5, 25)
            if start == "search": x = climb(x, ev)
            f0 = ev(x)
            reps = [ev(x) for _ in range(N_REP)] if reps_noisy else [f0] * N_REP
            edits = []
            for _ in range(N_EDIT):
                j = int(RNG.integers(0, 25)); edits.append({"op": "one_edit", "f": ev(edit(x, j))})
            anchors.append({"x": x.tolist(), "f": f0, "reps": reps, "edits": edits})
        out[start] = anchors
    return out

def prep_benchmarks():
    global RNG
    g = load_run1_module(); pest = g["pest_control"]
    RNG = np.random.default_rng(SEED)
    sim = iter(range(10**7))
    ev_p = lambda x: pest(x, np.random.default_rng(5000 + next(sim)))
    b = {"pest_control": bench_landscape(ev_p, other_level, True),
         "sphere_step": bench_landscape(sphere, sphere_step, False),
         "sphere_level": bench_landscape(sphere, other_level, False)}
    json.dump(b, open(PARTS / "benchmarks.json", "w"))
    print("benchmarks written", flush=True)

# ---------------------------------------------------------------- VGC edits
def mutation_module():
    src = (HERE / "mutation_closure_test.py").read_text()
    head = src.split("# ---------------------------------------------------------------- run")[0]
    g = {"__name__": "mut", "__file__": str(HERE / "mutation_closure_test.py")}
    sys.argv = [sys.argv[0], str(SEED)]
    exec(compile(head, "mutation_closure_test.py", "exec"), g)
    g["rng"] = np.random.default_rng(SEED)
    return g

def make_edits(g, val, base, stem, ops):
    txt = g["team_to_text"](base); edits = []
    for op in ops:
        got = tries = 0
        while got < PER_OP and tries < 60:
            tries += 1
            child = copy.deepcopy(base)
            if g["COND"][op](child): continue
            ct = g["team_to_text"](child)
            if ct == txt or val(ct) is not None: continue
            ef = PROP / f"{stem}_{op[0]}{got}.txt"; ef.write_text(ct)
            edits.append({"op": op, "file": str(ef)}); got += 1
    return edits

def prep_vgc():
    g = mutation_module()
    from corpus import parse_team, parse_team_text
    PROP.mkdir(parents=True, exist_ok=True)
    for f in PROP.glob("*.txt"): f.unlink()
    val = g["Validator"]()
    run1_dir = Path("/tmp/vgc-pilot/proposals/ruggedness")
    meta = []
    for a in range(N_RAND):
        af = run1_dir / f"a{a:02d}.txt"
        meta.append({"file": str(af), "edits": make_edits(g, val, parse_team(af), f"m{a:02d}", ["F_candidate_copy"])})
    top = json.load(open(RES / "rebattle_top.json"))["teams"]
    search = []
    for k, t in enumerate(top[:N_SEARCH]):
        base = parse_team_text(t["paste"]); txt = g["team_to_text"](base)
        err = val(txt)
        if err is not None: print(f"search anchor {k} invalid: {err}", flush=True); continue
        af = PROP / f"s{k:02d}.txt"; af.write_text(txt)
        search.append({"rank": t["rank"], "file": str(af), "edits": make_edits(g, val, base, f"s{k:02d}", OPS)})
        print(f"search anchor {k:02d}: {len(search[-1]['edits'])} edits", flush=True)
    val.close()
    json.dump({"meta_new": meta, "search": search}, open(PARTS / "vgc_plan.json", "w"), indent=1)
    print("meta F edits:", sum(len(m["edits"]) for m in meta), flush=True)

# ---------------------------------------------------------------- battles
def battle():
    plan = json.load(open(PARTS / "vgc_plan.json"))
    sys.path.insert(0, "/tmp/vgc-pilot/src"); import pool, top50
    opp = top50.files()      # all 50; raises if one is missing
    run = lambda files, seed: pool.score([(f, opp) for f in files], battles=BATTLES, conc=50,
                                         opp_schedule=opp[:], seed=seed)
    out = PARTS / "vgc_battles.json"
    res = json.load(open(out)) if out.exists() else {}
    jobs = [("anchor", [a["file"] for a in plan["search"]], 101),
            ("edits", [e["file"] for grp in ("meta_new", "search") for a in plan[grp] for e in a["edits"]], 202)]
    jobs += [(f"rep{s}", [a["file"] for a in plan["search"]], s) for s in (202, 303, 404)]
    for key, files, seed in jobs:
        if key in res: continue
        t0 = time.time(); res[key] = run(files, seed)
        json.dump(res, open(out, "w")); print(f"{key}: {len(files)} teams in {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- statistics
def vgc_anchors():
    r1 = json.load(open(RUN1))["vgc"]["anchors"]
    plan = json.load(open(PARTS / "vgc_plan.json")); bt = json.load(open(PARTS / "vgc_battles.json"))
    meta = []
    for a, new in zip(r1, plan["meta_new"]):
        assert a["file"] == new["file"]
        eds = [dict(e) for e in a["edits"]]
        eds += [{"op": e["op"], "file": e["file"], "f": bt["edits"][e["file"]]["win_rate"],
                 "n": bt["edits"][e["file"]]["battles"]} for e in new["edits"]]
        meta.append({"file": a["file"], "f": a["f"], "n": a["n"], "reps": a["reps"], "edits": eds})
    search = []
    for a in plan["search"]:
        f = a["file"]
        search.append({"file": f, "rank": a["rank"], "f": bt["anchor"][f]["win_rate"], "n": bt["anchor"][f]["battles"],
                       "reps": [bt[f"rep{s}"][f]["win_rate"] for s in (202, 303, 404)],
                       "edits": [{"op": e["op"], "file": e["file"], "f": bt["edits"][e["file"]]["win_rate"],
                                  "n": bt["edits"][e["file"]]["battles"]} for e in a["edits"]]})
    return {"meta": meta, "search": search}

def moments(groups, start, op):
    """E[d^2] for this start/op, E[r^2] for this start, pooled var_f over all starts."""
    A = groups[start]
    d2 = [(e["f"] - a["f"]) ** 2 for a in A for e in a["edits"] if op is None or e["op"] == op]
    r2 = np.mean([(r - a["f"]) ** 2 for a in A for r in a["reps"]])
    allr2 = np.mean([(r - a["f"]) ** 2 for G in groups.values() for a in G for r in a["reps"]])
    fa = np.array([a["f"] for G in groups.values() for a in G])
    var_f = fa.var(ddof=1) - allr2 / 2
    return np.mean(d2), r2, var_f

def stat(groups, start, op):
    d2, r2, var_f = moments(groups, start, op)
    s2 = max(d2 - r2, 0.0) / var_f
    # sensitivity, NOT pre-registered: var_f from this start type's anchors only
    var_w = np.var([a["f"] for a in groups[start]], ddof=1) - r2 / 2
    return {"rho": 1 - s2 / 2, "delta_star": float(np.sqrt(max(d2 - r2, 0.0))), "s": float(np.sqrt(s2)),
            "E_d2": float(d2), "E_r2": float(r2), "var_f": float(var_f),
            "var_f_within_start": float(var_w),
            "rho_within_start_var": float(1 - max(d2 - r2, 0.0) / (2 * var_w)) if var_w > 0 else None}

def resample(groups, rng):
    return {k: [G[i] for i in rng.integers(0, len(G), len(G))] for k, G in groups.items()}

def beyond_noise(A, op):
    fe = [abs(e["f"] - a["f"]) > 1.96 * np.sqrt(a["f"] * (1 - a["f"]) / a["n"] + e["f"] * (1 - e["f"]) / e["n"])
          for a in A for e in a["edits"] if e["op"] == op]
    fr = [abs(r - a["f"]) > 1.96 * np.sqrt(2 * a["f"] * (1 - a["f"]) / a["n"]) for a in A for r in a["reps"]]
    return float(np.mean(fe)), float(np.mean(fr))

def analyse():
    b = json.load(open(PARTS / "benchmarks.json")); V = vgc_anchors()
    rng = np.random.default_rng(7)
    boot_V = [resample(V, rng) for _ in range(BOOT)]
    boot_P = [resample(b["pest_control"], rng) for _ in range(BOOT)]
    out = {"design": {"battles": BATTLES, "edits_per_op": PER_OP, "bench_edits": N_EDIT, "replicates": N_REP,
                      "climb_proposals": CLIMB, "boot": BOOT,
                      "vgc_anchors": {k: len(v) for k, v in V.items()}}, "benchmarks": {}, "vgc": {}}
    pest_rho = {}
    for name, L in b.items():
        out["benchmarks"][name] = {}
        boots = boot_P if name == "pest_control" else [resample(L, rng) for _ in range(BOOT)]
        for start in L:
            s = stat(L, start, None); bs = [stat(g, start, None)["rho"] for g in boots]
            s["rho_ci95"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
            out["benchmarks"][name][start] = s
            if name == "pest_control": pest_rho[start] = bs
    pair = {"meta": "random", "search": "search"}
    for start in V:
        out["vgc"][start] = {}
        for op in OPS:
            s = stat(V, start, op); bs = np.array([stat(g, start, op)["rho"] for g in boot_V])
            fe, fr = beyond_noise(V[start], op)
            bdiff = [np.subtract(*beyond_noise(g[start], op)) for g in boot_V]
            diff = bs - np.array(pest_rho[pair[start]])
            s.update({"rho_ci95": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
                      "frac_beyond_noise": fe, "frac_replicates_beyond_noise": fr,
                      "frac_diff_ci95": [float(x) for x in np.percentile(bdiff, [2.5, 97.5])],
                      "rho_minus_pest_ci95": [float(x) for x in np.percentile(diff, [2.5, 97.5])],
                      "edits": sum(e["op"] == op for a in V[start] for e in a["edits"])})
            s["T1"] = bool(s["frac_diff_ci95"][0] > 0 and s["delta_star"] >= 0.05)
            s["T2"] = bool(s["rho_minus_pest_ci95"][1] < 0)
            out["vgc"][start][op] = s
            print(f"{start:6s} {op:18s} rho {s['rho']:+.3f} [{s['rho_ci95'][0]:+.3f},{s['rho_ci95'][1]:+.3f}] "
                  f"D* {s['delta_star']:.3f} beyond {fe:.2f} vs {fr:.2f}  T1 {s['T1']} T2 {s['T2']}", flush=True)
        out["vgc"][start]["anchor_mean_win_rate"] = float(np.mean([a["f"] for a in V[start]]))
    out["sanity_sphere_step_rho_ge_0.95"] = all(v["rho"] >= 0.95 for v in out["benchmarks"]["sphere_step"].values())
    t1 = [(s, o) for s in V for o in OPS if out["vgc"][s][o]["T1"]]
    t12 = [(s, o) for (s, o) in t1 if out["vgc"][s][o]["T2"]]
    out["verdict"] = ("rugged beyond battle noise and beyond Pest Control" if t12 else
                      "beyond battle noise, not more rugged than Pest Control" if t1 else "claim dropped")
    out["T1_cells"], out["T1_and_T2_cells"] = t1, t12
    for name in b: print(name, {k: round(v["rho"], 3) for k, v in out["benchmarks"][name].items()})
    print("verdict:", out["verdict"], t12 or t1)
    json.dump(out, open(RES / f"ruggedness2{TAG}.json", "w"), indent=1)

if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "prep": prep_benchmarks(); prep_vgc()
    elif stage == "battle": battle()
    elif stage == "analyse": analyse()
