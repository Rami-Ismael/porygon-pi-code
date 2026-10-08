"""Is win rate rugged under a single legal edit?  Measured against battle noise, and
compared with one combinatorial Bayesian-optimisation benchmark (COMBO's Pest Control)
and one smooth continuous function (Hartmann-6).  Pre-registration: docs/ruggedness.md.

Same design on all three landscapes:
  24 anchors; 15 single edits per anchor; each noisy anchor re-measured 3 times.
  VGC       anchors = real Reg M-B corpus teams outside the top-50 opponent pool;
            edits   = conditional legal operators A-E from mutation_closure_test.py
                      (move, ability, item, alignment, spread copy), 3 of each,
                      every child checked by Showdown's TeamValidator;
            f       = win rate vs the top-50 meta, behaviour-cloning policy both sides,
                      BATTLES battles, fixed ordered opponent schedule for every team.
  Pest      anchors = uniform over 5^25; edit = one stage set to another of its 5 values;
            f       = COMBO _pest_control_score (100 simulated trajectories), fresh seed.
  Hartmann  anchors = uniform on [0,1]^6; edit = one coordinate moved by +-STEP.

Noise: an edit delta and a replicate delta are built the same way (different RNG seed
from the anchor's), so the replicate deltas are the null distribution for edit deltas.

Run:  /tmp/vgc-pilot/.venv/bin/python scripts/ruggedness.py
"""
import copy, json, os, random, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / os.environ.get("RUG_OUT", "ruggedness.json")
N_ANCHOR = int(os.environ.get("RUG_ANCHORS", 24)); BATTLES = int(os.environ.get("RUG_BATTLES", 384))
PER_OP, N_REP, STEP = 3, 3, 0.01
OPS = ["A_move_swap", "B_ability_swap", "C_item_swap", "D_alignment_swap", "E_spread_copy"]
SEED = 0

# ---------------------------------------------------------------- smooth: Hartmann-6
H_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
H_A = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
H_P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]])
def hartmann6(x):
    return float(-np.sum(H_ALPHA * np.exp(-np.sum(H_A * (x - H_P) ** 2, axis=1))))

# ---------------------------------------------------------------- combinatorial: Pest Control
# Ported from COMBO (QUVA-Lab/COMBO, experiments/test_functions/multiple_categorical.py,
# _pest_control_score); only change is an explicit RNG instead of the global np.random.
def _pest_spread(curr, spread_rate, control_rate, apply_control):
    if apply_control:
        return (1.0 - control_rate) * curr
    return spread_rate * (1 - curr) + curr
def pest_control(x, rng):
    U, n_stages, n_sim = 0.1, x.size, 100
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}
    payed, above = 0.0, 0.0
    curr = rng.beta(1.0, 30.0, size=n_sim)
    for i in range(n_stages):
        spread_rate = rng.beta(1.0, 17.0 / 3.0, size=n_sim)
        if x[i] > 0:
            control_rate = rng.beta(1.0, control_beta[x[i]], size=n_sim)
            nxt = _pest_spread(curr, spread_rate, control_rate, True)
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            price = control_price[x[i]] * (1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            nxt = _pest_spread(curr, spread_rate, 0, False); price = 0
        payed += price; above += np.mean(curr > U); curr = nxt
    return float(payed + above)

def benchmarks():
    rng = np.random.default_rng(SEED)
    out = {}
    # Hartmann-6 (deterministic: replicates are exact, noise 0)
    anchors = []
    for a in range(N_ANCHOR):
        x = rng.random(6); f0 = hartmann6(x); edits = []
        for e in range(len(OPS) * PER_OP):
            y = x.copy(); j = int(rng.integers(0, 6))
            y[j] = np.clip(y[j] + (STEP if rng.random() < .5 else -STEP), 0, 1)
            edits.append({"f": hartmann6(y), "se": 0.0, "coord": j})
        anchors.append({"f": f0, "se": 0.0, "reps": [f0] * N_REP, "edits": edits})
    out["hartmann6"] = anchors
    # Pest Control, 25 stages x 5 choices
    anchors = []
    sim_seed = iter(range(10**6))
    ev = lambda x: pest_control(x, np.random.default_rng(1000 + next(sim_seed)))
    for a in range(N_ANCHOR):
        x = rng.integers(0, 5, 25); f0 = ev(x); edits = []
        for e in range(len(OPS) * PER_OP):
            y = x.copy(); j = int(rng.integers(0, 25))
            y[j] = rng.choice([v for v in range(5) if v != x[j]])
            edits.append({"f": ev(y), "stage": j})
        anchors.append({"f": f0, "reps": [ev(x) for _ in range(N_REP)], "edits": edits})
    out["pest_control"] = anchors
    return out

# ---------------------------------------------------------------- VGC
def vgc():
    src = (HERE / "mutation_closure_test.py").read_text()
    head = src.split("# ---------------------------------------------------------------- run")[0]
    g = {"__name__": "mut", "__file__": str(HERE / "mutation_closure_test.py")}
    sys.argv = [sys.argv[0], str(SEED)]
    exec(compile(head, "mutation_closure_test.py", "exec"), g)   # operators + tables, no run
    teams, names, team_to_text, Validator, COND = g["teams"], g["names"], g["team_to_text"], g["Validator"], g["COND"]
    g["rng"] = np.random.default_rng(SEED)

    sys.path.insert(0, "/tmp/vgc-pilot/src"); import top50
    top = top50.ids()
    top_names = {f"{i}.txt" for i in top}

    d = Path("/tmp/vgc-pilot/proposals/ruggedness"); d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.txt"): f.unlink()
    val = Validator()
    order = np.random.default_rng(SEED).permutation(len(teams))
    anchors = []
    for k in order:
        if len(anchors) == N_ANCHOR: break
        if names[k] in top_names: continue
        base = teams[k]; txt = team_to_text(base)
        if val(txt) is not None: continue
        a = len(anchors); af = d / f"a{a:02d}.txt"; af.write_text(txt)
        edits = []
        for op in OPS:
            got = tries = 0
            while got < PER_OP and tries < 60:
                tries += 1
                child = copy.deepcopy(base)
                if COND[op](child): continue                    # no-op draw
                ct = team_to_text(child)
                if ct == txt or val(ct) is not None: continue    # unchanged or illegal
                ef = d / f"a{a:02d}_{op[0]}{got}.txt"; ef.write_text(ct)
                edits.append({"op": op, "file": str(ef)}); got += 1
        anchors.append({"name": names[k], "file": str(af), "edits": edits})
        print(f"anchor {a:02d} {names[k]}: {len(edits)} edits", flush=True)
    val.close()
    return battle_vgc(anchors)

def battle_vgc(anchors):
    """Battle anchors, edits and replicates vs all 50 top-50 teams; fills f / n / reps in place."""
    sys.path.insert(0, "/tmp/vgc-pilot/src")
    import pool, top50
    opp = top50.files()                  # all 50; raises if one is missing
    sched = opp[:]                       # one fixed ordered schedule for every team
    def run(files, seed):
        return pool.score([(f, opp) for f in files], battles=BATTLES, conc=50,
                          opp_schedule=sched, seed=seed)
    t0 = time.time()
    r_anchor = run([a["file"] for a in anchors], 101)
    r_edit = run([e["file"] for a in anchors for e in a["edits"]], 202)
    r_reps = [run([a["file"] for a in anchors], s) for s in (202, 303, 404)[:N_REP]]
    for a in anchors:
        a["f"] = r_anchor[a["file"]]["win_rate"]; a["n"] = r_anchor[a["file"]]["battles"]
        a["reps"] = [r[a["file"]]["win_rate"] for r in r_reps]
        for e in a["edits"]:
            e["f"] = r_edit[e["file"]]["win_rate"]; e["n"] = r_edit[e["file"]]["battles"]
    print(f"battles done in {time.time()-t0:.0f}s", flush=True)
    return anchors, len(opp)

# ---------------------------------------------------------------- statistics
def neighbour_corr(anchors, noise_var_of):
    """Noise-corrected lag-1 neighbour correlation, rho = 1 - (E[d^2] - E[r^2]) / (2 var_f).
    d = edit - anchor, r = replicate - anchor (same construction, so E[r^2] is the noise part);
    var_f = var(anchor values) minus the mean single-measurement noise variance."""
    d2 = np.mean([(e["f"] - a["f"]) ** 2 for a in anchors for e in a["edits"]])
    r2 = np.mean([(r - a["f"]) ** 2 for a in anchors for r in a["reps"]])
    fa = np.array([a["f"] for a in anchors])
    var_f = fa.var(ddof=1) - np.mean([noise_var_of(a) for a in anchors])
    return 1 - (d2 - r2) / (2 * var_f), d2, r2, var_f

def summarise(anchors, noise_var_of, boot=2000):
    rho, d2, r2, var_f = neighbour_corr(anchors, noise_var_of)
    rng = np.random.default_rng(1); bs = []
    for _ in range(boot):
        s = [anchors[i] for i in rng.integers(0, len(anchors), len(anchors))]
        bs.append(neighbour_corr(s, noise_var_of)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"rho": float(rho), "rho_ci95": [float(lo), float(hi)], "mean_sq_edit_delta": float(d2),
            "mean_sq_replicate_delta": float(r2), "var_f_noise_corrected": float(var_f),
            "rms_edit_over_sd_f": float(np.sqrt(max(d2 - r2, 0) / var_f)) if var_f > 0 else None}

def main():
    res = {"design": {"anchors": N_ANCHOR, "edits_per_anchor": len(OPS) * PER_OP, "replicates": N_REP,
                      "battles": BATTLES, "hartmann_step": STEP, "seed": SEED}}
    b = benchmarks()
    res["hartmann6"] = {"anchors": b["hartmann6"], "summary": summarise(b["hartmann6"], lambda a: 0.0)}
    pest_noise = float(np.mean([np.var([a["f"]] + a["reps"], ddof=1) for a in b["pest_control"]]))
    res["pest_control"] = {"anchors": b["pest_control"], "noise_var": pest_noise,
                           "summary": summarise(b["pest_control"], lambda a: pest_noise)}
    for k in ("hartmann6", "pest_control"):
        print(k, json.dumps(res[k]["summary"]), flush=True)
    res["vgc"] = vgc_block(*vgc())
    print("vgc", json.dumps(res["vgc"]["summary"]), flush=True)
    OUT.parent.mkdir(exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)

def vgc_block(anchors, n_opp):
    binom = lambda a: a["f"] * (1 - a["f"]) / a["n"]
    res = {}
    res["vgc"] = {"anchors": anchors, "opponents": n_opp, "summary": summarise(anchors, binom)}
    # per-edit test against battle noise
    flags = []
    for a in anchors:
        for e in a["edits"]:
            se = np.sqrt(a["f"] * (1 - a["f"]) / a["n"] + e["f"] * (1 - e["f"]) / e["n"])
            e["delta"] = e["f"] - a["f"]; e["se"] = float(se); flags.append(abs(e["delta"]) > 1.96 * se)
    rep_d = [r - a["f"] for a in anchors for r in a["reps"]]
    res["vgc"]["summary"].update({
        "edits": len(flags), "edits_beyond_noise": int(np.sum(flags)),
        "frac_edits_beyond_noise": float(np.mean(flags)),
        "replicates_beyond_noise": int(np.sum([abs(r - a["f"]) > 1.96 * np.sqrt(2 * a["f"] * (1 - a["f"]) / a["n"])
                                               for a in anchors for r in a["reps"]])),
        "replicate_delta_sd": float(np.std(rep_d, ddof=1)),
        "anchor_mean_win_rate": float(np.mean([a["f"] for a in anchors]))})
    return res["vgc"]

def rebattle():
    """Re-battle run 1's existing VGC files (no new prep) and recompute the VGC block;
    benchmarks are copied. Added 2026-09-30: run 1 had battled 36 of the top-50."""
    src = HERE.parent / "results" / "ruggedness.json"
    if OUT.resolve() == src.resolve(): raise SystemExit("set RUG_OUT; rebattle never overwrites ruggedness.json")
    res = json.load(open(src))
    anchors = [{"name": a["name"], "file": a["file"], "edits": [{"op": e["op"], "file": e["file"]} for e in a["edits"]]}
               for a in res["vgc"]["anchors"]]
    res["vgc"] = vgc_block(*battle_vgc(anchors))
    res["design"]["rebattled_from"] = "ruggedness.json"
    print("vgc", json.dumps(res["vgc"]["summary"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)

if __name__ == "__main__":
    rebattle() if sys.argv[1:] == ["rebattle"] else main()
