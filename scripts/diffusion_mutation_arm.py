"""Mutation operator inside deep-surrogate MAP-Elites: hand-written A-F vs uniform refill vs v3 diffusion refill.
Pre-registration: docs/diffusion-mutation-arm.md. Everything except the operator is scripts/dsame_experiment.py's
dsame arm (surrogate, archive, insert rule, budget, pilot bin edges).

    python scripts/diffusion_mutation_arm.py probe     # operators only: valid rates and seconds, no battles
    python scripts/diffusion_mutation_arm.py smoke     # one tiny generation per arm, 8 teams battled, nothing saved
    python scripts/diffusion_mutation_arm.py run       # 3 arms x 3 seeds x 6 generations (resumable per generation)
    python scripts/diffusion_mutation_arm.py verify    # 16 new battles per top-50 opponent for each arm-seed's top 8
    python scripts/diffusion_mutation_arm.py heldout   # the same teams vs ranks 51-100
    python scripts/diffusion_mutation_arm.py report
(run with /tmp/vgc-pilot/.venv/bin/python; own Showdown servers on 8401-8406)
"""
import copy, importlib.util, json, os, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
_spec = importlib.util.spec_from_file_location("dsame_experiment", HERE / "dsame_experiment.py")
DS = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(DS)
A, MDB, HO, DM = DS.A, DS.MDB, DS.HO, DS.DM
import foundation as FD
from showdown_validator import VGC_BENCH
import lossdown as L
import diffusion as D
import hps_generate
from encode import canon, NF, NSLOT
from corpus import parse_team_text, dex_entry

def _dex(species):
    """Cosmetic formes (24 Alcremie / Vivillon patterns in the v3 vocabulary) carry no base stats: use the base species'."""
    e = dex_entry(species)
    return dex_entry(e["baseSpecies"]) if e and "baseStats" not in e and e.get("baseSpecies") else e
DS.dex_entry = _dex                                          # DS.ob_ratio's lookup; unchanged for every other species

OUT = REPO / "results" / "diffusion_mutation_arm.json"
DSAME = REPO / "results" / "dsame_experiment.json"          # pilot turns + bin edges, reused
WORK = Path("/tmp/vgc-pilot/dmut")
PORTS = [8401, 8402, 8403, 8404, 8405, 8406]                # this run's own servers (8123-8129 are shared)
A.pool.REPO = str(VGC_BENCH)   # as cem_v3
ARMS = ("hand", "uniform", "diffusion")
SEEDS, GENS, BATTLE, BATTLES, GRID, TOPV, VERIFY_CELL = DS.SEEDS, DS.GENS, DS.BATTLE, DS.BATTLES, DS.GRID, DS.TOPV, DS.VERIFY_CELL
CHILDREN, PBATCH, MAX_ATTEMPTS = DS.CHILDREN, 200, 20 * DS.CHILDREN
ASK, GUIDE, TEMP_SETS = 0.5, 2.0, 0.5
P_WHOLE, KMAX = 1 / 6, 4
HELDOUT_BATTLES = 384
HAND_SIZE = {"A": "1 field", "B": "1 field", "C": "1 field", "D": "1 field", "E": "spread only", "F": "whole Pokemon"}

# ---------------------------------------------------------------- B and C: one mask sampler, two refills
class Kernel:
    def __init__(self):
        self.G = FD.Gen("v3"); self.V, self.T = self.G.V, self.G.tables
        self.EMPTY = {"item": self.T.EIT, "move": self.T.EMV}       # arm A never empties an item or a move

    def masks(self, rng, n):
        out = []
        for _ in range(n):
            s = int(rng.integers(0, NSLOT))
            if rng.random() < P_WHOLE:
                out.append((s, [s * NF + j for j in range(NF)], True))
            else:
                k = int(rng.integers(1, KMAX + 1))
                out.append((s, sorted(s * NF + int(j) for j in rng.choice(np.arange(1, NF), k, replace=False)), False))
        return out

    @torch.no_grad()
    def refill(self, parents, masks, uniform, rng):
        teams = [canon(copy.deepcopy(p)) for p in parents]
        X = torch.as_tensor(np.stack([self.V.encode(t) for t in teams]), dtype=torch.long, device=L.DEV)
        seqs = []
        for i, (_, cols, _) in enumerate(masks):
            X[i, cols] = 0; seqs.append([c for c in D.ORDER if c in cols])
        n = len(teams)
        w = torch.full((n,), ASK, device=L.DEV); wn = torch.full((n,), float("nan"), device=L.DEV)
        for k in range(max(map(len, seqs))):
            rows = [i for i in range(n) if len(seqs[i]) > k]
            t = (X == 0).sum(1).float() / L.COLS
            al = self.T.allowed(X)
            if not uniform: lc, lu = self.G.f(X, t, w), self.G.f(X, t, wn)
            for key in sorted({L.KEY_OF_COL[seqs[i][k]] for i in rows}):
                R = [i for i in rows if L.KEY_OF_COL[seqs[i][k]] == key]
                cs = [seqs[i][k] for i in R]
                Ri = torch.tensor(R, device=L.DEV); Ji = torch.tensor([L.COLS_BY_KEY[key].index(c) for c in cs], device=L.DEV)
                ok = al[key][Ri, Ji].clone()
                e = self.EMPTY.get(key)
                if e is not None: ok[:, e] = False
                ok[~ok.any(1)] = True; ok[:, 0] = False
                lg = torch.zeros(ok.shape, device=L.DEV) if uniform else GUIDE * lc[key][Ri, Ji] + (1 - GUIDE) * lu[key][Ri, Ji]
                lg = lg.masked_fill(~ok, -float("inf")) / (1.0 if key == "species" else TEMP_SETS)
                X[Ri, torch.tensor(cs, device=L.DEV)] = torch.multinomial(torch.softmax(lg, -1), 1).squeeze(1)
        X = X.cpu().numpy()
        dec = lambda i, c: self.V.decode_field(c, int(X[i, c]))
        name = lambda v: self.G.look.get(v, v)
        out = []
        for i, (s, cols, whole) in enumerate(masks):
            sl = teams[i][s]
            for c in cols:
                j = c % NF
                if j in (0, 1, 2, 7): sl[{0: "species", 1: "ability", 2: "item", 7: "nature"}[j]] = name(dec(i, c))
            if any(3 <= c % NF <= 6 for c in cols):
                sl["moves"] = [name(dec(i, s * NF + j)) for j in range(3, 7) if dec(i, s * NF + j) not in ("", "[MASK]")]
            if whole:
                pl = self.G.spreads.get(dec(i, s * NF))
                sl["evs"] = dict(pl[int(rng.integers(0, len(pl)))]) if pl else hps_generate.sample_spread(rng)
            out.append(teams[i])
        return out

def mutate(arm, parents, kern, mut, rng, nrng, st, sizes):
    """Children as paste text; counts no-ops and edit sizes."""
    if arm == "hand":
        ops = [o for o in mut["COND"] if o[0] in "ABCDEF"]            # the pre-registered operators A-F only
        out = []
        for p in parents:
            child = copy.deepcopy(p); op = rng.choice(ops); sizes[HAND_SIZE[op[0]]] += 1; sizes["op " + op[0]] += 1
            if mut["COND"][op](child): st["noop"] += 1; continue
            out.append(mut["team_to_text"](child))
        return out
    ms = kern.masks(nrng, len(parents))
    for _, cols, whole in ms: sizes["whole Pokemon" if whole else f"{len(cols)} field" + ("s" if len(cols) > 1 else "")] += 1
    return [mut["team_to_text"](t) for t in kern.refill(parents, ms, arm == "uniform", nrng)]

# ---------------------------------------------------------------- one generation (DS.gen_dsame with the operator swapped)
def gen(ctx, arm, kern, mut, lab_t, lab_y, lab_turn, seen, edges, seed, g, val, children=CHILDREN, battle_n=BATTLE):
    rng = random.Random(seed * 100 + g); nrng = np.random.default_rng(seed * 100 + g); mut["rng"] = nrng
    F = A.Feats(lab_t + ctx.corpus)
    deep = DS.Deep(F.dim, seed * 100 + g)
    t_train = deep.fit(F.mat(lab_t), np.asarray(lab_y, float), np.asarray(lab_turn, float))
    t0 = time.perf_counter()
    cell = lambda r, t: (int(np.clip(np.digitize(r, edges["ratio"]), 0, GRID - 1)),
                         int(np.clip(np.digitize(t, edges["turns"]), 0, GRID - 1)))
    arch = {}
    pw, pt = deep.predict(F.mat(lab_t))
    for t, w_, tu in zip(lab_t, pw, pt):
        c = cell(DS.ob_ratio(t), tu)
        if c not in arch or w_ > arch[c][0]: arch[c] = (float(w_), t)
    st, sizes = Counter(), Counter()
    made = inserted = 0; t_op = t_val = 0.0
    while made < children and st["attempts"] < MAX_ATTEMPTS:
        el = list(arch.values())
        parents = [rng.choice(el)[1] for _ in range(PBATCH)]; st["attempts"] += PBATCH
        a = time.perf_counter(); cands = mutate(arm, parents, kern, mut, rng, nrng, st, sizes); t_op += time.perf_counter() - a
        a = time.perf_counter(); batch = []
        for txt in cands:
            st["submitted"] += 1
            if val(txt) is not None: continue
            st["valid"] += 1
            if HO.is_heldout_text(txt): st["heldout_dropped"] += 1; continue
            t = parse_team_text(txt)
            if len(t) != 6: st["parse_short"] += 1; continue
            batch.append(t)
        t_val += time.perf_counter() - a
        made += len(batch)
        if not batch: continue
        bw, bt = deep.predict(F.mat(batch))
        for t, w_, tu in zip(batch, bw, bt):
            c = cell(DS.ob_ratio(t), tu)
            if c not in arch or w_ > arch[c][0]: arch[c] = (float(w_), t); inserted += 1
    t_inner = time.perf_counter() - t0
    elites = sorted(arch.values(), key=lambda z: -z[0])
    pastes, ks = [], set()
    for w_, t in elites:
        p = mut["team_to_text"](t); k = DS.key(p)
        if k in seen or k in ks: continue
        ks.add(k); pastes.append(p)
        if len(pastes) == battle_n: break
    return pastes, dict(surrogate_train_seconds=t_train, inner_seconds=t_inner, operator_seconds=t_op, validator_seconds=t_val,
                        children=made, inserted=inserted, archive_cells=len(arch), counts=dict(st), edit_sizes=dict(sizes),
                        predicted_mean=float(np.mean([w_ for w_, _ in elites[:battle_n]])))

def battle(ctx, pastes, d):
    files = DS.write_files(pastes, d)
    res = A.pool.score([(f, ctx.opp) for f in files], battles=BATTLES, conc=50, ports=PORTS)
    return [(res.get(f) or {}) for f in files]

# ---------------------------------------------------------------- stages
def load():
    return json.load(open(OUT)) if OUT.exists() else {"config": dict(
        arms=ARMS, seeds=SEEDS, gens=GENS, battle=BATTLE, battles=BATTLES, children=CHILDREN, parent_batch=PBATCH, grid=GRID,
        ask=ASK, guidance=GUIDE, temp_sets=TEMP_SETS, p_whole=P_WHOLE, kmax=KMAX, members=FD.V3_MEMBERS, ports=PORTS,
        topv=TOPV, verify_cell=VERIFY_CELL, heldout_battles=HELDOUT_BATTLES), "runs": {}, "verify": {}, "heldout": {}}

def save(out):
    tmp = str(OUT) + ".tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, OUT)

def setup(ctx):
    pilot = json.load(open(DSAME))["pilot"]
    pt = pilot["turns"]
    if len(pt) == len(ctx.start_keep): pt = [t for t, k in zip(pt, ctx.start_keep) if k]
    assert len(pt) == len(ctx.start)
    return pilot["edges"], [t if t is not None else np.nan for t in pt]

def stage_probe(ctx):
    """Operators only, on the 313 starting teams: valid rate and seconds per 1,000 children. No battles."""
    kern, val = Kernel(), hps_generate.Validator(); mut = DS.mutation_module(5)
    try:
        for arm in ARMS:
            rng = random.Random(5); nrng = np.random.default_rng(5); mut["rng"] = nrng; st, sizes = Counter(), Counter()
            t0 = time.perf_counter(); cands = []
            for _ in range(5):
                cands += mutate(arm, [rng.choice(ctx.start) for _ in range(PBATCH)], kern, mut, rng, nrng, st, sizes)
            t_op = time.perf_counter() - t0
            ok = sum(val(c) is None for c in cands)
            print(f"{arm}: {len(cands)} children of {5 * PBATCH} ({st['noop']} no-ops) · valid {ok / max(len(cands), 1):.3f} · "
                  f"operator {t_op:.1f}s · sizes {dict(sizes)}", flush=True)
            print(cands[0][:400])
    finally:
        val.close()

def stage_run(ctx, out, smoke=False):
    edges, start_turn = setup(ctx)
    kern, val = Kernel(), hps_generate.Validator()
    seeds = SEEDS if not smoke else [7]
    try:
        for seed in seeds:
            mut = DS.mutation_module(seed)
            for arm in ARMS:
                rk = f"{arm}_{seed}"
                run = out["runs"].setdefault(rk, {"gens": {}}) if not smoke else {"gens": {}}
                lab_t, lab_y, lab_turn = list(ctx.start), list(ctx.start_y), list(start_turn)
                seen = {DS.key(p) for p in ctx.start_pastes}
                for g in range(1, (GENS if not smoke else 1) + 1):
                    gk = str(g)
                    if gk in run["gens"]:
                        r = run["gens"][gk]
                        lab_t += [parse_team_text(p) for p in r["pastes"]]; lab_y += r["y"]
                        lab_turn += [t if t is not None else np.nan for t in r["turns"]]; seen |= {DS.key(p) for p in r["pastes"]}
                        continue
                    t0 = time.perf_counter()
                    pastes, info = gen(ctx, arm, kern, mut, lab_t, lab_y, lab_turn, seen, edges, seed, g, val,
                                       *((400, 8) if smoke else ()))
                    tb = time.perf_counter()
                    r = battle(ctx, pastes, WORK / ("smoke" if smoke else rk) / f"g{g}")
                    info["battle_seconds"] = time.perf_counter() - tb
                    keep = [i for i, x in enumerate(r) if x.get("battles")]
                    rec = dict(pastes=[pastes[i] for i in keep], y=[r[i]["win_rate"] for i in keep],
                               turns=[r[i].get("turns") for i in keep], info=info, seconds=time.perf_counter() - t0)
                    c = info["counts"]
                    print(f"{rk} g{g}: n {len(keep)} mean {np.mean(rec['y']):.3f} max {max(rec['y']):.3f} · valid "
                          f"{c.get('valid', 0) / max(c.get('submitted', 1), 1):.3f} · {rec['seconds']:.0f}s (inner {info['inner_seconds']:.0f}, "
                          f"battles {info['battle_seconds']:.0f}) · {info['edit_sizes']}", flush=True)
                    if smoke: continue
                    run["gens"][gk] = rec; save(out)
                    lab_t += [parse_team_text(p) for p in rec["pastes"]]; lab_y += rec["y"]
                    lab_turn += [t if t is not None else np.nan for t in rec["turns"]]; seen |= {DS.key(p) for p in rec["pastes"]}
    finally:
        val.close()

def top_teams(run):
    allp, seenk = [], set()
    for y, p in sorted(((y, p) for g in run["gens"].values() for p, y in zip(g["pastes"], g["y"])), key=lambda z: -z[0]):
        k = DS.key(p)
        if k not in seenk: seenk.add(k); allp.append((y, p))
    return allp[:TOPV]

def stage_verify(ctx, out):
    con = MDB.connect(); pid = MDB.get_policy(con); val = MDB.Validator()
    try:
        for rk, run in out["runs"].items():
            if rk in out["verify"] or len(run["gens"]) < GENS: continue
            top = top_teams(run)
            with con: ids = [MDB.add_team(con, p, f"dmut:{rk}", val)[0] for _, p in top]
            note, cols = f"dmut verify {rk}", MDB.set_members(con, "top50")
            by_target = {}                                   # VERIFY_CELL NEW battles per cell (see dsame amendment 3)
            for t in ids:
                have = max([MDB.battles_done(con, pid, t, c) for c in cols if c != t] or [0])
                by_target.setdefault(have + VERIFY_CELL, []).append(t)
            for target, ts in sorted(by_target.items()):
                MDB.ensure(con, pid, ts, cols, target, seed=4242, note=note)
            rows = []
            for (y, p), t in zip(top, ids):
                f, n = DS.fresh_score(con, pid, t, note)
                rows.append(dict(search=y, fresh=f, fresh_battles=n, team_id=t, paste=p))
            out["verify"][rk] = rows; save(out)
            print(rk, [round(v["fresh"], 3) if v["fresh"] is not None else None for v in rows], flush=True)
    finally:
        val.close()

def stage_heldout(ctx, out):
    for rk, rows in out["verify"].items():
        if rk in out["heldout"]: continue
        files = DS.write_files([r["paste"] for r in rows], WORK / rk / "heldout")
        res = HO.evaluate(files, battles=HELDOUT_BATTLES, conc=50, ports=PORTS)
        out["heldout"][rk] = [dict(win_rate=res.get(f, {}).get("win_rate"), battles=res.get(f, {}).get("battles")) for f in files]
        save(out); print(rk, [round(r["win_rate"], 3) if r["win_rate"] is not None else None for r in out["heldout"][rk]], flush=True)

def boot(vals_by_seed, rng, B=10_000):
    """Mean over teams, resampled within seed (teams are the cluster: one value per team)."""
    arrs = [np.asarray(v, float) for v in vals_by_seed if len(v)]
    return np.array([np.concatenate([a[rng.integers(0, len(a), len(a))] for a in arrs]).mean() for _ in range(B)])

def stage_report(ctx, out):
    rng = np.random.default_rng(0); summ = {}
    fresh = {a: [[v["fresh"] for v in out["verify"].get(f"{a}_{s}", []) if v["fresh"] is not None] for s in SEEDS] for a in ARMS}
    held = {a: [[v["win_rate"] for v in out["heldout"].get(f"{a}_{s}", []) if v["win_rate"] is not None] for s in SEEDS] for a in ARMS}
    for a in ARMS:
        rks = [f"{a}_{s}" for s in SEEDS if f"{a}_{s}" in out["runs"]]
        gens = [g for rk in rks for g in out["runs"][rk]["gens"].values()]
        cnt, sizes = Counter(), Counter()
        for g in gens: cnt.update(g["info"]["counts"]); sizes.update(g["info"]["edit_sizes"])
        div = [DM.measure([parse_team_text(p) for g in out["runs"][rk]["gens"].values() for p in g["pastes"]], ctx.ref) for rk in rks]
        f_all = [x for v in fresh[a] for x in v]
        best = max(((v["fresh"], v) for s in SEEDS for v in out["verify"].get(f"{a}_{s}", []) if v["fresh"] is not None),
                   key=lambda z: z[0], default=(None, None))[1]
        bs = boot(fresh[a], rng) if f_all else None
        h_all = [x for v in held[a] for x in v]; hb = boot(held[a], rng) if h_all else None
        summ[a] = dict(
            verified_mean=float(np.mean(f_all)) if f_all else None, verified_ci95=[float(x) for x in np.percentile(bs, [2.5, 97.5])] if f_all else None,
            verified_n=len(f_all), verified_by_seed={s: float(np.mean(v)) for s, v in zip(SEEDS, fresh[a]) if v},
            heldout_mean=float(np.mean(h_all)) if h_all else None, heldout_ci95=[float(x) for x in np.percentile(hb, [2.5, 97.5])] if h_all else None,
            best_verified=dict(fresh=best["fresh"], search=best["search"], team_id=best["team_id"],
                               species=[t["species"] for t in parse_team_text(best["paste"])]) if best else None,
            distinct_compositions_per_seed=[int(d["composition"]["unique_count"]) for d in div],
            distinct_compositions=float(np.mean([d["composition"]["unique_count"] for d in div])) if div else None,
            valid_rate_before_gate=cnt["valid"] / max(cnt["submitted"], 1),
            noop_share=cnt["noop"] / max(cnt["attempts"], 1), counts=dict(cnt),
            edit_size_distribution={k: v / sum(c for kk, c in sizes.items() if not kk.startswith("op ")) for k, v in sizes.items() if not k.startswith("op ")},
            hand_operator_counts={k: v for k, v in sizes.items() if k.startswith("op ")} or None,
            seconds_per_gen=float(np.mean([g["seconds"] for g in gens])) if gens else None,
            inner_seconds_per_gen=float(np.mean([g["info"]["inner_seconds"] for g in gens])) if gens else None,
            operator_seconds_per_gen=float(np.mean([g["info"]["operator_seconds"] for g in gens])) if gens else None,
            battle_seconds_per_gen=float(np.mean([g["info"]["battle_seconds"] for g in gens])) if gens else None,
            search_gen_means={gk: float(np.mean([np.mean(out["runs"][rk]["gens"][gk]["y"]) for rk in rks]))
                              for gk in map(str, range(1, GENS + 1)) if rks and all(gk in out["runs"][rk]["gens"] for rk in rks)})
    def delta(x, y, src):
        if not all(src[x]) or not all(src[y]): return None
        d = boot(src[x], np.random.default_rng(1)) - boot(src[y], np.random.default_rng(2))
        pt = float(np.mean([v for s in src[x] for v in s]) - np.mean([v for s in src[y] for v in s]))
        ci = [float(v) for v in np.percentile(d, [2.5, 97.5])]
        return dict(delta=pt, ci95=ci, call="stronger" if ci[0] > 0 else "weaker" if ci[1] < 0 else "no detectable difference")
    summ["contrasts"] = {name: dict(top50=delta(x, y, fresh), heldout=delta(x, y, held))
                         for name, (x, y) in {"diffusion_minus_hand": ("diffusion", "hand"), "diffusion_minus_uniform": ("diffusion", "uniform"),
                                              "uniform_minus_hand": ("uniform", "hand")}.items()}
    p = summ["contrasts"]["diffusion_minus_hand"]
    if p["top50"]:
        h = p["heldout"]["delta"] if p["heldout"] else None
        summ["verdict"] = ("PASS: the diffusion kernel finds stronger teams than the hand-written operators at this budget"
                           if p["top50"]["ci95"][0] > 0 and h is not None and h > 0 else
                           "FAIL (worse): the diffusion kernel finds weaker teams than the hand-written operators" if p["top50"]["ci95"][1] < 0 else
                           "FAIL (no detectable difference): the diffusion kernel does not earn its compute at this budget")
    out["summary"] = summ; save(out); print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    stage = sys.argv[1]
    ctx = DS.Ctx()
    if stage == "probe": stage_probe(ctx); sys.exit()
    if stage == "smoke": stage_run(ctx, load(), smoke=True); sys.exit()
    out = load()
    {"run": lambda: stage_run(ctx, out), "verify": lambda: stage_verify(ctx, out),
     "heldout": lambda: stage_heldout(ctx, out), "report": lambda: stage_report(ctx, out)}[stage]()
