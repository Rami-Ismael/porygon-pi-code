"""Deep-surrogate MAP-Elites vs the existing search loop (gradloop `combined`), same start, same battles.
Pre-registration: docs/dsame-experiment.md.

    python scripts/dsame_experiment.py validate   # surrogate validation, battle-free
    python scripts/dsame_experiment.py smoke      # tiny search, no battles
    python scripts/dsame_experiment.py pilot      # turns for the 328 starting teams -> bin edges
    python scripts/dsame_experiment.py run        # both arms x seeds (resumable per arm-seed-generation)
    python scripts/dsame_experiment.py verify     # fresh battles for each arm's top 8
    python scripts/dsame_experiment.py report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import copy, importlib.util, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))   # diversity_metrics lives only in the repo
import diffusion as D
import activesearch as A
import gradguide as GG
import matchup_db as MDB
import diversity_metrics as DM
from corpus import parse_team, parse_team_text, dex_entry, norm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "results" / "dsame_experiment.json"
WORK = Path("/tmp/vgc-pilot/dsame")
CKPT = REPO / "results" / "temperature_p0.pt"
SEEDS, GENS, BATTLE, BATTLES, PROPOSE = [11, 22, 33], 6, 128, 24, 512
GRID, CHILDREN, ENSEMBLE, EPOCHS, TOPV, VERIFY_CELL = 12, 8000, 3, 150, 8, 16
MODE, LAM = "gloss", 108.0

# ---------------------------------------------------------------- measures
NAT = {"adamant": ("atk", "spa"), "bold": ("def", "atk"), "brave": ("atk", "spe"), "calm": ("spd", "atk"),
       "careful": ("spd", "spa"), "gentle": ("spd", "def"), "hasty": ("spe", "def"), "impish": ("def", "spa"),
       "jolly": ("spe", "spa"), "lax": ("def", "spd"), "lonely": ("atk", "def"), "mild": ("spa", "def"),
       "modest": ("spa", "atk"), "naive": ("spe", "spd"), "naughty": ("atk", "spd"), "quiet": ("spa", "spe"),
       "rash": ("spa", "spd"), "relaxed": ("def", "spe"), "sassy": ("spd", "spe"), "timid": ("spe", "atk")}
SPK = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}

def ob_ratio(team):
    """Static offense/bulk ratio: mean over slots of ln(max(Atk,SpA)) - 1/2 ln(HP*(Def+SpD)/2),
    level-50 Champions stats, base form."""
    vals = []
    for s in team:
        e = dex_entry(s["species"]); b = e["baseStats"]
        up, dn = NAT.get(norm(s.get("nature") or ""), (None, None))
        st = {}
        for k in ("hp", "atk", "def", "spa", "spd", "spe"):
            sp = int((s.get("evs") or {}).get(SPK[k], 0))
            if k == "hp": st[k] = b[k] + sp + 75
            else: st[k] = math.floor((b[k] + sp + 20) * (1.1 if k == up else 0.9 if k == dn else 1.0))
        vals.append(math.log(max(st["atk"], st["spa"])) - 0.5 * math.log(st["hp"] * (st["def"] + st["spd"]) / 2))
    return float(np.mean(vals))

# ---------------------------------------------------------------- deep surrogate
class Deep:
    def __init__(self, dim, seed):
        torch.manual_seed(seed)
        mk = lambda: torch.nn.Sequential(torch.nn.Linear(dim, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                         torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 2))
        self.nets = [mk() for _ in range(ENSEMBLE)]
        self.params = sum(p.numel() for n in self.nets for p in n.parameters())
    def fit(self, X, yw, yt):
        """yw win rate; yt turns (nan = unknown, masked)."""
        X = torch.tensor(X, dtype=torch.float32)
        tm = np.nanmean(yt) if np.isfinite(yt).any() else 0.0
        ts = np.nanstd(yt) if np.isfinite(yt).any() else 1.0
        self.tm, self.ts = float(tm), float(ts or 1.0)
        Y = torch.tensor(np.stack([yw, (yt - self.tm) / self.ts], 1), dtype=torch.float32)
        M = torch.isfinite(Y); Y = torch.nan_to_num(Y)
        t0 = time.perf_counter()
        for net in self.nets:
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            net.train()
            for ep in range(EPOCHS):
                perm = torch.randperm(len(X))
                for i in range(0, len(X), 128):
                    b = perm[i:i + 128]
                    loss = (((net(X[b]) - Y[b]) ** 2) * M[b]).sum() / M[b].sum().clamp(min=1)
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
        return time.perf_counter() - t0
    @torch.no_grad()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        P = torch.stack([n(X) for n in self.nets]).mean(0).numpy()
        return P[:, 0], P[:, 1] * self.ts + self.tm

# ---------------------------------------------------------------- shared context
def mutation_module(seed):
    src = (HERE / "mutation_closure_test.py").read_text()
    head = src.split("# ---------------------------------------------------------------- run")[0]
    g = {"__name__": "mut", "__file__": str(HERE / "mutation_closure_test.py")}
    argv = sys.argv; sys.argv = [argv[0], str(seed)]
    exec(compile(head, "mutation_closure_test.py", "exec"), g); sys.argv = argv
    return g

class Ctx:
    def __init__(self):
        cf = A.load_corpus_files(); self.corpus = [t for _, t in cf]
        self.V = A.Vocab(self.corpus); L = A.Legality(self.corpus); self.C = D.Constraints(self.V, L)
        self.grid = np.stack([self.V.encode(t) for t in self.corpus])
        self.look, self.spreads = A.decode_tables(self.corpus)
        self.opp = A.opponents()
        as1 = json.load(open(A.RESULTS))
        self.start_files = list(as1["anchor"])
        self.start = [parse_team(f) for f in as1["anchor"]] + [parse_team_text(p) for p in as1["gens"]["gen0"]["pastes"]]
        self.start_y = [float(y) for y in as1["anchor"].values()] + [float(y) for y in as1["gens"]["gen0"]["y"]]
        self.start_pastes = [MDB.canonical_paste(open(f).read()) for f in as1["anchor"]] + list(as1["gens"]["gen0"]["pastes"])
        self.ref = DM.Reference(self.corpus, scope="corpus", format_id=MDB.FORMAT)

def key(paste): return MDB.canon_hash(MDB.canonical_paste(paste))

def write_files(pastes, d):
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.txt"): f.unlink()
    out = []
    for i, p in enumerate(pastes):
        f = d / f"{i:04d}.txt"; f.write_text(p); out.append(str(f))
    return out

def battle(ctx, pastes, d, seed):
    files = write_files(pastes, d)
    res = A.pool.score([(f, ctx.opp) for f in files], battles=BATTLES, conc=50)
    return [(res.get(f) or {}) for f in files]

# ---------------------------------------------------------------- arms
def gen_baseline(ctx, lab_t, lab_y, seed, g, val, outdir):
    import hpsdiffusion as H
    t0 = time.perf_counter(); F, wv = GG.fit_ridge(lab_t, lab_y, ctx.corpus); G = GG.Guide(F, wv, ctx.V, D.DEV)
    t_fit = time.perf_counter() - t0
    k = max(A.ELITE_MIN, int(A.RHO * len(lab_y)))
    el = np.argsort(-np.asarray(lab_y))[:k]
    t0 = time.perf_counter(); model = A.resteer(ctx.V, [lab_t[i] for i in el], seed=seed); t_ft = time.perf_counter() - t0
    t0 = time.perf_counter()
    files, teams, pastes, validity, _ = GG.propose_guided(model, ctx.C, G, ctx.V, ctx.look, ctx.spreads, val, MODE, LAM,
                                                          PROPOSE, outdir / "proposals", 1000 * seed + g)
    t_prop = time.perf_counter() - t0
    mu = F.mat(teams) @ wv
    sel = [int(i) for i in np.argsort(-mu)[:BATTLE]]
    return [pastes[i] for i in sel], dict(ridge_seconds=t_fit, resteer_seconds=t_ft, proposal_seconds=t_prop,
                                          validity=validity, proposals=len(teams))

def gen_dsame(ctx, lab_t, lab_y, lab_turn, seen, edges, seed, g, val, mut):
    rng = random.Random(seed * 100 + g); mut["rng"] = np.random.default_rng(seed * 100 + g)
    F = A.Feats(lab_t + ctx.corpus)
    deep = Deep(F.dim, seed * 100 + g)
    t_train = deep.fit(F.mat(lab_t), np.asarray(lab_y, float), np.asarray(lab_turn, float))
    t0 = time.perf_counter()
    cell = lambda r, t: (int(np.clip(np.digitize(r, edges["ratio"]), 0, GRID - 1)),
                         int(np.clip(np.digitize(t, edges["turns"]), 0, GRID - 1)))
    arch = {}
    pw, pt = deep.predict(F.mat(lab_t))
    for t, w_, tu in zip(lab_t, pw, pt):
        c = cell(ob_ratio(t), tu)
        if c not in arch or w_ > arch[c][0]: arch[c] = (float(w_), t)
    made = tried = inserted = 0
    while made < CHILDREN:
        batch = []
        for _ in range(200):
            tried += 1
            child = copy.deepcopy(rng.choice(list(arch.values()))[1])
            op = rng.choice([o for o in mut["COND"] if o[0] in "ABCDEF"])   # pre-registered operators A-F only
            if mut["COND"][op](child): continue
            txt = mut["team_to_text"](child)
            if val(txt) is None: batch.append(parse_team_text(txt))
        made += len(batch)
        if not batch: continue
        bw, bt = deep.predict(F.mat(batch))
        for t, w_, tu in zip(batch, bw, bt):
            c = cell(ob_ratio(t), tu)
            if c not in arch or w_ > arch[c][0]: arch[c] = (float(w_), t); inserted += 1
    t_inner = time.perf_counter() - t0
    elites = sorted(arch.values(), key=lambda z: -z[0])
    pastes, ks = [], set()
    for w_, t in elites:
        p = mut["team_to_text"](t)
        k = key(p)
        if k in seen or k in ks: continue
        ks.add(k); pastes.append(p)
        if len(pastes) == BATTLE: break
    return pastes, dict(surrogate_train_seconds=t_train, surrogate_params=deep.params, epochs=EPOCHS, ensemble=ENSEMBLE,
                        inner_seconds=t_inner, children=made, attempts=tried, inserted=inserted,
                        archive_cells=len(arch), predicted_mean=float(np.mean([w_ for w_, _ in elites[:BATTLE]])))

# ---------------------------------------------------------------- stages
def load():
    return json.load(open(OUT)) if OUT.exists() else {"config": {}, "pilot": None, "runs": {}, "verify": {}, "validate": None}

def save(out):
    tmp = str(OUT) + ".tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, OUT)

def stage_validate(ctx, out):
    spearman = A.spearman
    lab_t, lab_y = ctx.start, ctx.start_y
    t0 = time.perf_counter(); F, wv = GG.fit_ridge(lab_t, lab_y, ctx.corpus); t_ridge = time.perf_counter() - t0
    Fd = A.Feats(lab_t + ctx.corpus); deep = Deep(Fd.dim, 1)
    t_deep = deep.fit(Fd.mat(lab_t), np.asarray(lab_y, float), np.full(len(lab_y), np.nan))
    ridge = lambda ts: F.mat(ts) @ wv
    deepf = lambda ts: deep.predict(Fd.mat(ts))[0]
    all_t, all_y, srcs = GG.all_labels()
    held = [(t, y) for t, y, s in zip(all_t, all_y, srcs) if s == "as2"]   # as2_anchor = the training anchors re-battled: not held out
    ht, hy = [t for t, _ in held], [y for _, y in held]
    spec = importlib.util.spec_from_file_location("rug2", HERE / "ruggedness2.py"); rug2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rug2)
    V = rug2.vgc_anchors()
    pairs = []
    for grp in V.values():
        for a in grp:
            at = parse_team(a["file"])
            for e in a["edits"]:
                if not os.path.exists(e["file"]): continue
                se = math.sqrt(a["f"] * (1 - a["f"]) / a["n"] + e["f"] * (1 - e["f"]) / e["n"])
                pairs.append((at, parse_team(e["file"]), e["f"] - a["f"], se, e["op"]))
    res = {"ridge_fit_seconds": t_ridge, "deep_fit_seconds": t_deep, "deep_params": deep.params,
           "heldout_n": len(hy), "edit_pairs": len(pairs)}
    for name, f in (("ridge", ridge), ("deep", deepf)):
        ph = f(ht); res[f"{name}_heldout_spearman"] = float(spearman(list(ph), list(hy)))
        pa = f([p[0] for p in pairs]); pe = f([p[1] for p in pairs])
        dp, dm = pe - pa, np.array([p[2] for p in pairs]); se = np.array([p[3] for p in pairs])
        big = np.abs(dm) > 1.96 * se
        res[f"{name}_edit_delta_spearman"] = float(spearman(list(dp), list(dm)))
        res[f"{name}_edit_sign_agreement_beyond_noise"] = float(np.mean(np.sign(dp[big]) == np.sign(dm[big]))) if big.any() else None
        res[f"{name}_edit_n_beyond_noise"] = int(big.sum())
        res[f"{name}_pred_delta_sd"] = float(np.std(dp))
    res["measured_delta_sd"] = float(np.std([p[2] for p in pairs]))
    out["validate"] = res; save(out); print(json.dumps(res, indent=1))

def stage_pilot(ctx, out):
    if out.get("pilot"): print("pilot cached"); return
    t0 = time.perf_counter()
    r = battle(ctx, ctx.start_pastes, WORK / "pilot", None)
    turns = [x.get("turns") for x in r]
    ratio = [ob_ratio(t) for t in ctx.start]
    ok = [i for i, t in enumerate(turns) if t is not None]
    tt = np.array([turns[i] for i in ok]); rr = np.array([ratio[i] for i in ok]); yy = np.array([ctx.start_y[i] for i in ok])
    edges = {"ratio": np.linspace(*np.percentile(ratio, [1, 99]), GRID + 1)[1:-1].tolist(),
             "turns": np.linspace(*np.percentile(tt, [1, 99]), GRID + 1)[1:-1].tolist()}
    corr = np.corrcoef(np.stack([tt, rr, yy])).tolist()
    out["pilot"] = dict(turns=turns, edges=edges, corr_turns_ratio_winrate=corr, battles=int(sum(x.get("battles", 0) for x in r)),
                        seconds=time.perf_counter() - t0)
    save(out); print("pilot", json.dumps({k: v for k, v in out["pilot"].items() if k != "turns"}, indent=1))

def stage_run(ctx, out, smoke=False):
    from hps_generate import Validator
    A.P0_CKPT = str(CKPT)                                    # restored shared p0 (the original is gone)
    val = Validator()
    edges = out["pilot"]["edges"] if out.get("pilot") else {"ratio": np.linspace(-0.6, 0.4, GRID + 1)[1:-1].tolist(),
                                                             "turns": np.linspace(4, 14, GRID + 1)[1:-1].tolist()}
    start_turn = [t if t is not None else np.nan for t in (out["pilot"]["turns"] if out.get("pilot") else [None] * len(ctx.start))]
    global CHILDREN, BATTLE, PROPOSE, GENS
    seeds, arms = (SEEDS, ("baseline", "dsame")) if not smoke else ([7], ("dsame", "baseline"))
    if smoke: CHILDREN, BATTLE, PROPOSE, GENS = 400, 8, 24, 1
    for seed in seeds:
        mut = mutation_module(seed)
        for arm in arms:
            rk = f"{arm}_{seed}"; run = out["runs"].setdefault(rk, {"gens": {}}) if not smoke else {"gens": {}}
            lab_t, lab_y, lab_turn = list(ctx.start), list(ctx.start_y), list(start_turn)
            seen = {key(p) for p in ctx.start_pastes}
            for g in range(1, GENS + 1):
                gk = str(g)
                if gk in run["gens"]:
                    r = run["gens"][gk]
                    lab_t += [parse_team_text(p) for p in r["pastes"]]; lab_y += r["y"]
                    lab_turn += [t if t is not None else np.nan for t in r["turns"]]; seen |= {key(p) for p in r["pastes"]}
                    continue
                t0 = time.perf_counter()
                if arm == "baseline":
                    pastes, info = gen_baseline(ctx, lab_t, lab_y, seed, g, val, WORK / rk / f"g{g}")
                else:
                    pastes, info = gen_dsame(ctx, lab_t, lab_y, lab_turn, seen, edges, seed, g, val, mut)
                if smoke:
                    print(arm, g, len(pastes), {k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items()}, flush=True)
                    continue
                r = battle(ctx, pastes, WORK / rk / f"g{g}" / "battled", seed)
                keep = [i for i, x in enumerate(r) if x.get("battles")]
                rec = dict(pastes=[pastes[i] for i in keep], y=[r[i]["win_rate"] for i in keep],
                           turns=[r[i].get("turns") for i in keep], info=info, seconds=time.perf_counter() - t0)
                run["gens"][gk] = rec; save(out)
                lab_t += [parse_team_text(p) for p in rec["pastes"]]; lab_y += rec["y"]
                lab_turn += [t if t is not None else np.nan for t in rec["turns"]]; seen |= {key(p) for p in rec["pastes"]}
                print(f"{rk} g{g}: mean {np.mean(rec['y']):.3f} max {max(rec['y']):.3f} · {rec['seconds']:.0f}s · "
                      + ", ".join(f"{k} {v:.1f}" for k, v in info.items() if k.endswith("seconds")), flush=True)
    val.close()

def fresh_score(con, pid, team, note):
    cols = dict(con.execute("SELECT team_id, weight FROM team_set WHERE set_name='top50'")); cols.pop(team, None)
    rows = con.execute("SELECT c.lo, c.hi, SUM(c.lo_wins), SUM(c.hi_wins), SUM(c.battles) FROM contribution c JOIN batch b "
                       "USING (batch_id) WHERE b.policy_id=? AND b.note=? AND (c.lo=? OR c.hi=?) GROUP BY c.lo, c.hi",
                       (pid, note, team, team)).fetchall()
    cells = {}
    for lo, hi, lw, hw, n in rows:
        c = hi if lo == team else lo
        if c in cols and n: cells[c] = ((lw if lo == team else hw), n)
    if not cells: return None, 0
    W = sum(cols[c] for c in cells)
    return sum(cols[c] * w / n for c, (w, n) in cells.items()) / W, sum(n for _, n in cells.values())

def stage_verify(ctx, out):
    con = MDB.connect(); pid = MDB.get_policy(con); val = MDB.Validator()
    for rk, run in out["runs"].items():
        if rk in out["verify"]: continue
        allp, seenk = [], set()
        for y, p in sorted(((y, p) for g in run["gens"].values() for p, y in zip(g["pastes"], g["y"])), key=lambda z: -z[0]):
            k = key(p)                                            # dedupe: one verification per distinct team
            if k not in seenk: seenk.add(k); allp.append((y, p))
        top = allp[:TOPV]
        with con: ids = [MDB.add_team(con, p, f"dsame:{rk}", val)[0] for _, p in top]
        note = f"dsame verify {rk}"
        cols = MDB.set_members(con, "top50")
        # force VERIFY_CELL NEW battles per cell: target = what the team already has + VERIFY_CELL
        # (ensure() only tops up a deficit, so a team already in the database would otherwise get none)
        by_target = {}
        for t in ids:
            have = max([MDB.battles_done(con, pid, t, c) for c in cols if c != t] or [0])
            by_target.setdefault(have + VERIFY_CELL, []).append(t)
        for target, ts in sorted(by_target.items()):
            MDB.ensure(con, pid, ts, cols, target, seed=4242, note=note)
        out["verify"][rk] = [dict(search=y, fresh=fresh_score(con, pid, t, note)[0], fresh_battles=fresh_score(con, pid, t, note)[1],
                                  team_id=t) for (y, _), t in zip(top, ids)]
        save(out); print(rk, [round(v["fresh"], 3) if v["fresh"] is not None else None for v in out["verify"][rk]], flush=True)
    val.close()

def stage_report(ctx, out):
    edges = out["pilot"]["edges"]
    cell = lambda r, t: (int(np.clip(np.digitize(r, edges["ratio"]), 0, GRID - 1)), int(np.clip(np.digitize(t, edges["turns"]), 0, GRID - 1)))
    summ, rng = {}, np.random.default_rng(0)
    for arm in ("baseline", "dsame"):
        rks = [f"{arm}_{s}" for s in SEEDS if f"{arm}_{s}" in out["runs"]]
        fresh = [v["fresh"] for rk in rks for v in out["verify"].get(rk, []) if v["fresh"] is not None]
        teams = [parse_team_text(p) for rk in rks for g in out["runs"][rk]["gens"].values() for p in g["pastes"]]
        div = [DM.measure([parse_team_text(p) for g in out["runs"][rk]["gens"].values() for p in g["pastes"]], ctx.ref) for rk in rks]
        qd = []
        for rk in rks:
            arch = {}
            for g in out["runs"][rk]["gens"].values():
                for p, y, tu in zip(g["pastes"], g["y"], g["turns"]):
                    if tu is None: continue
                    c = cell(ob_ratio(parse_team_text(p)), tu); arch[c] = max(arch.get(c, 0), y)
            qd.append((len(arch) / GRID ** 2, sum(arch.values())))
        infos = [g["info"] for rk in rks for g in out["runs"][rk]["gens"].values()]
        cost = sum(i.get("surrogate_train_seconds", 0) + i.get("ridge_seconds", 0) + i.get("resteer_seconds", 0) for i in infos)
        summ[arm] = dict(
            verified_mean=float(np.mean(fresh)) if fresh else None, verified_best=float(max(fresh)) if fresh else None, verified_n=len(fresh),
            search_gen_means={g: float(np.mean([np.mean(out["runs"][rk]["gens"][g]["y"]) for rk in rks])) for g in map(str, range(1, GENS + 1))
                              if all(g in out["runs"][rk]["gens"] for rk in rks)},
            unique_fraction_48=float(np.mean([d["categorical_48"]["unique_fraction"] for d in div])),
            novel_unique_fraction_48=float(np.mean([d["categorical_48"]["novel_unique_fraction"] for d in div])),
            distinct_compositions=float(np.mean([d["composition"]["unique_count"] for d in div])),
            largest_composition_share=float(np.mean([d["composition"]["largest_share"] for d in div])),
            coverage=float(np.mean([q[0] for q in qd])), qd_score=float(np.mean([q[1] for q in qd])),
            model_cost_seconds_total=float(cost), model_cost_seconds_per_gen=float(cost / max(len(infos), 1)),
            proposal_seconds_per_gen=float(np.mean([i.get("proposal_seconds", i.get("inner_seconds", 0)) for i in infos])),
            evaluated=len(teams))
    a = [v["fresh"] for s in SEEDS for v in out["verify"].get(f"dsame_{s}", []) if v["fresh"] is not None]
    b = [v["fresh"] for s in SEEDS for v in out["verify"].get(f"baseline_{s}", []) if v["fresh"] is not None]
    if a and b:
        boot = [np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b))) for _ in range(10_000)]
        ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
        summ["delta_dsame_minus_baseline"] = float(np.mean(a) - np.mean(b)); summ["delta_ci95"] = ci
        summ["verdict"] = ("deep-surrogate MAP-Elites found stronger teams at this budget" if ci[0] > 0 else
                           "deep-surrogate MAP-Elites found weaker teams at this budget" if ci[1] < 0 else
                           "no detectable difference at this budget")
    out["summary"] = summ; save(out); print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    stage = sys.argv[1]
    ctx = Ctx(); out = load()
    {"validate": lambda: stage_validate(ctx, out), "pilot": lambda: stage_pilot(ctx, out),
     "smoke": lambda: stage_run(ctx, out, smoke=True), "run": lambda: stage_run(ctx, out),
     "verify": lambda: stage_verify(ctx, out), "report": lambda: stage_report(ctx, out)}[stage]()
