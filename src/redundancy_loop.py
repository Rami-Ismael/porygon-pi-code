"""Redundancy penalty (maximal marginal relevance) in the elite step. Pre-reg: docs/redundancy-penalty.md.

    python redundancy_loop.py smoke     # tiny, scratch copy of the matchup DB
    python redundancy_loop.py run       # resumable per arm-seed-generation
    python redundancy_loop.py rebattle  # top 8 per arm-seed to 4 battles per opponent
    python redundancy_loop.py report
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hps_generate
# /tmp cleanup broke the /tmp copies (2026-10-04); durable runtime copies, same commits
hps_generate.SHOWDOWN = Path.home() / ".local/share/vgc-pilot-runtime/validator-913da36"
import activesearch as A
import entropyloop as E
import gradguide as GG
import matchup_db as MDB
import diffusion as D
from corpus import parse_team, parse_team_text, norm
from hps_generate import Validator

A.pool.REPO = str(Path.home() / ".local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2")
MDB.SHOWDOWN = hps_generate.SHOWDOWN
REPO = Path("/Users/ramiismael/Documents/code/vgc-team-generator-pilot")
A.P0_CKPT = str(REPO / "results/temperature_p0.pt")         # shared p0 (the original is gone)
OUT = REPO / "results/redundancy_loop.json"
WORK = Path("/tmp/vgc-pilot/redundancy")
ARMS = {"cut": 0.0, "mmr30": 0.30, "mmr60": 0.60}   # 0.10 dropped before any battle: near no-op (see doc amendment)
SEEDS, GENS, PROPOSE, BATTLE = [1, 2, 3], 6, 512, 64
PER_CELL, POLICY, TOP, TOP_CELL = 1, 1, 8, 4
MODE, LAM = E.MODE, E.LAM


def select_mmr(lab_teams, lab_y, k, lam):
    """Greedy maximal marginal relevance: argmax y_i - lam * max_{j kept} shared_species(i, j) / 6."""
    sp = [frozenset(norm(s["species"]) for s in t) for t in lab_teams]
    vocab = {s: i for i, s in enumerate(sorted(set().union(*sp)))}
    B = np.zeros((len(sp), len(vocab)), np.float32)
    for i, s in enumerate(sp):
        B[i, [vocab[x] for x in s]] = 1
    y = np.asarray(lab_y, float); maxsim = np.zeros(len(y)); taken = np.zeros(len(y), bool); idx = []
    for _ in range(min(k, len(y))):
        sc = y - lam * maxsim; sc[taken] = -np.inf
        i = int(np.argmax(sc)); idx.append(i); taken[i] = True
        if lam: maxsim = np.maximum(maxsim, (B @ B[i]) / 6.0)
    return np.array(idx)


def score_batch(con, pastes, origin, val):
    """Add to the matchup matrix (validated), battle one per top-50 column, return DB scores."""
    with con:
        ids = [MDB.add_team(con, p, origin, val)[0] for p in pastes]
    legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
    MDB.ensure(con, POLICY, sorted(legal), MDB.set_members(con, "top50"), PER_CELL, note=origin)
    sc = [MDB.score(con, POLICY, t) if t in legal else None for t in ids]
    return ids, [s["score"] if s and s["score"] is not None else None for s in sc]


def setup():
    cf = A.load_corpus_files(); corpus = [t for _, t in cf]
    V = A.Vocab(corpus); C = D.Constraints(V, A.Legality(corpus))
    look, spreads = A.decode_tables(corpus)
    grid = np.stack([V.encode(t) for t in corpus])
    as1 = json.load(open(REPO / "results/activesearch.json"))
    anchor = [parse_team(f) for f in as1["anchor"]]; anchor_y = [float(v) for v in as1["anchor"].values()]
    g0 = as1["gens"]["gen0"]
    start = anchor + [parse_team_text(p) for p in g0["pastes"]]
    start_y = anchor_y + [float(v) for v in g0["y"]]
    return corpus, V, C, look, spreads, grid, start, start_y, len(anchor)


def run(smoke=False):
    global GENS, PROPOSE, BATTLE
    if smoke:
        GENS, PROPOSE, BATTLE = 1, 8, 2
    corpus, V, C, look, spreads, grid, start, start_y, n_anchor = setup()
    corpus_sets = {A.species_set(t) for t in corpus}
    out_path = WORK / "smoke.json" if smoke else OUT
    out = json.load(open(out_path)) if out_path.exists() and not smoke else {"gens": {}}
    out["config"] = dict(arms=ARMS, seeds=SEEDS, gens=GENS, propose=PROPOSE, battle=BATTLE, per_cell=PER_CELL,
                         policy=POLICY, mode=MODE, lam_guidance=LAM, p0=A.P0_CKPT, rho=A.RHO, elite_min=A.ELITE_MIN)
    con = MDB.connect(); val = Validator()
    try:
        for seed in (SEEDS[:1] if smoke else SEEDS):
            for arm, lam in (list(ARMS.items())[:2] if smoke else ARMS.items()):
                lab_t, lab_y = list(start), list(start_y)
                for g in range(1, GENS + 1):
                    tag = f"{arm}_s{seed}_g{g}"
                    if tag in out["gens"]:
                        r = out["gens"][tag]
                        lab_t += [parse_team_text(p) for p, y in zip(r["pastes"], r["y"]) if y is not None]
                        lab_y += [y for y in r["y"] if y is not None]
                        continue
                    t0 = time.perf_counter()
                    k = max(A.ELITE_MIN, int(A.RHO * len(lab_y)))
                    F_, wv = GG.fit_ridge(lab_t, lab_y, corpus); G = GG.Guide(F_, wv, V, D.DEV)
                    idx = select_mmr(lab_t, lab_y, k, lam)
                    ss = E.selection_stats(idx, None, lab_t, lab_y, n_anchor)
                    model, steps = E.resteer(V, [lab_t[i] for i in idx], seed=100 * seed + g, checkpoint=A.P0_CKPT)
                    files, teams, pastes, validity, diag = GG.propose_guided(
                        model, C, G, V, look, spreads, val, MODE, LAM, PROPOSE, WORK / tag, 1000 * seed + g)
                    mu = F_.mat(teams) @ wv
                    sel = [int(i) for i in np.argsort(-mu)[:BATTLE]]
                    ids, y = score_batch(con, [pastes[i] for i in sel], f"redundancy:{tag}", val)
                    keep = [j for j, v in enumerate(y) if v is not None]
                    st = A.batch_stats([y[j] for j in keep], [teams[sel[j]] for j in keep], corpus_sets, E.REAL_MEAN)
                    st.update(validity=validity, selection=ss, proposal=A.memorisation(V, teams, grid),
                              n_labels_fit=len(lab_y), minutes=(time.perf_counter() - t0) / 60)
                    out["gens"][tag] = dict(stats=st, pastes=[pastes[i] for i in sel], team_id=ids, y=y,
                                            proposal_sets=sorted({"|".join(A.species_set(t)) for t in teams}))
                    json.dump(out, open(out_path, "w"))
                    lab_t += [teams[sel[j]] for j in keep]; lab_y += [y[j] for j in keep]
                    p = st["proposal"]
                    print(f"[{tag}] batch {st['mean']:.3f} ± {st['se'] or 0:.3f} · max {st['max']:.3f} · "
                          f"proposal sets {p['distinct_species_sets']}/{len(teams)} · elite sets {ss['distinct_species_sets']}/{ss['k']} "
                          f"· valid {validity:.2f} · {st['minutes']:.1f} min", flush=True)
    finally:
        val.close()


def rebattle():
    out = json.load(open(OUT)); con = MDB.connect(); res = {}
    for seed in SEEDS:
        for arm in ARMS:
            rows = [(y, tid) for g in range(1, GENS + 1) for y, tid in
                    zip(out["gens"][f"{arm}_s{seed}_g{g}"]["y"], out["gens"][f"{arm}_s{seed}_g{g}"]["team_id"]) if y is not None]
            top = [tid for _, tid in sorted(rows, reverse=True)[:TOP]]
            MDB.ensure(con, POLICY, top, MDB.set_members(con, "top50"), TOP_CELL, note=f"redundancy rebattle {arm}_s{seed}")
            res[f"{arm}_s{seed}"] = [dict(team_id=t, label=next(y for y, tt in rows if tt == t),
                                          **{k: v for k, v in MDB.score(con, POLICY, t).items() if k != "team_id"}) for t in top]
    out["rebattle"] = res; json.dump(out, open(OUT, "w"))
    print("rebattled", len(res), "arm-seeds")


def report():
    out = json.load(open(OUT)); rng = np.random.default_rng(0); G = out["config"]["gens"]
    print(f"{'arm':6s}" + "".join(f"   g{g} win / sets" for g in range(1, G + 1)))
    for arm in ARMS:
        line = f"{arm:6s}"
        for g in range(1, G + 1):
            rs = [out["gens"].get(f"{arm}_s{s}_g{g}") for s in SEEDS]
            if not all(rs): line += "        —        "; continue
            line += f"   {np.mean([r['stats']['mean'] for r in rs]):.3f} / {np.mean([r['stats']['proposal']['distinct_species_sets'] for r in rs]):4.0f}"
        print(line)
    summ = {}
    for arm in ARMS:
        if arm == "cut": continue
        d_w, d_s = [], []
        for s in SEEDS:
            a, c = out["gens"][f"{arm}_s{s}_g{G}"], out["gens"][f"cut_s{s}_g{G}"]
            ya = np.array([v for v in a["y"] if v is not None]); yc = np.array([v for v in c["y"] if v is not None])
            d_w.append((ya, yc)); d_s.append(a["stats"]["proposal"]["distinct_species_sets"] - c["stats"]["proposal"]["distinct_species_sets"])
        point = float(np.mean([ya.mean() - yc.mean() for ya, yc in d_w]))
        boots = [np.mean([ya[rng.integers(0, len(ya), len(ya))].mean() - yc[rng.integers(0, len(yc), len(yc))].mean() for ya, yc in d_w])
                 for _ in range(4000)]
        lo, hi = np.quantile(boots, [.025, .975])
        ok = all(x > 0 for x in d_s) and lo > -0.03
        summ[arm] = dict(win_diff=point, ci=[float(lo), float(hi)], sets_diff_by_seed=d_s, passes=bool(ok))
        print(f"{arm} vs cut at g{G}: win rate {point:+.3f} [{lo:+.3f}, {hi:+.3f}] · proposal sets by seed {d_s} · "
              f"{'PASSES' if ok else 'fails'} the pre-registered rule")
    if "rebattle" in out:
        for arm in ARMS:
            best = [max(r["score"] for r in out["rebattle"][f"{arm}_s{s}"]) for s in SEEDS]
            print(f"rebattle {arm:6s}: best per seed {[round(b, 3) for b in best]} (196 battles each)")
    out["summary"] = summ; json.dump(out, open(OUT, "w"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"smoke": lambda: run(smoke=True), "run": run, "rebattle": rebattle, "report": report}[cmd]()
