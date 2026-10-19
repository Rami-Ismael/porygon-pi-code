"""Four ways to keep composition diverse across generations (2026-10-08). Pre-registration: docs/diversity-methods.md.

Arms on the cem_v3 loop: control, taboo (reject seen compositions), peel (ban the winning core's species),
island (4 walled-off loops pinned to distant meta teams), big (4,096 proposals, top-128), dpp (4,096 proposals,
determinantal point process picks the 128). Every arm shares cem_v3's generation 0; generations 1-6; seeds 1-3.
Each generation logs the Metrics-for-Diversity measures on the first 512 accepted proposals.

    python battle_supervisor.py diversity_loops.py run --then rebattle --then report
    DIVM_SMOKE=1 python diversity_loops.py run        # tiny plumbing check, own file
(run with /tmp/vgc-pilot/.venv/bin/python)
"""
import json, math, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cem_v3 as CV
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import matchup_db as MDB
import diffusion as D
from diversity_metrics import Reference, measure
from hps_generate import Validator
from corpus import parse_team_text, norm

ARMS = ["control", "taboo", "peel", "island", "big", "dpp"]
SEEDS, GENS, PROPOSE, BIG, BATTLE, K_ISL, ISL_BATTLE = [int(x) for x in os.environ.get("DIVM_SEEDS", "1").split(",")], 6, 512, 4096, 128, 4, 32
CUT, TABOO_K, DPP_TAU, UNRANKED, TOP, TOP_CELL = 0.531, 2, 0.05, 32, 16, 4
CORE = {"basculegion", "chandelure", "floetteeternal", "garchomp", "kingambit", "whimsicott"}
ALL_SEEDS = [1, 2, 3]
OUT = L.RES / ("diversity_methods.json" if SEEDS == [1] else f"diversity_methods_s{SEEDS[0]}.json")
SMOKE = bool(os.environ.get("DIVM_SMOKE"))
if SMOKE:
    GENS, PROPOSE, BIG, BATTLE, ISL_BATTLE, UNRANKED, TOP, TOP_CELL = 1, 16, 48, 4, 2, 2, 2, 1
    SEEDS = [1]; CV.FT_EPOCHS = 1; OUT = L.RES / "diversity_methods_smoke.json"
    ARMS = ["taboo", "peel", "island", "dpp"]


def save(out): json.dump(out, open(OUT, "w"))
MERGED = L.RES / "diversity_methods_all.json"


def merged():
    """Combine the per-seed files (seed 1 in diversity_methods.json, seeds 2-3 in _s2/_s3)."""
    out = None
    for f in [L.RES / "diversity_methods.json"] + [L.RES / f"diversity_methods_s{s}.json" for s in (2, 3)]:
        if not f.exists(): continue
        o = json.load(open(f))
        if out is None: out = o; out["config"]["seeds"] = ALL_SEEDS
        else: out["gens"].update(o["gens"])
    return out
def comp(slots): return tuple(sorted(norm(s["species"]) for s in slots))
def comp_paste(p):
    t = parse_team_text(p); return tuple(sorted(norm(s["species"]) for s in t)) if len(t) == 6 else None


# ---------------------------------------------------------------- measurement (Metrics for Diversity)
def diversity(stream, ref, tried):
    """stream: list of (paste, slots) in sampling order; the first PROPOSE accepted, before ranking/dedup."""
    teams = [s for _, s in stream[:PROPOSE]]
    m = measure(teams, ref, expected_n=PROPOSE, attempts=tried)
    freq = sorted((f["count"] for f in m["composition"]["frequencies"]), reverse=True)
    n = sum(freq); H = -sum(c / n * math.log(c / n) for c in freq) if n else None
    top = sorted(m["composition"]["frequencies"], key=lambda f: -f["count"])[:3]
    return dict(status=m["status"], accepted=m["sampling"]["accepted"], attempts=tried,
                distinct_compositions=len(freq), effective=math.exp(H) if H is not None else None,
                largest_share=m["composition"]["largest_share"], top_compositions=top,
                composition_novel_draw=m["composition"]["novel_draw_fraction"],
                **{f"{k}_{f}": m[k][f] for k in ("categorical_48", "with_spreads")
                   for f in ("unique_fraction", "repeat_fraction", "novel_draw_fraction", "novel_unique_fraction")})


# ---------------------------------------------------------------- proposing
class Taboo:
    def __init__(self, seen): self.seen, self.here = set(seen), Counter()
    def accept(self, c):
        if c in self.seen or self.here[c] >= TABOO_K: return False
        self.here[c] += 1; return True


def masked(G, V, ban=(), pin=None):
    """Species-column logit mask: remove banned species; pin the chosen columns to their seed species."""
    sid = {s: i for i, s in enumerate(V.itos["species"])}
    bad = [sid[s] for s in ban if s in sid]; orig = G.ok
    def ok(x, c):
        o = orig(x, c)
        if c in FD.SPECIES_COLS and bad:
            o = o.clone(); o[:, bad] = False; o[~o.any(1)] = orig(x, c)[~o.any(1)]
        if pin and c in pin:
            o = torch.zeros_like(o); o[:, pin[c]] = True
        return o
    G.ok = ok
    return lambda: G.__dict__.pop("ok", None)


def propose(G, sur, n_accept, rng, taboo=None, batch=96):
    val = Validator(); raw, kept, tried, cap = [], [], 0, 8 * n_accept
    try:
        while len(kept) < n_accept and tried < cap:
            for r in G.sample(batch, 0.5, 2.0, 1.0, 1.0, sur, sur.lam).cpu().numpy():
                tried += 1
                p, slots = G.to_paste(r, rng, repair=True)
                if val(p) is not None: continue
                raw.append((p, slots))
                if taboo is None or taboo.accept(comp(slots)): kept.append((p, slots))
                if len(kept) >= n_accept: break
    finally:
        val.close()
    return raw, kept, tried


def dpp_map(q, B, k):
    """Greedy MAP of the DPP with L = diag(q) B B' diag(q) (Chen, Zhang & Zhou 2018, fast greedy)."""
    Phi = q[:, None] * B; n = len(q); d2 = (Phi ** 2).sum(1); C = np.zeros((k, n)); sel = []
    j = int(np.argmax(d2))
    while len(sel) < k and d2[j] > 1e-10:
        sel.append(j); i = len(sel) - 1
        e = (Phi @ Phi[j] - C[:i].T @ C[:i, j]) / math.sqrt(d2[j]); C[i] = e; d2 = d2 - e ** 2; d2[sel] = -np.inf
        j = int(np.argmax(d2))
    return sel


def select_dpp(stream, mu, k):
    sp = [frozenset(comp(s)) for _, s in stream]
    vocab = {x: i for i, x in enumerate(sorted(set().union(*sp)))}
    B = np.zeros((len(sp), len(vocab)))
    for i, s in enumerate(sp): B[i, [vocab[x] for x in s]] = 1 / math.sqrt(6)
    mu = np.asarray(mu); q = np.exp((mu - mu.max()) / DPP_TAU)
    sel = dpp_map(q, B, k)
    filled = len(sel)
    if filled < k:                                   # fewer than k distinct rosters: fill by ridge score
        rest = [i for i in np.argsort(-mu) if i not in set(sel)]; sel += rest[:k - filled]
    return sel, filled


# ---------------------------------------------------------------- battling
def battle(con, pastes, tag, seed):
    v = MDB.Validator()
    try: ids = [MDB.add_team(con, p, f"divm:{tag}", v)[0] for p in pastes]
    finally: v.close()
    legal = {t for (t,) in con.execute(f"SELECT team_id FROM team WHERE legal=1 AND team_id IN ({','.join('?' * len(ids))})", ids)}
    MDB.ensure(con, LB.POLICY, sorted(legal), MDB.set_members(con, "top50"), LB.PER_CELL, seed=seed, note=f"divm {tag}")
    sc = [MDB.score(con, LB.POLICY, t) if t in legal else None for t in ids]
    return ids, [s["score"] if s else None for s in sc]


def bstats(y, pastes):
    ok = [(v, p) for v, p in zip(y, pastes) if v is not None]; v = np.array([a for a, _ in ok], float)
    cs = [comp_paste(p) for _, p in ok]
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None,
                p90=float(np.quantile(v, .9)), max=float(v.max()), above_cut=int((v >= CUT).sum()), n=len(v),
                battled_compositions=len({c for c in cs if c}))


def unranked(con, stream, tag, seed, rng):
    idx = rng.choice(min(len(stream), PROPOSE), size=min(UNRANKED, len(stream)), replace=False)
    ps = [stream[i][0] for i in idx]; ids, y = battle(con, ps, f"{tag}:unranked", seed)
    v = [a for a in y if a is not None]
    return dict(mean=float(np.mean(v)), n=len(v), team_id=ids)


# ---------------------------------------------------------------- shared pieces of a generation
def ridge_for(ridge, rows, path):
    loop = [r for r in rows if r["src"] != "anchor"]
    FD.RIDGE = ridge.fit([parse_team_text(r["paste"]) for r in loop], [r["y"] for r in loop], path)
    return FD.Surrogate(CV_V[0])


def top_species(rows, n=16, k=3, exclude=()):
    best = sorted(rows, key=lambda r: -r["y"])[:n]; cnt = Counter()
    for r in best:
        c = comp_paste(r["paste"])
        if c: cnt.update(set(c))
    return [s for s, _ in cnt.most_common() if s not in exclude][:k]


def keep_unbanned(rows, ban):
    return [r for r in rows if (comp_paste(r["paste"]) or ()) and not (set(comp_paste(r["paste"])) & set(ban))]


CV_V = [None]


# ---------------------------------------------------------------- one arm-seed
def run_arm(arm, seed, out, base, G, V, tables, ref, con, ridge):
    CV_V[0] = V
    rows = list(base["rows"]); seen = set(base["g0_comps"]); banned = []
    isl = None
    for g in range(1, GENS + 1):
        tag = f"{arm}_s{seed}_g{g}"
        c = out["gens"].get(tag)
        if c is not None:                                           # resume: rebuild state from the saved generation
            rows += [dict(paste=p, y=y, src=f"g{g}") for p, y in zip(c["battled"], c["y"]) if y is not None]
            seen |= {tuple(x) for x in c["stream_comps"]}; banned = c.get("banned", banned)
            if arm == "island": isl = c["_isl"]
            continue
        t0 = time.perf_counter(); rng = np.random.default_rng(seed * 1000 + g); torch.manual_seed(seed * 1000 + g)
        CV.SEED = 1000 * seed; c = {}
        if arm == "island":
            c, new_rows, isl = island_gen(seed, g, rows, isl, out, base, G, V, tables, ref, con, ridge, rng)
        else:
            CV.TAG = f"divm_{arm}_s{seed}"
            if arm == "peel":
                if g in (1, 3, 5):
                    src = {"g0"} if g == 1 else {f"g{g - 2}", f"g{g - 1}"}
                    pool_rows = [r for r in rows if r["src"] in src]
                    banned = banned + top_species(pool_rows, exclude=banned)
                c["banned"] = list(banned)
            el_rows = keep_unbanned(rows, banned) if banned else rows
            el = CV.elites_from(el_rows, V)
            c["elites"] = dict(n=len(el), anchors=sum(e[2] == "anchor" for e in el), cut=min(e[1] for e in el))
            G.f = CV.members(V, tables, el, g)
            sur = ridge_for(ridge, rows, L.RES / f"divm_{arm}_s{seed}_ridge_g{g}.npz")
            restore = masked(G, V, ban=banned)
            try:
                raw, kept, tried = propose(G, sur, BIG if arm in ("big", "dpp") else PROPOSE, rng,
                                           Taboo(seen) if arm == "taboo" else None, batch=192 if arm in ("big", "dpp") else 96)
            finally: restore()
            mu = [float(sur.score(s)) for _, s in kept]
            if arm == "dpp": sel, filled = select_dpp(kept, mu, BATTLE); c["dpp_distinct_filled"] = filled
            else: sel = [int(i) for i in np.argsort(mu)[::-1][:BATTLE]]
            c["div"] = diversity(kept, ref, tried)
            if arm == "taboo": c["div_raw"] = diversity(raw, ref, tried); c["taboo_rejected"] = len(raw) - len(kept)
            c["valid"] = len(raw) / tried
            battled = [kept[i][0] for i in sel]; ids, y = battle(con, battled, tag, seed * 100 + g)
            c.update(battled=battled, team_id=ids, y=y, stats=bstats(y, battled),
                     stream_comps=[list(comp(s)) for _, s in kept[:PROPOSE]], stream_size=len(kept))
            if g in (3, GENS): c["unranked"] = unranked(con, kept, tag, seed * 100 + g, rng)
            new_rows = [dict(paste=p, y=v, src=f"g{g}") for p, v in zip(battled, y) if v is not None]
        c["minutes"] = (time.perf_counter() - t0) / 60
        if arm == "island": c["_isl"] = isl
        out["gens"][tag] = c; save(out)
        rows += new_rows; seen |= {tuple(x) for x in c["stream_comps"]}
        d, s = c["div"], c["stats"]
        print(f"[{tag}] battled {s['mean']:.3f} ± {s['se'] or 0:.3f} max {s['max']:.3f} ≥cut {s['above_cut']} · "
              f"compositions {d['distinct_compositions']}/{d['accepted']} eff {d['effective']:.0f} "
              f"largest {d['largest_share']:.1%} · battled comps {s['battled_compositions']} · {c['minutes']:.1f} min", flush=True)


def island_gen(seed, g, rows, isl, out, base, G, V, tables, ref, con, ridge, rng):
    """K islands, each pinned to a distant meta team, own re-steer on own elites, migration before g3 and g5."""
    sp = lambda r: set(comp_paste(r["paste"]) or ())
    if isl is None:
        anc = [r for r in base["rows"] if r["src"] == "anchor"]
        enc = []
        for r in anc:
            try: V.encode(parse_team_text(r["paste"])); enc.append(r)
            except ValueError: pass
        enc.sort(key=lambda r: -r["y"]); seeds = [enc[0]]
        while len(seeds) < K_ISL:                                   # farthest-first by species Jaccard
            seeds.append(max((r for r in enc if r not in seeds),
                             key=lambda r: min(1 - len(sp(r) & sp(s)) / len(sp(r) | sp(s)) for s in seeds)))
        isl = [dict(seed=s, rows=[s]) for s in seeds]
        for r in base["rows"]:
            if r["src"] == "g0": max(isl, key=lambda i: (len(sp(r) & sp(i["seed"])), rng.random()))["rows"].append(r)
    if g in (3, 5):
        for i, a in enumerate(isl):
            cnt = Counter(x for r in a["rows"] if r["src"] != "anchor" for x in sp(r))
            home = {s for s, _ in cnt.most_common(6)}
            for j, b in enumerate(isl):
                if j == i: continue
                cand = sorted((r for r in b["rows"] if r["src"] not in ("anchor", "mig")), key=lambda r: -r["y"])
                for r in cand:
                    if len(sp(r) & home) <= 2 and r not in a["rows"]: a["rows"].append(dict(r, src="mig")); break
    all_rows = [r for r in rows]
    sur = ridge_for(ridge, all_rows, L.RES / f"divm_island_s{seed}_ridge_g{g}.npz")
    enc_col = [c for c in D.ORDER if c in FD.SPECIES_COLS][:3]
    streams, raws, battled_all, per = [], [], [], []; tried_all = 0; CV.ELITE_MIN = 32
    try:
        for k, a in enumerate(isl):
            CV.TAG = f"divm_island_s{seed}_i{k}"; CV.SEED = 1000 * seed + 100 * k
            el = CV.elites_from(a["rows"], V)
            G.f = CV.members(V, tables, el, g)
            enc_row = V.encode(parse_team_text(a["seed"]["paste"]))
            restore = masked(G, V, pin={c: int(enc_row[c]) for c in enc_col})
            try: raw, kept, tried = propose(G, sur, PROPOSE // K_ISL, rng)
            finally: restore()
            tried_all += tried; mu = [float(sur.score(s)) for _, s in kept]
            sel = [int(i) for i in np.argsort(mu)[::-1][:ISL_BATTLE]]
            streams += kept; raws += raw
            per.append(dict(island=k, elites=len(el), proposals=len(kept), picked=[kept[i][0] for i in sel]))
    finally: CV.ELITE_MIN = 64
    c = dict(div=diversity(streams, ref, tried_all), valid=len(raws) / tried_all)
    battled = [p for x in per for p in x["picked"]]; tag = f"island_s{seed}_g{g}"
    ids, y = battle(con, battled, tag, seed * 100 + g)
    pos = 0; new_rows = []
    for k, x in enumerate(per):
        n = len(x["picked"]); yy = y[pos:pos + n]
        new = [dict(paste=p, y=v, src=f"g{g}") for p, v in zip(x["picked"], yy) if v is not None]
        isl[k]["rows"] += new; new_rows += new; pos += n
        x["picked"] = None; x["mean"] = float(np.mean([v for v in yy if v is not None])); x["qualifying"] = sum(v >= CUT for v in yy if v is not None)
    c.update(battled=battled, team_id=ids, y=y, stats=bstats(y, battled), islands=per,
             stream_comps=[list(comp(s)) for _, s in streams], stream_size=len(streams))
    if g in (3, GENS): c["unranked"] = unranked(con, streams, tag, seed * 100 + g, rng)
    return c, new_rows, isl


# ---------------------------------------------------------------- driver
def run():
    LB._stack(); con = MDB.connect()
    c3 = json.load(open(L.RES / "cem_v3.json")); g0 = c3["gens"]["0"]
    G = FD.Gen("v3"); V, tables = G.V, G.tables
    ref = Reference(G.corpus, scope="corpus", format_id="reg_mb")
    ridge = CV.Ridge()
    rows = list(c3["anchors"]) + [dict(paste=p, y=y, src="g0") for p, y in zip(g0["battled"], g0["y"]) if y is not None]
    base = dict(rows=rows, g0_comps=[comp_paste(p) for p in g0["proposals"] if comp_paste(p)])
    out = json.load(open(OUT)) if OUT.exists() else {"config": dict(arms=ARMS, seeds=SEEDS, gens=GENS, propose=PROPOSE, big=BIG,
        battle=BATTLE, islands=K_ISL, island_battle=ISL_BATTLE, taboo_k=TABOO_K, dpp_tau=DPP_TAU, cut=CUT), "gens": {}}
    out["g0"] = dict(div=None, sets=g0["sets"], stats=g0["stats"]); save(out)
    for seed in SEEDS:
        for arm in ARMS:
            run_arm(arm, seed, out, base, G, V, tables, ref, con, ridge)


def rebattle():
    LB._stack(); out = merged(); con = MDB.connect(); res = {}
    for seed in ALL_SEEDS:
        for arm in ARMS:
            rows = {}
            for g in range(1, GENS + 1):
                c = out["gens"].get(f"{arm}_s{seed}_g{g}")
                if c:
                    for y, t, p in zip(c["y"], c["team_id"], c["battled"]):
                        if y is not None: rows[t] = (y, p)
            top = [t for t, _ in sorted(rows.items(), key=lambda kv: -kv[1][0])[:TOP]]
            MDB.ensure(con, LB.POLICY, top, MDB.set_members(con, "top50"), TOP_CELL, note=f"divm rebattle {arm}_s{seed}")
            res[f"{arm}_s{seed}"] = [dict(team_id=t, label=rows[t][0], paste=rows[t][1], score=MDB.score(con, LB.POLICY, t)["score"]) for t in top]
    out["rebattle"] = res; json.dump(out, open(MERGED, "w"))


def report():
    out = json.load(open(MERGED)) if MERGED.exists() else merged(); cfg = out["config"]; rng = np.random.default_rng(0); G = cfg["gens"]
    get = lambda a, s, g: out["gens"].get(f"{a}_s{s}_g{g}")
    print(f"generation 0 (shared): compositions {out['g0']['sets']}/512, battled {out['g0']['stats']['mean']:.3f}")
    for key, name in (("distinct_compositions", "distinct compositions /512"), ("largest_share", "largest composition share")):
        print(f"\n{name} (mean of seeds)\n{'arm':8s}" + "".join(f"{'g' + str(g):>8s}" for g in range(1, G + 1)))
        for a in cfg["arms"]:
            line = f"{a:8s}"
            for g in range(1, G + 1):
                cs = [get(a, s, g) for s in cfg["seeds"]]
                line += f"{np.mean([c['div'][key] for c in cs]):8.3f}" if all(cs) else f"{'-':>8s}"
            print(line)
    print(f"\nbattled-batch win rate (mean of seeds)\n{'arm':8s}" + "".join(f"{'g' + str(g):>8s}" for g in range(1, G + 1)))
    for a in cfg["arms"]:
        line = f"{a:8s}"
        for g in range(1, G + 1):
            cs = [get(a, s, g) for s in cfg["seeds"]]
            line += f"{np.mean([c['stats']['mean'] for c in cs]):8.3f}" if all(cs) else f"{'-':>8s}"
        print(line)
    summ = {}
    print("\ndecision rule at generation", G)
    for a in cfg["arms"]:
        if a == "control": continue
        cs = [(get(a, s, G), get("control", s, G)) for s in cfg["seeds"]]
        if not all(x and y for x, y in cs): continue
        d_comp = [x["div"]["distinct_compositions"] - y["div"]["distinct_compositions"] for x, y in cs]
        d_share = [x["div"]["largest_share"] - y["div"]["largest_share"] for x, y in cs]
        floor = all(x["div"]["distinct_compositions"] >= out["g0"]["sets"] for x, _ in cs)
        ya = [np.array([v for v in x["y"] if v is not None]) for x, _ in cs]; yc = [np.array([v for v in y["y"] if v is not None]) for _, y in cs]
        point = float(np.mean([p.mean() - q.mean() for p, q in zip(ya, yc)]))
        boots = [np.mean([p[rng.integers(0, len(p), len(p))].mean() - q[rng.integers(0, len(q), len(q))].mean() for p, q in zip(ya, yc)]) for _ in range(4000)]
        lo, hi = np.quantile(boots, [.025, .975])
        ok = all(x > 0 for x in d_comp) and all(x < 0 for x in d_share) and floor and lo > -0.03
        un = [x.get("unranked", {}).get("mean") for x, _ in cs]; unc = [y.get("unranked", {}).get("mean") for _, y in cs]
        cores = {s: len({tuple(comp_paste(p)) for gg in range(1, G + 1) for p, v in zip(get(a, s, gg)["battled"], get(a, s, gg)["y"]) if v is not None and v >= CUT and comp_paste(p)}) for s in cfg["seeds"]}
        off = {s: len({tuple(comp_paste(p)) for gg in range(1, G + 1) for p, v in zip(get(a, s, gg)["battled"], get(a, s, gg)["y"])
                       if v is not None and v >= CUT and comp_paste(p) and len(set(comp_paste(p)) & CORE) <= 3}) for s in cfg["seeds"]}
        summ[a] = dict(win_diff=point, ci=[float(lo), float(hi)], comps_vs_control=d_comp, share_vs_control=d_share,
                       passes=bool(ok), unranked=un, unranked_control=unc, qualifying_cores=cores, off_core_cores=off)
        print(f"{a:7s} compositions vs control {d_comp} · share {[round(x, 3) for x in d_share]} · win {point:+.3f} [{lo:+.3f}, {hi:+.3f}] · "
              f"{'PASSES' if ok else 'fails'} · unranked {un} vs control {unc} · qualifying cores {cores} · off-core {off}")
    if "rebattle" in out:
        print("\nre-battled best of the 16 (196 battles)")
        for a in cfg["arms"]:
            print(f"{a:8s}", [round(max(r['score'] for r in out['rebattle'][f'{a}_s{s}']), 3) for s in cfg["seeds"]])
    out["summary"] = summ; json.dump(out, open(MERGED, "w"))


if __name__ == "__main__":
    {"run": run, "rebattle": rebattle, "report": report}[sys.argv[1]]()
